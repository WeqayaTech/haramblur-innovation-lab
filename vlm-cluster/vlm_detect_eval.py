#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-06: whole-image person detection + classification with
commercial VLMs, scored per dataset.

One frozen prompt (below) asks the model to find EVERY person in the image and
return box + gender + age_group + estimated_age + confidence as JSON. The same
prompt goes to every engine verbatim — only the model changes.

Scoring is per dataset via --gt, matching the component framework:
    --gt none        PASS / object_set — any detection is a false positive
    --gt crowdhuman  odgt vbox ground truth — recall / precision / duplicates
    --gt lagenda     YOLO labels + gt.jsonl — detection recall on the one
                     labeled person, then gender/age scored ONLY on matches
                     (Component 2/3 conditioned on Component 1, as always)

Usage (pod):
    python3 vlm_detect_eval.py --engine claude --model claude-sonnet-5 \
        --images /workspace/datasets/pass_3k --gt none \
        --max-images 3 --seed 42 --out /workspace/exp06/claude/pass

    python3 vlm_detect_eval.py --selftest      # no network, no data

Outputs in --out:
    detections.jsonl  one record per image (raw text + parsed people)
    summary.json      the per-dataset scores + parse/cost stats
    cost_report.json  token usage + USD (null until model_pricing.json filled)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from describe import extract_json
from run_model_children import iou, match_boxes

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# The frozen detection prompt (pre-registered in EXP-2026-06 before running).
# {width}/{height} are the only substitutions. The person ruling matches
# EXP-2026-03 §2.2: printed/photographic depictions count; dolls/statues don't.
# ---------------------------------------------------------------------------

BOX_FORMAT_PIXELS = """\
- "box_2d": the bounding box as [x1, y1, x2, y2] in pixels, where (x1, y1)
  is the top-left corner and (x2, y2) the bottom-right. The image is
  {width} pixels wide and {height} pixels tall."""

# Gemini's NATIVE detection format (per Google's own docs) — asking for it
# makes instruction-compliance and training instinct coincide, eliminating
# the per-response axis flipping observed from gemini-3.6-flash
BOX_FORMAT_GEMINI = """\
- "box_2d": the bounding box as [ymin, xmin, ymax, xmax], each value an
  integer normalized to a 0-1000 scale relative to the image size."""

DETECT_PROMPT = """\
You are annotating images for a person-detection dataset. Find EVERY person
in this image and return a JSON object, nothing else.

What counts as a person:
- Every real human being, of any age, including partially visible or
  occluded people (even if only a head, torso, or limb is visible).
- A printed or photographic depiction of a real human — a poster,
  billboard, magazine cover, or photo within the photo — COUNTS as a person.
- A doll, mannequin, statue, sculpture, toy, cartoon, or drawing is NOT a
  person. Do not include them.
- If the image contains no persons, return an empty list.

For each person report:
{box_format_instruction}
- "gender": "man" or "woman" as your best estimate from visible cues;
  "unknown" ONLY if no gender cue is visible at all (fully covered figure,
  too small, back turned with no other cues).
- "age_group": "child" if the person appears 12 years old or younger,
  otherwise "adult". Use "unknown" ONLY if you truly cannot tell.
- "estimated_age": your single best numeric age estimate (a number, not a
  range). Always give a number even when age_group is "unknown".
- "confidence": "high" if you are sure this is a real person (or printed
  photo of one), "low" if it might be a statue/doll/reflection or is barely
  visible.

Return EXACTLY this JSON structure:
{{"people": [{{"box_2d": {box_example}, "gender": "...",
"age_group": "...", "estimated_age": N, "confidence": "..."}}, ...]}}

Return only the JSON. Do not identify or name anyone.
"""


# ---------------------------------------------------------------------------
# Parsing + normalization
# ---------------------------------------------------------------------------

def parse_detections(text: str, w: int, h: int):
    """Parse model output into a clean people list.

    Returns (people, parse_ok, rescaled). Each person dict:
        {box_xyxy, gender, age_group, estimated_age, confidence}
    Malformed entries are dropped (counted by the caller via len diff of
    raw list vs returned list is not needed — we record drops explicitly).

    Rescale heuristic (pre-registered): if the image is larger than 1024px on
    a side but every coordinate is <= 1000, the model likely used 0-1000
    normalized coords (some vendors' native convention) — rescale instead of
    punishing it. Recorded per image so it's auditable.
    """
    obj = extract_json(text)
    raw = obj.get("people", None)
    if raw is None or not isinstance(raw, list):
        return [], False, False

    boxes = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        b = p.get("box_2d") or p.get("box") or p.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
        except (TypeError, ValueError):
            continue
        boxes.append((p, [x1, y1, x2, y2]))

    rescaled = False
    if boxes:
        peak = max(max(abs(v) for v in b) for _, b in boxes)
        # 0-1000-normalized output is detected two ways: on large images, all
        # coords <= 1000 while the image is bigger; on SMALL images (<=1024px,
        # where <=1000 is also a valid pixel range), coords that exceed the
        # image dimensions while staying <= 1000 give it away (seen from
        # qwen3.7-plus on a 402x600 image: x2=521, y2=893).
        if peak <= 1000 and (max(w, h) > 1024 or peak > max(w, h)):
            rescaled = True
            boxes = [(p, [b[0] * w / 1000, b[1] * h / 1000,
                          b[2] * w / 1000, b[3] * h / 1000]) for p, b in boxes]

    people = []
    for p, (x1, y1, x2, y2) in boxes:
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            continue
        try:
            est_age = float(p.get("estimated_age"))
        except (TypeError, ValueError):
            est_age = None
        people.append({
            "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "gender": str(p.get("gender", "unknown")).strip().lower(),
            "age_group": str(p.get("age_group", "unknown")).strip().lower(),
            "estimated_age": est_age,
            "confidence": str(p.get("confidence", "unknown")).strip().lower(),
        })
    return people, True, rescaled


# ---------------------------------------------------------------------------
# Ground-truth loaders
# ---------------------------------------------------------------------------

def load_odgt(path: Path) -> dict:
    """CrowdHuman odgt → {image_id: {"persons": [xyxy...], "ignores": [xyxy...]}}.
    Uses vbox (visible box), same convention as EXP-2026-03. Boxes tagged
    non-person or marked extra.ignore go to "ignores" — detections landing
    there are excluded from FP counts (standard CrowdHuman practice)."""
    gt = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            persons, ignores = [], []
            for g in rec.get("gtboxes", []):
                x, y, bw, bh = g.get("vbox", g.get("fbox", [0, 0, 0, 0]))
                box = [x, y, x + bw, y + bh]
                if g.get("tag") == "person" and not g.get("extra", {}).get("ignore", 0):
                    persons.append(box)
                else:
                    ignores.append(box)
            gt[rec["ID"]] = {"persons": persons, "ignores": ignores}
    return gt


def load_lagenda_gt(labels_dir: Path, manifest: Path) -> dict:
    """LAGENDA → {image_stem: [{box_xyxy_norm, gt_age, gt_gender}, ...]}.
    gt.jsonl rows carry id "<stem>_<box_index>" + age/gender; the box itself
    is line <box_index> of the YOLO label file (normalized cx cy w h)."""
    meta = {}
    with manifest.open() as fh:
        for line in fh:
            rec = json.loads(line)
            stem, _, idx = rec["id"].rpartition("_")
            meta.setdefault(stem, {})[int(idx)] = rec

    gt = {}
    for stem, rows in meta.items():
        lf = labels_dir / f"{stem}.txt"
        if not lf.exists():
            continue
        lines = lf.read_text().split("\n")
        people = []
        for idx, rec in rows.items():
            if idx >= len(lines) or not lines[idx].strip():
                continue
            parts = lines[idx].split()
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            people.append({
                "box_norm": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                "gt_age": rec["gt_age"],
                "gt_gender": rec["gt_gender"],   # "M" / "F"
            })
        if people:
            gt[stem] = people
    return gt


# ---------------------------------------------------------------------------
# Per-dataset scoring
# ---------------------------------------------------------------------------

def as_det_tuples(people):
    """match_boxes expects dets whose [1:5] slice is the box."""
    return [("person", *p["box_xyxy"], p) for p in people]


def score_negatives(records):
    """PASS / object_set: every detection is a false positive."""
    n_img = len(records)
    fps = sum(len(r["people"]) for r in records)
    fps_high = sum(1 for r in records for p in r["people"]
                   if p["confidence"] == "high")
    return {
        "mode": "negatives",
        "images": n_img,
        "images_with_fp": sum(1 for r in records if r["people"]),
        "false_persons_total": fps,
        "false_persons_high_conf": fps_high,
        "per_image": [{"image": r["image"], "n_fp": len(r["people"]),
                       "labels": [(p["gender"], p["age_group"], p["confidence"])
                                  for p in r["people"]]} for r in records],
    }


CONF_RANK = {"high": 2, "unknown": 1, "low": 0}   # VLMs give no numeric score


def average_precision(records, gt, iou_thr: float) -> float | None:
    """Single-class AP at one IoU threshold, VOC-style all-point interpolation.

    Caveat (documented in the experiment doc): our VLMs emit only a 2-level
    confidence ("high"/"low"), so the ranking — and therefore the PR curve —
    is coarse. Valid for comparing models run with the same prompt; NOT
    comparable to COCO mAPs of detectors with continuous scores.
    """
    dets = []      # (conf_rank, record_idx, person)
    total_gt = 0
    entries = []
    for i, r in enumerate(records):
        entry = gt.get(Path(r["image"]).stem)
        entries.append(entry)
        if entry is None:
            continue
        total_gt += len(entry["persons"])
        for p in r["people"]:
            dets.append((CONF_RANK.get(p["confidence"], 1), i, p))
    if total_gt == 0:
        return None
    dets.sort(key=lambda t: -t[0])

    matched = [set() for _ in records]
    curve = []                       # (recall, precision) after each det
    tp = fp = 0
    for _, i, p in dets:
        entry = entries[i]
        if entry is None:
            continue
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(entry["persons"]):
            if j in matched[i]:
                continue
            v = iou(p["box_xyxy"], g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr:
            matched[i].add(best_j)
            tp += 1
        else:
            # detections on ignore regions are neither TP nor FP
            if any(iou(p["box_xyxy"], ib) >= iou_thr for ib in entry["ignores"]):
                continue
            fp += 1
        curve.append((tp / total_gt, tp / (tp + fp)))

    ap = 0.0
    prev_r = 0.0
    for k, (r, _) in enumerate(curve):
        if r > prev_r:
            ap += (r - prev_r) * max(pr for _, pr in curve[k:])
            prev_r = r
    return round(ap, 4)


def box_tightness(records, gt, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Decompose 'missed at IoU 0.5' into loose-box vs truly-undetected.

    recall_at_iou: recall re-matched at each threshold — rising steeply as the
    threshold relaxes means the model FOUND people but boxed them loosely;
    flat means they were never detected (no threshold fixes that).
    matched_iou_*: fit quality of successful matches at 0.5 (mean / median /
    share at COCO-strict >=0.75).
    Crowd caveat (EXP-2026-03 §2.7): overlapping GT means loose boxes can
    match a neighbor (identity swap) — counting metrics are swap-proof, the
    IoU distribution can be slightly flattered; worst at the loosest
    thresholds, which is why the sweep and distribution ship together.
    """
    total_gt = 0
    matches_at = {t: 0 for t in thresholds}
    ious_at_50 = []
    for r in records:
        entry = gt.get(Path(r["image"]).stem)
        if entry is None:
            continue
        total_gt += len(entry["persons"])
        dets = as_det_tuples(r["people"])
        for t in thresholds:
            m = match_boxes(entry["persons"], dets, t)
            matches_at[t] += len(m)
            if t == 0.5:
                ious_at_50 += [v for (_, v) in m.values()]
    if total_gt == 0:
        return None
    dist = sorted(ious_at_50)
    n = len(dist)
    return {
        "recall_at_iou": {f"{t:.1f}": round(matches_at[t] / total_gt, 4)
                          for t in thresholds},
        "matched_iou_mean": round(sum(dist) / n, 3) if n else None,
        "matched_iou_median": round(dist[n // 2], 3) if n else None,
        "matched_iou_pct_ge_0.75": round(100 * sum(1 for v in dist
                                                   if v >= 0.75) / n, 1) if n else None,
    }


def score_crowdhuman(records, odgt_path: Path, match_iou: float):
    gt = load_odgt(odgt_path)
    tot_gt = tot_det = tot_match = tot_ignored = 0
    per_image = []
    for r in records:
        entry = gt.get(Path(r["image"]).stem)
        if entry is None:
            per_image.append({"image": r["image"], "error": "no GT in odgt"})
            continue
        dets = as_det_tuples(r["people"])
        matched = match_boxes(entry["persons"], dets, match_iou)
        used = {id(m[0][5]) for m in matched.values()}
        unmatched = [p for p in r["people"] if id(p) not in used]
        # unmatched dets overlapping an ignore region don't count as FPs
        ignored = [p for p in unmatched
                   if any(iou(p["box_xyxy"], ib) >= match_iou
                          for ib in entry["ignores"])]
        fps = len(unmatched) - len(ignored)
        tot_gt += len(entry["persons"])
        tot_det += len(r["people"])
        tot_match += len(matched)
        tot_ignored += len(ignored)
        per_image.append({
            "image": r["image"], "gt_persons": len(entry["persons"]),
            "detected": len(r["people"]), "matched": len(matched),
            "false_positives": fps, "ignored_region_hits": len(ignored),
        })
    ap50 = average_precision(records, gt, 0.50)
    ap75 = average_precision(records, gt, 0.75)
    ap_sweep = [average_precision(records, gt, t / 100)
                for t in range(50, 100, 5)]
    ap_coco = (round(sum(ap_sweep) / len(ap_sweep), 4)
               if all(a is not None for a in ap_sweep) else None)
    return {
        "mode": "crowdhuman", "match_iou": match_iou,
        "images": len(records),
        "gt_persons": tot_gt, "detections": tot_det, "matched": tot_match,
        "recall": round(tot_match / tot_gt, 4) if tot_gt else None,
        "precision": (round(tot_match / (tot_det - tot_ignored), 4)
                      if (tot_det - tot_ignored) else None),
        "AP50": ap50, "AP75": ap75, "AP_coco_50_95": ap_coco,
        "ap_note": "AP ranked on 2-level VLM confidence (high/low) — "
                   "comparable across our models, not to continuous-score "
                   "detector mAPs",
        "box_tightness": box_tightness(records, gt),
        "per_image": per_image,
    }


def score_lagenda(records, labels_dir: Path, manifest: Path, match_iou: float):
    gt = load_lagenda_gt(labels_dir, manifest)
    found = missed = 0
    gender_ok = gender_bad = gender_unk = 0
    age_ok = age_bad = age_unk = 0
    adult_as_child = 0
    per_person = []
    for r in records:
        entry = gt.get(Path(r["image"]).stem)
        if not entry:
            continue
        w, h = r["width"], r["height"]
        gt_boxes = [[g["box_norm"][0] * w, g["box_norm"][1] * h,
                     g["box_norm"][2] * w, g["box_norm"][3] * h] for g in entry]
        matched = match_boxes(gt_boxes, as_det_tuples(r["people"]), match_iou)
        for gi, g in enumerate(entry):
            gt_child = g["gt_age"] <= 12
            gt_gender = "man" if g["gt_gender"] == "M" else "woman"
            row = {"image": r["image"], "gt_age": g["gt_age"],
                   "gt_gender": gt_gender}
            if gi not in matched:
                missed += 1
                row["status"] = "missed"
                per_person.append(row)
                continue
            found += 1
            pred = matched[gi][0][5]
            row.update(status="matched", iou=round(matched[gi][1], 3),
                       pred_gender=pred["gender"], pred_age_group=pred["age_group"],
                       pred_estimated_age=pred["estimated_age"])
            # Component 3 — gender, scored only on detected people
            if pred["gender"] == "unknown":
                gender_unk += 1
            elif pred["gender"] == gt_gender:
                gender_ok += 1
            else:
                gender_bad += 1
            # Component 2 — age group at the production <=12 cutoff
            if pred["age_group"] == "unknown":
                age_unk += 1
            elif (pred["age_group"] == "child") == gt_child:
                age_ok += 1
            else:
                age_bad += 1
                if not gt_child and pred["age_group"] == "child":
                    adult_as_child += 1   # the direction that escapes the blur
            per_person.append(row)
    match_ious = sorted(r["iou"] for r in per_person if r.get("iou") is not None)
    return {
        "mode": "lagenda", "match_iou": match_iou,
        "gt_persons": found + missed, "found": found, "missed": missed,
        "gender": {"correct": gender_ok, "wrong": gender_bad, "unknown": gender_unk},
        "age_group_at_12": {"correct": age_ok, "wrong": age_bad,
                            "unknown": age_unk,
                            "adult_labeled_child": adult_as_child},
        "matched_iou_mean": (round(sum(match_ious) / len(match_ious), 3)
                             if match_ious else None),
        "per_person": per_person,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def select_images(images_dir: Path, n: int, seed: int):
    imgs = sorted(p for p in images_dir.rglob("*")
                  if p.suffix.lower() in IMG_EXTS)
    random.Random(seed).shuffle(imgs)
    return imgs[:n] if n > 0 else imgs


def run(engine, image_paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    jsonl = out_dir / "detections.jsonl"
    with jsonl.open("w") as fh:
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[detect] unreadable image skipped: {p}")
                continue
            h, w = img.shape[:2]
            gem = getattr(run, "_box_prompt", "pixels") == "gemini"
            box_instr = (BOX_FORMAT_GEMINI if gem
                         else BOX_FORMAT_PIXELS.format(width=w, height=h))
            prompt = DETECT_PROMPT.format(
                box_format_instruction=box_instr,
                box_example="[ymin, xmin, ymax, xmax]" if gem else "[x1, y1, x2, y2]")
            t_before = engine.cost.errors
            text, _, _ = _safe_call(engine, img, prompt)
            people, parse_ok, rescaled = parse_detections(text, w, h)
            rec = {"image": p.name, "width": w, "height": h,
                   "n_people": len(people), "people": people,
                   "parse_ok": parse_ok, "rescaled": rescaled,
                   "api_error": engine.cost.errors > t_before,
                   "raw_text": text}
            fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            print(f"[detect] {p.name}: {len(people)} people "
                  f"(parse_ok={parse_ok}, rescaled={rescaled})")
    return records


TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "overloaded", "high demand",
                     "timeout", "timed out", "connection",
                     # seen intermittently from OpenAI mid-run with identical
                     # calls succeeding before/after — treat as transient
                     "insufficient permissions")


def _safe_call(engine, img, prompt, retries=2, backoffs=(10, 30)):
    """engine._call with record-and-degrade behavior + retry on transient
    errors (503/overloaded/timeouts). Hard failures (bad key, quota limit 0,
    invalid model) are not retried — they won't heal in 30 seconds."""
    import time as _t
    for attempt in range(retries + 1):
        t0 = _t.time()
        try:
            text, in_tok, out_tok = engine._call(img, prompt)
            engine.cost.record(in_tok, out_tok, _t.time() - t0)
            return text, in_tok, out_tok
        except Exception as e:  # noqa: BLE001 — degrade to empty, never crash the run
            msg = str(e)
            transient = any(m in msg for m in TRANSIENT_MARKERS)
            if transient and attempt < retries:
                wait = backoffs[min(attempt, len(backoffs) - 1)]
                print(f"[detect] {engine.cost.model_id} transient error, "
                      f"retry {attempt + 1}/{retries} in {wait}s: {msg[:120]}")
                _t.sleep(wait)
                continue
            engine.cost.record(0, 0, _t.time() - t0, error=True)
            print(f"[detect] {engine.cost.model_id} call failed: {msg[:300]}")
            return "", 0, 0


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-06 VLM detection eval")
    ap.add_argument("--engine", choices=["openai", "gemini", "claude", "qwenapi"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", help="dataset images dir")
    ap.add_argument("--gt", choices=["none", "crowdhuman", "lagenda"],
                    default="none")
    ap.add_argument("--odgt", help="CrowdHuman annotation_val.odgt")
    ap.add_argument("--labels", help="LAGENDA YOLO labels dir")
    ap.add_argument("--manifest", help="LAGENDA gt.jsonl")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="output-token cap per call; must fit every person in "
                         "a crowd (~70 tokens each) PLUS thinking-model "
                         "reasoning, which spends from the same budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--box-prompt", choices=["pixels", "gemini"], default="pixels",
                    help="box-format instruction variant: 'pixels' = x-first "
                         "pixel coords (the frozen default); 'gemini' = the "
                         "y-first 0-1000 format Gemini is trained on (parse "
                         "its output with reparse_boxes --write yxyx_1000)")
    ap.add_argument("--media-resolution", choices=["low", "medium", "high"],
                    default=None,
                    help="Gemini-only: request a MediaResolution mode instead "
                         "of the API default used by every pre-EXP-2026-09 "
                         "run. 'high' = more image tokens (finer effective "
                         "resolution, higher input cost).")
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable model reasoning where the API allows it "
                         "(Gemini thinking_budget=0, GPT reasoning_effort="
                         "minimal, Qwen enable_thinking=false; no-op for "
                         "Claude). Cuts output-token cost; accuracy impact "
                         "is the experiment.")
    ap.add_argument("--out", default="./detect_run")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not (args.engine and args.images):
        raise SystemExit("--engine and --images are required (or --selftest)")
    if args.gt == "crowdhuman" and not args.odgt:
        raise SystemExit("--gt crowdhuman needs --odgt")
    if args.gt == "lagenda" and not (args.labels and args.manifest):
        raise SystemExit("--gt lagenda needs --labels and --manifest")

    import api_describers
    engine = api_describers.build_engine(args.engine, args.model,
                                         max_tokens=args.max_tokens,
                                         no_thinking=args.no_thinking,
                                         media_resolution=args.media_resolution)
    out_dir = Path(args.out)
    paths = select_images(Path(args.images), args.max_images, args.seed)
    print(f"[detect] {engine.cost.model_id} on {len(paths)} images "
          f"from {args.images} (seed {args.seed})")
    run._box_prompt = args.box_prompt
    records = run(engine, paths, out_dir)

    if args.gt == "none":
        summary = score_negatives(records)
    elif args.gt == "crowdhuman":
        summary = score_crowdhuman(records, Path(args.odgt), args.match_iou)
    else:
        summary = score_lagenda(records, Path(args.labels),
                                Path(args.manifest), args.match_iou)
    summary["model_id"] = engine.cost.model_id
    summary["parse_failures"] = sum(1 for r in records if not r["parse_ok"])
    summary["api_errors"] = engine.cost.errors
    summary["rescaled_images"] = sum(1 for r in records if r["rescaled"])

    # The production question: what does labeling cost with this system?
    # Two unit prices — per image (fixed: image+prompt tokens dominate) and
    # per person labeled (variable: ~60-70 output tokens per person entry).
    cost = engine.cost.report()
    n_img = len(records)
    n_persons = sum(len(r["people"]) for r in records)
    usd = cost["cost_usd"]
    summary["labeling_economics"] = {
        "images": n_img,
        "persons_labeled": n_persons,
        "input_tokens_per_image": round(cost["input_tokens"] / n_img) if n_img else None,
        "output_tokens_per_image": round(cost["output_tokens"] / n_img) if n_img else None,
        "usd_per_1000_images": (round(usd / n_img * 1000, 2)
                                if usd is not None and n_img else None),
        "usd_per_1000_persons": (round(usd / n_persons * 1000, 2)
                                 if usd is not None and n_persons else None),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "cost_report.json").write_text(json.dumps(cost, indent=2))
    print(f"[detect] summary -> {out_dir/'summary.json'}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_image", "per_person")}, indent=2))
    print(f"[detect] cost -> {json.dumps(cost)}")


# ---------------------------------------------------------------------------
# Selftest — no network, no data
# ---------------------------------------------------------------------------

def selftest():
    import numpy as np

    # 1. parsing + pixel passthrough
    text = ('{"people": [{"box_2d": [10, 20, 110, 220], "gender": "woman", '
            '"age_group": "adult", "estimated_age": 34, "confidence": "high"}]}')
    ppl, ok, rescaled = parse_detections(text, 640, 480)
    assert ok and not rescaled and len(ppl) == 1
    assert ppl[0]["box_xyxy"] == [10, 20, 110, 220], ppl

    # 2. 0-1000 rescale heuristic on a big image
    ppl, ok, rescaled = parse_detections(text, 2000, 1500)
    assert rescaled and ppl[0]["box_xyxy"] == [20, 30, 220, 330], ppl

    # 3. empty list + junk entries
    ppl, ok, _ = parse_detections('{"people": []}', 640, 480)
    assert ok and ppl == []
    ppl, ok, _ = parse_detections('{"people": [{"box_2d": [1,2], "x": 1}, 5]}',
                                  640, 480)
    assert ok and ppl == []
    ppl, ok, _ = parse_detections("no json here", 640, 480)
    assert not ok

    # 4. negatives scoring
    recs = [{"image": "a.jpg", "people": ppl_from(
                [(0, 0, 10, 10, "man", "adult", "high")]), "parse_ok": True},
            {"image": "b.jpg", "people": [], "parse_ok": True}]
    s = score_negatives(recs)
    assert s["false_persons_total"] == 1 and s["images_with_fp"] == 1
    assert s["false_persons_high_conf"] == 1

    # 5. lagenda matching end-to-end via temp files
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "img1.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        (td / "gt.jsonl").write_text(json.dumps(
            {"id": "img1_0", "image": "img1.jpg",
             "gt_age": 30, "gt_gender": "F"}) + "\n")
        rec = {"image": "img1.jpg", "width": 100, "height": 100,
               "people": ppl_from([(25, 25, 75, 75, "woman", "adult", "high")])}
        s = score_lagenda([rec], td, td / "gt.jsonl", 0.5)
        assert s["found"] == 1 and s["gender"]["correct"] == 1, s
        assert s["age_group_at_12"]["correct"] == 1, s
        # adult read as child = the bad direction
        rec["people"][0]["age_group"] = "child"
        s = score_lagenda([rec], td, td / "gt.jsonl", 0.5)
        assert s["age_group_at_12"]["adult_labeled_child"] == 1, s

    print("[vlm_detect_eval] selftest OK")


def ppl_from(tuples):
    return [{"box_xyxy": [x1, y1, x2, y2], "gender": g, "age_group": a,
             "estimated_age": 30.0, "confidence": c}
            for x1, y1, x2, y2, g, a, c in tuples]


if __name__ == "__main__":
    main()
