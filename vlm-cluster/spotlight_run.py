#!/usr/bin/env python3
"""
SPOTLIGHT — production auto-labeling runner (Stage 2-5).

  Stage 1 (separate): autolabel_sam_raw.py  -> YOLO labels + raw mask sidecars
  Stage 2 THIS FILE : spotlight crop per detection (mask outlined, upscaled)
  Stage 3 THIS FILE : one Gemini Flash-Lite verdict per detection
  Stage 4 THIS FILE : merge rules  (delete rejects, age>12 -> adult, gender wins)
  Stage 5 THIS FILE : EMIT cleaned YOLO labels + an audit trail

Deliberately self-contained: the only local import is api_describers (the
Gemini client). Small helpers copied verbatim from the validated experiment
code are marked with their origin so the lineage stays auditable.

    # verify + label + emit, with a hard spend ceiling
    python3 spotlight_run.py --images /workspace/open-images-v7/images/train \\
        --raw-labels /workspace/oiv7_raw/train \\
        --out /workspace/oiv7_clean --max-spend 500

    python3 spotlight_run.py --emit-only --raw-labels ... --out ...  # re-emit, no API
    python3 spotlight_run.py --selftest

Everything is resumable: verdicts.jsonl is appended and flushed per call, so
re-running the same command continues where it stopped and never re-pays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --- frozen production configuration (validated in EXP-2026-10) -------------
PROMPT_VERSION = "spotlight-e1"
PAD = 0.25                 # crop padding around the SAM3 box
MIN_CROP_SIDE = 320        # upscale small crops (free: same image tokens)
OUTLINE_RGB = (0, 220, 90)
OUTLINE_DARK = (0, 70, 20)
OUTLINE_W = 3
CHILD_AGE_MAX = 12         # final Child only if estimated_age <= this
CHILD_BY_AGE_MIN = 1       # --child-by-age floor: 0 and -1 are Gemini's
                           # placeholders for "no idea", not newborns
CLASS_ID = {"Woman": 0, "Man": 1, "Child": 2}
CLASS_NAME = {0: "Woman", 1: "Man", 2: "Child"}   # from run_autolabel_on_manifest
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROMPT = """\
One candidate in this image is outlined in green (it may be several separate
parts — all mark the SAME person). Judge ONLY that candidate.

What counts as a person:
- Any real human, of any age, even partly hidden.
- A printed or photographic depiction of a real human (poster, billboard,
  magazine, screen) COUNTS as a person.
- A doll, mannequin, statue, toy, cartoon or drawing does NOT.
Judge gender from the face and body, never from robes or clothing style.

Reply with ONLY this JSON, no markdown:
{"verdict": "real_person|depiction|not_person",
"gender": "man|woman|unknown",
"age_group": "child|adult|unknown", "estimated_age": <number>,
"highlight_quality": "good|covers_wrong_object|covers_multiple_people",
"confidence": "high|low",
"apparent_race": "white|black|east_asian|southeast_asian|south_asian|central_asian_turkic|middle_eastern_north_african|hispanic_latino|other|unknown",
"head_covering": "none|cap_hat|hijab|niqab|ghutra_keffiyeh|turban|helmet|other",
"exposed_body_parts": ["face","hair","neck","shoulders","arms","hands","chest","midriff","back","legs","knees","feet"] or ["none"]}

"child" means the person looks 12 or younger. "apparent_race" is a coarse
visual grouping from facial features and skin tone — an appearance guess,
never an identity claim. "exposed_body_parts" lists parts showing bare skin.
Use "unknown" only when genuinely not visible. Do not name anyone.
"""

ANALYSIS_KEYS = ["apparent_race", "head_covering", "exposed_body_parts"]


# --- helpers copied verbatim from the validated code -----------------------

def extract_json(text: str) -> dict:
    """From describe.py — tolerant JSON extraction (fences, prose, multiple
    blocks; the LAST complete object wins, per the Qwen self-correction fix)."""
    if not text:
        return {}
    blocks = re.findall(r"\{.*?\}", text, re.S)
    for blk in reversed(blocks or []):
        try:
            return json.loads(blk)
        except ValueError:
            continue
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group())
        except ValueError:
            pass
    return {}


def iou(a, b):
    """From run_model_children.py."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# --- Stage 2: the spotlight crop -------------------------------------------

def load_raw(raw_dir: Path, stem: str):
    """autolabel_sam_raw.py sidecar -> (detections, image w/h) or None."""
    f = raw_dir / f"{stem}.json"
    if not f.exists():
        return None
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # Sidecar is mid-write: the labeler is still running on this image.
        # Treat it as not-ready rather than crashing the whole scan -- the
        # next pass picks it up once the write completes. This is what makes
        # it safe to run verification while labeling is still in progress.
        return None
    return rec


def build_crop(img: Image.Image, box, parts):
    """25%-padded crop, upscaled if small, with the person's exact mask parts
    outlined (two-tone so it reads on any background). Returns (crop, scale)."""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    cx2, cy2 = int(min(img.width, x2 + pw)), int(min(img.height, y2 + ph))
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    scale = 1.0
    if max(crop.size) < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * scale)),
                            max(1, int(crop.height * scale))), Image.LANCZOS)
    loops = []
    for part in (parts or []):
        pts = [((px - cx1) * scale, (py - cy1) * scale) for px, py in part]
        if len(pts) >= 3:
            loops.append([(int(x), int(y)) for x, y in pts] +
                         [(int(pts[0][0]), int(pts[0][1]))])
    if not loops:                       # box-only fallback
        b = [(x1 - cx1) * scale, (y1 - cy1) * scale,
             (x2 - cx1) * scale, (y2 - cy1) * scale]
        loops = [[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]),
                  (b[0], b[1])]]
    d = ImageDraw.Draw(crop)
    for lp in loops:
        d.line(lp, fill=OUTLINE_DARK, width=OUTLINE_W + 2)
    for lp in loops:
        d.line(lp, fill=OUTLINE_RGB, width=OUTLINE_W)
    return crop, scale


def blurriness(img: Image.Image, box):
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    g = np.array(img.crop((x1, y1, x2, y2)).convert("L"))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# --- Stage 3/4: verdict parsing and merge rules ----------------------------

def parse_verdict(text: str):
    obj = extract_json(text)
    if not isinstance(obj, dict) or "verdict" not in obj:
        return None
    v = {"verdict": str(obj.get("verdict", "")).strip().lower(),
         "gender": str(obj.get("gender", "unknown")).strip().lower(),
         "age_group": str(obj.get("age_group", "unknown")).strip().lower(),
         "highlight_quality": str(obj.get("highlight_quality", "good")).strip().lower(),
         "confidence": str(obj.get("confidence",
                                   obj.get("verdict_confidence", "unknown"))).strip().lower()}
    try:
        v["estimated_age"] = float(obj.get("estimated_age"))
    except (TypeError, ValueError):
        v["estimated_age"] = None
    for k in ANALYSIS_KEYS:
        val = obj.get(k)
        if isinstance(val, str):
            val = val.strip()
        elif isinstance(val, list):
            val = [str(x).strip().lower() for x in val if str(x).strip()]
        v[k] = val
    return v


def merge(v, child_by_age=False):
    """Final decision for one detection. SAM3 geometry always wins (v1 policy);
    Gemini decides existence and class only.

    A child needs no gender -- Child is class 2 whatever the gender says -- and
    the age_group branch below already honours that. `child_by_age` extends the
    same policy one step: when Gemini abstained on age_group but still wrote a
    usable estimated_age, take the age rather than delete a real person from
    training for lack of a gender they never needed.

    It is a deliberately narrow rescue, because the age field in these verdicts
    is mostly NOT a reading (measured over the 1,579,649 train verdicts,
    2026-08-18): of the 2,943 dropped detections with estimated_age <= 12,
    1,601 carry estimated_age <= 0 (20 of them -1) inside a verdict that
    abstained on race and body parts too, and 603 carry
    highlight_quality != "good", meaning the crop judged was not this
    detection. Both are excluded. Calling either group Child would teach the
    model that an unreadable person is a child -- an adult that then escapes
    the blur, the one error direction this project cannot afford."""
    if v is None or v["verdict"] not in ("real_person", "depiction"):
        return {"keep": False, "final_class": None}
    age = v["estimated_age"]
    if v["age_group"] == "child" and (age is None or age <= CHILD_AGE_MAX):
        return {"keep": True, "final_class": "Child"}
    if v["gender"] == "man":
        return {"keep": True, "final_class": "Man"}
    if v["gender"] == "woman":
        return {"keep": True, "final_class": "Woman"}
    if child_by_age and child_age_only(v):
        return {"keep": True, "final_class": "Child", "via": "age_only"}
    return {"keep": True, "final_class": "Unknown"}   # dropped at emit


def child_rescue_allowed(r, require_sam_child=False, min_px=0):
    """Record-level guards for the --child-by-age rescue.

    merge() only ever sees the verdict `v`. SAM3's own class and the person's
    pixel height live on the detection RECORD, so they are checked here and
    folded into the flag handed to merge() -- which keeps merge() a pure
    function of the verdict instead of growing a record argument.

    `require_sam_child` demands a SECOND, independent opinion: SAM3 must have
    boxed this person as Child too. Owner's call after reviewing the gallery on
    2026-08-18 -- Gemini's read here is always low-confidence, so requiring the
    detector to agree is what turns a single weak signal into two. It cuts the
    rescue from 1,206 to 493 and raises the median person height from 77 to
    118 px, because SAM3 mostly agrees on the people big enough to see.
    """
    if require_sam_child and r.get("sam_class_id") != CLASS_ID["Child"]:
        return False
    if min_px:
        h = r.get("person_px_height")
        if not isinstance(h, (int, float)) or h < min_px:
            return False
    return True


def child_age_only(v):
    """True when a gender-less verdict still says 'child' credibly enough to
    keep. Separate from merge() so the counting tool and the emit apply one
    definition -- see CHILD_BY_AGE_MIN for why the floor exists."""
    age = v.get("estimated_age")
    return (isinstance(age, (int, float))
            and CHILD_BY_AGE_MIN <= age <= CHILD_AGE_MAX
            and v.get("highlight_quality") == "good")


# --- Stage 5: emit cleaned YOLO labels -------------------------------------

def emit(raw_dir: Path, verdicts_path: Path, out_dir: Path, keep_unknown: bool,
         allow_unverified: bool = False, child_by_age: bool = False,
         require_sam_child: bool = False, child_min_px: int = 0):
    """verdicts_path may be a single file or a run dir containing
    verdicts*.jsonl (one per shard)."""
    """Write cleaned YOLO labels: rejected detections removed, class replaced
    by the merged verdict, SAM3 polygon geometry preserved byte-for-byte."""
    by_img = {}
    files = (sorted(verdicts_path.parent.glob("verdicts*.jsonl"))
             if verdicts_path.name.startswith("verdicts") else [verdicts_path])
    lines_all = []
    for f in files:
        if f.exists():
            lines_all += f.read_text().splitlines()
    for line in lines_all:
        if not line.strip():
            continue
        r = json.loads(line)
        by_img.setdefault(r["image_stem"], {})[r["det_index"]] = r
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "kept": 0, "deleted": 0, "relabeled": 0,
             "dropped_unknown": 0, "no_verdict": 0,
             "images_skipped_unverified": 0, "child_by_age": 0}
    audit = (out_dir / "_audit.jsonl").open("w")
    for txt in sorted(raw_dir.glob("*.txt")):
        stem = txt.stem
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        verdicts = by_img.get(stem, {})
        # An image is only emitted when EVERY detection in it has a verdict.
        # Otherwise the output would silently mix verified labels with raw
        # SAM3 passthrough and look verified when it is not.
        if not allow_unverified and len(verdicts) < len(lines):
            stats["images_skipped_unverified"] += 1
            stats["no_verdict"] += len(lines) - len(verdicts)
            continue
        kept_lines = []
        for i, line in enumerate(lines):
            r = verdicts.get(i)
            if r is None:
                stats["no_verdict"] += 1
                kept_lines.append(line)         # unjudged -> leave untouched
                continue
            m = merge(r.get("v"), child_by_age and child_rescue_allowed(
                r, require_sam_child, child_min_px))
            if not m["keep"]:
                stats["deleted"] += 1
                audit.write(json.dumps({"image": stem, "det": i, "action": "delete",
                                        "was": CLASS_NAME.get(r["sam_class_id"]),
                                        "verdict": (r.get("v") or {}).get("verdict")}) + "\n")
                continue
            if m.get("via") == "age_only":
                # a gender-less child kept on its age alone: always audited,
                # because this is the population the rule exists to recover
                stats["child_by_age"] += 1
                audit.write(json.dumps({
                    "image": stem, "det": i, "action": "child_by_age",
                    "was": CLASS_NAME.get(r["sam_class_id"]),
                    "estimated_age": (r.get("v") or {}).get("estimated_age"),
                    "person_px_height": r.get("person_px_height")}) + "\n")
            if m["final_class"] == "Unknown":
                if not keep_unknown:
                    stats["dropped_unknown"] += 1
                    audit.write(json.dumps({"image": stem, "det": i,
                                            "action": "drop_unknown_gender"}) + "\n")
                    continue
                new_id = r["sam_class_id"]
            else:
                new_id = CLASS_ID[m["final_class"]]
            if new_id != r["sam_class_id"]:
                stats["relabeled"] += 1
                audit.write(json.dumps({"image": stem, "det": i, "action": "relabel",
                                        "was": CLASS_NAME.get(r["sam_class_id"]),
                                        "now": m["final_class"]}) + "\n")
            stats["kept"] += 1
            kept_lines.append(" ".join([str(new_id)] + line.split()[1:]))
        (out_dir / f"{stem}.txt").write_text("\n".join(kept_lines))
        stats["images"] += 1
    audit.close()
    (out_dir / "_emit_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if stats["images_skipped_unverified"]:
        print(f"\n[emit] {stats['images_skipped_unverified']} images were NOT "
              f"emitted because some of their detections have no verdict yet "
              f"({stats['no_verdict']} detections). Finish the run, or pass "
              f"--allow-unverified to emit them with raw SAM3 labels passed "
              f"through (NOT recommended for training data).")
    return stats


# --- the run loop ----------------------------------------------------------

def run(engine, images_dir: Path, raw_dir: Path, out_dir: Path,
        max_spend, rate_in, rate_out, report_every, max_images,
        shards=1, shard_index=0):
    from api_describers import build_engine  # noqa: F401  (import check)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each shard owns its own verdicts file so parallel processes never
    # interleave writes; emit() reads verdicts*.jsonl together.
    vpath = out_dir / (f"verdicts_shard{shard_index}.jsonl" if shards > 1
                       else "verdicts.jsonl")
    done = set()
    for f in sorted(out_dir.glob("verdicts*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["image_stem"], r["det_index"]))
    sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
    (out_dir / "run_meta.json").write_text(json.dumps(
        {"prompt_version": PROMPT_VERSION, "prompt_sha": sha,
         "prompt_text": PROMPT, "model": engine.cost.model_id,
         "pad": PAD, "min_crop_side": MIN_CROP_SIDE,
         "child_age_max": CHILD_AGE_MAX, "images": str(images_dir),
         "raw_labels": str(raw_dir)}, indent=2))

    imgs = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if max_images:
        imgs = imgs[:max_images]
    if shards > 1:
        total = len(imgs)
        imgs = [p for p in imgs
                if int(hashlib.md5(str(p).encode()).hexdigest(), 16)
                % shards == shard_index]
        print(f"[run] shard {shard_index}/{shards}: {len(imgs)} of {total} images")
    t0, n_det, n_img, stop = time.time(), 0, 0, False

    def spend():
        c = engine.cost
        return (c.input_tokens * rate_in + c.output_tokens * rate_out) / 1e6

    with vpath.open("a") as fh:
        for p in imgs:
            rec = load_raw(raw_dir, p.stem)
            if rec is None:
                continue
            n_img += 1
            img = None
            for i, det in enumerate(rec.get("detections", [])):
                if (p.stem, i) in done:
                    continue
                if img is None:
                    img = Image.open(p).convert("RGB")
                parts = [[(x, y) for x, y in part] for part in det.get("parts", [])]
                crop, scale = build_crop(img, det["box"], parts)
                bgr = np.array(crop)[:, :, ::-1].copy()
                t = time.time()
                try:
                    text, ti, to = engine._call(bgr, PROMPT)
                    engine.cost.record(ti, to, time.time() - t)
                except Exception as e:                       # noqa: BLE001
                    engine.cost.record(0, 0, time.time() - t, error=True)
                    print(f"[run] call failed {p.stem}_{i}: {str(e)[:160]}")
                    text = ""
                v = parse_verdict(text)
                fh.write(json.dumps({
                    "image_stem": p.stem, "det_index": i, "image": p.name,
                    "img_wh": [rec.get("width", img.width),
                               rec.get("height", img.height)],
                    "sam_class_id": det["cls"], "sam_conf": det.get("conf"),
                    "box": det["box"], "n_parts": len(parts),
                    "blurriness": round(blurriness(img, det["box"]) or 0, 1),
                    "person_px_height": round(det["box"][3] - det["box"][1], 1),
                    "prompt_sha": sha, "parse_ok": v is not None,
                    "v": v, "raw_text": text}) + "\n")
                fh.flush()
                n_det += 1
                if n_det % report_every == 0:
                    el = max(1e-6, time.time() - t0)
                    st = {"detections": n_det, "images": n_img,
                          "usd_spent": round(spend(), 4),
                          "usd_per_1k": round(spend() / n_det * 1000, 4),
                          "dets_per_min": round(n_det / el * 60, 1),
                          "api_errors": engine.cost.errors,
                          "max_spend": max_spend}
                    (out_dir / "status.json").write_text(json.dumps(st, indent=2))
                    b = (f" · {100*spend()/max_spend:.1f}% of ${max_spend:g}"
                         if max_spend else "")
                    print(f"[run] {n_det} dets · {n_img} imgs · ${spend():.2f} "
                          f"(${st['usd_per_1k']:.2f}/1k) · "
                          f"{st['dets_per_min']:.0f}/min{b}", flush=True)
                if max_spend and spend() >= max_spend:
                    print(f"\n[run] *** SPEND LIMIT ${max_spend:g} REACHED "
                          f"(${spend():.2f}). {n_det} detections saved. "
                          f"Re-run to continue. ***", flush=True)
                    stop = True
                    break
            if img is not None:
                img.close()
            if stop:
                break
    print(f"[run] done: {n_img} images, {n_det} new verdicts, ${spend():.2f}")
    (out_dir / "cost_report.json").write_text(json.dumps(engine.cost.report(), indent=2))


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "img").mkdir(); (root / "raw").mkdir()
        Image.new("RGB", (200, 200), (120,) * 3).save(root / "img/a.jpg")
        (root / "raw/a.txt").write_text(
            "1 0.10 0.10 0.40 0.10 0.40 0.90 0.10 0.90\n"
            "0 0.60 0.60 0.90 0.60 0.90 0.90 0.60 0.90\n")
        json.dump({"image": "a.jpg", "width": 200, "height": 200, "detections": [
            {"cls": 1, "conf": 0.9, "box": [20, 20, 80, 180],
             "parts": [[[20, 20], [80, 20], [80, 180]]]},
            {"cls": 0, "conf": 0.8, "box": [120, 120, 180, 180], "parts": []}]},
            open(root / "raw/a.json", "w"))

        assert merge({"verdict": "not_person", "gender": "u", "age_group": "u",
                      "estimated_age": None})["keep"] is False
        assert merge(parse_verdict('{"verdict":"real_person","gender":"woman",'
                                   '"age_group":"adult","estimated_age":30}')
                     )["final_class"] == "Woman"
        assert merge(parse_verdict('{"verdict":"real_person","gender":"man",'
                                   '"age_group":"child","estimated_age":15}')
                     )["final_class"] == "Man", "teen must default to adult"
        assert merge(parse_verdict('{"verdict":"depiction","gender":"man",'
                                   '"age_group":"child","estimated_age":8}')
                     )["final_class"] == "Child"

        with Image.open(root / "img/a.jpg") as im:
            crop, sc = build_crop(im, [20, 20, 80, 180], [[(20, 20), (80, 20), (80, 180)]])
            assert max(crop.size) >= MIN_CROP_SIDE and sc > 1

        # det 0 -> relabel Man->Woman, det 1 -> deleted
        (root / "verd.jsonl").write_text(
            json.dumps({"image_stem": "a", "det_index": 0, "sam_class_id": 1,
                        "v": {"verdict": "real_person", "gender": "woman",
                              "age_group": "adult", "estimated_age": 30,
                              "highlight_quality": "good", "confidence": "high"}}) + "\n" +
            json.dumps({"image_stem": "a", "det_index": 1, "sam_class_id": 0,
                        "v": {"verdict": "not_person", "gender": "unknown",
                              "age_group": "unknown", "estimated_age": None,
                              "highlight_quality": "good", "confidence": "high"}}) + "\n")
        # partial coverage must NOT emit (silent raw-passthrough guard)
        (root / "partial.jsonl").write_text(
            json.dumps({"image_stem": "a", "det_index": 0, "sam_class_id": 1,
                        "v": {"verdict": "real_person", "gender": "woman",
                              "age_group": "adult", "estimated_age": 30,
                              "highlight_quality": "good",
                              "confidence": "high"}}) + "\n")
        sp = emit(root / "raw", root / "partial.jsonl", root / "out_partial",
                  keep_unknown=False)
        assert sp["images"] == 0 and sp["images_skipped_unverified"] == 1, sp
        assert not (root / "out_partial/a.txt").exists(), "emitted an unverified image!"
        sp2 = emit(root / "raw", root / "partial.jsonl", root / "out_partial2",
                   keep_unknown=False, allow_unverified=True)
        assert sp2["images"] == 1 and sp2["no_verdict"] == 1, sp2

        st = emit(root / "raw", root / "verd.jsonl", root / "out", keep_unknown=False)
        assert st == {"images": 1, "kept": 1, "deleted": 1, "relabeled": 1,
                      "dropped_unknown": 0, "no_verdict": 0,
                      "images_skipped_unverified": 0, "child_by_age": 0}, st
        # --- --child-by-age: a gender-less child kept on its age alone -----
        V = lambda **kw: {"verdict": "real_person", "gender": "unknown",
                          "age_group": "unknown", "estimated_age": 3.0,
                          "highlight_quality": "good", "confidence": "low", **kw}
        # off by default: every existing label file stays reproducible
        assert merge(V())["final_class"] == "Unknown"
        assert merge(V(), True)["final_class"] == "Child"
        assert merge(V(), True)["via"] == "age_only"
        # the guards
        assert merge(V(estimated_age=0.0), True)["final_class"] == "Unknown", \
            "0 is Gemini's placeholder, not a newborn"
        assert merge(V(estimated_age=-1.0), True)["final_class"] == "Unknown"
        assert merge(V(estimated_age=None), True)["final_class"] == "Unknown"
        assert merge(V(estimated_age=13.0), True)["final_class"] == "Unknown", \
            "13 is the teen band, not a child"
        assert merge(V(highlight_quality="covers_wrong_object"),
                     True)["final_class"] == "Unknown", "judged the wrong crop"
        assert merge(V(highlight_quality="covers_multiple_people"),
                     True)["final_class"] == "Unknown"
        # the rule is STRICTLY ADDITIVE: it can only turn a dropped detection
        # into a Child, never change what an already-kept detection was called.
        # Nothing that answers earlier in merge() may move.
        for g in ("man", "woman"):
            for grp, a in (("adult", 30.0), ("unknown", 3.0), ("child", 6.0)):
                base = merge(V(gender=g, age_group=grp, estimated_age=a))
                assert merge(V(gender=g, age_group=grp, estimated_age=a),
                             True) == base, (g, grp, a)
        assert merge({"verdict": "not_person", "gender": "unknown",
                      "age_group": "unknown", "estimated_age": 3.0,
                      "highlight_quality": "good"}, True)["keep"] is False

        # the record-level guards, checked outside merge()
        R = lambda **kw: {"sam_class_id": 2, "person_px_height": 120.0, **kw}
        assert child_rescue_allowed(R())
        assert child_rescue_allowed(R(sam_class_id=1))            # off by default
        assert not child_rescue_allowed(R(sam_class_id=1), require_sam_child=True)
        assert not child_rescue_allowed(R(sam_class_id=0), require_sam_child=True)
        assert child_rescue_allowed(R(sam_class_id=2), require_sam_child=True)
        assert child_rescue_allowed(R(), min_px=120)
        assert not child_rescue_allowed(R(), min_px=121)
        assert not child_rescue_allowed(R(person_px_height=None), min_px=1)
        assert child_rescue_allowed(R(person_px_height=None))     # no floor set

        # and end-to-end through emit(): the dropped toddler comes back as
        # class 2, the unreadable adult next to it still goes
        (root / "raw_kid").mkdir()
        (root / "raw_kid/b.txt").write_text(
            "1 0.10 0.10 0.40 0.10 0.40 0.90 0.10 0.90\n"
            "0 0.60 0.60 0.90 0.60 0.90 0.90 0.60 0.90\n")
        (root / "verd_kid.jsonl").write_text("\n".join(
            json.dumps({"image_stem": "b", "det_index": i, "sam_class_id": c,
                        "v": v}) for i, c, v in [
                (0, 1, V()),                                  # toddler
                (1, 0, V(estimated_age=44.0))]) + "\n")       # unreadable adult
        off = emit(root / "raw_kid", root / "verd_kid.jsonl", root / "off",
                   keep_unknown=False)
        on = emit(root / "raw_kid", root / "verd_kid.jsonl", root / "on",
                  keep_unknown=False, child_by_age=True)
        assert off["dropped_unknown"] == 2 and off["kept"] == 0, off
        assert on["dropped_unknown"] == 1 and on["kept"] == 1, on
        assert on["child_by_age"] == 1, on
        # --child-require-sam: det 0's SAM class is 1 (Man), so the same
        # toddler is NOT rescued once a second opinion is demanded
        sam_on = emit(root / "raw_kid", root / "verd_kid.jsonl", root / "on_sam",
                      keep_unknown=False, child_by_age=True,
                      require_sam_child=True)
        assert sam_on["child_by_age"] == 0 and sam_on["kept"] == 0, sam_on
        assert sam_on["dropped_unknown"] == 2, sam_on
        kid = (root / "on/b.txt").read_text().strip().splitlines()
        assert len(kid) == 1 and kid[0].split()[0] == "2", kid
        # geometry byte-for-byte, only the class id differs
        assert kid[0].split()[1:] == (root / "raw_kid/b.txt").read_text(
            ).splitlines()[0].split()[1:]
        assert json.loads((root / "on/_audit.jsonl").read_text().splitlines()[0]
                          )["action"] == "child_by_age"

        out = (root / "out/a.txt").read_text().strip().split("\n")
        assert len(out) == 1 and out[0].startswith("0 "), out
        # geometry preserved byte-for-byte apart from the class id
        assert out[0].split()[1:] == \
            (root / "raw/a.txt").read_text().splitlines()[0].split()[1:]
    print("spotlight_run.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="Spotlight production runner")
    ap.add_argument("--images"); ap.add_argument("--raw-labels"); ap.add_argument("--out")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--rate-in", type=float, default=0.30)
    ap.add_argument("--rate-out", type=float, default=2.50)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--keep-unknown-gender", action="store_true",
                    help="keep SAM3's class when Gemini can't tell gender "
                         "(default: drop those detections from training)")
    ap.add_argument("--shards", type=int, default=1,
                    help="run N processes in parallel; same command on each, "
                         "only --shard-index differs (recovers throughput when "
                         "API latency is high)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="emit images whose detections were not all verified, "
                         "passing raw SAM3 labels through (NOT for training)")
    ap.add_argument("--emit-only", action="store_true",
                    help="re-emit labels from existing verdicts, no API calls")
    ap.add_argument("--child-require-sam", action="store_true",
                    help="with --child-by-age: only rescue when SAM3 ALSO "
                         "classed the detection Child (two independent "
                         "opinions instead of one low-confidence one)")
    ap.add_argument("--child-min-px", type=int, default=0,
                    help="with --child-by-age: skip rescues whose "
                         "person_px_height is below this")
    ap.add_argument("--child-by-age", action="store_true",
                    help="when Gemini gave no gender but a usable estimated_age "
                         f"({CHILD_BY_AGE_MIN}-{CHILD_AGE_MAX}) and judged the right crop, "
                         "keep the detection as Child instead of dropping it")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.raw_labels and a.out):
        ap.error("--raw-labels and --out are required")
    out = Path(a.out)
    if not a.emit_only:
        if not a.images:
            ap.error("--images is required unless --emit-only")
        import api_describers
        engine = api_describers.build_engine("gemini", a.model,
                                             max_tokens=a.max_tokens)
        if not (0 <= a.shard_index < a.shards):
            ap.error("--shard-index must be in [0, --shards)")
        run(engine, Path(a.images), Path(a.raw_labels), out, a.max_spend,
            a.rate_in, a.rate_out, a.report_every, a.max_images,
            a.shards, a.shard_index)
    emit(Path(a.raw_labels), out / "verdicts.jsonl", out / "labels",
         a.keep_unknown_gender, a.allow_unverified, a.child_by_age,
         a.child_require_sam, a.child_min_px)


if __name__ == "__main__":
    main()
