#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-08: crop-level verifier-gate evaluation.

Feeds SAM3's actual detections (conf-logged polygon labels from EXP-2026-07)
as padded crops to a VLM with the pipeline's arbitration prompt, and scores
the verdicts against what the crop is known to be:

    --expect reject   FP arms (objects / PASS): every crop is a false positive
                      by construction -> verdict "not_person" = gate success
    --expect keep     TP arm (crowd): crops are GT-matched real people
                      (matched inside this script, same convention as
                      conf_zoom_report) -> "real_person"/"depiction" = success

Usage (pod):
    python3 gate_eval.py --engine gemini --model gemini-3.5-flash-lite \
        --labels /workspace/exp07_conf/sam_labels/objects \
        --images /workspace/datasets/object_set \
        --expect reject --pad 0.25 \
        --out /workspace/exp08/lite/objects

    # TP arm: add odgt matching + sampling
    python3 gate_eval.py --engine gemini --model gemini-3.5-flash-lite \
        --labels /workspace/exp07_conf/sam_labels/crowd \
        --images /workspace/datasets/crowdhuman/Images_sample500 \
        --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
        --expect keep --only-matched --max-crops 800 --seed 42 \
        --out /workspace/exp08/lite/crowd_tp

    python3 gate_eval.py --selftest      # no network, no data

Outputs in --out: verdicts.jsonl (per crop: source, box, sam class+conf,
verdict, gender, ages, raw_text), summary.json, cost_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2

from describe import extract_json
from run_model_children import iou, match_boxes

CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child"}
KEEP_VERDICTS = {"real_person", "depiction"}

GATE_PROMPT = """\
You are verifying a single detection for a person-detection dataset.
This crop was flagged by an automated detector as possibly containing a person.

First decide what this is:
- "real_person": a real human being, of any age, even partially visible
- "depiction": a printed or photographic image of a real human — a poster,
  billboard, magazine cover, screen, or photo within the photo
- "not_person": a doll, mannequin, statue, sculpture, toy, cartoon,
  drawing, anime figure, or anything else that is not a real human
  or photographic depiction of one

If real_person or depiction, then classify:
- "gender": "man" or "woman", from visible cues (face, build, facial hair,
  clothing). Note: men in traditional Gulf dress (thobe, ghutra/keffiyeh)
  are men — judge by face and facial hair, not by the robe.
- "estimated_age": single best numeric age estimate.
- "age_group": "child" ONLY if this person is clearly 12 years old or
  younger (prepubescent). If the person could plausibly be a teenager or
  older, answer "adult". When uncertain between child and adult, answer
  "adult".
- "confidence": "high" or "low" for the overall judgment.

Return only JSON:
{"verdict": "real_person|depiction|not_person", "gender": "man|woman|unknown",
 "estimated_age": N, "age_group": "child|adult", "confidence": "high|low"}
"""


def parse_labels(label_dir: Path):
    """{stem: [(cls, conf_or_None, box_norm_xyxy), ...]} from polygon labels
    (trailing conf optional — even token count means it is present)."""
    out = {}
    for lf in sorted(Path(label_dir).rglob("*.txt")):
        dets = []
        for line in lf.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            cls = int(float(t[0]))
            if len(t) % 2 == 0:                      # cls + 2N coords + conf
                conf, coords = float(t[-1]), t[1:-1]
            else:                                    # cls + 2N coords
                conf, coords = None, t[1:]
            xs = [float(v) for v in coords[0::2]]
            ys = [float(v) for v in coords[1::2]]
            if not xs or not ys:
                continue
            dets.append((cls, conf, [min(xs), min(ys), max(xs), max(ys)]))
        if dets:
            out[lf.stem] = dets
    return out


def build_crop_list(args):
    """[(crop_id, image_path, cls, conf, box_px), ...] per the arm's rules."""
    labels = parse_labels(Path(args.labels))
    img_index = {p.stem: p for p in Path(args.images).rglob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")}
    odgt = {}
    if args.odgt:
        with open(args.odgt) as fh:
            for line in fh:
                rec = json.loads(line)
                persons = [[g["vbox"][0], g["vbox"][1],
                            g["vbox"][0] + g["vbox"][2], g["vbox"][1] + g["vbox"][3]]
                           for g in rec.get("gtboxes", [])
                           if g.get("tag") == "person"
                           and not g.get("extra", {}).get("ignore", 0)]
                odgt[rec["ID"]] = persons

    crops = []
    for stem in sorted(labels):
        if stem not in img_index:
            continue
        path = img_index[stem]
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = labels[stem]
        det_tuples = [("d", b[0] * w, b[1] * h, b[2] * w, b[3] * h, cls, conf)
                      for cls, conf, b in dets]
        if args.only_matched:
            if stem not in odgt:
                continue
            matched = match_boxes(odgt[stem], det_tuples, 0.5)
            det_tuples = [m[0] for m in matched.values()]
        for i, dt in enumerate(det_tuples):
            crops.append((f"{stem}_{i}", path, dt[5], dt[6],
                          [dt[1], dt[2], dt[3], dt[4]]))
    if args.max_crops and len(crops) > args.max_crops:
        crops = random.Random(args.seed).sample(crops, args.max_crops)
    return crops


def cut_crop(img, box, pad: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def run_gate(engine, crops, pad, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (out_dir / "verdicts.jsonl").open("w") as fh:
        cache_path, cache_img = None, None
        for n, (cid, path, cls, conf, box) in enumerate(crops, 1):
            if path != cache_path:
                cache_path, cache_img = path, cv2.imread(str(path))
            crop = cut_crop(cache_img, box, pad)
            if crop is None:
                continue
            t0 = time.time()
            try:
                text, in_tok, out_tok = engine._call(crop, GATE_PROMPT)
                engine.cost.record(in_tok, out_tok, time.time() - t0)
            except Exception as e:  # noqa: BLE001 — degrade, never crash
                engine.cost.record(0, 0, time.time() - t0, error=True)
                text = ""
                print(f"[gate] call failed on {cid}: {str(e)[:150]}")
            obj = extract_json(text)
            rec = {"crop_id": cid, "image": path.name,
                   "sam_class": CLASS_NAMES.get(cls, str(cls)), "sam_conf": conf,
                   "box_px": [round(v, 1) for v in box],
                   "verdict": str(obj.get("verdict", "PARSE_FAIL")).strip().lower(),
                   "gender": str(obj.get("gender", "unknown")).strip().lower(),
                   "estimated_age": obj.get("estimated_age"),
                   "age_group": str(obj.get("age_group", "unknown")).strip().lower(),
                   "confidence": str(obj.get("confidence", "unknown")).strip().lower(),
                   "raw_text": text}
            fh.write(json.dumps(rec) + "\n")
            results.append(rec)
            if n % 50 == 0:
                print(f"[gate] {n}/{len(crops)} crops")
    return results


def summarize(results, expect: str):
    n = len(results)
    verdict_counts = {}
    for r in results:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1
    kept = sum(1 for r in results if r["verdict"] in KEEP_VERDICTS)
    rejected = verdict_counts.get("not_person", 0)
    parse_fail = sum(1 for r in results if r["verdict"] == "parse_fail")
    s = {"crops": n, "expect": expect, "verdicts": verdict_counts,
         "kept": kept, "rejected": rejected, "parse_fail": parse_fail}
    if n:
        if expect == "reject":
            s["fp_kill_rate_pct"] = round(100 * rejected / n, 1)
            s["fp_survivors"] = kept          # FPs the gate would let into training
            by_cls = {}
            for r in results:
                if r["verdict"] in KEEP_VERDICTS:
                    by_cls[r["sam_class"]] = by_cls.get(r["sam_class"], 0) + 1
            s["survivors_by_sam_class"] = by_cls
        elif expect == "keep":
            s["tp_keep_rate_pct"] = round(100 * kept / n, 1)
            s["false_rejections"] = rejected  # real people the gate deleted
    return s


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-08 crop-level gate eval")
    ap.add_argument("--engine", choices=["gemini", "openai", "claude", "qwenapi"],
                    default="gemini")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--labels", help="SAM3 polygon label dir")
    ap.add_argument("--images", help="matching images dir")
    ap.add_argument("--odgt", help="CrowdHuman odgt (needed with --only-matched)")
    ap.add_argument("--only-matched", action="store_true",
                    help="keep only GT-matched detections (the verified-TP arm)")
    ap.add_argument("--expect", choices=["reject", "keep", "none"], default="none")
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--max-crops", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default="./gate_run")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.labels and args.images):
        raise SystemExit("--labels and --images required (or --selftest)")
    if args.only_matched and not args.odgt:
        raise SystemExit("--only-matched needs --odgt")

    import api_describers
    engine = api_describers.build_engine(args.engine, args.model,
                                         max_tokens=args.max_tokens)
    crops = build_crop_list(args)
    print(f"[gate] {engine.cost.model_id}: {len(crops)} crops "
          f"(pad {args.pad}, expect {args.expect})")
    out_dir = Path(args.out)
    results = run_gate(engine, crops, args.pad, out_dir)

    summary = summarize(results, args.expect)
    summary["model_id"] = engine.cost.model_id
    summary["pad"] = args.pad
    cost = engine.cost.report()
    summary["cost_per_crop_usd"] = (round(cost["cost_usd"] / len(results), 6)
                                    if cost["cost_usd"] and results else None)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "cost_report.json").write_text(json.dumps(cost, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[gate] cost -> {json.dumps(cost)}")


def selftest():
    import numpy as np
    import tempfile

    class FakeCost:
        def record(self, *a, **k): pass
    class FakeEngine:
        cost = FakeCost()
        def __init__(self): self.cost = type("C", (), {
            "model_id": "fake", "record": lambda *a, **k: None})()
        def _call(self, img, prompt):
            return ('{"verdict": "not_person", "gender": "unknown", '
                    '"estimated_age": 0, "age_group": "adult", '
                    '"confidence": "high"}', 100, 20)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "img").mkdir(); (td / "lab").mkdir()
        cv2.imwrite(str(td / "img" / "a.jpg"),
                    np.full((200, 300, 3), 180, dtype="uint8"))
        # polygon + conf (even count) and one without conf (odd count)
        (td / "lab" / "a.txt").write_text(
            "0 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8 0.91\n"
            "1 0.1 0.1 0.3 0.1 0.3 0.3\n")
        labels = parse_labels(td / "lab")
        assert len(labels["a"]) == 2
        assert labels["a"][0][1] == 0.91 and labels["a"][1][1] is None

        args = argparse.Namespace(labels=str(td / "lab"), images=str(td / "img"),
                                  odgt=None, only_matched=False, max_crops=0, seed=42)
        crops = build_crop_list(args)
        assert len(crops) == 2, crops
        out = td / "run"
        results = run_gate(FakeEngine(), crops, 0.25, out)
        assert len(results) == 2 and all(r["verdict"] == "not_person" for r in results)
        s = summarize(results, "reject")
        assert s["fp_kill_rate_pct"] == 100.0 and s["fp_survivors"] == 0
        s2 = summarize(results, "keep")
        assert s2["tp_keep_rate_pct"] == 0.0 and s2["false_rejections"] == 2
        assert (out / "verdicts.jsonl").exists()
    print("[gate_eval] selftest OK")


if __name__ == "__main__":
    main()
