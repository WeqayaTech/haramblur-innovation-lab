#!/usr/bin/env python3
"""
describe HARD NEGATIVES (confident misclassifications).

Input: a disagreements.jsonl (model vs label), each record:
    {image_path, score, num_disagreements, max_confidence,
     disagreements:[{gt_class, pred_class, confidence, iou}, ...]}

For each case we describe the most prominent person in the (whole) image with the
same person schema, and tag it with the CONFUSION PAIR `<gt>_as_<pred>` (truth
labeled as what the model predicted). Then analyze.py / cluster.py reveal what
features characterize each confident misclassification — e.g. "Child_as_Man"
crops are mostly tall + short hair, "Man_as_Woman" are long hair + no beard.

    # extract the metadata file once (no unzip needed):
    python3 -c "import zipfile; zipfile.ZipFile('/workspace/label_review/review_compressed.zip').extract('disagreements.jsonl','/workspace/label_review/')"

    python describe_hardneg.py --yaml /workspace/open-images-v7/dataset.yaml \
        --disagreements /workspace/label_review/disagreements.jsonl \
        --max-records 5000 --batch-size 16 --out ./hardneg

Then:
    CUDA_VISIBLE_DEVICES="" python analyze.py --in ./hardneg --out ./hardneg/analysis
    CUDA_VISIBLE_DEVICES="" python cluster.py --in ./hardneg --out ./hardneg/clusters
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2

import dataset_utils as du
from describe import (MockDescriber, QwenDescriber, OBJECT_SCHEMA, schema_str,
                      load_done, fmt_dur, save_thumb)

# Prompt for the focused crop of the disputed detection.
HN_OBJECT_PROMPT = (
    "This crop shows the person a detector flagged in a label review (its class "
    "prediction disagreed with the dataset label). Describe THIS person. Judge "
    "every attribute from visible cues only; use 'unknown' when unsure. Do not "
    "identify individuals.\n"
    "- apparent_age_band: pick the closest band as a best estimate — infant (<1), "
    "toddler (1-3), child (4-9), preteen (10-12), teenager (13-17), young_adult "
    "(18-29), adult (30-49), middle_aged (50-64), elderly (65-79), senior (80+).\n"
    "- age_read_confidence / gender_read_confidence: 'ambiguous' ONLY when the "
    "broad life stage / apparent gender genuinely cannot be read.\n"
    "- facial_hair: none/stubble/moustache/short_beard/full_beard. "
    "head_covering / garment_type: name the actual garment. "
    "skin_tone_mst: Monk 1 (lightest)-10 (darkest); set skin_tone_confidence="
    "'uncertain_lighting' if exposure is poor.\n"
    "Return a single JSON object with EXACTLY these keys:\n{schema}\n"
    "Return only the JSON."
)


def conf_band(c: float) -> str:
    if c >= 0.97:
        return "very_high(>=.97)"
    if c >= 0.93:
        return "high(.93-.97)"
    return "moderate(<.93)"


def top_disagreement(rec: dict) -> dict:
    ds = rec.get("disagreements") or []
    return max(ds, key=lambda d: d.get("confidence", 0)) if ds else {}


def label_path_for_image(image_path: Path) -> Path:
    """Open Images / Ultralytics layout: .../images/<split>/x.jpg -> labels/.../x.txt"""
    s = str(image_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    return Path(s).with_suffix(".txt")


def disputed_box(image_path: Path, gt_class, w, h):
    """Find the GT box of the disputed class. Returns (xyxy or None, ambiguous).
    ambiguous = there were multiple boxes of that class (can't be 100% sure which)."""
    boxes = du.yolo_boxes(label_path_for_image(image_path), w, h)
    same = [(x1, y1, x2, y2) for (c, x1, y1, x2, y2) in boxes if c == gt_class]
    if not same:
        return None, False
    same.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)  # largest
    return same[0], (len(same) > 1)


def main():
    ap = argparse.ArgumentParser(description="describe hard negatives")
    ap.add_argument("--yaml", required=True, help="dataset.yaml (for class names)")
    ap.add_argument("--disagreements", required=True, help="disagreements.jsonl")
    ap.add_argument("--out", default="./hardneg")
    ap.add_argument("--max-records", type=int, default=0, help="0 = all (sorted by score desc)")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--thumb", type=int, default=320,
                    help="thumbnail size (larger so the highlighted box is visible)")
    args = ap.parse_args()

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out / "descriptions.jsonl"
    done = load_done(jsonl)
    if done:
        print(f"[hardneg] resuming — {len(done)} already described")

    names = du.class_names(du.load_yaml(Path(args.yaml)))
    print(f"[hardneg] classes={names}")

    # load + sort disagreements by score (highest-confidence failures first)
    recs = []
    for line in Path(args.disagreements).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    recs.sort(key=lambda r: r.get("score", 0), reverse=True)

    # build pending work list (skip done, missing images, low confidence)
    pending = []
    for r in recs:
        d = top_disagreement(r)
        if d.get("confidence", 0) < args.min_confidence:
            continue
        img_path = Path(r["image_path"])
        stem = img_path.stem
        if stem in done:
            continue
        pending.append((img_path, stem, d))
        if args.max_records and len(pending) >= args.max_records:
            break
    total = len(pending)
    print(f"[hardneg] {total} cases to describe")
    if total == 0:
        print("[hardneg] nothing to do.")
        return

    if args.mock:
        engine = MockDescriber()
        print("[hardneg] MOCK engine")
    else:
        engine = QwenDescriber(model_id=args.model, device=args.device,
                               max_pixels=args.max_pixels,
                               max_new_tokens=args.max_new_tokens)
        print(f"[hardneg] Qwen2.5-VL on {engine.device} (batch={args.batch_size})")

    n = 0
    t0 = time.time()
    f = jsonl.open("a")
    for start in range(0, total, args.chunk):
        chunk = pending[start:start + args.chunk]
        full_imgs, crops, anns, metas = [], [], [], []
        for img_path, stem, d in chunk:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            box, ambiguous = disputed_box(img_path, d.get("gt_class"), w, h)
            if box:
                x1, y1, x2, y2 = box
                px1, py1, px2, py2 = du.pad_box(x1, y1, x2, y2, w, h, 0.3)
                crop = img[py1:py2, px1:px2]
                if crop.size == 0:
                    crop = img
                ann = img.copy()  # full image with the disputed box drawn
                cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 0, 255), max(2, w // 250))
                fb = [int(x1), int(y1), int(x2), int(y2)]
            else:  # no GT box of that class found -> fall back to whole image
                crop, ann, fb, ambiguous = img, img, None, False
            full_imgs.append(img)
            crops.append(crop)
            anns.append(ann)
            metas.append((img_path, stem, d, fb, ambiguous))
        if not full_imgs:
            continue

        scenes = engine.scene_batch(full_imgs, args.batch_size)   # scene from full image
        objs = engine.describe_batch([(c, "person") for c in crops],  # desc from focused crop
                                     args.batch_size,
                                     prompt_template=HN_OBJECT_PROMPT)

        for (img_path, stem, d, fb, ambiguous), scene, obj, ann in zip(
                metas, scenes, objs, anns):
            pred = names.get(d.get("pred_class"), str(d.get("pred_class")))
            gt = names.get(d.get("gt_class"), str(d.get("gt_class")))
            conf = float(d.get("confidence", 0))
            obj = dict(obj)
            obj["model_pred"] = pred
            obj["true_label"] = gt
            obj["conf_band"] = conf_band(conf)
            obj["focus"] = "box" if fb else "whole_image"
            obj["focus_ambiguous"] = "yes" if ambiguous else "no"
            # thumbnail = full image with the disputed box highlighted
            save_thumb(ann, crops_dir / f"{stem}.jpg", args.thumb)
            f.write(json.dumps({
                "id": stem,
                "image": str(img_path),
                "class": f"{gt}_as_{pred}",          # confusion pair = main axis
                "confidence": conf,
                "iou": float(d.get("iou", 0)),
                "score": float(d.get("confidence", 0)),
                "focus_box_xyxy": fb,
                "scene": scene,
                "object": obj,
            }) + "\n")
            n += 1
        f.flush()

        elapsed = time.time() - t0
        rate = n / elapsed if elapsed else 0
        eta = (total - n) / rate if rate else 0
        print(f"[hardneg] {n}/{total} ({100*n/total:.1f}%) "
              f"· {rate:.2f} rec/s · elapsed {fmt_dur(elapsed)} "
              f"· ETA {fmt_dur(eta)}", flush=True)

    f.close()
    print(f"[hardneg] done: {n} new -> {jsonl}")


if __name__ == "__main__":
    main()
