#!/usr/bin/env python3
"""
HARAMBLUR — post-process: add the disputed bounding box to EXISTING hard-neg
descriptions, without re-running the VLM.

The VLM description is the expensive part and is already done. Drawing the box
is pure geometry (the GT box comes from the dataset label file, matched by the
disputed class). This:
  * redraws crops/<id>.jpg = the full image with the disputed box in red,
  * adds focus_box_xyxy + object.focus + object.focus_ambiguous to each record.

So an old whole-image run becomes reviewable (you can SEE which object the
label/prediction refers to) with zero GPU time. Idempotent — by default it only
touches records that don't already have a focus box (so it won't redo records
produced by the new box-aware describe_hardneg).

    CUDA_VISIBLE_DEVICES="" python annotate_focus.py --in ./hardneg \
        --yaml /workspace/open-images-v7/dataset.yaml

Caveat: this only adds the box overlay. For records made on the WHOLE image, the
description text is still whole-image based — if focus_ambiguous=yes (multiple
same-class boxes) the description may be of the wrong person; review those via
the highlighted thumbnail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import dataset_utils as du
from describe import save_thumb
from describe_hardneg import disputed_box


def main():
    ap = argparse.ArgumentParser(description="post-hoc focus-box annotation")
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--yaml", required=True, help="dataset.yaml (for class names)")
    ap.add_argument("--thumb", type=int, default=320)
    ap.add_argument("--all", action="store_true",
                    help="re-annotate every record (default: only those missing a box)")
    args = ap.parse_args()

    in_dir = Path(args.indir)
    crops_dir = in_dir / "crops"
    jsonl = in_dir / "descriptions.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"No descriptions.jsonl in {in_dir}")

    names = du.class_names(du.load_yaml(Path(args.yaml)))
    name2idx = {v: k for k, v in names.items()}

    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    print(f"[annotate] {len(recs)} records")

    updated = boxed = ambiguous = no_box = missing_img = skipped = 0
    for r in recs:
        if r.get("focus_box_xyxy") is not None and not args.all:
            skipped += 1
            continue
        o = r.get("object", {})
        gt_idx = name2idx.get(o.get("true_label"))
        img = cv2.imread(r["image"]) if r.get("image") else None
        if img is None or gt_idx is None:
            r["focus_box_xyxy"] = None
            o["focus"] = "missing_image" if img is None else "no_class"
            o["focus_ambiguous"] = "no"
            missing_img += 1
            continue
        h, w = img.shape[:2]
        box, amb = disputed_box(Path(r["image"]), gt_idx, w, h)
        if box:
            x1, y1, x2, y2 = box
            ann = img.copy()
            cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), max(2, w // 250))
            save_thumb(ann, crops_dir / f"{r['id']}.jpg", args.thumb)
            r["focus_box_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
            o["focus"] = "box"
            o["focus_ambiguous"] = "yes" if amb else "no"
            boxed += 1
            ambiguous += 1 if amb else 0
        else:
            r["focus_box_xyxy"] = None
            o["focus"] = "whole_image"
            o["focus_ambiguous"] = "no"
            no_box += 1
        r["object"] = o
        updated += 1

    # atomic rewrite
    tmp = jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    tmp.replace(jsonl)

    print(f"[annotate] updated {updated} (skipped {skipped} already-boxed)")
    print(f"[annotate] boxed {boxed} (ambiguous focus: {ambiguous}) · "
          f"no GT box: {no_box} · missing image/class: {missing_img}")
    print(f"[annotate] thumbnails redrawn in {crops_dir}")


if __name__ == "__main__":
    main()
