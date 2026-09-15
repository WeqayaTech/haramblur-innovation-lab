#!/usr/bin/env python3
"""Negatives arm of haramblur_holdout from a raw dump: over the `randoms__*` (person-free) images,
count images with >=1 kept detection at conf >= 0.25 / 0.45 and the box totals.  usage: randoms_fp.py <raw_dir> <out.json>"""
import json, os, sys

raw, out = sys.argv[1], sys.argv[2]
ths = (0.25, 0.45)
n = 0; img_hit = {t: 0 for t in ths}; boxes = {t: 0 for t in ths}
for f in os.listdir(raw):
    if not f.startswith("randoms__") or not f.endswith(".json"): continue
    n += 1
    dets = [d for d in json.load(open(os.path.join(raw, f)))["detections"] if d.get("kept")]
    for t in ths:
        k = sum(1 for d in dets if d["conf"] >= t)
        boxes[t] += k; img_hit[t] += 1 if k else 0
res = {"n_images": n}
for t in ths:
    res[f"img_fp_rate@{t}"] = round(img_hit[t] / n, 4) if n else None
    res[f"boxes_per_100@{t}"] = round(100 * boxes[t] / n, 2) if n else None
json.dump(res, open(out, "w"), indent=1); print(res)
