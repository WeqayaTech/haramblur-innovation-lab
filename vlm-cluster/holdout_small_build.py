#!/usr/bin/env python3
"""Mine the QA holdout for clearly-small people and emit a scorable slice.

Rule (states its height space explicitly): a person is SMALL when its
full-label bbox height, converted to model-input pixels
(native_h * 640 / max(W, H)), is <= 96 px — the same convention as
smallperson_v1. Everyone else in a member image becomes an IGNORE region
(never deleted), and the original unknown-gender ignore rows are carried
over, so the score is a pure small-person number.

Outputs under /workspace/deploycmp/holdout_small/:
  labels/<stem>.txt   original small-person rows, verbatim
  ignore/<stem>.txt   original non-small rows + original ignore rows
  raw_<alias>/        per-model sidecar symlink views (map_eval walks PRED stems)
  census.json
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOLD = Path("/workspace/datasets/haramblur_holdout/labeling/full")
GT = HOLD / "labels_std"
IGN = HOLD / "ignore"
DIMS_RAW = Path("/workspace/holdout_eval/v11n_shipped/raw")  # width/height source
OUT = Path("/workspace/deploycmp/holdout_small")
ALIASES = ["v11n_shipped", "y26n_sop50", "y26n_warm50",
           "gelan_r4fw_v2", "gelan_r6_v2"]
BAND_PX = 96.0  # model-input pixels at imgsz 640

(OUT / "labels").mkdir(parents=True, exist_ok=True)
(OUT / "ignore").mkdir(exist_ok=True)


def row_height_frac(line):
    """Normalized bbox height of one YOLO row (box or polygon). None if bad."""
    parts = line.split()
    if len(parts) < 5:
        return None, None
    try:
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None, None
    if len(vals) == 4:  # cx cy w h
        return cls, vals[3]
    ys = vals[1::2]
    if len(ys) < 3:
        return None, None
    return cls, max(ys) - min(ys)


def dims(stem):
    try:
        d = json.loads((DIMS_RAW / (stem + ".json")).read_text())
        return stem, d["width"], d["height"]
    except Exception:
        return stem, None, None


label_files = sorted(GT.glob("*.txt"))
stems = [f.stem for f in label_files]
with ThreadPoolExecutor(16) as ex:
    dim_map = {s: (w, h) for s, w, h in ex.map(dims, stems)}

census = {"band_px": BAND_PX, "space": "model-input (native_h*640/max(W,H))",
          "n_images_total": len(label_files), "no_dims": 0,
          "n_people_total": 0, "n_small": 0, "n_small_le64": 0,
          "n_small_le48": 0, "small_by_class": {"0": 0, "1": 0, "2": 0},
          "small_by_collection": {}, "member_images": 0,
          "members_by_collection": {}}
members = []
for f in label_files:
    stem = f.stem
    w, h = dim_map.get(stem, (None, None))
    if not w or not h:
        census["no_dims"] += 1
        continue
    scale = 640.0 / max(w, h)
    small_rows, big_rows = [], []
    for line in f.read_text().splitlines():
        cls, hfrac = row_height_frac(line)
        if hfrac is None:
            continue
        census["n_people_total"] += 1
        h_in = hfrac * h * scale
        if h_in <= BAND_PX:
            small_rows.append(line)
            census["n_small"] += 1
            census["small_by_class"][str(cls)] = \
                census["small_by_class"].get(str(cls), 0) + 1
            coll = stem.split("__")[0]
            census["small_by_collection"][coll] = \
                census["small_by_collection"].get(coll, 0) + 1
            if h_in <= 64:
                census["n_small_le64"] += 1
            if h_in <= 48:
                census["n_small_le48"] += 1
        else:
            big_rows.append(line)
    if not small_rows:
        continue
    members.append(stem)
    coll = stem.split("__")[0]
    census["members_by_collection"][coll] = \
        census["members_by_collection"].get(coll, 0) + 1
    (OUT / "labels" / (stem + ".txt")).write_text("\n".join(small_rows) + "\n")
    ign_rows = list(big_rows)
    orig_ign = IGN / (stem + ".txt")
    if orig_ign.exists():
        ign_rows += [l for l in orig_ign.read_text().splitlines() if l.strip()]
    (OUT / "ignore" / (stem + ".txt")).write_text(
        ("\n".join(ign_rows) + "\n") if ign_rows else "")

census["member_images"] = len(members)

n_links = 0
for a in ALIASES:
    view = OUT / f"raw_{a}"
    view.mkdir(exist_ok=True)
    src_root = Path(f"/workspace/holdout_eval/{a}/raw")
    for stem in members:
        src = src_root / (stem + ".json")
        dst = view / (stem + ".json")
        if src.exists() and not dst.is_symlink():
            dst.symlink_to(src)
            n_links += 1
census["raw_links_created"] = n_links

(OUT / "census.json").write_text(json.dumps(census, indent=2))
print(json.dumps(census, indent=2))
print("BUILD_DONE")
