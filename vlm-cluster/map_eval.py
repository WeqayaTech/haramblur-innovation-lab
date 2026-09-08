#!/usr/bin/env python3
"""
EXP-2026-12 — COCO-style mAP over run_ultralytics_labels.py raw sidecars.

One mAP pipeline for every model (team request 2026-08-05): reads the per-image
raw sidecar JSONs (every detection >= the 0.05 floor, with conf — the log-raw
principle paying off: no re-run needed) and YOLO-format GT label files, and
computes per-class AP50, AP75 and AP@[.5:.95] (COCO 101-point interpolation),
class-aware greedy matching per image, predictions sorted by confidence.

    python3 map_eval.py \
        --raw /workspace/exp12/<arm>/oiv7val/raw \
        --gt-labels /workspace/spotlight/run/oiv7_val/labels \
        --out /workspace/exp12/eval/<arm>/oiv7val_map.json

    python3 map_eval.py --selftest    # synthetic, no data

Boxes: sidecars carry pixel xyxy + image width/height, GT is normalized YOLO
cxcywh — both are converted to normalized xyxy so no image files are read.
Detections with an `excluded` reason (distractor classes etc.) are skipped;
the `kept` flag (the 0.45 working threshold) is deliberately IGNORED — mAP
sweeps confidence by construction.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]        # 0.50 .. 0.95
CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child"}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(gt_dir: Path, stems):
    """{stem: {cls: [xyxy_norm, ...]}}"""
    gt = {}
    for s in stems:
        f = gt_dir / f"{s}.txt"
        boxes = defaultdict(list)
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                vals = [float(x) for x in p[1:]]
                if len(vals) == 4:                     # cx cy w h
                    cx, cy, w, h = vals
                    boxes[c].append((cx - w / 2, cy - h / 2,
                                     cx + w / 2, cy + h / 2))
                else:                                  # polygon
                    xs, ys = vals[0::2], vals[1::2]
                    if len(xs) >= 3 and len(xs) == len(ys):
                        boxes[c].append((min(xs), min(ys), max(xs), max(ys)))
        gt[s] = boxes
    return gt


def load_preds(raw_dir: Path, stem_prefix: str | None = None):
    """{stem: {cls: [(conf, xyxy_norm), ...]}}

    `stem_prefix` restricts the evaluation to one slice of an arm whose sizes
    share a directory (e.g. `a56_` for the 56 px avatars) — the GT and ignore
    dirs follow the same stems, so the whole computation narrows with it."""
    preds = {}
    for f in sorted(raw_dir.glob("*.json")):
        if stem_prefix and not f.stem.startswith(stem_prefix):
            continue
        d = json.loads(f.read_text())
        w, h = d["width"], d["height"]
        boxes = defaultdict(list)
        for det in d["detections"]:
            if det.get("excluded"):
                continue
            x1, y1, x2, y2 = det["box_xyxy"]
            boxes[det["cls"]].append(
                (det["conf"], (x1 / w, y1 / h, x2 / w, y2 / h)))
        preds[f.stem] = boxes
    return preds


def _area_px(box, wh):
    """Box area in ORIGINAL image pixels (boxes are normalized xyxy)."""
    w, h = wh
    return max(0.0, (box[2] - box[0]) * w) * max(0.0, (box[3] - box[1]) * h)


def _ioa(box, region):
    """Intersection over the DETECTION's own area."""
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    return inter / a if a > 0 else 0.0


IGNORE_IOA = 0.5


def ap_per_class(preds, gt, cls, thr, dims=None, area_rng=None, max_dets=0,
                 ignore=None):
    """AP for one class at one IoU threshold, COCO 101-point interpolation.

    area_rng: (lo, hi) in squared pixels -> COCO-style S/M/L slice. GT outside
      the range is EXCLUDED from n_gt and predictions matching it are dropped
      (not counted as FP); predictions outside the range that match nothing are
      also dropped. This mirrors pycocotools' ignore semantics closely enough
      for a slice metric — it is an approximation, stated in the output.
    max_dets: per-image cap applied AFTER confidence sorting (COCO uses 100);
      0 = no cap (use every logged detection).
    ignore: {stem: [xyxy_norm]} regions (e.g. real people the dataset detected
      but never labeled). An unmatched detection with IoA >= 0.5 against one is
      DROPPED — neither TP nor FP — COCO's iscrowd/ignore semantics. Without
      this, a correct detection of an unlabeled person is scored as a false
      positive and AP is meaningless on partially-labeled data.
    """
    rows = []                                  # (conf, is_tp) over all images
    n_gt = 0
    for stem, gboxes in gt.items():
        wh = (dims or {}).get(stem, (1.0, 1.0))
        g = gboxes.get(cls, [])
        in_rng = [area_rng is None or area_rng[0] <= _area_px(b, wh) < area_rng[1]
                  for b in g]
        n_gt += sum(in_rng)
        p = sorted(preds.get(stem, {}).get(cls, []), reverse=True)
        if max_dets:
            p = p[:max_dets]
        used = [False] * len(g)
        for conf, box in p:
            best, best_i = 0.0, -1
            for i, gb in enumerate(g):
                if used[i]:
                    continue
                v = iou(box, gb)
                if v > best:
                    best, best_i = v, i
            if best >= thr and best_i >= 0:
                used[best_i] = True
                if in_rng[best_i]:
                    rows.append((conf, 1))
                # matched an out-of-range GT -> ignored, neither TP nor FP
            elif any(_ioa(box, ig) >= IGNORE_IOA
                     for ig in (ignore or {}).get(stem, ())):
                pass                     # landed on an unlabeled real person
            elif area_rng is None or area_rng[0] <= _area_px(box, wh) < area_rng[1]:
                rows.append((conf, 0))
            # else: out-of-range unmatched prediction -> ignored
    if n_gt == 0:
        return None
    rows.sort(reverse=True)
    tp = fp = 0
    rec_prec = []
    for _, is_tp in rows:
        tp += is_tp
        fp += 1 - is_tp
        rec_prec.append((tp / n_gt, tp / (tp + fp)))
    ap = 0.0
    for r in [i / 100 for i in range(101)]:
        p_at = max((p for rec, p in rec_prec if rec >= r), default=0.0)
        ap += p_at / 101
    return ap


AREA_RANGES = {"small": (0.0, 32.0 ** 2),
               "medium": (32.0 ** 2, 96.0 ** 2),
               "large": (96.0 ** 2, float("inf"))}


def load_dims(raw_dir: Path, stem_prefix: str | None = None):
    """{stem: (width, height)} from the sidecars — needed for COCO area ranges."""
    dims = {}
    for f in sorted(raw_dir.glob("*.json")):
        if stem_prefix and not f.stem.startswith(stem_prefix):
            continue
        d = json.loads(f.read_text())
        dims[f.stem] = (d["width"], d["height"])
    return dims


def load_ignore(ign_dir: Path, stems):
    """{stem: [xyxy_norm]} — every box in the ignore label dir, class ignored."""
    out = {}
    for s in stems:
        f = ign_dir / f"{s}.txt"
        boxes = []
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cx, cy, w, h = [float(x) for x in p[1:5]]
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        out[s] = boxes
    return out


def evaluate(raw_dir: Path, gt_dir: Path, max_dets: int = 100,
             conf_floor: float | None = None, ignore_dir: Path | None = None,
             stem_prefix: str | None = None):
    preds = load_preds(raw_dir, stem_prefix)
    dims = load_dims(raw_dir, stem_prefix)
    gt = load_gt(gt_dir, preds.keys())
    ignore = load_ignore(ignore_dir, preds.keys()) if ignore_dir else None
    lowest = min((c for b in preds.values() for v in b.values() for c, _ in v),
                 default=None)
    out = {"n_images": len(preds), "stem_prefix": stem_prefix,
           "n_gt": sum(len(v) for g in gt.values() for v in g.values()),
           "protocol": {
               "interpolation": "COCO 101-point (mean of interpolated precision "
                                "envelope) — pycocotools convention, NOT "
                                "ultralytics' trapezoid variant",
               "matching": "greedy, confidence-ordered, one GT per detection "
                           "(pycocotools convention; ultralytics orders by IoU)",
               "iou_thresholds": "0.50:0.05:0.95",
               "max_dets_per_image": max_dets or "uncapped",
               "lowest_logged_conf": round(lowest, 5) if lowest is not None else None,
               "area_ranges": "COCO S/M/L in original-image pixels; out-of-range "
                              "GT and unmatched out-of-range predictions are "
                              "ignored (approximation of pycocotools' ignore flags)",
               "ignore_regions": (f"{sum(len(v) for v in ignore.values())} boxes "
                                  f"(IoA>=0.5 -> neither TP nor FP)") if ignore
                                 else "none",
           }}
    per_class = {}
    for cls, name in CLASS_NAMES.items():
        aps = [ap_per_class(preds, gt, cls, t, dims, None, max_dets, ignore)
               for t in IOU_THRESHOLDS]
        if aps[0] is None:
            continue
        per_class[name] = {"ap50": round(aps[0], 4),
                           "ap75": round(aps[5], 4),
                           "ap50_95": round(sum(aps) / len(aps), 4)}
    out["per_class"] = per_class
    for k in ("ap50", "ap75", "ap50_95"):
        out[f"m{k}"] = round(sum(v[k] for v in per_class.values())
                             / len(per_class), 4) if per_class else 0.0

    by_area = {}
    for label, rng in AREA_RANGES.items():
        vals = []
        for cls in CLASS_NAMES:
            aps = [ap_per_class(preds, gt, cls, t, dims, rng, max_dets, ignore)
                   for t in IOU_THRESHOLDS]
            if aps[0] is not None:
                vals.append(sum(aps) / len(aps))
        by_area[f"mAP50_95_{label}"] = round(sum(vals) / len(vals), 4) if vals else None
    out["by_area"] = by_area
    if conf_floor is not None and lowest is not None and lowest > conf_floor * 5:
        out["WARNING"] = (f"lowest logged confidence is {lowest:.4f}; sidecars were "
                          f"dumped with a higher floor than {conf_floor} — AP is "
                          f"truncated and reads low. Re-dump with --floor {conf_floor}.")
    return out


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "raw").mkdir()
        (td / "gt").mkdir()
        # img a: one Woman GT, predicted perfectly at conf .9 + one FP at .3
        (td / "gt" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (td / "raw" / "a.json").write_text(json.dumps({
            "image": "a.jpg", "width": 100, "height": 100, "detections": [
                {"det_index": 0, "cls": 0, "cls_name": "Woman",
                 "box_xyxy": [40, 40, 60, 60], "conf": 0.9, "kept": True,
                 "excluded": None},
                {"det_index": 1, "cls": 0, "cls_name": "Woman",
                 "box_xyxy": [0, 0, 10, 10], "conf": 0.3, "kept": False,
                 "excluded": None},
                {"det_index": 2, "cls": 1, "cls_name": "statue",
                 "box_xyxy": [0, 0, 90, 90], "conf": 0.99, "kept": False,
                 "excluded": "distractor"}]}))
        # img b: one Man GT, missed entirely
        (td / "gt" / "b.txt").write_text("1 0.5 0.5 0.4 0.4\n")
        (td / "raw" / "b.json").write_text(json.dumps({
            "image": "b.jpg", "width": 100, "height": 100, "detections": []}))
        r_noign = evaluate(td / "raw", td / "gt")
        (td / "ign").mkdir()
        # the conf-.3 box at (0,0,10,10) is a real-but-unlabeled person
        (td / "ign" / "a.txt").write_text("0 0.05 0.05 0.10 0.10\n")
        (td / "ign" / "b.txt").write_text("")
        r_ign = evaluate(td / "raw", td / "gt", ignore_dir=td / "ign")
        assert r_ign["per_class"]["Woman"]["ap50"] >= r_noign["per_class"]["Woman"]["ap50"], \
            "ignore regions must not lower AP"
        assert "boxes" in r_ign["protocol"]["ignore_regions"], r_ign["protocol"]
        r = r_noign
        assert r["n_images"] == 2 and r["n_gt"] == 2, r
        # Woman: TP at conf .9 before the FP -> AP50 = 1.0 (precision holds at
        # full recall); Man: missed -> 0
        assert abs(r["per_class"]["Woman"]["ap50"] - 1.0) < 1e-6, r
        assert r["per_class"]["Man"]["ap50"] == 0.0, r
        assert abs(r["map50"] - 0.5) < 1e-6, r
        # excluded distractor must not have poisoned Man as an FP-only class
        assert r["per_class"]["Woman"]["ap50_95"] > 0.5, r
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", help="raw sidecar dir from run_ultralytics_labels")
    ap.add_argument("--gt-labels", help="YOLO GT label dir")
    ap.add_argument("--out", help="output json")
    ap.add_argument("--max-dets", type=int, default=100,
                    help="per-image detection cap after confidence sort "
                         "(COCO uses 100; 0 = uncapped)")
    ap.add_argument("--ignore-labels",
                    help="dir of YOLO boxes to treat as ignore regions "
                         "(unlabeled real people) — COCO iscrowd semantics")
    ap.add_argument("--stem-prefix",
                    help="only score images whose stem starts with this "
                         "(e.g. a56_ for the 56px avatars, h48_ for that paste size)")
    ap.add_argument("--expect-floor", type=float, default=0.001,
                    help="warn if the sidecars were dumped above this floor")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.raw and args.gt_labels and args.out):
        ap.error("--raw, --gt-labels, --out required (or --selftest)")
    r = evaluate(Path(args.raw), Path(args.gt_labels),
                 max_dets=args.max_dets, conf_floor=args.expect_floor,
                 ignore_dir=Path(args.ignore_labels) if args.ignore_labels else None,
                 stem_prefix=args.stem_prefix)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
