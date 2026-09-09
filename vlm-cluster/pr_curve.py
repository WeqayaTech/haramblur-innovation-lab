#!/usr/bin/env python3
"""
pr_curve.py — full per-class precision/recall curve (COCO greedy matching,
IoU>=0.5 by default), the analogue of Ultralytics' own `metrics.box.px/rx`
but built from this repo's own scorer instead of `model.val()`.

Reuses map_eval.py's exact loading + matching primitives (frozen-core:
`load_preds`, `load_gt`, `load_ignore`, `iou`, `_ioa`) so this curve is
consistent with every AP number already in docs/MODEL_COMPARISON.md — only
the reduction step differs (keep the full curve instead of collapsing to a
single AP scalar).

Unlike `conf_sweep.py`'s LAGENDA arm (`--lagenda`), this precision is a REAL
FP-based precision: any detection that doesn't land on a correct-class GT box
(and isn't covered by an ignore region) counts against it, exactly like
CrowdHuman/PASS — but per-class, which those arms are not. This is the
curve to eyeball for picking an operating point.

    python3 pr_curve.py \
        --raw /workspace/mapdump/y26n_humanshaped_v2/lagenda/raw \
        --gt-labels  /workspace/datasets/lagenda_full/eval_v2/labels_3class \
        --ignore-labels /workspace/datasets/lagenda_full/eval_v2/ignore \
        --out /workspace/pr_curves/y26n_humanshaped_v2.json

    python3 pr_curve.py --selftest    # synthetic, no data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_eval import CLASS_NAMES, _ioa, iou, load_gt, load_ignore, load_preds

THRESH_GRID = [round(0.05 * i, 2) for i in range(1, 20)]      # 0.05 .. 0.95


def class_curve(preds, gt, cls, thr, ignore=None):
    """(conf, is_tp) for every kept-or-FP detection of `cls`, confidence-desc,
    plus the running (recall, precision) after each — same logic as
    map_eval.ap_per_class, minus the AP collapse."""
    rows = []
    n_gt = 0
    for stem, gboxes in gt.items():
        g = gboxes.get(cls, [])
        n_gt += len(g)
        p = sorted(preds.get(stem, {}).get(cls, []), reverse=True)
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
                rows.append((conf, 1))
            elif any(_ioa(box, ig) >= 0.5 for ig in (ignore or {}).get(stem, ())):
                pass                                 # unlabeled real person
            else:
                rows.append((conf, 0))
    rows.sort(reverse=True)
    tp = fp = 0
    curve = []                                       # (conf, recall, precision)
    for conf, is_tp in rows:
        tp += is_tp
        fp += 1 - is_tp
        curve.append((conf, tp / n_gt if n_gt else 0.0,
                      tp / (tp + fp) if (tp + fp) else 0.0))
    return curve, n_gt


def at_thresholds(curve, grid):
    """(recall, precision, f1) at each grid confidence — last point whose
    conf >= t (curve is confidence-descending)."""
    out = {}
    for t in grid:
        rec = prec = 0.0
        for conf, r, p in curve:
            if conf < t:
                break
            rec, prec = r, p
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        out[f"{t:.2f}"] = {"recall": round(rec, 4), "precision": round(prec, 4),
                           "f1": round(f1, 4)}
    return out


def build(raw_dir, gt_dir, ignore_dir, iou_thr, grid, downsample):
    preds = load_preds(Path(raw_dir))
    gt = load_gt(Path(gt_dir), preds.keys())
    ignore = load_ignore(Path(ignore_dir), preds.keys()) if ignore_dir else None
    out = {"iou_threshold": iou_thr, "n_images": len(preds), "classes": {}}
    for cls, name in CLASS_NAMES.items():
        curve, n_gt = class_curve(preds, gt, cls, iou_thr, ignore)
        if n_gt == 0:
            continue
        thin = curve[::downsample] if downsample > 1 else curve
        out["classes"][name] = {
            "n_gt": n_gt,
            "curve": [{"conf": round(c, 4), "recall": round(r, 4),
                       "precision": round(p, 4)} for c, r, p in thin],
            "at_thresholds": at_thresholds(curve, grid),
        }
    return out


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "raw").mkdir(); (td / "gt").mkdir()
        (td / "gt" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (td / "raw" / "a.json").write_text(json.dumps({
            "width": 100, "height": 100, "detections": [
                {"cls": 0, "box_xyxy": [40, 40, 60, 60], "conf": 0.9},
                {"cls": 0, "box_xyxy": [0, 0, 10, 10], "conf": 0.3}]}))
        r = build(td / "raw", td / "gt", None, 0.5, [0.2, 0.5, 0.8], 1)
        c = r["classes"]["Woman"]["curve"]
        assert c == [{"conf": 0.9, "recall": 1.0, "precision": 1.0},
                     {"conf": 0.3, "recall": 1.0, "precision": 0.5}], c
        t = r["classes"]["Woman"]["at_thresholds"]
        assert t["0.50"] == {"recall": 1.0, "precision": 1.0, "f1": 1.0}, t
        assert t["0.20"] == {"recall": 1.0, "precision": 0.5,
                             "f1": round(2 / 3, 4)}, t
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--gt-labels")
    ap.add_argument("--ignore-labels")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--downsample", type=int, default=1,
                    help="keep every Nth curve point (curves can be thousands "
                         "of points; at_thresholds is unaffected)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.raw and args.gt_labels and args.out):
        ap.error("--raw, --gt-labels, --out required (or --selftest)")
    r = build(args.raw, args.gt_labels, args.ignore_labels, args.iou,
             THRESH_GRID, args.downsample)
    Path(args.out).write_text(json.dumps(r, indent=2))
    for name, c in r["classes"].items():
        print(f"{name}: n_gt={c['n_gt']}  curve_points={len(c['curve'])}")


if __name__ == "__main__":
    main()
