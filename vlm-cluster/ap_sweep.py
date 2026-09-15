#!/usr/bin/env python3
"""
ap_sweep.py — per-class AP (AP50 / AP75 / AP50-95) as a function of the deployment
confidence threshold, replayed offline from run_ultralytics_labels.py raw sidecars.

"AP at threshold t" = the AP map_eval.py would report if every detection with
conf < t were deleted from the dumps first. It answers: how much of each class's
AP survives at the operating point actually shipped (0.45), and where an export's
confidence ceiling (docs/MODEL_COMPARISON.md, INT8 sections) kills the class
outright.

Frozen-core reuse: loading, IoU, ignore regions, greedy confidence-ordered
matching, per-image max_dets and the 101-point interpolation are map_eval.py's
(`load_preds`, `load_gt`, `load_ignore`, `iou`, `_ioa`, `IGNORE_IOA`,
`IOU_THRESHOLDS`). Only the reduction differs: matching is done ONCE per class
and IoU threshold, and the confidence threshold is applied as a prefix of the
confidence-descending (conf, is_tp) rows — legal because greedy matching in
descending confidence never lets a lower-confidence box change the outcome of a
higher-confidence one, so deleting low boxes == truncating the rows. At t = 0
this reproduces map_eval.py's numbers exactly (--crosscheck asserts it).

    python3 ap_sweep.py --raw <dump>/raw --gt-labels <labels> --ignore-labels <ignore> \
        --grid-step 0.05 --out <cell>.json [--crosscheck <cell>_map.json]
    python3 ap_sweep.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from map_eval import (CLASS_NAMES, IGNORE_IOA, IOU_THRESHOLDS, _ioa, iou, load_gt,
                      load_ignore, load_preds)

MAX_DETS = 100          # map_eval.evaluate default (COCO)


def rows_per_class(preds, gt, cls, thr, ignore=None, max_dets=MAX_DETS):
    """map_eval.ap_per_class's matching (area_rng=None), returning the sorted
    (conf, is_tp) rows and n_gt instead of collapsing to AP."""
    rows = []
    n_gt = 0
    for stem, gboxes in gt.items():
        g = gboxes.get(cls, [])
        n_gt += len(g)
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
                rows.append((conf, 1))
            elif any(_ioa(box, ig) >= IGNORE_IOA for ig in (ignore or {}).get(stem, ())):
                pass
            else:
                rows.append((conf, 0))
    rows.sort(reverse=True)
    return rows, n_gt


def ap_at(conf, tp, n_gt, conf_min):
    """COCO 101-point AP over the rows with conf >= conf_min (a prefix of the
    confidence-descending rows). Same value as map_eval.ap_per_class on the
    filtered detections."""
    if n_gt == 0:
        return None
    k = int(np.searchsorted(-conf, -conf_min, side="right"))   # rows with conf >= conf_min
    if k == 0:
        return 0.0
    t = tp[:k]
    ctp = np.cumsum(t)
    cfp = np.cumsum(1 - t)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    # max precision over all points with rec >= r == reverse cumulative max, looked up at the first such point
    revmax = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.arange(101) / 100.0
    idx = np.searchsorted(rec, r, side="left")
    p_at = np.where(idx < k, revmax[np.minimum(idx, k - 1)], 0.0)
    return float(p_at.sum() / 101)


def sweep(raw_dir, gt_dir, ignore_dir, grid, max_dets=MAX_DETS):
    preds = load_preds(Path(raw_dir))
    gt = load_gt(Path(gt_dir), preds.keys())
    ignore = load_ignore(Path(ignore_dir), preds.keys()) if ignore_dir else None
    out = {"n_images": len(preds), "n_gt": {}, "max_dets_per_image": max_dets,
           "iou_thresholds": IOU_THRESHOLDS, "grid": grid, "at_thresholds": {}}
    per = {}   # (cls_name, iou) -> (conf array, tp array, n_gt)
    for cls, name in CLASS_NAMES.items():
        for thr in IOU_THRESHOLDS:
            rows, n_gt = rows_per_class(preds, gt, cls, thr, ignore, max_dets)
            conf = np.array([c for c, _ in rows], dtype=np.float64)
            tp = np.array([t for _, t in rows], dtype=np.float64)
            per[(name, thr)] = (conf, tp, n_gt)
        out["n_gt"][name] = per[(name, IOU_THRESHOLDS[0])][2]
        out.setdefault("max_conf", {})[name] = float(per[(name, IOU_THRESHOLDS[0])][0][0]) if len(per[(name, IOU_THRESHOLDS[0])][0]) else 0.0
    for t in grid:
        cell = {}
        for cls, name in CLASS_NAMES.items():
            aps = [ap_at(*per[(name, thr)], t) for thr in IOU_THRESHOLDS]
            if aps[0] is None:
                continue
            cell[name] = {"ap50": round(aps[0], 4), "ap75": round(aps[5], 4),
                          "ap50_95": round(sum(aps) / len(aps), 4)}
        present = [n for n in CLASS_NAMES.values() if n in cell]
        for k in ("ap50", "ap75", "ap50_95"):          # map50 / map75 / map50_95, as map_eval names them
            cell[f"m{k}"] = round(sum(cell[n][k] for n in present) / len(present), 4) if present else 0.0
        out["at_thresholds"][f"{t:.3f}"] = cell
    return out


def make_grid(step):
    n = int(round(1.0 / step))
    return [0.0] + [round(step * i, 4) for i in range(1, n)]


def crosscheck(result, map_json):
    """The t=0 column must equal map_eval.py's per-class numbers for the same dump."""
    ref = json.loads(Path(map_json).read_text())["per_class"]
    got = result["at_thresholds"]["0.000"]
    bad = [(n, k, ref[n][k], got[n][k]) for n in ref for k in ("ap50", "ap75", "ap50_95")
           if abs(ref[n][k] - got[n][k]) > 0.00015]
    assert not bad, f"crosscheck FAILED vs {map_json}: {bad}"
    return {n: {k: ref[n][k] for k in ("ap50", "ap75", "ap50_95")} for n in ref}


def selftest():
    import tempfile
    from map_eval import ap_per_class
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); (td / "raw").mkdir(); (td / "gt").mkdir()
        # 30 synthetic images, 3 classes, random GT + jittered/false detections
        for i in range(30):
            gts, dets = [], []
            for _ in range(rng.integers(1, 4)):
                c = int(rng.integers(0, 3)); cx, cy = rng.uniform(0.2, 0.8, 2); w, h = rng.uniform(0.1, 0.3, 2)
                gts.append(f"{c} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
                if rng.random() < 0.85:      # a detection near it, sometimes the wrong class
                    j = rng.uniform(-0.03, 0.03, 4); cc = c if rng.random() < 0.85 else int(rng.integers(0, 3))
                    dets.append({"cls": cc, "box_xyxy": [(cx - w / 2 + j[0]) * 100, (cy - h / 2 + j[1]) * 100, (cx + w / 2 + j[2]) * 100, (cy + h / 2 + j[3]) * 100], "conf": float(rng.uniform(0.05, 0.99))})
            for _ in range(rng.integers(0, 3)):  # false positives
                x, y = rng.uniform(0, 80, 2)
                dets.append({"cls": int(rng.integers(0, 3)), "box_xyxy": [x, y, x + 10, y + 10], "conf": float(rng.uniform(0.05, 0.6))})
            (td / "gt" / f"i{i}.txt").write_text("\n".join(gts) + "\n")
            (td / "raw" / f"i{i}.json").write_text(json.dumps({"width": 100, "height": 100, "detections": dets}))
        preds = load_preds(td / "raw"); gt = load_gt(td / "gt", preds.keys())
        res = sweep(td / "raw", td / "gt", None, [0.0, 0.3, 0.6])
        for t in (0.0, 0.3, 0.6):
            # reference: filter the detections, then map_eval.ap_per_class
            fp = {s: {c: [(cf, b) for cf, b in v if cf >= t] for c, v in d.items()} for s, d in preds.items()}
            for cls, name in CLASS_NAMES.items():
                for thr, key in ((0.5, "ap50"), (0.75, "ap75")):
                    ref = ap_per_class(fp, gt, cls, thr, None, None, MAX_DETS, None)
                    got = res["at_thresholds"][f"{t:.3f}"][name][key]
                    assert ref is not None and abs(ref - got) < 1e-4, (t, name, key, ref, got)
                ref95 = sum(ap_per_class(fp, gt, cls, thr, None, None, MAX_DETS, None) for thr in IOU_THRESHOLDS) / 10
                assert abs(ref95 - res["at_thresholds"][f"{t:.3f}"][name]["ap50_95"]) < 1e-4
        assert res["at_thresholds"]["0.600"]["map50"] <= res["at_thresholds"]["0.000"]["map50"]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw"); ap.add_argument("--gt-labels"); ap.add_argument("--ignore-labels")
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--max-dets", type=int, default=MAX_DETS)
    ap.add_argument("--crosscheck", help="map_eval _map.json for the same dump; asserts the t=0 column equals it")
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.raw and a.gt_labels and a.out):
        ap.error("--raw, --gt-labels, --out required (or --selftest)")
    res = sweep(a.raw, a.gt_labels, a.ignore_labels, make_grid(a.grid_step), a.max_dets)
    if a.crosscheck:
        res["crosscheck"] = {"against": a.crosscheck, "per_class": crosscheck(res, a.crosscheck)}
        print("crosscheck OK vs", a.crosscheck)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"n_images={res['n_images']} n_gt={res['n_gt']} max_conf={res['max_conf']}")
    for t in ("0.000", "0.450", "0.700"):
        if t in res["at_thresholds"]:
            c = res["at_thresholds"][t]
            print(t, " ".join(f"{n}={c[n]['ap50_95']:.4f}" for n in CLASS_NAMES.values() if n in c), f"mAP50-95={c['map50_95']:.4f}")


if __name__ == "__main__":
    main()
