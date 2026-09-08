#!/usr/bin/env python3
"""
conf_sweep.py — find the confidence threshold(s) that best serve a stated
objective, by REPLAYING detections that were already dumped. No model re-run,
no GPU, no image reads.

Why an offline replay is faithful: every Stage-A dump logs each detection down
to a low floor (`run_ultralytics_labels.py --floor 0.05`, the mapdump runs
0.001) into a per-image sidecar. NMS is greedy in DESCENDING confidence, so
adding lower-confidence boxes never changes what higher-confidence boxes do —
the survivors above any threshold t >= floor are exactly the survivors a fresh
run at `--conf t` would have written. `--verify-at 0.45` proves that on your own
data: it replays the production threshold and diffs against the emitted
`labels/` files. (One caveat it also checks: if an image hit the model's
`max_det` cap at the floor, low-confidence detections were truncated and only
the LOW end of the sweep is affected.)

Every dataset answers its own question and results are NEVER pooled
(docs/COMPONENT_FRAMEWORK.md). Matching is the same greedy, mutually-exclusive
IoU >= 0.5 matcher every other scorer in this repo uses — the replay matcher is
a fast re-implementation and `--selftest` asserts it agrees with
`run_model_children.match_boxes` exactly, ties included.

    # 1. the usual question: fewer false persons, more women actually blurred
    python3 conf_sweep.py \
        --lagenda    /workspace/exp12/y26n_gradsupp/lagenda/raw \
        --lagenda-gt /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
        --lagenda-gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
        --negatives objects:/workspace/exp12/y26n_gradsupp/object_set/raw \
        --negatives pass:/workspace/exp12/y26n_gradsupp/pass_3k/raw \
        --crowd      /workspace/exp12/y26n_gradsupp/crowd/raw \
        --crowd-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
        --out /workspace/sweep/y26n_gradsupp

    # 2. a different target: minimise false persons, but never drop below 90%
    #    of the women the model blurs today
    ... --objective "-objects.img_fp_rate" \
        --constraint "lagenda.woman_recall_e2e >= 0.90"

    # 3. per-class thresholds (Woman/Man/Child get their own cut)
    ... --mode per-class

    # 4. a report you can actually look at: ONE self-contained HTML file
    #    (baseline-vs-chosen table + 3 charts, no CDN, no fonts to fetch) —
    #    written automatically to <out>/report.html, or point --report anywhere
    ... --out /workspace/sweep/<arm> --label "<arm> · male viewer"

    # 5. what metrics can an objective reference?
    python3 conf_sweep.py --list-metrics

    python3 conf_sweep.py --selftest      # no data, no GPU, no network
    python3 conf_sweep.py --crosscheck   # prove the replay reproduces
                                         # eval_negatives_crowd.py and
                                         # run_autolabel_on_manifest.py
                                         # number-for-number at conf 0.45

The objective is any arithmetic expression over `dataset.metric` names, so
"the target of optimisation" is a command-line argument, not a code edit.
Constraints are comparisons in the same language; infeasible thresholds are
still evaluated and reported, just never chosen.
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import dataset_utils as du
from eval_negatives_crowd import (DUP_IOU, IGNORE_IOA, PARTIAL_IOU, _ioa,
                                  _occ_band, load_odgt)
from run_autolabel_on_manifest import gt_class, load_gt, seg_boxes
from eval_taxonomy import wilson

# production taxonomy; override with --classes for other label spaces
DEFAULT_CLASSES = ("Woman", "Man", "Child")
MATCH_IOU = 0.5          # same as every scorer in this repo
# GT person height in pixels. The small/medium split sits just above the ~58px
# median of the 196,119 unknown-gender detections Spotlight dropped to
# background, so "did that data choice cost us recall?" reads straight off the
# small band.
SIZE_BANDS = (("small", 0, 64), ("med", 64, 160), ("large", 160, 10 ** 9))
# GT age at/above which a "Child" call is a real adult escaping the blur.
# (LAGENDA's human apparent-age labels carry +-2-3 yrs, so the 13-17 band is
#  annotation noise, not escaping adults — EXP-2026-10's owner ruling.)
ADULT_LEAK_AGE = 20


# ---------------------------------------------------------------------------
# loading — one record per image, geometry kept in whatever space it arrives in
# ---------------------------------------------------------------------------
class ImageDets:
    """Detections for one image. `boxes` are xyxy; `space` is 'px' when they
    are pixels (sidecars carry width/height) or 'norm' when they are 0-1.
    IoU is invariant under a shared axis rescale, so GT is simply converted
    into whichever space the detections live in — no image is ever opened."""

    __slots__ = ("stem", "w", "h", "cls", "conf", "boxes", "space")

    def __init__(self, stem, w, h, cls, conf, boxes, space):
        self.stem, self.w, self.h = stem, w, h
        self.cls, self.conf, self.boxes, self.space = cls, conf, boxes, space


def _pack(stem, w, h, cls, conf, boxes, space):
    return ImageDets(stem, w, h,
                     np.asarray(cls, dtype=int),
                     np.asarray(conf, dtype=float),
                     np.asarray(boxes, dtype=float).reshape(-1, 4),
                     space)


def load_sidecars(raw_dir: Path, max_images: int = 0):
    """`run_ultralytics_labels.py` raw/<stem>.json -> {stem: ImageDets}.

    Detections carrying an `excluded` reason (YOLOE distractor vocabulary) are
    dropped: they are never emitted at ANY threshold, so they are not part of
    the threshold question."""
    recs, n_excluded = {}, 0
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no sidecars (*.json) under {raw_dir} — point "
                         f"--*-raw at a run_ultralytics_labels.py 'raw' dir, "
                         f"or use --format labels-conf for 6-field labels")
    for p in files:
        d = json.loads(p.read_text())
        cls, conf, boxes = [], [], []
        for r in d.get("detections", []):
            if r.get("excluded"):
                n_excluded += 1
                continue
            cls.append(int(r["cls"]))
            conf.append(float(r["conf"]))
            boxes.append(r["box_xyxy"])
        recs[p.stem] = _pack(p.stem, d.get("width"), d.get("height"),
                             cls, conf, boxes, "px")
        if max_images and len(recs) >= max_images:
            break
    return recs, {"n_images": len(recs), "n_excluded_dets": n_excluded}


def load_labels_conf(lbl_dir: Path, max_images: int = 0):
    """6-field YOLO labels 'cls cx cy w h conf' (normalized) -> {stem: ImageDets}.

    This is the shape `autolabel_sam_conf.py` (EXP-2026-07) writes. Boxes stay
    normalized, which is fine for LAGENDA (its GT is normalized too) and for
    negatives (no geometry at all), but CrowdHuman GT is in pixels — use
    sidecars for the crowd arm."""
    recs = {}
    files = sorted(lbl_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no label files (*.txt) under {lbl_dir}")
    for p in files:
        cls, conf, boxes = [], [], []
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) != 6:
                continue
            c, cx, cy, bw, bh, cf = (float(x) for x in parts)
            cls.append(int(c))
            conf.append(cf)
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        recs[p.stem] = _pack(p.stem, None, None, cls, conf, boxes, "norm")
        if max_images and len(recs) >= max_images:
            break
    return recs, {"n_images": len(recs), "n_excluded_dets": 0}


def load_dets(path: Path, fmt: str, max_images: int = 0):
    return (load_sidecars(path, max_images) if fmt == "sidecar"
            else load_labels_conf(path, max_images))


def norm_boxes(rec: ImageDets):
    """Detection boxes in 0-1 space (for GT that is normalized)."""
    if rec.space == "norm" or not rec.w or not rec.h:
        return rec.boxes
    return rec.boxes / np.array([rec.w, rec.h, rec.w, rec.h], dtype=float)


def check_classes(recs, nc, where):
    bad = sorted({int(c) for r in recs.values() for c in r.cls
                  if c < 0 or c >= nc})
    if bad:
        raise SystemExit(
            f"{where}: detections carry class ids {bad} outside 0..{nc - 1}. "
            f"Pass --classes with one name per id (the sweep needs a threshold "
            f"per class), or dump with --map coco-person / identity so the ids "
            f"are the production {DEFAULT_CLASSES} space.")


def observed_floor(recs):
    lo = [float(r.conf.min()) for r in recs.values() if r.conf.size]
    return min(lo) if lo else None


# ---------------------------------------------------------------------------
# matching — a fast replay of run_model_children.match_boxes
# ---------------------------------------------------------------------------
def iou_matrix(gt_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """(n_gt, n_det) IoU. Same formula as run_model_children.iou."""
    if gt_boxes.size == 0 or det_boxes.size == 0:
        return np.zeros((len(gt_boxes), len(det_boxes)))
    g, d = gt_boxes[:, None, :], det_boxes[None, :, :]
    ix1 = np.maximum(g[..., 0], d[..., 0])
    iy1 = np.maximum(g[..., 1], d[..., 1])
    ix2 = np.minimum(g[..., 2], d[..., 2])
    iy2 = np.minimum(g[..., 3], d[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ga = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    da = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    union = ga + da - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def candidate_pairs(gt_boxes, det_boxes, min_iou):
    """Every (iou, gt_index, det_index) above min_iou, ordered exactly as
    match_boxes orders them: descending IoU, ties broken by insertion order
    (gt-major, det-minor). Threshold-independent, so this is computed once per
    image and reused for every threshold in the sweep."""
    m = iou_matrix(np.asarray(gt_boxes, float).reshape(-1, 4),
                   np.asarray(det_boxes, float).reshape(-1, 4))
    pairs = [(float(m[gi, di]), gi, di)
             for gi in range(m.shape[0]) for di in range(m.shape[1])
             if m[gi, di] >= min_iou]
    pairs.sort(key=lambda p: p[0], reverse=True)     # stable, like match_boxes
    best_per_det = m.max(axis=0) if m.size else np.zeros(len(det_boxes))
    return pairs, best_per_det


def greedy_match(pairs, active):
    """{gt_index: det_index} over the detections `active[di]` allows."""
    used_gt, used_det, out = set(), set(), {}
    for _, gi, di in pairs:
        if gi in used_gt or di in used_det or not active[di]:
            continue
        used_gt.add(gi)
        used_det.add(di)
        out[gi] = di
    return out


def active_mask(rec_cls, rec_conf, thr):
    """Which detections survive the per-class thresholds `thr`."""
    if rec_conf.size == 0:
        return np.zeros(0, dtype=bool)
    return rec_conf >= np.asarray(thr, dtype=float)[rec_cls]


# ---------------------------------------------------------------------------
# dataset evaluators — each returns {metric: value} plus {metric: (k, n)} so
# the reporter can put a Wilson CI on every rate it prints
# ---------------------------------------------------------------------------
def _rate(k, n):
    return (k / n) if n else 0.0


def _size_band(h_px):
    if h_px is None:
        return None
    for name, lo, hi in SIZE_BANDS:
        if lo <= h_px < hi:
            return name
    return None


class NegativeSet:
    """PASS / object_set: no people, so every surviving detection is a false
    positive by construction (eval_negatives_crowd.py --mode negatives)."""

    kind = "negatives"
    METRICS = {
        "fp_per_100": "false persons per 100 images",
        "img_fp_rate": "fraction of images with >=1 false person "
                       "(the 'blurs my book' symptom)",
        "fp_total": "raw count of false persons",
        "fp_<class>_per_100": "false persons per 100 images, by class written",
        "n_images": "images scored",
    }

    def __init__(self, name, recs, classes):
        self.name, self.classes, self.nc = name, classes, len(classes)
        self.n_images = len(recs)
        # per-image max confidence per class: an image "has an FP" iff any
        # class's max clears that class's threshold. Fully vectorized.
        self.max_by_img = np.full((self.n_images, self.nc), -1.0)
        per_cls = [[] for _ in range(self.nc)]
        for i, r in enumerate(recs.values()):
            for c, cf in zip(r.cls, r.conf):
                self.max_by_img[i, c] = max(self.max_by_img[i, c], cf)
                per_cls[c].append(cf)
        self.per_cls = [np.asarray(v, dtype=float) for v in per_cls]

    def evaluate(self, thr):
        thr = np.asarray(thr, dtype=float)
        by_cls = [int((self.per_cls[c] >= thr[c]).sum()) for c in range(self.nc)]
        total = sum(by_cls)
        n_img_fp = int((self.max_by_img >= thr).any(axis=1).sum())
        n = self.n_images
        m = {"n_images": n, "fp_total": total,
             "fp_per_100": 100.0 * total / n if n else 0.0,
             "img_fp_rate": _rate(n_img_fp, n)}
        for c, name in enumerate(self.classes):
            m[f"fp_{name.lower()}_per_100"] = 100.0 * by_cls[c] / n if n else 0.0
        return m, {"img_fp_rate": (n_img_fp, n)}


class CrowdSet:
    """CrowdHuman: exhaustively boxed, so an unmatched detection is provably
    wrong and recall/precision are both real (eval_negatives_crowd.py --mode
    crowd). Ignore regions are dropped before matching, exactly as there."""

    kind = "crowd"
    METRICS = {
        "recall": "GT persons found (visible-box IoU >= match-iou)",
        "precision": "matched / kept detections",
        "f1": "harmonic mean of the two",
        "recall_light|partial|heavy": "recall stratified by occlusion band",
        "dup_rate": "duplicate boxes per matched person (fragmentation)",
        "clear_fp_per_100": "unmatched dets with <0.1 IoU on any person, /100 imgs",
        "clear_fp_<class>_per_100": "same, split by the class written",
        "n_gt_persons / n_images / dets_kept": "coverage counters",
    }

    def __init__(self, recs, odgt_path, classes, match_iou=MATCH_IOU):
        self.classes, self.nc = classes, len(classes)
        gt = load_odgt(Path(odgt_path))
        self.images, self.n_gt = [], 0
        self.n_missing_gt = 0
        for stem, r in recs.items():
            rec = gt.get(stem)
            if rec is None:
                self.n_missing_gt += 1
                continue
            if r.space != "px":
                raise SystemExit("crowd GT is in pixels — the crowd arm needs "
                                 "sidecars (--crowd-format sidecar), not "
                                 "normalized label files")
            keep = [i for i, b in enumerate(r.boxes)
                    if not any(_ioa(list(b), ig) > IGNORE_IOA
                               for ig in rec["ignores"])]
            boxes = r.boxes[keep] if keep else np.zeros((0, 4))
            vboxes = [p["vbox"] for p in rec["persons"]]
            pairs, best = candidate_pairs(vboxes, boxes, match_iou)
            self.images.append({
                "cls": r.cls[keep] if keep else np.zeros(0, int),
                "conf": r.conf[keep] if keep else np.zeros(0),
                "pairs": pairs, "best_iou": best,
                "bands": [_occ_band(p["occ_ratio"]) for p in rec["persons"]],
            })
            self.n_gt += len(vboxes)

    def evaluate(self, thr):
        n_img = len(self.images)
        n_matched = n_kept = n_dup = n_partial = n_clear = 0
        band = {b: [0, 0] for b in ("light", "partial", "heavy")}
        clear_by_cls = [0] * self.nc
        for im in self.images:
            act = active_mask(im["cls"], im["conf"], thr)
            n_kept += int(act.sum())
            matched = greedy_match(im["pairs"], act)
            n_matched += len(matched)
            for gi, b in enumerate(im["bands"]):
                band[b][0] += 1
                band[b][1] += (gi in matched)
            taken = set(matched.values())
            for di in range(len(im["cls"])):
                if not act[di] or di in taken:
                    continue
                best = float(im["best_iou"][di]) if len(im["best_iou"]) else 0.0
                if best >= DUP_IOU:
                    n_dup += 1
                elif best >= PARTIAL_IOU:
                    n_partial += 1
                else:
                    n_clear += 1
                    clear_by_cls[int(im["cls"][di])] += 1
        rec_, prec = _rate(n_matched, self.n_gt), _rate(n_matched, n_kept)
        m = {"n_images": n_img, "n_gt_persons": self.n_gt, "dets_kept": n_kept,
             "recall": rec_, "precision": prec,
             "f1": (2 * rec_ * prec / (rec_ + prec)) if (rec_ + prec) else 0.0,
             "dup_rate": _rate(n_dup, n_matched),
             "partial_overlap": n_partial,
             "clear_fp_total": n_clear,
             "clear_fp_per_100": 100.0 * n_clear / n_img if n_img else 0.0}
        for b, (tot, found) in band.items():
            m[f"recall_{b}"] = _rate(found, tot)
        for c, name in enumerate(self.classes):
            m[f"clear_fp_{name.lower()}_per_100"] = (
                100.0 * clear_by_cls[c] / n_img if n_img else 0.0)
        counts = {"recall": (n_matched, self.n_gt),
                  "precision": (n_matched, n_kept)}
        for b, (tot, found) in band.items():
            counts[f"recall_{b}"] = (found, tot)
        return m, counts


class LagendaSet:
    """LAGENDA: human apparent age + gender per labeled person. Label-anchored,
    so these are recall-side numbers only — an unmatched detection here is NOT
    a false positive (that is what PASS/objects/CrowdHuman measure)."""

    kind = "lagenda"
    METRICS = {
        "det_recall": "labeled people found at all",
        "det_recall_<class>": "same, split by GT class",
        "woman_recall_e2e": "GT women found AND written 'Woman' — the fraction "
                            "of women a male viewer actually gets blurred",
        "man_recall_e2e / child_recall_e2e": "same for the other classes",
        "<class>_precision_matched": "of matched people written that class, how "
                                     "many really are — the 'don't blur the "
                                     "wrong people' guard (matched-only proxy; "
                                     "LAGENDA cannot measure true precision)",
        "<class>_f1_e2e": "harmonic mean of that class's recall and precision",
        "acc3": "3-class accuracy on matched people",
        "gender_acc_adults": "Woman/Man accuracy on matched adults",
        "leak_adult20_child": "matched GT age >= 20 written 'Child' — adults "
                              "escaping the blur (the consequential direction)",
        "leak_adult20_child_count": "the same as an absolute count, with "
                                    "n_adults20_matched as its denominator — "
                                    "constrain THESE, not the rate, when the "
                                    "thresholds move the denominator",
        "n_matched": "labeled people matched at this threshold",
        "det_recall_small|med|large": "detection recall by GT person HEIGHT in "
                                      "pixels (<64 / 64-160 / >=160) — the "
                                      "small band is where dropping Spotlight's "
                                      "unknown-gender people would show up",
        "woman_recall_e2e_small|med|large": "the same split for women actually "
                                            "blurred",
        "n_scored / dets_kept": "coverage counters",
    }

    def __init__(self, recs, gt_manifest, gt_labels_dir, classes,
                 match_iou=MATCH_IOU, limit=0):
        self.classes, self.nc = classes, len(classes)
        by_image = load_gt(Path(gt_manifest), limit)
        gt_labels_dir = Path(gt_labels_dir)
        self.images = []
        self.n_untranslatable = self.n_not_processed = self.n_bad_index = 0
        for img_name, rows in sorted(by_image.items()):
            stem = Path(img_name).stem
            r = recs.get(stem)
            if r is None:
                self.n_not_processed += len(rows)
                continue
            # GT labels are normalized YOLO; parse them in 0-1 space and put
            # the detections in the same space (IoU is invariant to a shared
            # axis rescale) — no image is opened.
            gt_all = du.yolo_boxes(gt_labels_dir / f"{stem}.txt", 1.0, 1.0)
            gboxes, meta = [], []
            for row in rows:
                if row["box_index"] >= len(gt_all):
                    self.n_bad_index += 1
                    continue
                gcls = gt_class(row)
                if gcls is None:
                    self.n_untranslatable += 1
                    continue
                box = list(gt_all[row["box_index"]][1:5])
                gboxes.append(box)
                # box is normalized; r.h is the image height from the sidecar
                h_px = (box[3] - box[1]) * r.h if r.h else None
                meta.append({"gt_class": gcls, "gt_age": row.get("gt_age"),
                             "band": _size_band(h_px)})
            if not gboxes:
                continue
            pairs, _ = candidate_pairs(gboxes, norm_boxes(r), match_iou)
            self.images.append({"cls": r.cls, "conf": r.conf,
                                "pairs": pairs, "meta": meta})
        self.n_scored = sum(len(im["meta"]) for im in self.images)

    def evaluate(self, thr):
        n_matched = n_correct = n_kept = 0
        det_by_cls = {c: [0, 0] for c in self.classes}      # [n_gt, n_found]
        e2e_by_cls = {c: [0, 0] for c in self.classes}      # [n_gt, n_correct]
        pred_by_cls = {c: [0, 0] for c in self.classes}   # [n_pred, n_right]
        band_det = {b: [0, 0] for b, _, _ in SIZE_BANDS}   # [n_gt, n_found]
        band_woman = {b: [0, 0] for b, _, _ in SIZE_BANDS}  # [n_gt_w, n_right]
        n_adults = n_adults_right = 0
        n_age20 = n_age20_child = 0
        child = self.classes[2] if self.nc > 2 else None
        for im in self.images:
            act = active_mask(im["cls"], im["conf"], thr)
            n_kept += int(act.sum())
            matched = greedy_match(im["pairs"], act)
            for gi, meta in enumerate(im["meta"]):
                g = meta["gt_class"]
                det_by_cls[g][0] += 1
                e2e_by_cls[g][0] += 1
                band = meta.get("band")
                if band:
                    band_det[band][0] += 1
                    if g == self.classes[0]:
                        band_woman[band][0] += 1
                di = matched.get(gi)
                if di is None:
                    continue
                if band:
                    band_det[band][1] += 1
                p = self.classes[int(im["cls"][di])]
                n_matched += 1
                det_by_cls[g][1] += 1
                ok = (p == g)
                n_correct += ok
                e2e_by_cls[g][1] += ok
                pred_by_cls[p][0] += 1
                pred_by_cls[p][1] += ok
                if band and g == self.classes[0]:
                    band_woman[band][1] += (p == g)
                if g in self.classes[:2]:                  # adult GT
                    n_adults += 1
                    n_adults_right += ok
                age = meta.get("gt_age")
                if age is not None and age >= ADULT_LEAK_AGE:
                    n_age20 += 1
                    n_age20_child += (child is not None and p == child)
        m = {"n_scored": self.n_scored, "dets_kept": n_kept,
             "det_recall": _rate(n_matched, self.n_scored),
             "acc3": _rate(n_correct, n_matched),
             "gender_acc_adults": _rate(n_adults_right, n_adults),
             "leak_adult20_child": _rate(n_age20_child, n_age20),
             # absolute counts too: every rate above has a denominator that
             # MOVES with the thresholds (fewer matches = smaller denominator),
             # so a rate can climb without a single new error. Constrain the
             # count when you mean "no more mistakes than today".
             "leak_adult20_child_count": n_age20_child,
             "n_adults20_matched": n_age20, "n_matched": n_matched}
        counts = {"det_recall": (n_matched, self.n_scored),
                  "acc3": (n_correct, n_matched),
                  "gender_acc_adults": (n_adults_right, n_adults),
                  "leak_adult20_child": (n_age20_child, n_age20)}
        for b, _, _ in SIZE_BANDS:
            m[f"det_recall_{b}"] = _rate(*band_det[b][::-1])
            m[f"woman_recall_e2e_{b}"] = _rate(*band_woman[b][::-1])
            counts[f"det_recall_{b}"] = (band_det[b][1], band_det[b][0])
            counts[f"woman_recall_e2e_{b}"] = (band_woman[b][1],
                                               band_woman[b][0])
        for c in self.classes:
            key = c.lower()
            r = _rate(*e2e_by_cls[c][::-1])
            pr = _rate(*pred_by_cls[c][::-1])
            m[f"det_recall_{key}"] = _rate(*det_by_cls[c][::-1])
            m[f"{key}_recall_e2e"] = r
            m[f"{key}_precision_matched"] = pr
            m[f"{key}_f1_e2e"] = (2 * r * pr / (r + pr)) if (r + pr) else 0.0
            counts[f"det_recall_{key}"] = (det_by_cls[c][1], det_by_cls[c][0])
            counts[f"{key}_recall_e2e"] = (e2e_by_cls[c][1], e2e_by_cls[c][0])
            counts[f"{key}_precision_matched"] = (pred_by_cls[c][1],
                                                  pred_by_cls[c][0])
        return m, counts


# ---------------------------------------------------------------------------
# the objective language — arithmetic + comparisons over `dataset.metric`
# ---------------------------------------------------------------------------
_FUNCS = {"min": min, "max": max, "abs": abs}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b if b else 0.0,
           ast.Pow: lambda a, b: a ** b}
_CMPOPS = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
           ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
           ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


def _node(n, metrics):
    if isinstance(n, ast.Expression):
        return _node(n.body, metrics)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.Name) and n.id in _ALIASES:
        key = _ALIASES[n.id]
        if key not in metrics:
            if key.startswith("baseline."):
                raise SystemExit(
                    f"{key!r}: no baseline metrics available. `baseline.X` "
                    f"resolves X at --baseline, and needs the baseline point "
                    f"to have been evaluated first.")
            arm, _, metric = key.partition(".")
            near = [k for k in metrics if k.endswith("." + metric)] or \
                   [k for k in metrics if k.startswith(arm + ".")]
            raise SystemExit(f"unknown metric {key!r}. "
                             + (f"Did you mean: {', '.join(sorted(near)[:6])}?"
                                if near else
                                "Run --list-metrics, and check the dataset is "
                                "actually passed on the command line."))
        return metrics[key]
    if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
        return _BINOPS[type(n.op)](_node(n.left, metrics), _node(n.right, metrics))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _node(n.operand, metrics)
        return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.Compare) and all(type(o) in _CMPOPS for o in n.ops):
        left = _node(n.left, metrics)
        for op, comp in zip(n.ops, n.comparators):
            right = _node(comp, metrics)
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(n, ast.BoolOp):
        vals = [_node(v, metrics) for v in n.values]
        return all(vals) if isinstance(n.op, ast.And) else any(vals)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
            and n.func.id in _FUNCS and not n.keywords:
        return _FUNCS[n.func.id](*[_node(a, metrics) for a in n.args])
    raise SystemExit(f"unsupported expression element: {ast.dump(n)[:80]}. "
                     f"Objectives are arithmetic over dataset.metric names, "
                     f"plus min/max/abs.")


# `dataset.metric` is rewritten to an opaque alias before the expression is
# parsed. Python's grammar would otherwise reject perfectly good arm names —
# `pass.img_fp_rate` is a syntax error, and PASS is one of our datasets.
_METRIC_RE = re.compile(
    r"\b(baseline\.)?([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
_ALIASES: dict[str, str] = {}


def _rewrite(expr: str):
    """expr with every dataset.metric replaced by a stable alias; the alias ->
    key mapping is global because aliases must survive into _node()."""
    out, keys = [], []

    def sub(m):
        key = f"{m.group(1) or ''}{m.group(2)}.{m.group(3)}"
        for alias, k in _ALIASES.items():
            if k == key:
                keys.append(key)
                return alias
        alias = f"_metric_{len(_ALIASES)}"
        _ALIASES[alias] = key
        keys.append(key)
        return alias

    out = _METRIC_RE.sub(sub, expr)
    return out, keys


def eval_expr(expr: str, metrics: dict):
    rewritten, _ = _rewrite(expr)
    return _node(ast.parse(rewritten, mode="eval"), metrics)


def referenced_metrics(expr: str):
    seen, order = set(), []
    for k in _rewrite(expr)[1]:
        if k not in seen:
            seen.add(k)
            order.append(k)
    return order


PRESETS = {
    # the default question: blur more women, write fewer false persons
    "woman_recall_vs_fp": "lagenda.woman_recall_e2e - {w}*({fp})",
    "min_fp": "-({fp})",
    "woman_f1": "lagenda.woman_f1_e2e",
    "acc3": "lagenda.acc3",
    "crowd_f1": "crowd.f1",
    "balanced_recall_vs_fp":
        "(lagenda.woman_recall_e2e + lagenda.man_recall_e2e "
        "+ lagenda.child_recall_e2e)/3 - {w}*({fp})",
}


def expand_preset(name, neg_names, fp_weight):
    """Presets are written against whatever negatives arms were supplied, so
    the FP term is built at runtime — and printed, so it is never a mystery."""
    if name not in PRESETS:
        raise SystemExit(f"unknown preset @{name}. "
                         f"Available: {', '.join('@' + k for k in PRESETS)}")
    if not neg_names and "{fp}" in PRESETS[name]:
        raise SystemExit(f"preset @{name} needs at least one --negatives arm "
                         f"(that is where false positives are measurable).")
    fp = " + ".join(f"{n}.img_fp_rate" for n in neg_names) or "0"
    if len(neg_names) > 1:
        fp = f"({fp})/{len(neg_names)}"
    return PRESETS[name].format(w=fp_weight, fp=fp)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def frange(spec: str):
    """'0.05:0.95:0.01' -> [0.05, 0.06, ...] (inclusive of stop when it lands)."""
    a, b, s = (float(x) for x in spec.split(":"))
    n = int(round((b - a) / s))
    return [round(a + i * s, 6) for i in range(n + 1)]


def evaluate_datasets(datasets, thr):
    """{arm.metric: value} plus {arm.metric: (k, n)} at one threshold vector."""
    metrics, counts = {}, {}
    for name, ds in datasets.items():
        m, c = ds.evaluate(thr)
        metrics.update({f"{name}.{k}": v for k, v in m.items()})
        counts.update({f"{name}.{k}": v for k, v in c.items()})
    return metrics, counts


class Sweeper:
    """Evaluates a threshold vector across every dataset, caches by vector.

    `baseline_metrics` (the metrics at --baseline) are exposed to expressions
    under a `baseline.` prefix, so "don't regress" is written exactly —
    `objects.fp_per_100 <= baseline.objects.fp_per_100` — instead of pasting a
    rounded literal that the baseline itself then fails."""

    def __init__(self, datasets, objective, constraints, baseline_metrics=None):
        self.datasets, self.objective, self.constraints = \
            datasets, objective, constraints
        self.base = {f"baseline.{k}": v
                     for k, v in (baseline_metrics or {}).items()}
        self.cache = {}

    def __call__(self, thr):
        key = tuple(round(t, 6) for t in thr)
        if key in self.cache:
            return self.cache[key]
        metrics, counts = evaluate_datasets(datasets=self.datasets, thr=key)
        lookup = {**metrics, **self.base}
        failed = [c for c in self.constraints
                  if not bool(eval_expr(c, lookup))]
        point = {"thresholds": list(key), "metrics": metrics, "counts": counts,
                 "objective": float(eval_expr(self.objective, lookup)),
                 "feasible": not failed, "failed_constraints": failed}
        self.cache[key] = point
        return point


def search_global(sweep, grid, nc):
    return [sweep([t] * nc) for t in grid]


def search_per_class(sweep, coarse, fine, nc, log=print):
    """Coarse full grid, then a local fine refinement around the best feasible
    point. Reported honestly: this is a coarse-to-fine search, not a proof of
    global optimality."""
    pts = []
    log(f"  coarse grid: {len(coarse)}^{nc} = {len(coarse) ** nc} points")
    def rec(prefix):
        if len(prefix) == nc:
            pts.append(sweep(prefix))
            return
        for t in coarse:
            rec(prefix + [t])
    rec([])
    best = pick_best(pts)
    if best is None:
        return pts
    step = coarse[1] - coarse[0] if len(coarse) > 1 else 0.1
    axes = []
    for c in range(nc):
        lo, hi = best["thresholds"][c] - step, best["thresholds"][c] + step
        axes.append([t for t in fine if lo - 1e-9 <= t <= hi + 1e-9])
    log(f"  fine refine: {'x'.join(str(len(a)) for a in axes)} points "
        f"around {best['thresholds']}")
    def rec2(prefix, depth):
        if depth == nc:
            pts.append(sweep(prefix))
            return
        for t in axes[depth]:
            rec2(prefix + [t], depth + 1)
    rec2([], 0)
    return pts


def search_coord(sweep, grid, nc, start, log=print):
    """Coordinate ascent: cycle the classes, take the best single-class move,
    stop when a full pass changes nothing. Cheap; a LOCAL optimum."""
    cur, pts = list(start), []
    for it in range(10):
        improved = False
        for c in range(nc):
            cand = []
            for t in grid:
                v = list(cur)
                v[c] = t
                p = sweep(v)
                pts.append(p)
                cand.append(p)
            best = pick_best(cand)
            if best and tuple(best["thresholds"]) != tuple(cur):
                cur = list(best["thresholds"])
                improved = True
        log(f"  pass {it + 1}: thresholds {cur}")
        if not improved:
            break
    return pts


def dedupe(points):
    """A coarse grid and its refinement overlap, and coordinate ascent revisits
    points — the Sweeper cache hands back the same dict each time. Collapse to
    one entry per threshold vector before reporting or counting."""
    return list({tuple(p["thresholds"]): p for p in points}.values())


def pick_best(points):
    """Best FEASIBLE point, or None if every point failed a constraint.

    Ties are broken toward the HIGHER thresholds: if two cuts score the same,
    the one that emits fewer boxes is the one to ship — same measured
    behaviour, fewer false persons, which is the complaint users actually
    file (docs/MODEL_EVAL_OVERVIEW.md)."""
    pool = [p for p in points if p["feasible"]]
    if not pool:
        return None
    return max(pool, key=lambda p: (p["objective"], sum(p["thresholds"])))


def pareto_front(points, axes):
    """axes = [('max', 'lagenda.woman_recall_e2e'), ('min', 'objects.img_fp_rate')]"""
    def vec(p):
        return [p["metrics"][k] if d == "max" else -p["metrics"][k]
                for d, k in axes]
    out = []
    for p in points:
        v = vec(p)
        if not any(all(x >= y for x, y in zip(vec(q), v))
                   and any(x > y for x, y in zip(vec(q), v)) for q in points):
            out.append(p)
    seen, uniq = set(), []
    for p in sorted(out, key=lambda p: vec(p)[0], reverse=True):
        k = tuple(p["thresholds"])
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
HEADLINE = {
    "lagenda": ["woman_recall_e2e", "det_recall", "acc3", "gender_acc_adults",
                "leak_adult20_child"],
    "crowd": ["recall", "precision", "clear_fp_per_100"],
    "negatives": ["img_fp_rate", "fp_per_100"],
}


def headline_keys(datasets):
    keys = []
    for name, ds in datasets.items():
        keys += [f"{name}.{k}" for k in HEADLINE[ds.kind]]
    return keys


def thr_str(thresholds):
    """Thresholds as text. 3dp, because a --grid finer than 0.01 makes 2dp
    print distinct rows identically (0.005/0.010/0.015 all become '0.01')."""
    return "/".join(f"{t:.3f}" for t in thresholds)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def print_table(points, keys, title, log=print):
    log(f"\n{title}")
    head = ["thresholds", "objective"] + keys
    widths = [max(len(head[0]), 14)] + [max(len(h), 12) for h in head[1:]]
    log("  " + "  ".join(h.rjust(w) for h, w in zip(head, widths)))
    for p in points:
        row = [thr_str(p["thresholds"]),
               f"{p['objective']:.4f}" + ("" if p["feasible"] else " X")]
        row += [_fmt(p["metrics"].get(k)) for k in keys]
        log("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths)))


def print_compare(best, baseline, keys, log=print):
    log(f"\nBEST {thr_str(best['thresholds'])}  vs  "
        f"BASELINE {thr_str(baseline['thresholds'])}"
        f"   (objective {baseline['objective']:.4f} -> "
        f"{best['objective']:.4f})")
    log(f"  {'metric':<34} {'baseline':>12} {'best':>12} {'delta':>10}   95% CI (best)")
    for k in keys:
        b, n = best["metrics"].get(k), baseline["metrics"].get(k)
        if not isinstance(b, (int, float)) or not isinstance(n, (int, float)):
            continue
        ci = ""
        if k in best["counts"]:
            kk, nn = best["counts"][k]
            _, lo, hi = wilson(kk, nn)
            ci = f"   [{max(0.0, lo):.4f}-{hi:.4f}]  ({kk}/{nn})"
        log(f"  {k:<34} {n:>12.4f} {b:>12.4f} {b - n:>+10.4f}{ci}")


def write_outputs(out_dir: Path, meta, points, best, baseline, pareto):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(
        {"meta": meta, "best": best, "baseline": baseline,
         "pareto_front": pareto, "points": points}, indent=2))
    keys = sorted({k for p in points for k in p["metrics"]})
    with open(out_dir / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thresholds", "objective", "feasible"] + keys)
        for p in sorted(points, key=lambda p: p["objective"], reverse=True):
            w.writerow([thr_str(p["thresholds"]),
                        f"{p['objective']:.6f}", int(p["feasible"])]
                       + [p["metrics"].get(k, "") for k in keys])


def plot_svg(points, keys, best_thr, baseline_thr, path: Path):
    """Self-contained SVG (explicit colors, white background) — repo chart
    convention. Global mode only: x = the single threshold."""
    pts = sorted(points, key=lambda p: p["thresholds"][0])
    xs = [p["thresholds"][0] for p in pts]
    if len(xs) < 2:
        return None
    W, H, L, R, T, B = 820, 420, 70, 210, 40, 55
    colors = ["#1a6fb5", "#c2381f", "#2e8b57", "#7a4fb5", "#b58a1a"]
    def sx(x):
        return L + (x - xs[0]) / (xs[-1] - xs[0]) * (W - L - R)
    body = [f'<rect width="{W}" height="{H}" fill="white"/>',
            f'<text x="{L}" y="24" font-family="sans-serif" font-size="15" '
            f'fill="#111">confidence threshold sweep (each metric scaled to '
            f'its own range)</text>']
    for i, x in enumerate(xs):
        if i % max(1, len(xs) // 10):
            continue
        body.append(f'<line x1="{sx(x):.1f}" y1="{T}" x2="{sx(x):.1f}" '
                    f'y2="{H - B}" stroke="#eee" stroke-width="1"/>')
        body.append(f'<text x="{sx(x):.1f}" y="{H - B + 18}" font-size="11" '
                    f'font-family="sans-serif" text-anchor="middle" '
                    f'fill="#555">{x:.2f}</text>')
    for ci, k in enumerate(keys[:5]):
        ys = [p["metrics"].get(k, 0.0) for p in pts]
        lo, hi = min(ys), max(ys)
        rng = (hi - lo) or 1.0
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},"
                     f"{H - B - (y - lo) / rng * (H - T - B):.1f}"
                     for i, (x, y) in enumerate(zip(xs, ys)))
        body.append(f'<path d="{d}" fill="none" stroke="{colors[ci % 5]}" '
                    f'stroke-width="2"/>')
        body.append(f'<text x="{W - R + 8}" y="{T + 16 + ci * 18}" '
                    f'font-size="11" font-family="sans-serif" '
                    f'fill="{colors[ci % 5]}">{k} [{lo:.3g}-{hi:.3g}]</text>')
    for x, col, lab in ((baseline_thr, "#999", "baseline"),
                        (best_thr, "#111", "best")):
        if xs[0] <= x <= xs[-1]:
            body.append(f'<line x1="{sx(x):.1f}" y1="{T}" x2="{sx(x):.1f}" '
                        f'y2="{H - B}" stroke="{col}" stroke-width="1.5" '
                        f'stroke-dasharray="5,4"/>')
            body.append(f'<text x="{sx(x):.1f}" y="{T - 6}" font-size="11" '
                        f'font-family="sans-serif" text-anchor="middle" '
                        f'fill="{col}">{lab} {x:.2f}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
                    f'height="{H}" viewBox="0 0 {W} {H}">'
                    + "".join(body) + "</svg>")
    return path


# ---------------------------------------------------------------------------
# visual report — ONE self-contained HTML file (inline SVG, no CDN, no fonts to
# fetch), same convention as build_error_gallery.py: scp it off the pod and
# open it anywhere.
# ---------------------------------------------------------------------------
PALETTE = ["#1a6fb5", "#c2381f", "#2e8b57", "#7a4fb5", "#b58a1a"]
INK, MUTED, GRID = "#111", "#666", "#e6e6e6"


def _ticks(lo, hi, n=5):
    """Ticks on round numbers (1/2/2.5/5 x 10^k), not raw range fractions."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), mag)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def thr_str(thresholds):
    """Thresholds as text. 3dp, because a --grid finer than 0.01 makes 2dp
    print distinct rows identically (0.005/0.010/0.015 all become '0.01')."""
    return "/".join(f"{t:.3f}" for t in thresholds)


def _fmt(v):
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


class Panel:
    """Minimal linear-axes SVG panel: .x()/.y() map data -> pixels."""

    def __init__(self, w, h, xlo, xhi, ylo, yhi, title, xlabel, ylabel,
                 pad=(64, 22, 46, 58), yticks=True):
        self.w, self.h, self.pad = w, h, pad
        # pad a hair so points never sit exactly on an axis
        if xhi == xlo:
            xhi = xlo + 1e-9
        if yhi == ylo:
            yhi = ylo + 1e-9
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = ylo, yhi
        self.el = [f'<text x="{pad[0]}" y="16" font-size="13" fill="{INK}" '
                   f'font-weight="600">{title}</text>']
        L, R, T, B = pad
        for t in (_ticks(ylo, yhi) if yticks else []):
            y = self.y(t)
            self.el.append(f'<line x1="{L}" y1="{y:.1f}" x2="{w - R}" '
                           f'y2="{y:.1f}" stroke="{GRID}"/>')
            self.el.append(f'<text x="{L - 6}" y="{y + 4:.1f}" font-size="10" '
                           f'fill="{MUTED}" text-anchor="end">{_fmt(t)}</text>')
        for t in _ticks(xlo, xhi):
            x = self.x(t)
            self.el.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" '
                           f'y2="{h - B}" stroke="{GRID}"/>')
            self.el.append(f'<text x="{x:.1f}" y="{h - B + 15}" font-size="10" '
                           f'fill="{MUTED}" text-anchor="middle">{_fmt(t)}</text>')
        self.el.append(f'<text x="{(L + w - R) / 2:.0f}" y="{h - 8}" '
                       f'font-size="11" fill="{MUTED}" text-anchor="middle">'
                       f'{xlabel}</text>')
        self.el.append(f'<text transform="translate(14,{(T + h - B) / 2:.0f}) '
                       f'rotate(-90)" font-size="11" fill="{MUTED}" '
                       f'text-anchor="middle">{ylabel}</text>')

    def x(self, v):
        L, R = self.pad[0], self.pad[1]
        return L + (v - self.xlo) / (self.xhi - self.xlo) * (self.w - L - R)

    def y(self, v):
        T, B = self.pad[2], self.pad[3]
        return (self.h - B) - (v - self.ylo) / (self.yhi - self.ylo) * \
            (self.h - T - B)

    def add(self, e):
        self.el.append(e)

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="{self.w}" height="{self.h}" fill="white"/>'
                + "".join(self.el) + "</svg>")


def _legend(panel, entries, x, y, width=130):
    # opaque backdrop: a legend must never be overprinted by the data or the
    # axis labels, whatever the value range happens to be
    panel.add(f'<rect x="{x - 9}" y="{y - 14}" width="{width}" '
              f'height="{len(entries) * 15 + 6}" fill="white" opacity="0.92" '
              f'rx="3"/>')
    for i, (color, label, filled) in enumerate(entries):
        yy = y + i * 15
        panel.add(f'<circle cx="{x}" cy="{yy - 3}" r="4" fill="'
                  f'{color if filled else "white"}" stroke="{color}"/>')
        panel.add(f'<text x="{x + 10}" y="{yy}" font-size="10" fill="{MUTED}">'
                  f'{label}</text>')


def panel_tradeoff(points, best, baseline, xkey, ykey):
    """Every evaluated threshold as one dot: the shape of the tradeoff, with
    the Pareto front (up-and-left is better) drawn through it."""
    xs = [p["metrics"][xkey] for p in points]
    ys = [p["metrics"][ykey] for p in points]
    pan = Panel(760, 330, min(xs), max(xs), min(ys), max(ys),
                f"Every threshold evaluated  ·  up and to the LEFT is better",
                f"{xkey}  (lower = fewer false persons)",
                f"{ykey}  (higher = more blurred)")
    for p in points:
        feas = p["feasible"]
        pan.add(f'<circle cx="{pan.x(p["metrics"][xkey]):.1f}" '
                f'cy="{pan.y(p["metrics"][ykey]):.1f}" r="2.6" '
                f'fill="{"#1a6fb5" if feas else "white"}" '
                f'stroke="{"#1a6fb5" if feas else "#cfcfcf"}" '
                f'stroke-width="1" opacity="{0.75 if feas else 0.6}"/>')
    front = pareto_front(points, [("max", ykey), ("min", xkey)])
    if len(front) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}"
                     f"{pan.x(p['metrics'][xkey]):.1f},"
                     f"{pan.y(p['metrics'][ykey]):.1f}"
                     for i, p in enumerate(
                         sorted(front, key=lambda q: q["metrics"][xkey])))
        pan.add(f'<path d="{d}" fill="none" stroke="#2e8b57" '
                f'stroke-width="1.6" stroke-dasharray="4,3"/>')
    for p, color, label in ((baseline, "#c2381f", "baseline"),
                            (best, "#111", "chosen")):
        cx, cy = pan.x(p["metrics"][xkey]), pan.y(p["metrics"][ykey])
        pan.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="none" '
                f'stroke="{color}" stroke-width="2.5"/>')
        pan.add(f'<text x="{cx + 10:.1f}" y="{cy - 8:.1f}" font-size="11" '
                f'fill="{color}" font-weight="600">{label} '
                f'{"/".join(f"{t:.2f}" for t in p["thresholds"])}</text>')
    _legend(pan, [("#1a6fb5", "meets every constraint", True),
                  ("#cfcfcf", "fails a constraint", False),
                  ("#2e8b57", "Pareto front", True)], 596, 214, 152)
    return pan.svg()


def panel_response(sweep, grid, best, classes, objective_label):
    """What each class's threshold is actually buying: sweep one class over the
    whole grid with the other two held at the chosen values."""
    curves = []
    for c in range(len(classes)):
        pts = []
        for t in grid:
            v = list(best["thresholds"])
            v[c] = t
            pts.append((t, sweep(v)))
        curves.append(pts)
    ys = [p["objective"] for cur in curves for _, p in cur]
    pan = Panel(760, 330, grid[0], grid[-1], min(ys), max(ys),
                "One class at a time (others held at the chosen values)",
                "that class's confidence threshold", objective_label)
    labels = []
    for c, cur in enumerate(curves):
        d = " ".join(f"{'M' if i == 0 else 'L'}{pan.x(t):.1f},"
                     f"{pan.y(p['objective']):.1f}"
                     for i, (t, p) in enumerate(cur))
        pan.add(f'<path d="{d}" fill="none" stroke="{PALETTE[c % 5]}" '
                f'stroke-width="2"/>')
        for t, p in cur:
            if not p["feasible"]:
                pan.add(f'<circle cx="{pan.x(t):.1f}" '
                        f'cy="{pan.y(p["objective"]):.1f}" r="1.7" '
                        f'fill="#cfcfcf"/>')
        bt = best["thresholds"][c]
        pan.add(f'<circle cx="{pan.x(bt):.1f}" '
                f'cy="{pan.y(sweep(best["thresholds"])["objective"]):.1f}" '
                f'r="5" fill="none" stroke="{PALETTE[c % 5]}" '
                f'stroke-width="2.5"/>')
        labels.append((PALETTE[c % 5], f"{classes[c]} @ {bt:.2f}"))
    pan.add(f'<rect x="530" y="{44}" width="200" '
            f'height="{len(labels) * 16 + 22}" fill="white" opacity="0.92" '
            f'rx="3"/>')
    for c, (color, text) in enumerate(labels):
        pan.add(f'<text x="722" y="{62 + c * 16}" font-size="11" '
                f'fill="{color}" text-anchor="end">{text}</text>')
    pan.add(f'<text x="722" y="{62 + len(labels) * 16}" font-size="10" '
            f'fill="{MUTED}" text-anchor="end">grey = fails a constraint</text>')
    return pan.svg()


def panel_fp_composition(best, baseline, arms, classes):
    """Which class the false persons are written as — the thing that decides
    whether per-class thresholds can decouple FPs from recall at all."""
    rows = []
    for arm in arms:
        for label, p in (("baseline", baseline), ("chosen", best)):
            vals = [p["metrics"].get(f"{arm}.fp_{c.lower()}_per_100", 0.0)
                    for c in classes]
            rows.append((f"{arm} · {label}", vals))
    hi = max((sum(v) for _, v in rows), default=1.0) or 1.0
    h = 92 + len(rows) * 34
    pan = Panel(760, h, 0, hi, 0, 1, "False persons per 100 images, by the "
                "class the model wrote", "", "",
                pad=(150, 40, 34, 56), yticks=False)
    for i, (label, vals) in enumerate(rows):
        y = 44 + i * 34
        pan.add(f'<text x="142" y="{y + 13}" font-size="11" fill="{INK}" '
                f'text-anchor="end">{label}</text>')
        x0 = pan.x(0)
        for c, v in enumerate(vals):
            wpx = pan.x(v) - pan.x(0)
            if wpx > 0.4:
                pan.add(f'<rect x="{x0:.1f}" y="{y}" width="{wpx:.1f}" '
                        f'height="18" fill="{PALETTE[c % 5]}" opacity="0.85"/>')
            x0 += wpx
        pan.add(f'<text x="{x0 + 6:.1f}" y="{y + 13}" font-size="10" '
                f'fill="{MUTED}">{sum(vals):.1f}</text>')
    for c, name in enumerate(classes):
        pan.add(f'<rect x="{150 + c * 90}" y="{h - 16}" width="10" '
                f'height="10" fill="{PALETTE[c % 5]}" opacity="0.85"/>')
        pan.add(f'<text x="{164 + c * 90}" y="{h - 7}" font-size="10" '
                f'fill="{MUTED}">{name}</text>')
    return pan.svg()


# ---------------------------------------------------------------------------
# the four charts: what each class's own confidence cut does to detection of
# that class, and to the false persons written as that class
# ---------------------------------------------------------------------------
def class_curves(sweep, grid, base_thr, nc):
    """{class_index: [(threshold, point), ...]} — sweep ONE class over the
    whole grid with the others held at base_thr, so each class is isolated.
    One pass feeds all four charts (the Sweeper caches, so nothing is
    evaluated twice)."""
    out = {}
    for c in range(nc):
        rows = []
        for t in grid:
            v = list(base_thr)
            v[c] = t
            rows.append((t, sweep(v)))
        out[c] = rows
    return out


def panel_class(rows, cls_name, base_t, best_t, series):
    """One class: its detection metrics against its own confidence threshold.
    Y is fixed 0-1 so the three class charts can be read side by side."""
    grid = [t for t, _ in rows]
    pan = Panel(760, 340, grid[0], grid[-1], 0.0, 1.0,
                f"{cls_name}: what its own confidence threshold does",
                f"{cls_name} confidence threshold", "rate")
    for i, (key, label) in enumerate(series):
        pts = [(t, p["metrics"].get(key)) for t, p in rows]
        pts = [(t, v) for t, v in pts if isinstance(v, (int, float))]
        if not pts:
            continue
        d = " ".join(f"{'M' if j == 0 else 'L'}{pan.x(t):.1f},{pan.y(v):.1f}"
                     for j, (t, v) in enumerate(pts))
        pan.add(f'<path d="{d}" fill="none" stroke="{PALETTE[i % 5]}" '
                f'stroke-width="2"/>')
    for t, col, lab in ((base_t, "#c2381f", "prod"),
                        (best_t, "#111", "chosen")):
        if t is not None and grid[0] <= t <= grid[-1]:
            # keep the caption inside the frame at either edge
            frac = (t - grid[0]) / (grid[-1] - grid[0])
            anchor = ("start" if frac < 0.06 else
                      "end" if frac > 0.94 else "middle")
            dx = 3 if anchor == "start" else (-3 if anchor == "end" else 0)
            pan.add(f'<line x1="{pan.x(t):.1f}" y1="{pan.pad[2]}" '
                    f'x2="{pan.x(t):.1f}" y2="{pan.h - pan.pad[3]}" '
                    f'stroke="{col}" stroke-width="1.4" '
                    f'stroke-dasharray="5,4"/>')
            pan.add(f'<text x="{pan.x(t) + dx:.1f}" y="{pan.pad[2] - 5}" '
                    f'font-size="10" fill="{col}" text-anchor="{anchor}">'
                    f'{lab} {t:.2f}</text>')
    n = len(series)
    pan.add(f'<rect x="{500}" y="{pan.h - 44 - n * 15}" width="235" '
            f'height="{n * 15 + 8}" fill="white" opacity="0.92" rx="3"/>')
    for i, (key, label) in enumerate(series):
        y = pan.h - 46 - (n - 1 - i) * 15
        pan.add(f'<line x1="510" y1="{y - 4}" x2="530" y2="{y - 4}" '
                f'stroke="{PALETTE[i % 5]}" stroke-width="2.5"/>')
        pan.add(f'<text x="536" y="{y}" font-size="10.5" fill="{MUTED}">'
                f'{label}</text>')
    return pan.svg()


def panel_fp_by_class(curves, classes, arm, base_t, best_thr):
    """False persons written as each class, as THAT class's cut moves — the
    other half of the tradeoff the three class charts show."""
    grid = [t for t, _ in curves[0]]
    series = []
    for c, name in enumerate(classes):
        key = f"{arm}.fp_{name.lower()}_per_100"
        vals = [(t, p["metrics"].get(key)) for t, p in curves[c]]
        if all(isinstance(v, (int, float)) for _, v in vals):
            series.append((name, vals))
    hi = max((v for _, vals in series for _, v in vals), default=1.0) or 1.0
    pan = Panel(760, 340, grid[0], grid[-1], 0.0, hi,
                f"False persons on '{arm}' (no real people in it), by the "
                f"class written", "that class's confidence threshold",
                "false persons per 100 images")
    for i, (name, vals) in enumerate(series):
        d = " ".join(f"{'M' if j == 0 else 'L'}{pan.x(t):.1f},{pan.y(v):.1f}"
                     for j, (t, v) in enumerate(vals))
        pan.add(f'<path d="{d}" fill="none" stroke="{PALETTE[i % 5]}" '
                f'stroke-width="2"/>')
        if best_thr and i < len(best_thr):
            bt = best_thr[i]
            bv = next((v for t, v in vals if abs(t - bt) < 1e-9), None)
            if bv is not None:
                pan.add(f'<circle cx="{pan.x(bt):.1f}" cy="{pan.y(bv):.1f}" '
                        f'r="5" fill="none" stroke="{PALETTE[i % 5]}" '
                        f'stroke-width="2.5"/>')
    if grid[0] <= base_t <= grid[-1]:
        pan.add(f'<line x1="{pan.x(base_t):.1f}" y1="{pan.pad[2]}" '
                f'x2="{pan.x(base_t):.1f}" y2="{pan.h - pan.pad[3]}" '
                f'stroke="#c2381f" stroke-width="1.4" stroke-dasharray="5,4"/>')
        pan.add(f'<text x="{pan.x(base_t):.1f}" y="{pan.pad[2] - 5}" '
                f'font-size="10" fill="#c2381f" text-anchor="middle">'
                f'prod {base_t:.2f}</text>')  # always mid-grid, never clipped
    n = len(series)
    pan.add(f'<rect x="{560}" y="{60}" width="175" height="{n * 15 + 8}" '
            f'fill="white" opacity="0.92" rx="3"/>')
    for i, (name, _) in enumerate(series):
        pan.add(f'<line x1="570" y1="{72 + i * 15}" x2="590" '
                f'y2="{72 + i * 15}" stroke="{PALETTE[i % 5]}" '
                f'stroke-width="2.5"/>')
        pan.add(f'<text x="596" y="{76 + i * 15}" font-size="10.5" '
                f'fill="{MUTED}">written "{name}"</text>')
    return pan.svg()


def write_class_charts(out_dir: Path, sweep, grid, base_thr, best_thr, classes,
                       fp_arm):
    """The four charts: one per class + the false-positive counterpart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = class_curves(sweep, grid, base_thr, len(classes))
    written = []
    for c, name in enumerate(classes):
        k = name.lower()
        svg = panel_class(curves[c], name, base_thr[c],
                          best_thr[c] if best_thr else None,
                          [(f"lagenda.{k}_recall_e2e",
                            f"{name} blurred correctly (recall, end-to-end)"),
                           (f"lagenda.det_recall_{k}",
                            f"{name} detected at all"),
                           (f"lagenda.{k}_precision_matched",
                            f"written {name} and really is (precision)")])
        f = out_dir / f"chart_class_{k}.svg"
        f.write_text(svg)
        written.append(f)
    if fp_arm:
        f = out_dir / "chart_fp_by_class.svg"
        f.write_text(panel_fp_by_class(curves, classes, fp_arm,
                                       base_thr[0], best_thr))
        written.append(f)
    return written

def panel_scorecard(best, baseline, keys):
    """Baseline -> chosen for every headline metric. Each row is scaled to its
    own two values, so rates and per-100 counts can share one figure; the
    numbers are printed because the graph is for direction, not for reading
    values off an axis."""
    rows = []
    for k in keys:
        b, n = best["metrics"].get(k), baseline["metrics"].get(k)
        if isinstance(b, (int, float)) and isinstance(n, (int, float)):
            lower_better = any(t in k for t in ("fp", "leak", "dup"))
            rows.append((k, n, b, lower_better))
    h = 64 + len(rows) * 26
    L, R = 250, 96
    el = [f'<rect width="760" height="{h}" fill="white"/>',
          f'<text x="16" y="20" font-size="13" font-weight="600" fill="{INK}">'
          f'Baseline {"/".join(f"{t:.2f}" for t in baseline["thresholds"])} '
          f'&#8594; chosen {"/".join(f"{t:.2f}" for t in best["thresholds"])}'
          f'</text>',
          f'<text x="16" y="36" font-size="10" fill="{MUTED}">each row scaled '
          f'to its own range · green = moved the way the product wants</text>']
    for i, (k, n, b, lower_better) in enumerate(rows):
        y = 58 + i * 26
        lo, hi = min(n, b), max(n, b)
        span = (hi - lo) or 1.0
        xs = lambda v: L + (v - lo) / span * (760 - L - R)
        good = (b < n) == lower_better and abs(b - n) > 1e-12
        col = "#1d7a3e" if good else ("#c2381f" if abs(b - n) > 1e-12
                                      else MUTED)
        el.append(f'<text x="{L - 10}" y="{y + 4}" font-size="11" '
                  f'fill="{INK}" text-anchor="end">{k}</text>')
        el.append(f'<line x1="{xs(lo):.1f}" y1="{y}" x2="{xs(hi):.1f}" '
                  f'y2="{y}" stroke="#dcdcdc" stroke-width="3" '
                  f'stroke-linecap="round"/>')
        el.append(f'<circle cx="{xs(n):.1f}" cy="{y}" r="4" fill="white" '
                  f'stroke="{MUTED}" stroke-width="2"/>')
        el.append(f'<circle cx="{xs(b):.1f}" cy="{y}" r="5" fill="{col}"/>')
        el.append(f'<text x="{L - 10 - 0}" y="{y + 4}" font-size="11" '
                  f'fill="none">.</text>')
        el.append(f'<text x="{760 - R + 8}" y="{y + 4}" font-size="10.5" '
                  f'fill="{col}" font-variant-numeric="tabular-nums">'
                  f'{_fmt(n)} &#8594; {_fmt(b)}</text>')
    el.append(f'<circle cx="{L + 6}" cy="{h - 14}" r="4" fill="white" '
              f'stroke="{MUTED}" stroke-width="2"/>')
    el.append(f'<text x="{L + 16}" y="{h - 10}" font-size="10" fill="{MUTED}">'
              f'baseline</text>')
    el.append(f'<circle cx="{L + 90}" cy="{h - 14}" r="5" fill="{MUTED}"/>')
    el.append(f'<text x="{L + 100}" y="{h - 10}" font-size="10" '
              f'fill="{MUTED}">chosen</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{h}" '
            f'viewBox="0 0 760 {h}">' + "".join(el) + "</svg>")


def write_charts(out_dir: Path, points, best, baseline, sweep, grid, classes,
                 keys, objective, constraints, arms):
    """Standalone .svg files — one chart per file, self-contained, white
    background (renders in GitHub/VS Code/ClickUp like every other chart in
    this repo)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    charts = {"chart_scorecard.svg": panel_scorecard(best, baseline, keys),
              "chart_response.svg": panel_response(sweep, grid, best, classes,
                                                   f"objective: {objective}")}
    xkey = next((k for k in (referenced_metrics(" ".join(constraints))
                             + list(best["metrics"]))
                 if ".fp_" in k or k.endswith("fp_per_100")), None)
    ykey = next((k for k in referenced_metrics(objective)
                 if not k.startswith("baseline.")), None)
    if xkey and ykey and xkey != ykey:
        charts["chart_tradeoff.svg"] = panel_tradeoff(points, best, baseline,
                                                      xkey, ykey)
    if arms:
        charts["chart_fp_composition.svg"] = panel_fp_composition(
            best, baseline, arms, classes)
    for name, svg in charts.items():
        (out_dir / name).write_text(svg)
        written.append(out_dir / name)
    return written


def build_report(path: Path, meta, points, best, baseline, sweep, grid,
                 classes, keys, objective, constraints, arms):
    esc = lambda t: (str(t).replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;"))
    # rows: every headline metric, baseline -> chosen, with a Wilson CI where
    # the metric has counts behind it
    rows = []
    for k in keys:
        b, n = best["metrics"].get(k), baseline["metrics"].get(k)
        if not isinstance(b, (int, float)) or not isinstance(n, (int, float)):
            continue
        ci = ""
        if k in best["counts"]:
            kk, nn = best["counts"][k]
            _, lo, hi = wilson(kk, nn)
            ci = f"{max(0.0, lo):.4f}–{hi:.4f} <span class=n>({kk}/{nn})</span>"
        d = b - n
        # "better" is only knowable for the metrics we name; everything else
        # is reported without a verdict rather than guessed at
        lower_better = any(s in k for s in ("fp", "leak", "dup"))
        cls = "" if abs(d) < 1e-12 else \
            ("good" if (d < 0) == lower_better else "bad")
        rows.append(f"<tr><td>{esc(k)}</td><td>{n:.4f}</td><td>{b:.4f}</td>"
                    f"<td class='{cls}'>{d:+.4f}</td><td>{ci}</td></tr>")

    xkey = next((k for k in (referenced_metrics(" ".join(constraints))
                             + list(best["metrics"]))
                 if ".fp_" in k or k.endswith("fp_per_100")), None)
    ykey = next((k for k in referenced_metrics(objective)
                 if not k.startswith("baseline.")), None)
    charts = [panel_response(sweep, grid, best, classes,
                             f"objective: {objective}")]
    if xkey and ykey and xkey != ykey:
        charts.insert(0, panel_tradeoff(points, best, baseline, xkey, ykey))
    if arms:
        charts.append(panel_fp_composition(best, baseline, arms, classes))

    thr = " / ".join(f"<b>{c} {t:.2f}</b>"
                     for c, t in zip(classes, best["thresholds"]))
    cons = "".join(f"<li><code>{esc(c)}</code></li>" for c in constraints) \
        or "<li class=n>none</li>"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>conf_sweep — {esc(meta.get('run_label', 'threshold sweep'))}</title>
<style>
 body{{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
   margin:0;padding:28px 34px;color:#111;background:#fff;max-width:860px}}
 h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:14px;margin:28px 0 8px}}
 .n{{color:#666}} code{{background:#f4f4f4;padding:1px 4px;border-radius:3px;
   font-size:12px}}
 .card{{border:1px solid #e6e6e6;border-radius:8px;padding:14px 16px;
   margin:14px 0;background:#fafafa}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px}}
 th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #eee}}
 th{{color:#666;font-weight:600}} td:nth-child(2),td:nth-child(3),
 td:nth-child(4){{text-align:right;font-variant-numeric:tabular-nums}}
 .good{{color:#1d7a3e;font-weight:600}} .bad{{color:#c2381f;font-weight:600}}
 svg{{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px;
   margin:10px 0}}
 ul{{margin:6px 0;padding-left:20px}}
</style>
<h1>Confidence-threshold sweep</h1>
<div class=n>{esc(meta.get('run_label',''))} · {len(points)} thresholds
 evaluated · matcher IoU {meta.get('match_iou')} ·
 mode {esc(meta.get('mode'))}</div>
<div class=card>
 <div style="font-size:16px">Chosen: {thr}</div>
 <div class=n style="margin-top:6px">baseline
  {" / ".join(f"{t:.2f}" for t in baseline["thresholds"])} ·
  {"meets every constraint" if best["feasible"]
   else "<b style='color:#c2381f'>NO feasible threshold — this is the "
        "unconstrained best, not a recommendation</b>"}</div>
 <div style="margin-top:8px">maximise <code>{esc(objective)}</code></div>
 <ul>{cons}</ul>
</div>
<h2>Baseline vs chosen</h2>
<table><tr><th>metric</th><th>baseline</th><th>chosen</th><th>delta</th>
<th>95% CI (chosen)</th></tr>{''.join(rows)}</table>
<p class=n>Each dataset answers its own question and results are never pooled.
 LAGENDA is label-anchored, so its unmatched detections are NOT false
 positives — PASS, the object set and CrowdHuman measure those.</p>
{''.join(f'<h2>{t}</h2>{c}' for t, c in zip(
    ["The tradeoff", "What each class threshold buys", "Where the false persons come from"][:len(charts)],
    charts))}
<p class=n>Replayed offline from log-raw sidecars; no model was re-run.
 Reproduce every number with
 <code>python3 conf_sweep.py --crosscheck</code>.</p>
""")
    return path

# ---------------------------------------------------------------------------
# replay fidelity: does the sweep at t reproduce the labels the run emitted?
# ---------------------------------------------------------------------------
# sidecar confidences are stored rounded to 4 decimals; a detection sitting
# within this of the threshold can land on the wrong side of it in the replay
_ROUND_EPS = 5e-5


def verify_at(recs, labels_dir: Path, thr, log=print, max_report=5):
    n_checked = n_mismatch = n_missing = n_boundary = 0
    examples = []
    for stem, r in recs.items():
        lf = labels_dir / f"{stem}.txt"
        boxes = seg_boxes(lf, 1.0, 1.0)
        if boxes is None:
            n_missing += 1
            continue
        n_checked += 1
        act = active_mask(r.cls, r.conf, thr)
        replay = sorted(int(c) for c in r.cls[act]) if act.size else []
        emitted = sorted(int(b[0]) for b in boxes)
        if replay != emitted:
            n_mismatch += 1
            n_boundary += (len(replay) != len(emitted)
                           and any(abs(float(cf) - thr[int(c)]) <= _ROUND_EPS
                                   for c, cf in zip(r.cls, r.conf)))
            if len(examples) < max_report:
                # sidecars store conf rounded to 4dp, so a detection whose true
                # confidence sits a hair under the cut can round UP onto it and
                # the replay keeps a box the run dropped. Surface the confs at
                # the boundary so that is diagnosable, not mysterious.
                near = [round(float(cf), 4) for c, cf in zip(r.cls, r.conf)
                        if abs(float(cf) - thr[int(c)]) <= _ROUND_EPS]
                examples.append({"stem": stem, "replay_classes": replay,
                                 "emitted_classes": emitted,
                                 "confs_at_threshold": near})
    ok = n_mismatch == 0 and n_checked > 0
    log(f"  replay@{thr[0]:.2f} vs emitted labels: checked {n_checked}, "
        f"mismatched {n_mismatch}, no-label-file {n_missing} "
        f"-> {'OK' if ok else 'MISMATCH'}")
    if n_mismatch and n_boundary == n_mismatch:
        log(f"    all {n_mismatch} explained by 4dp rounding at the exact "
            f"threshold — harmless, and only ever at the cut itself")
    for e in examples:
        log(f"    {e['stem']}: replay {e['replay_classes']} != "
            f"emitted {e['emitted_classes']}"
            + (f"  confs at the cut: {e['confs_at_threshold']}"
               if e["confs_at_threshold"] else ""))
    return {"n_checked": n_checked, "n_mismatch": n_mismatch,
            "n_missing_label_file": n_missing,
            "n_mismatch_explained_by_rounding": n_boundary,
            "examples": examples, "ok": ok}


def list_metrics(log=print):
    for cls in (LagendaSet, CrowdSet, NegativeSet):
        log(f"\n{cls.kind}.<metric>   ({cls.__name__})")
        for k, v in cls.METRICS.items():
            log(f"  {k:<34} {v}")
    log("\nThe prefix is the arm's name: 'lagenda', 'crowd', and whatever you "
        "named each --negatives arm (e.g. objects, pass).")
    log("Objectives: any arithmetic over those names, e.g.")
    log("  --objective \"lagenda.woman_recall_e2e - 2*objects.img_fp_rate\"")
    log("  --constraint \"crowd.precision >= 0.85\" "
        "--constraint \"lagenda.leak_adult20_child <= 0.01\"")
    log("Presets: " + ", ".join("@" + k for k in PRESETS))


# ---------------------------------------------------------------------------
# selftest — synthetic, no data, no GPU, no network
# ---------------------------------------------------------------------------
def _write_sidecars(raw: Path, dets_by_stem, w=100, h=100):
    raw.mkdir(parents=True, exist_ok=True)
    for stem, dets in dets_by_stem.items():
        rows = [{"det_index": i, "cls": c, "cls_name": str(c),
                 "box_xyxy": list(b), "conf": cf, "kept": cf >= 0.45,
                 "excluded": None}
                for i, (c, b, cf) in enumerate(dets)]
        (raw / f"{stem}.json").write_text(json.dumps(
            {"image": f"{stem}.jpg", "width": w, "height": h,
             "detections": rows}))


def _selftest():
    import random
    import tempfile
    from run_model_children import iou, match_boxes

    # 1. the replay matcher must agree with the repo's matcher, ties
    #    included. Coarse integer boxes on purpose: they manufacture IoU ties,
    #    which is exactly where a re-implementation drifts. Each detection
    #    carries a unique trailing id so the comparison is by identity, not by
    #    value (match_boxes returns the det tuple, not its index).
    rng = random.Random(20260813)
    n_tie_trials = 0
    for trial in range(300):
        def box():
            x1, y1 = rng.randint(0, 3), rng.randint(0, 3)
            return [x1, y1, x1 + rng.randint(1, 2), y1 + rng.randint(1, 2)]
        gts = [box() for _ in range(rng.randint(0, 4))]
        dets = [[rng.randint(0, 2)] + box() + [uid]
                for uid in range(rng.randint(0, 5))]
        det_boxes = [d[1:5] for d in dets]
        # both a strict and a loose IoU gate: the loose one lets many more
        # pairs through, which is where equal-IoU ties actually show up
        for min_iou in (MATCH_IOU, 0.1):
            pairs, best_iou = candidate_pairs(gts, det_boxes, min_iou)
            n_tie_trials += len({j for j, _, _ in pairs}) < len(pairs)

            ref = match_boxes(gts, dets, min_iou)
            mine = greedy_match(pairs, [True] * len(dets))
            assert {gi: d[5] for gi, (d, _) in ref.items()} == \
                   {gi: dets[di][5] for gi, di in mine.items()}, \
                f"matcher drift on trial {trial} @{min_iou}: {ref} vs {mine}"

            # and with an arbitrary subset active, it must equal match_boxes
            # run on exactly that subset — this is what a threshold does
            active = [rng.random() > 0.4 for _ in dets]
            sub = [d for d, a in zip(dets, active) if a]
            ref2 = match_boxes(gts, sub, min_iou)
            mine2 = greedy_match(pairs, active)
            assert {gi: d[5] for gi, (d, _) in ref2.items()} == \
                   {gi: dets[di][5] for gi, di in mine2.items()}, \
                f"subset drift on trial {trial} @{min_iou}: {ref2} vs {mine2}"

            # best_iou (the crowd FP buckets) is threshold-independent
            for di, d in enumerate(dets):
                assert abs(best_iou[di] - max((iou(g, d[1:5]) for g in gts),
                                              default=0.0)) < 1e-9

    assert n_tie_trials > 20, \
        f"only {n_tie_trials} trials produced IoU ties — the tie-break path " \
        f"is barely exercised, tighten the box grid"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        classes = list(DEFAULT_CLASSES)

        # 2. negatives ------------------------------------------------------
        _write_sidecars(root / "obj" / "raw", {
            "a": [(0, [10, 10, 40, 40], 0.90)],
            "b": [(1, [10, 10, 40, 40], 0.30)],
            "c": [(0, [1, 1, 20, 20], 0.60), (2, [50, 50, 90, 90], 0.50)],
            "d": [],
        })
        recs_o, _ = load_dets(root / "obj" / "raw", "sidecar")
        check_classes(recs_o, 3, "objects")
        neg = NegativeSet("objects", recs_o, classes)
        m, _c = neg.evaluate([0.45] * 3)
        assert m["fp_total"] == 3 and m["fp_per_100"] == 75.0, m
        assert m["img_fp_rate"] == 0.5, m
        assert m["fp_woman_per_100"] == 50.0 and m["fp_child_per_100"] == 25.0, m
        assert neg.evaluate([0.95] * 3)[0]["fp_total"] == 0
        # per-class thresholds move each class independently
        m2, _ = neg.evaluate([0.70, 0.20, 0.20])
        assert m2["fp_total"] == 3 and m2["img_fp_rate"] == 0.75, m2

        # 3. LAGENDA --------------------------------------------------------
        lag = root / "lag"
        (lag / "labels").mkdir(parents=True)
        (lag / "labels" / "L1.txt").write_text("0 0.3 0.5 0.2 0.6\n")
        (lag / "labels" / "L2.txt").write_text("0 0.5 0.5 0.4 0.8\n")
        (lag / "gt.jsonl").write_text(
            json.dumps({"id": "L1_0", "image": "L1.jpg", "gt_age": 30,
                        "gt_gender": "female"}) + "\n"
            + json.dumps({"id": "L2_0", "image": "L2.jpg", "gt_age": 25,
                          "gt_gender": "male"}) + "\n")
        _write_sidecars(lag / "raw", {
            "L1": [(0, [20, 20, 40, 80], 0.80), (1, [50, 20, 70, 80], 0.20)],
            "L2": [(2, [30, 10, 70, 90], 0.60)],
        })
        recs_l, _ = load_dets(lag / "raw", "sidecar")
        ls = LagendaSet(recs_l, lag / "gt.jsonl", lag / "labels", classes)
        assert ls.n_scored == 2, (ls.n_scored, ls.n_untranslatable)
        m, c = ls.evaluate([0.45] * 3)
        assert m["det_recall"] == 1.0 and m["acc3"] == 0.5, m
        assert m["woman_recall_e2e"] == 1.0 and m["man_recall_e2e"] == 0.0, m
        assert m["gender_acc_adults"] == 0.5, m
        assert m["leak_adult20_child"] == 0.5, m       # the 25-yr-old man
        assert m["leak_adult20_child_count"] == 1 and m["n_adults20_matched"] == 2
        assert m["n_matched"] == 2, m
        # size bands: images are 100px tall; L1's GT box is 0.6 of that (60px,
        # small), L2's is 0.8 (80px, medium). Both are found at 0.45.
        assert c["det_recall_small"] == (1, 1) and c["det_recall_med"] == (1, 1), c
        assert c["det_recall_large"] == (0, 0), c
        # only L1 is a woman, and she is correctly written Woman
        assert c["woman_recall_e2e_small"] == (1, 1), c
        assert c["woman_recall_e2e_med"] == (0, 0), c
        # raise the Woman cut past her confidence: the small band collapses,
        # the medium band (the man) is untouched
        ms, cs = ls.evaluate([0.85, 0.45, 0.45])
        assert cs["det_recall_small"] == (0, 1), cs
        assert cs["det_recall_med"] == (1, 1), cs
        assert ms["woman_recall_e2e_small"] == 0.0, ms
        assert c["woman_recall_e2e"] == (1, 1), c
        assert m["woman_precision_matched"] == 1.0, m     # 1 of 1 written Woman
        assert m["child_precision_matched"] == 0.0, m     # the misread man
        assert c["child_precision_matched"] == (0, 1), c
        m, _ = ls.evaluate([0.65] * 3)
        assert m["det_recall"] == 0.5 and m["acc3"] == 1.0, m
        assert m["leak_adult20_child"] == 0.0, m       # he is no longer detected
        # raising ONLY the Child threshold rescues him as a detection loss,
        # not as a Man — the per-class knob does what it says
        m, _ = ls.evaluate([0.45, 0.45, 0.65])
        assert m["det_recall"] == 0.5 and m["leak_adult20_child"] == 0.0, m
        # rate vs count: dropping the woman leaves the leaking man as the only
        # matched adult, so the RATE doubles to 1.0 while the COUNT is
        # unchanged at 1 — constraining the rate would call that a regression
        m, _ = ls.evaluate([0.85, 0.45, 0.45])
        assert m["leak_adult20_child"] == 1.0 and \
            m["leak_adult20_child_count"] == 1 and \
            m["n_adults20_matched"] == 1, m

        # 4. crowd ----------------------------------------------------------
        crowd = root / "crowd"
        crowd.mkdir()
        (crowd / "gt.odgt").write_text(json.dumps({
            "ID": "c1", "gtboxes": [
                {"tag": "person", "vbox": [10, 10, 50, 50],
                 "fbox": [10, 10, 50, 50], "extra": {"ignore": 0}},
                {"tag": "person", "vbox": [100, 100, 40, 40],
                 "fbox": [100, 100, 80, 80], "extra": {"ignore": 0}},
                {"tag": "mask", "fbox": [150, 10, 40, 40]}]}) + "\n")
        _write_sidecars(crowd / "raw", {
            "c1": [(0, [10, 10, 60, 60], 0.90),      # exact match, person 1
                   (1, [155, 15, 185, 45], 0.50),    # inside the ignore region
                   (0, [0, 150, 30, 180], 0.60)],    # clear false positive
        }, w=200, h=200)
        recs_c, _ = load_dets(crowd / "raw", "sidecar")
        cs = CrowdSet(recs_c, crowd / "gt.odgt", classes)
        m, c = cs.evaluate([0.45] * 3)
        assert cs.n_gt == 2 and m["dets_kept"] == 2, (cs.n_gt, m)
        assert m["recall"] == 0.5 and m["precision"] == 0.5, m
        assert m["recall_light"] == 1.0 and m["recall_heavy"] == 0.0, m
        assert m["clear_fp_total"] == 1 and m["clear_fp_per_100"] == 100.0, m
        m, _ = cs.evaluate([0.70] * 3)
        assert m["precision"] == 1.0 and m["clear_fp_total"] == 0, m

        # 5. objective language ---------------------------------------------
        datasets = {"lagenda": ls, "crowd": cs, "objects": neg}
        obj = expand_preset("woman_recall_vs_fp", ["objects"], 1.0)
        assert obj == "lagenda.woman_recall_e2e - 1.0*(objects.img_fp_rate)", obj
        sweep = Sweeper(datasets, obj, ["crowd.precision >= 0.9"])
        p = sweep([0.45] * 3)
        assert abs(p["objective"] - (1.0 - 0.5)) < 1e-9, p["objective"]
        assert p["feasible"] is False and p["failed_constraints"], p
        assert sweep([0.70] * 3)["feasible"] is True
        assert referenced_metrics(obj) == ["lagenda.woman_recall_e2e",
                                           "objects.img_fp_rate"]
        # an arm named after a Python keyword must still work: PASS is one of
        # our datasets and `pass.img_fp_rate` is a syntax error to Python
        kw = Sweeper({"lagenda": ls, "pass": neg},
                     "lagenda.woman_recall_e2e - 2*pass.img_fp_rate",
                     ["pass.fp_per_100 <= 80"])
        pk = kw([0.45] * 3)
        assert abs(pk["objective"] - (1.0 - 2 * 0.5)) < 1e-9, pk["objective"]
        assert pk["feasible"] is True, pk
        assert referenced_metrics("pass.img_fp_rate - pass.img_fp_rate") == \
            ["pass.img_fp_rate"]

        for bad in ("__import__('os')", "open('x')", "lagenda.nope",
                    "objects.img_fp_rate.__class__"):
            try:
                eval_expr(bad, p["metrics"])
                raise AssertionError(f"unsafe/unknown expr accepted: {bad}")
            except SystemExit:
                pass

        # a no-regression constraint written against the baseline must hold
        # AT the baseline (equality) — pasting a rounded literal is what makes
        # a whole run come back "NO FEASIBLE THRESHOLD"
        base_m, _ = evaluate_datasets(datasets, (0.45, 0.45, 0.45))
        nr = Sweeper(datasets, "lagenda.woman_recall_e2e",
                     ["objects.fp_per_100 <= baseline.objects.fp_per_100"],
                     base_m)
        assert nr([0.45] * 3)["feasible"] is True, nr([0.45] * 3)
        assert nr([0.95] * 3)["feasible"] is True     # fewer FPs, still fine
        assert nr([0.10] * 3)["feasible"] is False, "more FPs must be blocked"
        nr2 = Sweeper(datasets, "lagenda.woman_recall_e2e",
                      ["lagenda.acc3 >= baseline.lagenda.acc3"], base_m)
        assert nr2([0.45] * 3)["feasible"] is True
        # a threshold that detects nobody scores accuracy 0, not "undefined" —
        # so an accuracy floor correctly rejects it instead of being vacuous
        assert nr2([0.95] * 3)["metrics"]["lagenda.acc3"] == 0.0
        assert nr2([0.95] * 3)["feasible"] is False
        # the truncated-literal trap itself: 4dp below the true value
        trap = Sweeper(datasets, "lagenda.woman_recall_e2e",
                       [f"objects.fp_per_100 <= {base_m['objects.fp_per_100']:.4f}"],
                       base_m)
        assert base_m["objects.fp_per_100"] == 75.0  # exact here, so no trap
        assert trap([0.45] * 3)["feasible"] is True
        try:
            Sweeper(datasets, "baseline.lagenda.acc3", [], None)([0.45] * 3)
            raise AssertionError("baseline.X without baselines must fail loudly")
        except SystemExit:
            pass

        # 6. search + best-pick ---------------------------------------------
        grid = frange("0.10:0.90:0.10")
        pts = search_global(sweep, grid, 3)
        assert len(pts) == 9 and all(len(set(p["thresholds"])) == 1 for p in pts)
        best = pick_best(pts)
        assert best is not None and best["feasible"], best
        # ties resolve toward the higher (fewer-boxes) threshold
        flat = Sweeper(datasets, "0*lagenda.acc3 + 1", [])
        tied = search_global(flat, grid, 3)
        assert pick_best(tied)["thresholds"] == [grid[-1]] * 3, \
            pick_best(tied)["thresholds"]
        front = pareto_front(pts, [("max", "lagenda.woman_recall_e2e"),
                                   ("min", "objects.img_fp_rate")])
        assert front and all(p in pts for p in front)
        pc = search_per_class(sweep, frange("0.20:0.80:0.30"),
                              frange("0.10:0.90:0.10"), 3, log=lambda *a: None)
        assert len(pc) > 9 and any(len(set(p["thresholds"])) > 1 for p in pc)
        # the coarse grid and its refinement overlap -> duplicates, which would
        # be double-counted in the report
        assert len(dedupe(pc)) < len(pc), "expected overlap to dedupe"
        assert len(dedupe(pc)) == len({tuple(p["thresholds"]) for p in pc})
        co = search_coord(sweep, grid, 3, [0.45] * 3, log=lambda *a: None)
        assert co and pick_best(co) is not None

        # 7. replay fidelity vs the labels a real run emitted ----------------
        lbl = root / "obj" / "labels"
        lbl.mkdir()
        for stem, dets in (("a", [(0, 0.90)]), ("b", []), ("c", [(0, 0.60),
                                                                 (2, 0.50)]),
                           ("d", [])):
            (lbl / f"{stem}.txt").write_text(
                "".join(f"{c} 0.5 0.5 0.1 0.1\n" for c, _ in dets))
        v = verify_at(recs_o, lbl, [0.45] * 3, log=lambda *a: None)
        assert v["ok"] and v["n_checked"] == 4 and v["n_mismatch"] == 0, v
        (lbl / "b.txt").write_text("1 0.5 0.5 0.1 0.1\n")   # corrupt one
        v = verify_at(recs_o, lbl, [0.45] * 3, log=lambda *a: None)
        assert not v["ok"] and v["n_mismatch"] == 1, v
        assert v["n_mismatch_explained_by_rounding"] == 0, v

        # a real 4dp-rounding boundary case: the run saw 0.44998 (dropped it),
        # the sidecar stored 0.45 (the replay keeps it) — flagged AND explained
        rnd = root / "rnd"
        _write_sidecars(rnd / "raw", {"r": [(0, [1, 1, 9, 9], 0.45)]})
        (rnd / "labels").mkdir(parents=True)
        (rnd / "labels" / "r.txt").write_text("")
        recs_r, _ = load_dets(rnd / "raw", "sidecar")
        v = verify_at(recs_r, rnd / "labels", [0.45] * 3, log=lambda *a: None)
        assert v["n_mismatch"] == 1 and \
            v["n_mismatch_explained_by_rounding"] == 1, v
        assert v["examples"][0]["confs_at_threshold"] == [0.45], v

        # 8. outputs write and re-read --------------------------------------
        out = root / "out"
        write_outputs(out, {"objective": obj}, pts, best, pts[0], front)
        j = json.loads((out / "sweep.json").read_text())
        assert j["best"]["thresholds"] == best["thresholds"]
        assert (out / "sweep.csv").read_text().count("\n") == len(pts) + 1
        svg = plot_svg(pts, referenced_metrics(obj), best["thresholds"][0],
                       0.45, out / "sweep.svg")
        assert svg and svg.read_text().startswith("<svg") and \
            "woman_recall_e2e" in svg.read_text()

        svgs = write_charts(out / "charts", pts, best, sweep([0.45] * 3),
                            sweep, grid, classes, headline_keys(datasets),
                            obj, ["crowd.precision >= 0.9"], ["objects"])
        assert len(svgs) == 4, svgs
        for f in svgs:
            t = f.read_text()
            assert t.startswith("<svg") and t.endswith("</svg>"), f
            for bad in ("<script", "src=", 'href="http', "@import"):
                assert bad not in t, f"{f.name} must be self-contained: {bad}"
        cc = write_class_charts(out / "charts", sweep, grid, [0.45] * 3,
                                best["thresholds"], classes, "objects")
        assert len(cc) == 4, cc
        assert {f.name for f in cc} == {"chart_class_woman.svg",
                                        "chart_class_man.svg",
                                        "chart_class_child.svg",
                                        "chart_fp_by_class.svg"}, cc
        for f in cc:
            t = f.read_text()
            assert t.startswith("<svg") and t.endswith("</svg>")
            for bad in ("<script", "src=", 'href="http', "@import"):
                assert bad not in t, f"{f.name} must be self-contained"
        w = (out / "charts" / "chart_class_woman.svg").read_text()
        assert "Woman" in w and "prod 0.45" in w
        card = (out / "charts" / "chart_scorecard.svg").read_text()
        assert "woman_recall_e2e" in card and "baseline" in card

        rep = build_report(out / "report.html",
                           {"run_label": "selftest", "match_iou": 0.5,
                            "mode": "global"},
                           pts, best, sweep([0.45] * 3), sweep, grid, classes,
                           headline_keys(datasets), obj,
                           ["crowd.precision >= 0.9"], ["objects"])
        html = rep.read_text()
        assert html.startswith("<!doctype html>") and html.count("<svg") == 3
        # self-contained: nothing to FETCH, or it will not render off the pod.
        # (the xmlns="http://www.w3.org/2000/svg" namespace is not a fetch)
        for bad in ("<script", "src=", 'href="http', "@import", "url(http"):
            assert bad not in html, f"report must be self-contained: {bad}"
        assert "woman_recall_e2e" in html and "selftest" in html

        # 9. the 6-field label-with-confidence loader ------------------------
        lc = root / "lc"
        lc.mkdir()
        (lc / "x.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n"
                                  "1 0.2 0.2 0.1 0.1 0.3\n")
        r2, _ = load_dets(lc, "labels-conf")
        assert len(r2["x"].cls) == 2 and r2["x"].space == "norm"
        n2 = NegativeSet("lc", r2, classes)
        assert n2.evaluate([0.45] * 3)[0]["fp_total"] == 1

    print("selftest OK")


# ---------------------------------------------------------------------------
# crosscheck — the replay must reproduce the repo's OWN scorers, number for
# number, on the same data. Builds a synthetic corpus, materializes the labels
# conf 0.45 would have emitted, runs eval_negatives_crowd.py and
# run_autolabel_on_manifest.py over them, and diffs against this tool's
# metrics at 0.45. No GPU, no network; needs PIL (already a repo dependency).
# ---------------------------------------------------------------------------
def _crosscheck(log=print):
    import random
    import tempfile
    from PIL import Image

    import eval_negatives_crowd as ENC
    import run_autolabel_on_manifest as RAM

    rng = random.Random(4242)
    classes = list(DEFAULT_CLASSES)
    names = RAM.DEFAULT_CLASS_NAMES
    W, H = 640, 480

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        def materialize(arm, dets_by_stem):
            """images/ + raw/ + pred_labels/ exactly as a real run leaves them."""
            _write_sidecars(root / arm / "raw", dets_by_stem, W, H)
            (root / arm / "images").mkdir(parents=True, exist_ok=True)
            (root / arm / "pred_labels").mkdir(parents=True, exist_ok=True)
            for stem, dets in dets_by_stem.items():
                Image.new("RGB", (W, H), (30, 30, 30)).save(
                    root / arm / "images" / f"{stem}.jpg")
                lines = [f"{c} {(b[0] + b[2]) / 2 / W:.6f} "
                         f"{(b[1] + b[3]) / 2 / H:.6f} "
                         f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
                         for c, b, cf in dets if cf >= 0.45]
                (root / arm / "pred_labels" / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))

        def rbox(w=(40, 120), h=(60, 160)):
            x, y = rng.randint(0, W - w[1]), rng.randint(0, H - h[1])
            return [x, y, x + rng.randint(*w), y + rng.randint(*h)]

        checks = []

        # --- negatives ----------------------------------------------------
        neg = {}
        for i in range(60):
            dets = [(rng.choice([0, 1, 2]), rbox(), rng.betavariate(2, 2))
                    for _ in range(rng.choice([0, 0, 1, 1, 2]))]
            neg[f"n_{i:03d}"] = dets
        materialize("neg", neg)
        recs, _ = load_dets(root / "neg" / "raw", "sidecar")
        m, _c = NegativeSet("neg", recs, classes).evaluate([0.45] * 3)
        o = ENC.run_negatives(root / "neg" / "images",
                              root / "neg" / "pred_labels",
                              root / "canon_neg", names, 0)
        checks += [("negatives fp_total", m["fp_total"], o["fp_total"]),
                   ("negatives fp_per_100", round(m["fp_per_100"], 2),
                    o["fp_per_100_images"]),
                   ("negatives img_fp_rate", round(m["img_fp_rate"], 4),
                    o["pct_images_with_fp"]["rate"])]

        # --- crowd --------------------------------------------------------
        crowd, odgt = {}, []
        for i in range(25):
            stem, persons, dets = f"c_{i:03d}", [], []
            for _ in range(rng.randint(1, 6)):
                x, y = rng.randint(0, 520), rng.randint(0, 340)
                vw, vh = rng.randint(50, 100), rng.randint(70, 130)
                persons.append({"tag": "person", "vbox": [x, y, vw, vh],
                                "fbox": [x, y, vw,
                                         int(vh * rng.uniform(1.0, 2.4))],
                                "extra": {"ignore": 0}})
                if rng.random() < 0.75:      # found, with a jittered box
                    j = rng.randint(0, 8)
                    dets.append((rng.choice([0, 1]),
                                 [x + j, y + j, x + vw - j, y + vh - j],
                                 rng.betavariate(3, 2)))
            if rng.random() < 0.4:           # a hallucination in the corner
                dets.append((0, [600, 440, 639, 479], rng.betavariate(2, 2)))
            if rng.random() < 0.3:           # an ignore region + a det inside it
                persons.append({"tag": "mask", "fbox": [0, 0, 90, 90]})
                dets.append((1, [5, 5, 80, 80], 0.9))
            odgt.append({"ID": stem, "gtboxes": persons})
            crowd[stem] = dets
        materialize("crowd", crowd)
        (root / "crowd" / "gt.odgt").write_text(
            "".join(json.dumps(r) + "\n" for r in odgt))
        recs, _ = load_dets(root / "crowd" / "raw", "sidecar")
        m, cnt = CrowdSet(recs, root / "crowd" / "gt.odgt", classes).evaluate(
            [0.45] * 3)
        o = ENC.run_crowd(root / "crowd" / "images", root / "crowd" / "gt.odgt",
                          root / "crowd" / "pred_labels", root / "canon_crowd",
                          names, MATCH_IOU, 0)
        ub = o["unmatched_breakdown"]
        checks += [("crowd n_gt_persons", m["n_gt_persons"], o["n_gt_persons"]),
                   ("crowd dets_kept", m["dets_kept"], o["dets_total_kept"]),
                   ("crowd recall", round(m["recall"], 4),
                    o["detection_recall"]["rate"]),
                   ("crowd recall k", cnt["recall"][0],
                    o["detection_recall"]["k"]),
                   ("crowd precision", round(m["precision"], 4),
                    o["detection_precision"]["rate"]),
                   ("crowd dup_rate", round(m["dup_rate"], 4),
                    ub["duplicates_vs_matched_persons"]["rate"]),
                   ("crowd partial_overlap", m["partial_overlap"],
                    ub["partial_overlap_ambiguous"]),
                   ("crowd clear_fp_total", m["clear_fp_total"],
                    ub["clear_false_positives"]),
                   ("crowd clear_fp_per_100", round(m["clear_fp_per_100"], 2),
                    ub["clear_fp_per_100_images"])]
        for band, v in o["recall_by_occlusion"].items():
            checks.append((f"crowd recall_{band}",
                           round(m[f"recall_{band}"], 4), v["rate"]))

        # --- LAGENDA ------------------------------------------------------
        lag, gt_rows = {}, []
        (root / "lag" / "gt_labels").mkdir(parents=True)
        for i in range(120):
            stem = f"l_{i:03d}"
            n_people = rng.choice([1, 1, 1, 2])
            gt_lines, dets = [], []
            for k in range(n_people):
                cls_true, age, gender = rng.choice(
                    [(0, rng.randint(19, 70), "female"),
                     (1, rng.randint(19, 70), "male"),
                     (2, rng.randint(2, 11), rng.choice(["female", "male"]))])
                cx, cy = 0.25 + 0.45 * k, 0.5
                gt_lines.append(f"0 {cx:.4f} {cy:.4f} 0.20 0.70")
                gt_rows.append({"id": f"{stem}_{k}", "image": f"{stem}.jpg",
                                "gt_age": age, "gt_gender": gender})
                if rng.random() < 0.85:      # detected, sometimes misclassified
                    pred = cls_true if rng.random() > 0.15 \
                        else rng.choice([0, 1, 2])
                    jx = rng.uniform(-0.02, 0.02)
                    dets.append((pred, [(cx - 0.10 + jx) * W, 0.15 * H,
                                        (cx + 0.10 + jx) * W, 0.85 * H],
                                 rng.betavariate(3, 2)))
            if rng.random() < 0.3:
                dets.append((rng.choice([0, 1, 2]), rbox(), rng.betavariate(2, 3)))
            (root / "lag" / "gt_labels" / f"{stem}.txt").write_text(
                "\n".join(gt_lines) + "\n")
            lag[stem] = dets
        materialize("lag", lag)
        (root / "lag" / "gt.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in gt_rows))
        recs, _ = load_dets(root / "lag" / "raw", "sidecar")
        m, cnt = LagendaSet(recs, root / "lag" / "gt.jsonl",
                            root / "lag" / "gt_labels", classes).evaluate(
            [0.45] * 3)
        o = RAM.score(RAM.load_gt(root / "lag" / "gt.jsonl", 0),
                      root / "lag" / "gt_labels", root / "lag" / "images",
                      root / "lag" / "pred_labels", root / "canon_lag",
                      names, MATCH_IOU, 0.1, 0)
        d, cl = o["detection"], o["classification_on_matched"]
        checks += [("lagenda det_recall", round(m["det_recall"], 4),
                    d["recall"]["rate"]),
                   ("lagenda det_recall k", cnt["det_recall"][0],
                    d["recall"]["k"]),
                   ("lagenda det_recall n", cnt["det_recall"][1],
                    d["recall"]["n"]),
                   ("lagenda acc3", round(m["acc3"], 4),
                    cl["accuracy_3class"]["rate"]),
                   ("lagenda acc3 k", cnt["acc3"][0],
                    cl["accuracy_3class"]["k"]),
                   ("lagenda gender_acc_adults",
                    round(m["gender_acc_adults"], 4),
                    cl["gender_accuracy_adults"]["rate"]),
                   ("lagenda gender_adults n", cnt["gender_acc_adults"][1],
                    cl["gender_accuracy_adults"]["n"]),
                   ("lagenda child_recall_e2e", round(m["child_recall_e2e"], 4),
                    cl["child_recall_end_to_end"]["rate"])]
        for c in classes:
            checks.append((f"lagenda det_recall_{c}",
                           round(m[f"det_recall_{c.lower()}"], 4),
                           d["recall_by_class"][c]["rate"]))

        bad = [(n, a, b) for n, a, b in checks if a != b]
        for n, a, b in checks:
            log(f"  {'OK  ' if a == b else 'DIFF'} {n:<28} "
                f"replay={a!r:<10} canonical={b!r}")
        log(f"\n{len(checks) - len(bad)}/{len(checks)} metrics identical to "
            f"eval_negatives_crowd.py / run_autolabel_on_manifest.py "
            f"at conf 0.45")
        if bad:
            raise SystemExit(f"CROSSCHECK FAILED on {len(bad)} metric(s): "
                             + ", ".join(n for n, _, _ in bad))
    print("crosscheck OK")


# ---------------------------------------------------------------------------
def build_datasets(args, classes, log=print):
    datasets, meta, raw_by_arm = {}, {}, {}
    nc = len(classes)
    if args.lagenda:
        if not (args.lagenda_gt and args.lagenda_gt_labels):
            raise SystemExit("--lagenda needs --lagenda-gt and "
                             "--lagenda-gt-labels")
        recs, info = load_dets(Path(args.lagenda), args.format, args.max_images)
        check_classes(recs, nc, "lagenda")
        ds = LagendaSet(recs, args.lagenda_gt, args.lagenda_gt_labels, classes,
                        args.match_iou, args.limit_gt)
        datasets["lagenda"] = ds
        raw_by_arm["lagenda"] = (recs, Path(args.lagenda))
        meta["lagenda"] = {**info, "n_scored_persons": ds.n_scored,
                           "n_gt_not_processed": ds.n_not_processed,
                           "n_gt_untranslatable": ds.n_untranslatable,
                           "n_gt_bad_box_index": ds.n_bad_index,
                           "observed_conf_floor": observed_floor(recs)}
    if args.crowd:
        if not args.crowd_odgt:
            raise SystemExit("--crowd needs --crowd-odgt")
        recs, info = load_dets(Path(args.crowd), args.format, args.max_images)
        check_classes(recs, nc, "crowd")
        ds = CrowdSet(recs, args.crowd_odgt, classes, args.match_iou)
        datasets["crowd"] = ds
        raw_by_arm["crowd"] = (recs, Path(args.crowd))
        meta["crowd"] = {**info, "n_gt_persons": ds.n_gt,
                         "n_images_without_gt_entry": ds.n_missing_gt,
                         "observed_conf_floor": observed_floor(recs)}
    for spec in args.negatives or []:
        if ":" not in spec:
            raise SystemExit(f"--negatives wants NAME:PATH, got {spec!r}")
        name, path = spec.split(":", 1)
        if name in datasets:
            raise SystemExit(f"duplicate arm name {name!r}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise SystemExit(f"arm name {name!r} must be a plain identifier "
                             f"(letters/digits/underscore) — it is used in "
                             f"objective expressions as {name}.<metric>")
        recs, info = load_dets(Path(path), args.format, args.max_images)
        check_classes(recs, nc, name)
        datasets[name] = NegativeSet(name, recs, classes)
        raw_by_arm[name] = (recs, Path(path))
        meta[name] = {**info, "observed_conf_floor": observed_floor(recs)}
    if not datasets:
        raise SystemExit("nothing to sweep — pass --lagenda / --crowd / "
                         "--negatives (see --help)")
    for name, m in meta.items():
        log(f"[data] {name}: {m['n_images']} images"
            + (f", {m['n_scored_persons']} labeled people" if "n_scored_persons" in m else "")
            + (f", {m['n_gt_persons']} GT persons" if "n_gt_persons" in m else "")
            + f", conf floor observed {m['observed_conf_floor']}")
    return datasets, meta, raw_by_arm


def default_objective(datasets, neg_names, fp_weight):
    if "lagenda" in datasets and neg_names:
        return expand_preset("woman_recall_vs_fp", neg_names, fp_weight), True
    if "lagenda" in datasets:
        return PRESETS["woman_f1"], True
    if neg_names:
        return expand_preset("min_fp", neg_names, fp_weight), True
    return PRESETS["crowd_f1"], True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--lagenda", help="raw sidecar dir for the LAGENDA arm")
    ap.add_argument("--lagenda-gt", help="gt.jsonl (human age + gender)")
    ap.add_argument("--lagenda-gt-labels", help="GT YOLO label dir")
    ap.add_argument("--crowd", help="raw sidecar dir for the CrowdHuman arm")
    ap.add_argument("--crowd-odgt", help="CrowdHuman annotation .odgt")
    ap.add_argument("--negatives", action="append", metavar="NAME:RAWDIR",
                    help="a people-free arm, repeatable "
                         "(e.g. objects:/workspace/exp12/<arm>/object_set/raw)")
    ap.add_argument("--format", default="sidecar",
                    choices=["sidecar", "labels-conf"],
                    help="sidecar = run_ultralytics_labels.py raw/*.json; "
                         "labels-conf = 6-field 'cls cx cy w h conf' labels")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="class names in id order (one threshold per class)")
    ap.add_argument("--mode", default="global",
                    choices=["global", "per-class"])
    ap.add_argument("--search", default="grid", choices=["grid", "coord"],
                    help="per-class only: coarse grid + local refine, or "
                         "coordinate ascent from --baseline (faster, local)")
    ap.add_argument("--grid", default="0.05:0.95:0.01",
                    help="start:stop:step for the fine grid")
    ap.add_argument("--coarse-step", type=float, default=0.10,
                    help="per-class grid search: coarse step before refining")
    ap.add_argument("--objective", default=None,
                    help="expression over dataset.metric, or @preset "
                         "(see --list-metrics)")
    ap.add_argument("--fp-weight", type=float, default=1.0,
                    help="weight on the false-positive term in the presets")
    ap.add_argument("--constraint", action="append", default=[],
                    help="comparison that a threshold must satisfy to be "
                         "chosen, repeatable")
    ap.add_argument("--pareto", default=None, metavar="max:A,min:B",
                    help="also report the Pareto front over two metrics")
    ap.add_argument("--baseline", type=float, default=0.45,
                    help="threshold to compare against (production = 0.45)")
    ap.add_argument("--match-iou", type=float, default=MATCH_IOU)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--report-metrics", default=None,
                    help="extra metrics to add to the printed tables, "
                         "comma-separated (they are all in sweep.json anyway)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--limit-gt", type=int, default=0)
    ap.add_argument("--verify-at", type=float, default=None,
                    help="replay this threshold and diff against each arm's "
                         "sibling labels/ dir — proves the offline replay "
                         "reproduces what the run actually emitted")
    ap.add_argument("--out", help="output dir (sweep.json/.csv/.svg)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--report", default=None,
                    help="ALSO write the charts + table as one HTML file here "
                         "(optional; --out already writes standalone .svg)")
    ap.add_argument("--fp-arm", default=None,
                    help="which negatives arm the false-positive chart uses "
                         "(default: the first --negatives arm)")
    ap.add_argument("--label", default=None,
                    help="run label shown in the report header")
    ap.add_argument("--list-metrics", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--crosscheck", action="store_true",
                    help="prove the replay reproduces this repo's own scorers "
                         "number-for-number on synthetic data (no GPU/network)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.crosscheck:
        _crosscheck()
        return
    if args.list_metrics:
        list_metrics()
        return

    classes = [c.strip() for c in args.classes.split(",")]
    nc = len(classes)
    datasets, meta, raw_by_arm = build_datasets(args, classes)
    neg_names = [n for n, d in datasets.items() if d.kind == "negatives"]

    if args.objective and args.objective.startswith("@"):
        objective = expand_preset(args.objective[1:], neg_names, args.fp_weight)
        chosen = f"preset {args.objective}"
    elif args.objective:
        objective, chosen = args.objective, "user"
    else:
        objective, _ = default_objective(datasets, neg_names, args.fp_weight)
        chosen = "DEFAULT (no --objective given)"
    print(f"\n[objective] {chosen}: maximise  {objective}")
    for c in args.constraint:
        print(f"[constraint] {c}")

    if args.verify_at is not None:
        print(f"\n[verify] replaying {args.verify_at:.2f} against the emitted "
              f"label files")
        verification = {}
        for name, (recs, raw_dir) in raw_by_arm.items():
            lbl = raw_dir.parent / "labels"
            if not lbl.is_dir():
                print(f"  {name}: no sibling labels/ dir at {lbl} — skipped")
                continue
            if args.lagenda_gt_labels and \
                    lbl.resolve() == Path(args.lagenda_gt_labels).resolve():
                print(f"  {name}: {lbl} is the human GT label dir, not the "
                      f"model's emitted labels — skipped (verifying the replay "
                      f"against GT would compare two different things)")
                continue
            print(f"  {name}:")
            verification[name] = verify_at(recs, lbl, [args.verify_at] * nc)
        meta["verification"] = verification

    fine = frange(args.grid)
    lo = min(m.get("observed_conf_floor") or 1.0 for m in meta.values()
             if isinstance(m, dict) and "observed_conf_floor" in m)
    if fine[0] < lo - 1e-9:
        print(f"\nWARNING: grid starts at {fine[0]:.3f} but the dumps only log "
              f"down to {lo:.3f} — thresholds below that are NOT measurable "
              f"from these sidecars (re-dump with a lower --floor to explore "
              f"them). Points below {lo:.3f} all reproduce the floor.")

    base_metrics, _ = evaluate_datasets(datasets, tuple([args.baseline] * nc))
    sweep = Sweeper(datasets, objective, args.constraint, base_metrics)
    if args.mode == "global":
        points = search_global(sweep, fine, nc)
    elif args.search == "grid":
        coarse = frange(f"{fine[0]}:{fine[-1]}:{args.coarse_step}")
        points = search_per_class(sweep, coarse, fine, nc)
    else:
        points = search_coord(sweep, fine, nc, [args.baseline] * nc)

    points = dedupe(points)
    baseline = sweep([args.baseline] * nc)
    best = pick_best(points)
    if best is None:
        print("\nNO FEASIBLE THRESHOLD — every point failed a --constraint.")
        blocked = Counter(c for p in points for c in p["failed_constraints"])
        for c, n in blocked.most_common():
            print(f"  blocked {n}/{len(points)} points:  {c}")
        if baseline["failed_constraints"]:
            print(f"\n  The BASELINE ({args.baseline}) fails these too:")
            for c in baseline["failed_constraints"]:
                lhs = referenced_metrics(c)[0]
                print(f"    {c}   <- {lhs} is "
                      f"{baseline['metrics'].get(lhs)!r} at the baseline")
            print("  A constraint the baseline itself cannot meet is almost "
                  "always a hand-copied bound rounded the wrong way — write "
                  "it as `<= baseline.<arm>.<metric>` instead.")
        print("\nReporting the best by objective regardless (NOT a "
              "recommendation — it ignores every constraint):")
        best = max(points, key=lambda p: p["objective"])

    edge = [t for t in best["thresholds"] if t <= fine[0] + 1e-9
            or t >= fine[-1] - 1e-9]
    if edge:
        print(f"\nWARNING: the chosen threshold sits at the EDGE of the grid "
              f"({edge}). That usually means one term dominates the objective "
              f"— the search would keep going if the grid allowed it. Widen "
              f"--grid, re-weight (--fp-weight), or express the other side as "
              f"a --constraint instead of a weight.")

    keys = headline_keys(datasets)
    # whatever you optimised is always shown: a comparison table that omits
    # the objective's own metric makes the result unreadable
    for k in referenced_metrics(objective):
        if not k.startswith("baseline.") and k not in keys \
                and k in best["metrics"]:
            keys.append(k)
    if args.report_metrics:
        for k in (x.strip() for x in args.report_metrics.split(",")):
            if k and k not in keys:
                if k not in best["metrics"]:
                    raise SystemExit(f"--report-metrics: unknown metric {k!r} "
                                     f"(see --list-metrics)")
                keys.append(k)
    ranked = sorted(points, key=lambda p: (p["objective"],
                                           sum(p["thresholds"])), reverse=True)
    print_table(ranked[:args.top], keys,
                f"top {min(args.top, len(ranked))} of {len(points)} thresholds "
                f"by objective  (X = fails a constraint)")
    print_compare(best, baseline, keys)

    front = []
    if args.pareto:
        axes = []
        for part in args.pareto.split(","):
            d, k = part.split(":")
            axes.append((d.strip(), k.strip()))
        front = pareto_front(points, axes)
        print_table(front, keys, f"Pareto front over {args.pareto}")

    if args.report and not args.out:
        meta["run_label"] = args.label or "threshold sweep"
        meta["match_iou"], meta["mode"] = args.match_iou, args.mode
        build_report(Path(args.report), meta, points, best, baseline, sweep,
                     fine, classes, keys, objective, args.constraint,
                     [n for n, d in datasets.items() if d.kind == "negatives"])
        print(f"\n[out] {args.report}   <- open this one")
    if args.out:
        out = Path(args.out)
        meta.update({"objective": objective, "objective_source": chosen,
                     "constraints": args.constraint, "mode": args.mode,
                     "grid": args.grid, "classes": classes,
                     "match_iou": args.match_iou, "baseline": args.baseline,
                     "n_points": len(points), "argv": sys.argv})
        write_outputs(out, meta, points, best, baseline, front)
        print(f"\n[out] {out}/sweep.json  {out}/sweep.csv")
        arms = [n for n, d in datasets.items() if d.kind == "negatives"]
        for f in write_charts(out, points, best, baseline, sweep, fine,
                              classes, keys, objective, args.constraint, arms):
            print(f"[out] {f}")
        if "lagenda" in datasets:
            for f in write_class_charts(out, sweep, fine,
                                        [args.baseline] * nc,
                                        best["thresholds"], classes,
                                        args.fp_arm or (arms[0] if arms
                                                        else None)):
                print(f"[out] {f}")
        if args.report:
            meta["run_label"] = args.label or out.name
            build_report(Path(args.report), meta, points, best, baseline,
                         sweep, fine, classes, keys, objective,
                         args.constraint, arms)
            print(f"[out] {args.report}  (HTML, optional)")
        if args.mode == "global" and not args.no_plot:
            plot_keys = [k for k in referenced_metrics(objective)
                         if not k.startswith("baseline.")] or keys[:3]
            p = plot_svg(points, plot_keys,
                         best["thresholds"][0], args.baseline,
                         out / "sweep.svg")
            if p:
                print(f"[out] {p}")


if __name__ == "__main__":
    main()
