#!/usr/bin/env python3
"""
EXP-2026-02 — score the production SAM3 auto-labeler against LAGENDA human GT.

Two-stage design (so production code runs untouched):

  Stage A (pod, GPU): run the production `autolabel_sam.py` UNCHANGED over the
      LAGENDA images — its native output is YOLO label .txt files. Frozen
      production settings (conf, mask-IoU NMS, mask_to_polygon). Nothing here
      re-implements SAM; we score exactly what production would have written.

  Stage B (this script, CPU-only): compare those predicted label files against
      the human answer key (`gt.jsonl`, per-person gt_age/gt_gender — the same
      manifest EXP-2026-01 used). GT (age, gender) is translated to the
      production {Woman, Man, Child} taxonomy via translation.py, so both
      experiments score on identical definitions.

    python3 run_autolabel_on_manifest.py \
        --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
        --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
        --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
        --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
        --out         /workspace/lagenda_eval/sam_autolabel/eval

    python3 run_autolabel_on_manifest.py --selftest   # synthetic end-to-end check

Format notes (verified against the real data/code on the pod, 2026-07-09):
  - gt.jsonl rows are {id, image, gt_age, gt_gender}; the GT *box* lives in the
    YOLO label file next to the image (--gt-labels), and the trailing "_N" in
    the id is the box's line index in that file.
  - autolabel_sam.py writes YOLO *segment* lines (class + normalized polygon,
    variable length), not boxes — pred boxes are the polygon's extent. Images
    the labeler has processed always have a .txt (possibly empty = zero
    detections); a MISSING .txt means "not processed yet" and the person is
    excluded from metrics (lets a pilot be scored while a run is partial).

Outputs (in --out):
  sam_matches.jsonl  one record per GT person:
      {id, image, gt_class, pred_class, match_iou, dual_classes, status}
      status = correct | wrong_class | missed | gt_untranslatable
  summary.json       detection recall/precision, 3-class confusion + accuracy,
                     child recall (detected-only AND end-to-end), gender
                     accuracy on adults, dual-class survivor rate, matched-IoU
                     histogram — each rate with a Wilson 95% CI.
  overlays/          debug images for mismatches/misses (green GT, red pred).

Reuses (does NOT duplicate): match_boxes/iou/draw_match from
run_model_children.py, yolo_boxes from dataset_utils.py, wilson from
eval_taxonomy.py, production_3class from translation.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import dataset_utils as du
import translation as T
from eval_taxonomy import wilson
from run_model_children import draw_match, iou, match_boxes

# production class ids == the --classes prompt order in run_autolabel.sh
DEFAULT_CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child"}

_PROD3 = T.get("production_3class").heads[0]


def gt_class(row) -> str | None:
    """Human (age, gender) -> Woman/Man/Child, or None if undecidable."""
    return _PROD3.label(T.norm_gender(row.get("gt_gender")),
                        T.age_bucket_from_years(row.get("gt_age")))


def seg_boxes(label_file: Path, w: int, h: int):
    """Parse a YOLO label file that may contain segment polygons (class + 2N
    normalized coords) or plain boxes (class + cxcywh). Returns
    [(cls, x1, y1, x2, y2)] in pixels; polygon boxes are the polygon extent."""
    if not label_file.exists():
        return None                       # not processed (distinct from empty)
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:                # box: cx cy w h
            cx, cy, bw, bh = vals
            boxes.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
        else:                             # polygon: x y x y ...
            xs, ys = vals[0::2], vals[1::2]
            if len(xs) < 3 or len(xs) != len(ys):
                continue
            boxes.append((cls, min(xs) * w, min(ys) * h,
                          max(xs) * w, max(ys) * h))
    return boxes


def load_gt(path: Path, limit: int):
    """gt.jsonl -> {image_name: [row, ...]}, row = {id, box_index, gt_*}."""
    by_image = defaultdict(list)
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uid = r["id"]
            try:
                box_index = int(uid.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                raise SystemExit(f"gt id {uid!r}: expected '<stem>_<boxindex>'")
            by_image[Path(r["image"]).name].append({
                "id": uid, "box_index": box_index,
                "gt_age": r.get("gt_age"), "gt_gender": r.get("gt_gender"),
            })
            n += 1
            if limit and n >= limit:
                break
    return by_image


def score(gt_by_image, gt_labels_dir: Path, images_dir: Path, pred_dir: Path,
          out_dir: Path, class_names, match_iou_thr, dual_iou_thr,
          max_overlays):
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = out_dir / "overlays"
    matches_f = open(out_dir / "sam_matches.jsonl", "w")

    rows_out = []           # per-person records
    conf = Counter()        # (gt_class, pred_class) on matched+translatable
    ious = []               # matched IoUs
    n_dets_total = 0
    n_dets_matched = 0
    n_overlays = 0
    n_not_processed = 0     # GT persons in images the labeler hasn't reached
    missing_images = []
    bad_gt = []             # id whose box_index isn't in the GT label file

    for img_name, gt_rows in sorted(gt_by_image.items()):
        img_path = images_dir / img_name
        if not img_path.exists():
            missing_images.append(img_name)
            continue
        with Image.open(img_path) as im:
            w, h = im.size

        dets = seg_boxes(pred_dir / (Path(img_name).stem + ".txt"), w, h)
        if dets is None:                  # no .txt = labeler hasn't run here
            n_not_processed += len(gt_rows)
            continue
        n_dets_total += len(dets)

        gt_all = du.yolo_boxes(gt_labels_dir / (Path(img_name).stem + ".txt"),
                               w, h)
        gt_rows_ok = []
        for r in gt_rows:
            if r["box_index"] >= len(gt_all):
                bad_gt.append(r["id"])
                continue
            r["box"] = list(gt_all[r["box_index"]][1:5])
            gt_rows_ok.append(r)
        gt_rows = gt_rows_ok

        matched = match_boxes([r["box"] for r in gt_rows], dets, match_iou_thr)
        n_dets_matched += len(matched)

        for gi, r in enumerate(gt_rows):
            gcls = gt_class(r)
            det, j = matched.get(gi, (None, None))
            pcls = class_names.get(det[0], f"class_{det[0]}") if det else None
            # every det overlapping this person — the cross-class NMS leak check.
            # But two SAM boxes both clipping this GT box doesn't necessarily
            # mean one physical person got double-labeled — it could be two
            # DIFFERENT nearby people (e.g. a man sitting behind a child) each
            # legitimately detected under their own class. dual_mutual_iou (the
            # overlap the two SAM boxes have with EACH OTHER, not just with the
            # GT box) tells them apart: high -> same object, low -> two people.
            dual_dets = [d for d in dets if iou(r["box"], d[1:5]) >= dual_iou_thr]
            dual = sorted({class_names.get(d[0], f"class_{d[0]}") for d in dual_dets})
            dual_mutual_iou = None
            if len(dual) > 1:
                dual_mutual_iou = 0.0
                for i1 in range(len(dual_dets)):
                    for i2 in range(i1 + 1, len(dual_dets)):
                        d1, d2 = dual_dets[i1], dual_dets[i2]
                        if class_names.get(d1[0]) != class_names.get(d2[0]):
                            dual_mutual_iou = max(dual_mutual_iou, iou(d1[1:5], d2[1:5]))
            if gcls is None:
                status = "gt_untranslatable"
            elif det is None:
                status = "missed"
            elif pcls == gcls:
                status = "correct"
            else:
                status = "wrong_class"
            if status in ("correct", "wrong_class"):
                conf[(gcls, pcls)] += 1
                ious.append(j)
            # for a wrong_class call: was there ALSO a same-class detection
            # nearby that lost the greedy IoU match (e.g. a mom holding a
            # child — SAM correctly boxes both, but only one wins the match)?
            # A non-trivial value here means the "error" may be a matching
            # artifact of a crowded/overlapping scene, not a real misread.
            near_miss_iou = None
            if status == "wrong_class":
                same_cls_dets = [d for d in dets if d is not det
                                 and class_names.get(d[0], f"class_{d[0]}") == gcls]
                near_miss_iou = max((iou(r["box"], d[1:5]) for d in same_cls_dets),
                                    default=0.0)
            rec = {"id": r["id"], "image": img_name, "gt_age": r.get("gt_age"),
                   "gt_gender": r.get("gt_gender"),
                   "gt_class": gcls, "pred_class": pcls,
                   "match_iou": round(j, 4) if j is not None else None,
                   "dual_classes": dual, "status": status,
                   "dual_mutual_iou": round(dual_mutual_iou, 4) if dual_mutual_iou is not None else None,
                   "near_miss_iou": round(near_miss_iou, 4) if near_miss_iou is not None else None,
                   "n_gt_in_image": len(gt_rows), "n_dets_in_image": len(dets)}
            rows_out.append(rec)
            matches_f.write(json.dumps(rec) + "\n")

            if (status in ("wrong_class", "missed") or len(dual) > 1) \
                    and n_overlays < max_overlays:
                ov_dir.mkdir(exist_ok=True)
                with Image.open(img_path) as im:
                    ann = draw_match(im.convert("RGB"), r["box"], det,
                                     j or 0.0, pcls or "MISS")
                ann.save(ov_dir / f"{r['id']}_{status}.jpg", quality=90)
                n_overlays += 1

    matches_f.close()
    if missing_images:
        print(f"WARNING: {len(missing_images)} GT images not found under "
              f"{images_dir} (first: {missing_images[:3]})")
    if bad_gt:
        print(f"WARNING: {len(bad_gt)} GT ids whose box_index isn't in the GT "
              f"label file (first: {bad_gt[:3]})")
    s = summarize(rows_out, conf, ious, n_dets_total, n_dets_matched,
                  len(missing_images))
    s["n_gt_not_processed_yet"] = n_not_processed
    s["n_gt_bad_box_index"] = len(bad_gt)
    return s


def _rate(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(p, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


# an image with this many more SAM detections than labeled GT persons is
# treated as a real crowd scene (LAGENDA only labels the "main" person(s)),
# not evidence of hallucination — see crowd_diagnostics in the summary.
CROWD_EXCESS_THRESHOLD = 3
# IoU a same-class-as-GT detection needs, elsewhere in the same image, to
# count as "plausibly the real match that lost to a nearby overlapping person"
NEAR_MISS_IOU_THRESHOLD = 0.3
# how much two different-class SAM boxes need to overlap EACH OTHER (not just
# the GT box) to count as a genuine same-object double-label, vs. two
# different nearby people each legitimately boxed under their own class
DUAL_MUTUAL_IOU_THRESHOLD = 0.5


def summarize(rows, conf, ious, n_dets_total, n_dets_matched, n_missing_imgs):
    scored = [r for r in rows if r["status"] != "gt_untranslatable"]
    matched = [r for r in scored if r["status"] in ("correct", "wrong_class")]
    correct = [r for r in matched if r["status"] == "correct"]

    def excess(r):
        return r["n_dets_in_image"] - r["n_gt_in_image"]

    matched_simple = [r for r in matched if excess(r) < CROWD_EXCESS_THRESHOLD]
    matched_crowded = [r for r in matched if excess(r) >= CROWD_EXCESS_THRESHOLD]
    images_seen = {r["image"]: excess(r) for r in rows}
    wrong_class_rows = [r for r in matched if r["status"] == "wrong_class"]
    near_miss = [r for r in wrong_class_rows
                if (r.get("near_miss_iou") or 0.0) >= NEAR_MISS_IOU_THRESHOLD]

    per_class_recall = {}   # detection recall by GT class
    for c in ("Woman", "Man", "Child"):
        cls_rows = [r for r in scored if r["gt_class"] == c]
        per_class_recall[c] = _rate(
            sum(r["status"] != "missed" for r in cls_rows), len(cls_rows))

    child_det = [r for r in matched if r["gt_class"] == "Child"]
    child_all = [r for r in scored if r["gt_class"] == "Child"]
    adults = [r for r in matched if r["gt_class"] in ("Woman", "Man")]
    dual = [r for r in matched if len(r["dual_classes"]) > 1]
    dual_confirmed = [r for r in dual
                      if (r.get("dual_mutual_iou") or 0.0) >= DUAL_MUTUAL_IOU_THRESHOLD]

    hist = Counter()
    for j in ious:
        hist[f"{int(j * 10) / 10:.1f}"] += 1

    summary = {
        "n_gt_persons": len(rows),
        "n_gt_untranslatable": len(rows) - len(scored),
        "n_gt_images_missing": n_missing_imgs,
        "detection": {
            "recall": _rate(len(matched), len(scored)),
            "recall_by_class": per_class_recall,
            # LAGENDA doesn't annotate every person in frame, so unmatched
            # detections are an UPPER BOUND on hallucination, not a measurement
            "dets_total": n_dets_total,
            "dets_unmatched_upper_bound": _rate(
                n_dets_total - n_dets_matched, n_dets_total),
        },
        "classification_on_matched": {
            "accuracy_3class": _rate(len(correct), len(matched)),
            "confusion": {f"{g}->{p}": n
                          for (g, p), n in sorted(conf.items())},
            "child_recall_detected_only": _rate(
                sum(r["status"] == "correct" for r in child_det),
                len(child_det)),
            "child_recall_end_to_end": _rate(
                sum(r["status"] == "correct" for r in child_all),
                len(child_all)),
            "gender_accuracy_adults": _rate(
                sum(r["status"] == "correct" for r in adults), len(adults)),
        },
        "pathologies": {
            # naive: any 2 different-class SAM boxes both clipping this GT
            # person. Some of these are actually two DIFFERENT nearby people
            # (e.g. a man sitting behind a child), each legitimately boxed —
            # see dual_class_survivors_confirmed for the mutual-IoU-filtered,
            # more trustworthy version of this number.
            "dual_class_survivors_naive": _rate(len(dual), len(matched)),
            "dual_mutual_iou_threshold": DUAL_MUTUAL_IOU_THRESHOLD,
            "dual_class_survivors_confirmed": _rate(len(dual_confirmed), len(matched)),
            "matched_iou_histogram": dict(sorted(hist.items())),
            "matched_iou_below_0.75": _rate(
                sum(j < 0.75 for j in ious), len(ious)),
        },
        "child_age_diagnostics": child_age_diagnostics(matched),
        "crowd_diagnostics": {
            # LAGENDA labels only the "main" person(s), so multi-person /
            # crowd images get penalized by the naive unmatched-detection
            # count above. This splits detection *and* classification
            # accuracy by whether the image had a plausible real crowd.
            "crowd_excess_threshold": CROWD_EXCESS_THRESHOLD,
            "n_images_scored": len(images_seen),
            "n_crowded_images": sum(1 for e in images_seen.values() if e >= CROWD_EXCESS_THRESHOLD),
            "accuracy_3class_simple_images": _rate(
                sum(r["status"] == "correct" for r in matched_simple), len(matched_simple)),
            "accuracy_3class_crowded_images": _rate(
                sum(r["status"] == "correct" for r in matched_crowded), len(matched_crowded)),
            # of the wrong_class errors: how many had a same-class SAM box
            # elsewhere in the image that lost the greedy match (IoU >= 0.3)?
            # A high rate here means the "error" is often a matching artifact
            # from overlapping/crowded people (e.g. mom holding a child), not
            # a genuine misread — the near-miss box is the real answer.
            "near_miss_iou_threshold": NEAR_MISS_IOU_THRESHOLD,
            "wrong_class_plausible_crowd_artifact": _rate(
                len(near_miss), len(wrong_class_rows)),
        },
    }
    return summary


# ---------------------------------------------------------------------------
# where does SAM3's "child" prompt actually draw the line? Pure post-hoc
# analysis of already-collected predictions — no re-run of the labeler.
# Cutoffs are the project's standard band edges (see CLAUDE.md: 9/10, 12/13,
# 17/18 — the only points a clean child/adult split can be measured at).
# ---------------------------------------------------------------------------
_CUTOFF_SCHEMES = {9: "child_vs_adult_9", 12: "child_vs_adult", 17: "child_vs_adult_18"}


def child_age_diagnostics(matched_rows):
    # SAM's own binary call, regardless of what cutoff we score it against
    pred_binary = [(r, "Child" if r["pred_class"] == "Child" else "Adult")
                   for r in matched_rows if r.get("gt_age") is not None]

    # 1. child-call rate by true age (5-yr bands) — SAM's implicit boundary
    by_band = defaultdict(lambda: [0, 0])   # band -> [n, n_called_child]
    for r, pb in pred_binary:
        b = int(r["gt_age"] // 5) * 5
        by_band[b][0] += 1
        by_band[b][1] += (pb == "Child")
    call_rate_by_age = {
        str(b): _rate(n_child, n) for b, (n, n_child) in sorted(by_band.items())
    }

    # 2. accuracy/recall of SAM's "Child" call under each candidate cutoff
    sweep = {}
    for cutoff, scheme in _CUTOFF_SCHEMES.items():
        head = T.get(scheme).heads[0]
        pairs = []
        for r, pb in pred_binary:
            gb = head.label(T.norm_gender(r.get("gt_gender")),
                            T.age_bucket_from_years(r["gt_age"]))
            if gb is not None:
                pairs.append((pb, gb))
        n = len(pairs)
        acc = sum(pb == gb for pb, gb in pairs)
        gt_child = [gb for _, gb in pairs if gb == "Child"]
        gt_adult = [gb for _, gb in pairs if gb == "Adult"]
        child_recall = sum(pb == "Child" for pb, gb in pairs if gb == "Child")
        adult_recall = sum(pb == "Adult" for pb, gb in pairs if gb == "Adult")
        sweep[f"cutoff_{cutoff}"] = {
            "binary_accuracy": _rate(acc, n),
            "catches_kids": _rate(child_recall, len(gt_child)),
            "catches_adults": _rate(adult_recall, len(gt_adult)),
        }

    return {"call_rate_by_age": call_rate_by_age, "cutoff_sweep": sweep}


# ---------------------------------------------------------------------------
# synthetic self-test — no data or GPU needed
# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for d in ("images", "gt_labels", "pred_labels"):
            (root / d).mkdir()
        # image a (100x100): adult woman GT box 0 at (10,10,50,90),
        #                    child GT box 1 at (60,10,90,90)
        # image b: labeler hasn't processed it (no pred .txt)
        # image c: a real crowd — 1 labeled GT, 4 extra unrelated SAM detections
        #          elsewhere in frame (correctly matched, should NOT look "wrong")
        # image d: mom-holding-child style overlap — SAM's Man box wins the
        #          match over GT's true Child box, but SAM's OWN Child box is
        #          sitting right next to it (near-miss, a likely matching
        #          artifact rather than a genuine misread)
        # image e: the "man sitting behind a child" case (from a real gallery
        # screenshot) — GT is drawn generously wide; SAM's Child box (left
        # side, correctly matched) and Man box (right side, a genuinely
        # different real person) both clip >=50% of the GT box, so the naive
        # dual-class check flags this GT person — but the two SAM boxes barely
        # overlap EACH OTHER, so it should NOT count as a confirmed same-object
        # double-label.
        for name in ("a", "b", "c", "d", "e"):
            Image.new("RGB", (100, 100), (128,) * 3).save(root / f"images/{name}.jpg")
        (root / "gt_labels/a.txt").write_text(
            "0 0.30 0.50 0.40 0.80\n"     # box_index 0 (class id here is unused)
            "0 0.75 0.50 0.30 0.80\n"     # box_index 1
            "0 0.03 0.03 0.04 0.04\n")    # box_index 2 (missed by preds)
        (root / "gt_labels/b.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        (root / "gt_labels/c.txt").write_text("0 0.30 0.30 0.40 0.40\n")   # box [10,10,50,50]
        (root / "gt_labels/d.txt").write_text("0 0.50 0.50 0.20 0.20\n")  # box [40,40,60,60]
        (root / "gt_labels/e.txt").write_text("0 0.50 0.50 0.80 0.80\n")  # box [10,10,90,90]
        with open(root / "gt.jsonl", "w") as f:
            f.write(json.dumps({"id": "a_0", "image": "a.jpg",
                                "gt_age": 30, "gt_gender": "F"}) + "\n")
            f.write(json.dumps({"id": "a_1", "image": "a.jpg",
                                "gt_age": 8, "gt_gender": "M"}) + "\n")
            f.write(json.dumps({"id": "a_2", "image": "a.jpg",
                                "gt_age": 40, "gt_gender": "M"}) + "\n")
            f.write(json.dumps({"id": "b_0", "image": "b.jpg",
                                "gt_age": 25, "gt_gender": "F"}) + "\n")
            f.write(json.dumps({"id": "c_0", "image": "c.jpg",
                                "gt_age": 35, "gt_gender": "F"}) + "\n")
            f.write(json.dumps({"id": "d_0", "image": "d.jpg",
                                "gt_age": 8, "gt_gender": "M"}) + "\n")
            f.write(json.dumps({"id": "e_0", "image": "e.jpg",
                                "gt_age": 8, "gt_gender": "M"}) + "\n")
        # predictions in autolabel_sam.py's segment format (class + polygon):
        # woman polygon over a_0 -> correct; Man polygon over a_1 -> wrong_class;
        # plus a Woman polygon also over a_1 -> dual-class survivor; a_2 missed.
        (root / "pred_labels/a.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.90 0.10 0.90\n"
            "1 0.60 0.10 0.90 0.10 0.90 0.90 0.60 0.90\n"
            "0 0.59 0.09 0.91 0.09 0.91 0.91 0.59 0.91\n")
        # image c: Woman box exactly on GT (correct) + 4 tiny unrelated Man
        # boxes far away (a real, uncounted crowd -> excess=4 dets over 1 GT)
        (root / "pred_labels/c.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n"
            "1 0.60 0.60 0.70 0.60 0.70 0.70 0.60 0.70\n"
            "1 0.71 0.60 0.81 0.60 0.81 0.70 0.71 0.70\n"
            "1 0.60 0.71 0.70 0.71 0.70 0.81 0.60 0.81\n"
            "1 0.71 0.71 0.81 0.71 0.81 0.81 0.71 0.81\n")
        # image d: Man box exactly on GT (IoU 1.0, wins the match -> wrong_class)
        # + SAM's own Child box shifted alongside it (IoU ~0.33 with GT — a
        # same-class near-miss that lost the greedy match)
        (root / "pred_labels/d.txt").write_text(
            "1 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60\n"
            "2 0.50 0.40 0.70 0.40 0.70 0.60 0.50 0.60\n")
        # image e: Child box [10,10,60,90] (IoU 0.625 with GT -> wins the
        # match, correct) + Man box [45,10,95,90] (IoU 0.529 with GT -> also
        # clears the dual-leak threshold) but the two SAM boxes only overlap
        # EACH OTHER by IoU ~0.18 -> should NOT be a "confirmed" duplicate
        (root / "pred_labels/e.txt").write_text(
            "2 0.10 0.10 0.60 0.10 0.60 0.90 0.10 0.90\n"
            "1 0.45 0.10 0.95 0.10 0.95 0.90 0.45 0.90\n")
        s = score(load_gt(root / "gt.jsonl", 0), root / "gt_labels",
                  root / "images", root / "pred_labels", root / "eval",
                  DEFAULT_CLASS_NAMES, 0.5, 0.5, 10)
        det = s["detection"]["recall"]
        acc = s["classification_on_matched"]["accuracy_3class"]
        dual_naive = s["pathologies"]["dual_class_survivors_naive"]
        dual_confirmed = s["pathologies"]["dual_class_survivors_confirmed"]
        assert det["k"] == 5 and det["n"] == 6, det          # only a_2 missed
        assert acc["k"] == 3 and acc["n"] == 5, acc          # a_0, c_0, e_0 correct; a_1, d_0 wrong
        assert dual_naive["k"] == 2, dual_naive              # a_1 (real dup) + e_0 (2 diff people)
        assert dual_confirmed["k"] == 1, dual_confirmed      # only a_1 survives the mutual-IoU filter
        assert s["classification_on_matched"]["confusion"] == \
            {"Child->Child": 1, "Child->Man": 2, "Woman->Woman": 2}
        assert s["n_gt_not_processed_yet"] == 1              # b_0 excluded
        n_lines = sum(1 for _ in open(root / "eval/sam_matches.jsonl"))
        assert n_lines == 6

        diag = s["child_age_diagnostics"]
        assert diag["call_rate_by_age"]["30"]["rate"] == 0.0   # a_0 -> Woman
        # a_1, d_0 -> Man (not Child); e_0 -> Child (correct) -> 1/3 in this band
        r5 = diag["call_rate_by_age"]["5"]
        assert r5["k"] == 1 and r5["n"] == 3, r5
        # SAM misses a_1, d_0 but correctly catches e_0 -> 1/3 at every cutoff
        for c in (9, 12, 17):
            sw = diag["cutoff_sweep"][f"cutoff_{c}"]
            assert sw["catches_kids"]["k"] == 1 and sw["catches_kids"]["n"] == 3, (c, sw)

        cd = s["crowd_diagnostics"]
        # image c is the only one with excess dets over the threshold (4 >= 3)
        assert cd["n_crowded_images"] == 1, cd
        assert cd["accuracy_3class_crowded_images"]["k"] == 1 and \
            cd["accuracy_3class_crowded_images"]["n"] == 1, cd          # c_0 correct
        assert cd["accuracy_3class_simple_images"]["k"] == 2 and \
            cd["accuracy_3class_simple_images"]["n"] == 4, cd           # a_0, e_0 correct; a_1, d_0 wrong
        # 2 wrong_class errors total (a_1, d_0); only d_0 has a same-class
        # detection nearby (the near-miss Child box) -> 1/2 plausible artifact
        assert cd["wrong_class_plausible_crowd_artifact"]["k"] == 1 and \
            cd["wrong_class_plausible_crowd_artifact"]["n"] == 2, cd
    print("run_autolabel_on_manifest.py self-test passed")


def main():
    ap = argparse.ArgumentParser(
        description="score SAM auto-labeler YOLO labels against LAGENDA gt.jsonl")
    ap.add_argument("--gt-manifest", help="gt.jsonl with per-person gt_age/gt_gender")
    ap.add_argument("--gt-labels", help="dir with the LAGENDA GT YOLO .txt label files")
    ap.add_argument("--images", help="dir with the LAGENDA images")
    ap.add_argument("--pred-labels", help="dir with autolabel_sam.py YOLO .txt output")
    ap.add_argument("--out", help="output dir")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N GT persons; 0 = all (images the "
                         "labeler hasn't processed are auto-excluded anyway)")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--dual-iou", type=float, default=0.5,
                    help="IoU for the dual-class-survivor check")
    ap.add_argument("--overlays", type=int, default=40,
                    help="max debug overlay images to write")
    ap.add_argument("--class-names", nargs="+", default=None,
                    help="pred class names by id, default: Woman Man Child")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    for req in ("gt_manifest", "gt_labels", "images", "pred_labels", "out"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required (or use --selftest)")

    names = (dict(enumerate(args.class_names)) if args.class_names
             else DEFAULT_CLASS_NAMES)
    gt = load_gt(Path(args.gt_manifest), args.limit)
    print(f"GT: {sum(len(v) for v in gt.values())} persons in {len(gt)} images")
    summary = score(gt, Path(args.gt_labels), Path(args.images),
                    Path(args.pred_labels), Path(args.out), names,
                    args.match_iou, args.dual_iou, args.overlays)
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
