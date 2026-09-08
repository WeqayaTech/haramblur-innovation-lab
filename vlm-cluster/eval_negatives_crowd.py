#!/usr/bin/env python3
"""
EXP-2026-03 — score the production SAM3 auto-labeler on negatives and crowds.

Part 2 of the SAM3 audit. Two modes, run separately per dataset (results are
never pooled — each dataset answers its own question):

  --mode negatives   (PASS empty scenes; hand-verified toy/mannequin set)
      There is NO ground truth because there are no people: every SAM
      detection is a false positive by construction. Reports FPs per 100
      images, % of images with >=1 FP, and which class prompt fired.

  --mode crowd       (CrowdHuman val: exhaustive person boxes + ignore regions)
      Every person is labeled, so an unmatched SAM detection is finally
      *provably* wrong. Reports detection precision, recall stratified by
      occlusion (visible/full-body area ratio), duplicates-per-matched-person
      (fragmentation), and buckets each unmatched detection as
      duplicate / partial-overlap / clear false positive.

Stage A is the unmodified production autolabel_sam.py run over each image dir
(frozen settings, YOLO segment-polygon output). This script is Stage B,
CPU-only. Reuses (does NOT duplicate): seg_boxes + DEFAULT_CLASS_NAMES + _rate
(Wilson CIs) from run_autolabel_on_manifest.py, match_boxes/iou from
run_model_children.py.

Datasets live under /workspace/datasets/<name>/ (shared, reused across experiments);
run outputs live under /workspace/exp03/ (experiment-specific).

    # negatives (PASS or the toy/mannequin set)
    python3 eval_negatives_crowd.py --mode negatives \
        --images /workspace/datasets/pass_3k \
        --pred-labels /workspace/exp03/sam_labels/pass \
        --out /workspace/exp03/eval/pass --overlays 40

    # crowd (CrowdHuman val)
    python3 eval_negatives_crowd.py --mode crowd \
        --images /workspace/datasets/crowdhuman/Images \
        --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
        --pred-labels /workspace/exp03/sam_labels/crowd \
        --out /workspace/exp03/eval/crowd --overlays 40

    python3 eval_negatives_crowd.py --selftest   # both modes, synthetic, no GPU

CrowdHuman ground truth (.odgt = one JSON per line):
  {"ID": "...", "gtboxes": [{"tag": "person"|"mask", "hbox": [x,y,w,h],
   "vbox": [x,y,w,h], "fbox": [x,y,w,h], "extra": {"ignore": 0|1, ...}}, ...]}
  - tag "mask" or extra.ignore==1 -> IGNORE region (dense crowd blobs etc.):
    detections mostly inside one are EXCLUDED from scoring (standard practice),
    ignored GT is excluded from recall.
  - we match against the VISIBLE box (vbox) — pairs better with SAM masks.
  - occlusion = vbox area / fbox area: >=0.7 light, 0.3-0.7 partial, <0.3 heavy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from run_autolabel_on_manifest import DEFAULT_CLASS_NAMES, _rate, seg_boxes
from run_model_children import iou, match_boxes

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# occlusion bands from vbox/fbox area ratio (pre-registered in the EXP doc)
OCC_LIGHT, OCC_HEAVY = 0.7, 0.3
# a detection mostly inside an ignore region (intersection / det area) is excluded
IGNORE_IOA = 0.5
# unmatched-detection buckets, by its best IoU against any GT person
DUP_IOU = 0.5        # >= this vs an already-matched person -> duplicate/fragment
PARTIAL_IOU = 0.1    # in [PARTIAL_IOU, DUP_IOU) -> partial overlap (ambiguous)
                     # < PARTIAL_IOU -> clear false-positive candidate


def list_images(images_dir: Path):
    return sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)


def draw_boxes(img_path: Path, boxes, out_path: Path, max_dim=560, quality=70):
    """Annotate boxes = [(xyxy, label, rgb)] on a resized copy of the image."""
    with Image.open(img_path) as im:
        w, h = im.size
        scale = min(1.0, max_dim / max(w, h))
        canvas = im.convert("RGB")
        if scale < 1.0:
            canvas = canvas.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        draw = ImageDraw.Draw(canvas)
        for box, label, color in boxes:
            b = [v * scale for v in box]
            draw.rectangle(b, outline=color, width=2)
            draw.text((b[0], max(0, b[1] - 11)), label, fill=color)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=quality)


# ---------------------------------------------------------------------------
# negatives mode — every detection is a false positive
# ---------------------------------------------------------------------------
def run_negatives(images_dir: Path, pred_dir: Path, out_dir: Path,
                  class_names, max_overlays: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_f = open(out_dir / "fp_per_image.jsonl", "w")

    n_processed = n_not_processed = 0
    n_images_with_fp = 0
    fp_total = 0
    fp_by_class = Counter()
    n_overlays = 0

    for img_path in list_images(images_dir):
        dets = seg_boxes(pred_dir / (img_path.stem + ".txt"), 1, 1)
        if dets is None:                  # no .txt -> Stage A hasn't run here
            n_not_processed += 1
            continue
        # re-parse at real pixel scale only if there are detections to report
        if dets:
            with Image.open(img_path) as im:
                w, h = im.size
            dets = seg_boxes(pred_dir / (img_path.stem + ".txt"), w, h)
        n_processed += 1
        classes = [class_names.get(d[0], f"class_{d[0]}") for d in dets]
        fp_total += len(dets)
        fp_by_class.update(classes)
        if dets:
            n_images_with_fp += 1
            rows_f.write(json.dumps({
                "image": img_path.name, "n_fp": len(dets), "classes": classes,
                "boxes_xyxy": [[round(v, 1) for v in d[1:5]] for d in dets],
            }) + "\n")
            if n_overlays < max_overlays:
                draw_boxes(img_path,
                           [(list(d[1:5]), f"SAM: {c}", (220, 0, 0))
                            for d, c in zip(dets, classes)],
                           out_dir / "overlays" / f"{img_path.stem}_fp.jpg")
                n_overlays += 1

    rows_f.close()
    summary = {
        "mode": "negatives",
        "n_images_processed": n_processed,
        "n_images_not_processed_yet": n_not_processed,
        "fp_total": fp_total,
        "fp_per_100_images": round(100.0 * fp_total / n_processed, 2) if n_processed else None,
        "pct_images_with_fp": _rate(n_images_with_fp, n_processed),
        "fp_by_class": dict(fp_by_class),
    }
    return summary


# ---------------------------------------------------------------------------
# crowd mode — CrowdHuman odgt ground truth
# ---------------------------------------------------------------------------
def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def load_odgt(path: Path):
    """odgt -> {image_id: {persons: [{vbox, occ_ratio}], ignores: [xyxy]}}."""
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            persons, ignores = [], []
            for gb in r.get("gtboxes", []):
                extra = gb.get("extra") or {}
                if gb.get("tag") != "person" or extra.get("ignore") == 1:
                    ib = gb.get("fbox") or gb.get("vbox") or gb.get("hbox")
                    if ib:
                        ignores.append(_xywh_to_xyxy(ib))
                    continue
                vbox = gb.get("vbox") or gb.get("fbox")
                fbox = gb.get("fbox") or vbox
                if not vbox:
                    continue
                v, fb = _xywh_to_xyxy(vbox), _xywh_to_xyxy(fbox)
                ratio = _area(v) / _area(fb) if _area(fb) > 0 else 1.0
                persons.append({"vbox": v, "fbox": fb,
                                "occ_ratio": min(1.0, ratio)})
            recs[r["ID"]] = {"persons": persons, "ignores": ignores}
    return recs


def _occ_band(ratio):
    if ratio >= OCC_LIGHT:
        return "light"
    if ratio >= OCC_HEAVY:
        return "partial"
    return "heavy"


def _ioa(det_box, region):
    """Intersection over the DETECTION's area (how much of it sits in region)."""
    ix1, iy1 = max(det_box[0], region[0]), max(det_box[1], region[1])
    ix2, iy2 = min(det_box[2], region[2]), min(det_box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    da = _area(det_box)
    return inter / da if da > 0 else 0.0


def run_crowd(images_dir: Path, odgt_path: Path, pred_dir: Path, out_dir: Path,
              class_names, match_iou_thr: float, max_overlays: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_f = open(out_dir / "crowd_per_image.jsonl", "w")

    gt = load_odgt(odgt_path)
    n_imgs = n_not_processed = n_missing_img = 0
    n_gt = n_matched = 0
    n_dets_kept = n_dets_ignored = 0
    dup_count = partial_count = clear_fp_count = 0
    recall_by_band = defaultdict(lambda: [0, 0])   # band -> [n_gt, n_found]
    n_overlays = 0

    for img_id, rec in sorted(gt.items()):
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            n_missing_img += 1
            continue
        lbl = pred_dir / f"{img_id}.txt"
        dets = seg_boxes(lbl, 1, 1)
        if dets is None:
            n_not_processed += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        dets = seg_boxes(lbl, w, h)
        n_imgs += 1

        # drop detections mostly inside an ignore region (standard CrowdHuman eval)
        kept, ignored = [], []
        for d in dets:
            if any(_ioa(list(d[1:5]), ig) > IGNORE_IOA for ig in rec["ignores"]):
                ignored.append(d)
            else:
                kept.append(d)
        n_dets_kept += len(kept)
        n_dets_ignored += len(ignored)

        persons = rec["persons"]
        vboxes = [p["vbox"] for p in persons]
        matched = match_boxes(vboxes, kept, match_iou_thr)
        n_gt += len(persons)
        n_matched += len(matched)
        for gi, p in enumerate(persons):
            band = _occ_band(p["occ_ratio"])
            recall_by_band[band][0] += 1
            recall_by_band[band][1] += (gi in matched)

        matched_dets = {id(det) for det, _ in matched.values()}
        img_dups, img_partials, img_clear = [], [], []
        for d in kept:
            if id(d) in matched_dets:
                continue
            best = max((iou(v, d[1:5]) for v in vboxes), default=0.0)
            if best >= DUP_IOU:
                img_dups.append(d)
            elif best >= PARTIAL_IOU:
                img_partials.append(d)
            else:
                img_clear.append(d)
        dup_count += len(img_dups)
        partial_count += len(img_partials)
        clear_fp_count += len(img_clear)

        rows_f.write(json.dumps({
            "image": f"{img_id}.jpg", "n_gt": len(persons),
            "n_dets_kept": len(kept), "n_dets_ignored": len(ignored),
            "n_matched": len(matched), "n_duplicates": len(img_dups),
            "n_partial": len(img_partials), "n_clear_fp": len(img_clear),
        }) + "\n")

        if (img_clear or img_dups) and n_overlays < max_overlays:
            boxes = [(p["vbox"], "GT", (0, 190, 0)) for p in persons]
            boxes += [(list(d[1:5]),
                       f"dup:{class_names.get(d[0], '?')}", (160, 0, 200))
                      for d in img_dups]
            boxes += [(list(d[1:5]),
                       f"FP:{class_names.get(d[0], '?')}", (220, 0, 0))
                      for d in img_clear]
            draw_boxes(img_path, boxes,
                       out_dir / "overlays" / f"{img_id}_crowd.jpg")
            n_overlays += 1

    rows_f.close()
    summary = {
        "mode": "crowd",
        "n_images_scored": n_imgs,
        "n_images_not_processed_yet": n_not_processed,
        "n_images_missing": n_missing_img,
        "n_gt_persons": n_gt,
        "detection_recall": _rate(n_matched, n_gt),
        "recall_by_occlusion": {
            band: _rate(found, total)
            for band, (total, found) in sorted(recall_by_band.items())
        },
        "dets_total_kept": n_dets_kept,
        "dets_excluded_by_ignore_regions": n_dets_ignored,
        "detection_precision": _rate(n_matched, n_dets_kept),
        "unmatched_breakdown": {
            # duplicates = fragmentation (extra boxes on an already-found person)
            "duplicates_vs_matched_persons": _rate(dup_count, n_matched),
            "partial_overlap_ambiguous": partial_count,
            # candidates for hand-check: poster/statue-of-a-person vs hallucination
            # (per the pre-registered "person includes posters" ruling, some of
            #  these may be CORRECT detections CrowdHuman didn't annotate)
            "clear_false_positives": clear_fp_count,
            "clear_fp_per_100_images": round(100.0 * clear_fp_count / n_imgs, 2) if n_imgs else None,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# self-test — both modes, synthetic, no data or GPU needed
# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # ---- negatives mode -------------------------------------------------
        (root / "n_imgs").mkdir(); (root / "n_lbls").mkdir()
        for name in ("a", "b", "c"):
            Image.new("RGB", (100, 100), (128,) * 3).save(root / f"n_imgs/{name}.jpg")
        # a: SAM fired twice (woman + child) -> 2 FPs
        (root / "n_lbls/a.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.90 0.10 0.90\n"
            "2 0.60 0.10 0.90 0.10 0.90 0.90 0.60 0.90\n")
        # b: processed, clean (empty file) -> 0 FPs
        (root / "n_lbls/b.txt").write_text("")
        # c: Stage A hasn't reached it (no .txt) -> not processed
        s = run_negatives(root / "n_imgs", root / "n_lbls", root / "n_out",
                          DEFAULT_CLASS_NAMES, 10)
        assert s["n_images_processed"] == 2 and s["n_images_not_processed_yet"] == 1, s
        assert s["fp_total"] == 2 and s["fp_per_100_images"] == 100.0, s
        assert s["pct_images_with_fp"]["k"] == 1 and s["pct_images_with_fp"]["n"] == 2, s
        assert s["fp_by_class"] == {"Woman": 1, "Child": 1}, s
        assert (root / "n_out/overlays/a_fp.jpg").exists()

        # ---- crowd mode ------------------------------------------------------
        (root / "c_imgs").mkdir(); (root / "c_lbls").mkdir()
        Image.new("RGB", (100, 100), (128,) * 3).save(root / "c_imgs/x.jpg")
        # GT: person A light occlusion, person B partial (vbox/fbox ~ 0.67),
        #     one ignore region bottom-left.
        odgt = {"ID": "x", "gtboxes": [
            {"tag": "person", "vbox": [10, 10, 30, 80], "fbox": [10, 10, 30, 80],
             "extra": {}},
            {"tag": "person", "vbox": [60, 10, 30, 80], "fbox": [50, 10, 45, 80],
             "extra": {}},
            {"tag": "mask", "fbox": [0, 92, 20, 7], "extra": {"ignore": 1}},
        ]}
        (root / "anno.odgt").write_text(json.dumps(odgt) + "\n")
        # dets: d0 matches A exactly; d1 = near-duplicate on A (unmatched -> dup);
        #       d2 matches B; d3 sits inside the ignore region -> excluded;
        #       d4 overlaps nothing -> clear FP.
        (root / "c_lbls/x.txt").write_text(
            "0 0.10 0.10 0.40 0.10 0.40 0.90 0.10 0.90\n"
            "1 0.12 0.12 0.42 0.12 0.42 0.92 0.12 0.92\n"
            "2 0.60 0.10 0.90 0.10 0.90 0.90 0.60 0.90\n"
            "0 0.02 0.93 0.18 0.93 0.18 0.98 0.02 0.98\n"
            "1 0.45 0.40 0.55 0.40 0.55 0.60 0.45 0.60\n")
        s = run_crowd(root / "c_imgs", root / "anno.odgt", root / "c_lbls",
                      root / "c_out", DEFAULT_CLASS_NAMES, 0.5, 10)
        assert s["n_gt_persons"] == 2, s
        assert s["detection_recall"]["k"] == 2 and s["detection_recall"]["n"] == 2, s
        assert s["dets_excluded_by_ignore_regions"] == 1, s
        assert s["dets_total_kept"] == 4, s
        assert s["detection_precision"]["k"] == 2 and s["detection_precision"]["n"] == 4, s
        ub = s["unmatched_breakdown"]
        assert ub["duplicates_vs_matched_persons"]["k"] == 1, ub   # d1
        assert ub["clear_false_positives"] == 1, ub                 # d4
        rb = s["recall_by_occlusion"]
        assert rb["light"]["n"] == 1 and rb["partial"]["n"] == 1, rb
        assert (root / "c_out/overlays/x_crowd.jpg").exists()

    print("eval_negatives_crowd.py self-test passed (both modes)")


def main():
    ap = argparse.ArgumentParser(
        description="EXP-2026-03 Stage B: score SAM labels on negatives / crowds")
    ap.add_argument("--mode", choices=["negatives", "crowd"])
    ap.add_argument("--images", help="image directory")
    ap.add_argument("--pred-labels", help="autolabel_sam.py YOLO .txt output dir")
    ap.add_argument("--gt-odgt", help="CrowdHuman annotation .odgt (crowd mode)")
    ap.add_argument("--out", help="output dir")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--overlays", type=int, default=40,
                    help="max annotated proof images to write")
    ap.add_argument("--class-names", nargs="+", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.mode:
        ap.error("--mode is required (or use --selftest)")
    for req in ("images", "pred_labels", "out"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required")
    if args.mode == "crowd" and not args.gt_odgt:
        ap.error("--gt-odgt is required in crowd mode")

    names = (dict(enumerate(args.class_names)) if args.class_names
             else DEFAULT_CLASS_NAMES)
    if args.mode == "negatives":
        summary = run_negatives(Path(args.images), Path(args.pred_labels),
                                Path(args.out), names, args.overlays)
    else:
        summary = run_crowd(Path(args.images), Path(args.gt_odgt),
                            Path(args.pred_labels), Path(args.out), names,
                            args.match_iou, args.overlays)
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
