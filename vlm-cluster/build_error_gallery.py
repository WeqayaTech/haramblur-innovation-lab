#!/usr/bin/env python3
"""
EXP-2026-02 — visual proof gallery: a single self-contained HTML file with
embedded thumbnails of where the SAM3 auto-labeler got it wrong.

Four sections:
  1. Wrong classification  — GT box (green) vs SAM's box (color by error type).
     Teen-band (13-19) cases are sorted first, since that's the headline
     finding (SAM3's "child" concept fades out through the teens).
  2. Missed detections      — GT box SAM found nothing for.
  3. Dual-class survivors   — one person kept alive under two SAM classes at
     once (the cross-class NMS leak).
  4. Unmatched SAM detections ("boxes on the wrong things") — full-frame
     views showing every SAM box vs every GT box, so you can eyeball whether
     the extra boxes are real uncounted people or genuine false positives.
     Sorted by unmatched-box count (worst offenders first).

Must run where the images live (the pod) — it embeds real image crops as
base64, so the output is one portable .html file, no separate assets to move.

    python3 build_error_gallery.py \
        --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
        --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
        --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
        --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
        --out         /workspace/lagenda_eval/sam_autolabel/gallery.html

    python3 build_error_gallery.py --selftest   # synthetic, no data/GPU needed

Reuses (does NOT duplicate): load_gt/seg_boxes/gt_class from
run_autolabel_on_manifest.py, match_boxes/iou from run_model_children.py,
yolo_boxes/pad_box from dataset_utils.py.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

import dataset_utils as du
from run_autolabel_on_manifest import gt_class, load_gt, seg_boxes
from run_model_children import iou, match_boxes

GT_COLOR = (0, 190, 0)
STATUS_COLOR = {"correct": (0, 140, 255), "wrong_class": (235, 120, 0),
                "dual_leak": (160, 0, 200), "unmatched": (220, 0, 0)}
TEEN_MIN, TEEN_MAX = 13, 19


# ---------------------------------------------------------------------------
# per-image analysis (mirrors run_autolabel_on_manifest.score()'s matching,
# but collects everything needed to *draw* a case instead of aggregate stats)
# ---------------------------------------------------------------------------
def analyze(gt_by_image, gt_labels_dir: Path, images_dir: Path, pred_dir: Path,
            class_names, match_iou_thr, dual_iou_thr):
    wrong_class, missed, dual_leak, unmatched_frames = [], [], [], []

    for img_name, gt_rows in sorted(gt_by_image.items()):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size

        dets = seg_boxes(pred_dir / (Path(img_name).stem + ".txt"), w, h)
        if dets is None:
            continue

        gt_all = du.yolo_boxes(gt_labels_dir / (Path(img_name).stem + ".txt"), w, h)
        gt_rows_ok = []
        for r in gt_rows:
            if r["box_index"] < len(gt_all):
                r["box"] = list(gt_all[r["box_index"]][1:5])
                gt_rows_ok.append(r)
        gt_rows = gt_rows_ok
        if not gt_rows:
            continue

        matched = match_boxes([r["box"] for r in gt_rows], dets, match_iou_thr)
        matched_det_idx = set()
        for gi, r in enumerate(gt_rows):
            det, j = matched.get(gi, (None, None))
            if det is not None:
                matched_det_idx.add(dets.index(det))

        # a real crowd (LAGENDA labels only the "main" person(s)) has far more
        # SAM detections than GT boxes — flag it so the unmatched-detections
        # section can be judged in context instead of looking like noise
        is_crowd = (len(dets) - len(gt_rows)) >= 3

        for gi, r in enumerate(gt_rows):
            gcls = gt_class(r)
            det, j = matched.get(gi, (None, None))
            pcls = class_names.get(det[0], f"class_{det[0]}") if det else None
            dual = sorted({class_names.get(d[0], f"class_{d[0]}")
                           for d in dets if iou(r["box"], d[1:5]) >= dual_iou_thr})
            if gcls is None:
                continue
            case = {"id": r["id"], "image": img_name, "img_path": img_path,
                    "gt_box": r["box"], "pred_box": list(det[1:5]) if det else None,
                    "gt_class": gcls, "pred_class": pcls,
                    "gt_age": r.get("gt_age"), "gt_gender": r.get("gt_gender"),
                    "match_iou": j, "is_crowd_image": is_crowd}
            if det is None:
                missed.append(case)
            elif pcls != gcls:
                # was there ALSO a same-class SAM box nearby that lost the
                # greedy match (e.g. mom holding child — SAM boxed both
                # correctly, but only one wins)? Show it if so.
                near = [d for d in dets if d is not det
                       and class_names.get(d[0], f"class_{d[0]}") == gcls]
                near_iou, near_box = 0.0, None
                for d in near:
                    ji = iou(r["box"], d[1:5])
                    if ji > near_iou:
                        near_iou, near_box = ji, list(d[1:5])
                case["near_miss_iou"] = near_iou
                case["near_miss_box"] = near_box if near_iou >= 0.3 else None
                wrong_class.append(case)
            if len(dual) > 1:
                dual_dets = [d for d in dets if iou(r["box"], d[1:5]) >= dual_iou_thr]
                # do the two overlapping-class boxes overlap EACH OTHER much,
                # or are they plausibly two different real people (e.g. a man
                # sitting behind a child) that each legitimately clip the GT
                # box on their own? Only the former is a genuine double-label.
                mutual_iou = 0.0
                for i1 in range(len(dual_dets)):
                    for i2 in range(i1 + 1, len(dual_dets)):
                        d1, d2 = dual_dets[i1], dual_dets[i2]
                        if class_names.get(d1[0]) != class_names.get(d2[0]):
                            mutual_iou = max(mutual_iou, iou(d1[1:5], d2[1:5]))
                case2 = dict(case)
                case2["dual_boxes"] = [list(d[1:5]) for d in dual_dets]
                case2["dual_classes"] = dual
                case2["dual_mutual_iou"] = mutual_iou
                dual_leak.append(case2)

        unmatched = [d for i, d in enumerate(dets) if i not in matched_det_idx]
        if unmatched:
            unmatched_frames.append({
                "image": img_name, "img_path": img_path, "is_crowd": is_crowd,
                "gt_boxes": [(r["box"], gt_class(r)) for r in gt_rows],
                "matched_boxes": [(list(det[1:5]), class_names.get(det[0], "?"))
                                  for gi, (det, j) in matched.items()],
                "unmatched_boxes": [(list(d[1:5]), class_names.get(d[0], "?"))
                                    for d in unmatched],
            })

    return wrong_class, missed, dual_leak, unmatched_frames


# ---------------------------------------------------------------------------
# rendering: crop/frame -> base64 JPEG thumbnail
# ---------------------------------------------------------------------------
def _b64_jpeg(img: Image.Image, quality=78) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize_max(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def case_thumb(case, max_dim=220, quality=62) -> str:
    """Crop around GT + pred (+ dual + near-miss) boxes, padded, boxes drawn."""
    with Image.open(case["img_path"]) as im:
        w, h = im.size
        boxes = [case["gt_box"]] + ([case["pred_box"]] if case.get("pred_box") else [])
        boxes += case.get("dual_boxes", [])
        if case.get("near_miss_box"):
            boxes += [case["near_miss_box"]]
        ux1 = min(b[0] for b in boxes); uy1 = min(b[1] for b in boxes)
        ux2 = max(b[2] for b in boxes); uy2 = max(b[3] for b in boxes)
        cx1, cy1, cx2, cy2 = du.pad_box(ux1, uy1, ux2, uy2, w, h, 0.35)
        crop = im.convert("RGB").crop((cx1, cy1, cx2, cy2))
        draw = ImageDraw.Draw(crop)
        gx = [v - cx1 if i % 2 == 0 else v - cy1 for i, v in enumerate(case["gt_box"])]
        draw.rectangle(gx, outline=GT_COLOR, width=3)
        draw.text((gx[0], max(0, gx[1] - 12)), f"GT: {case['gt_class']}", fill=GT_COLOR)
        if case.get("pred_box"):
            px = [v - cx1 if i % 2 == 0 else v - cy1 for i, v in enumerate(case["pred_box"])]
            col = STATUS_COLOR["wrong_class"] if case["pred_class"] != case["gt_class"] else STATUS_COLOR["correct"]
            draw.rectangle(px, outline=col, width=2)
            draw.text((px[0], px[3] + 2), f"SAM: {case['pred_class']}", fill=col)
        else:
            draw.text((gx[0], gx[3] + 2), "SAM: (missed)", fill=(200, 0, 0))
        for db, dcls in zip(case.get("dual_boxes", []), case.get("dual_classes", [])):
            dx = [v - cx1 if i % 2 == 0 else v - cy1 for i, v in enumerate(db)]
            draw.rectangle(dx, outline=STATUS_COLOR["dual_leak"], width=2)
        if case.get("near_miss_box"):
            nb = case["near_miss_box"]
            nx = [v - cx1 if i % 2 == 0 else v - cy1 for i, v in enumerate(nb)]
            draw.rectangle(nx, outline=(255, 230, 0), width=2)
            draw.text((nx[0], nx[1] - 12 if nx[1] > 12 else nx[3] + 2),
                      f"SAM also boxed: {case['gt_class']} (IoU {case['near_miss_iou']:.2f})",
                      fill=(255, 230, 0))
        crop = _resize_max(crop, max_dim)
        return _b64_jpeg(crop, quality)


def frame_thumb(frame, max_dim=480, quality=58) -> str:
    with Image.open(frame["img_path"]) as im:
        w, h = im.size
        scale = min(1.0, max_dim / max(w, h))
        canvas = im.convert("RGB")
        if scale < 1.0:
            canvas = canvas.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        def sb(box):  # scale a full-res box into canvas coordinates
            return [v * scale for v in box]

        draw = ImageDraw.Draw(canvas)
        for box, cls in frame["gt_boxes"]:
            b = sb(box)
            draw.rectangle(b, outline=GT_COLOR, width=2)
            draw.text((b[0], max(0, b[1] - 11)), f"GT:{cls}", fill=GT_COLOR)
        for box, cls in frame["matched_boxes"]:
            draw.rectangle(sb(box), outline=STATUS_COLOR["correct"], width=1)
        for box, cls in frame["unmatched_boxes"]:
            b = sb(box)
            draw.rectangle(b, outline=STATUS_COLOR["unmatched"], width=2)
            draw.text((b[0], b[3] + 1), f"SAM:{cls}", fill=STATUS_COLOR["unmatched"])
        return _b64_jpeg(canvas, quality)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:24px}
h1{font-size:22px} h2{font-size:18px;margin-top:40px;border-top:1px solid #333;padding-top:20px}
.nav a{color:#4f8ef7;margin-right:16px;text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:14px}
.card{background:#1a1d25;border-radius:6px;padding:6px;font-size:11px}
.card img{width:100%;max-width:220px;border-radius:4px;display:block;margin:0 auto}
.meta{margin-top:6px;line-height:1.5}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;margin-right:4px}
.teen{background:#5b3a00;color:#ffb84d}
.artifact{background:#4a3d00;color:#ffe14d}
.crowd{background:#1f3a5b;color:#7fbfff}
.distinct{background:#1a3d33;color:#6fd9b8}
.count{color:#8899aa;font-weight:normal;font-size:14px}
"""


def render_gallery(wrong_class, missed, dual_leak, unmatched_frames, out_path: Path,
                    max_per_cat: int, max_frames: int, seed: int):
    rng = random.Random(seed)

    def teen_first(cases):
        def is_teen(c):
            a = c.get("gt_age")
            return a is not None and TEEN_MIN <= a <= TEEN_MAX
        teens = [c for c in cases if is_teen(c)]
        others = [c for c in cases if not is_teen(c)]
        rng.shuffle(others)
        return (teens + others)[:max_per_cat]

    # split out "ambiguous" wrong_class cases — SAM had its own same-class box
    # nearby that lost the greedy match (e.g. mom holding a child). These are
    # plausible matching artifacts, not necessarily genuine misreads.
    ambiguous = [c for c in wrong_class if c.get("near_miss_box")]
    clear_wrong = [c for c in wrong_class if not c.get("near_miss_box")]

    wrong_class = teen_first(clear_wrong)
    ambiguous = teen_first(ambiguous)
    rng.shuffle(missed); missed = missed[:max_per_cat]
    # a dual-class leak is only "confirmed" (a true same-object double-label)
    # if the two overlapping-class SAM boxes overlap EACH OTHER a lot too —
    # otherwise it's plausibly two different real people (e.g. a man sitting
    # behind a child), each legitimately boxed under their own class.
    dual_confirmed = sorted([c for c in dual_leak if c.get("dual_mutual_iou", 0) >= 0.5],
                            key=lambda c: -c["dual_mutual_iou"])[:max_per_cat]
    dual_distinct = sorted([c for c in dual_leak if c.get("dual_mutual_iou", 0) < 0.5],
                           key=lambda c: c["dual_mutual_iou"])[:max_per_cat]
    unmatched_frames = sorted(unmatched_frames, key=lambda f: -len(f["unmatched_boxes"]))[:max_frames]
    n_crowd_frames = sum(1 for f in unmatched_frames if f.get("is_crowd"))

    parts = [f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<title>EXP-2026-02 error gallery</title><style>{_CSS}</style></head><body>"]
    parts.append("<h1>EXP-2026-02 — SAM3 auto-labeler visual error gallery</h1>")
    parts.append("<div class='nav'><a href='#ambiguous'>Ambiguous (crowd-artifact?)</a>"
                 "<a href='#wrong'>Wrong classification</a>"
                 "<a href='#missed'>Missed</a><a href='#dual'>Dual-class leak (confirmed)</a>"
                 "<a href='#dual-distinct'>Dual-class flags (likely 2 people)</a>"
                 "<a href='#unmatched'>Unmatched SAM boxes</a></div>")
    parts.append("<p style='color:#8899aa'>Green = ground truth. Blue = SAM correct. "
                 "Orange = SAM wrong class. Purple = dual-class leak. Yellow = a same-class SAM box "
                 "found nearby but not credited. Red = missed / unmatched SAM box.</p>")

    parts.append(f"<h2 id='ambiguous'>Ambiguous wrong-class &mdash; SAM had a same-class box nearby too "
                 f"<span class='count'>({len(ambiguous)} shown)</span></h2>"
                 f"<p style='color:#8899aa'>These count as classification errors in the summary numbers, "
                 f"but SAM also drew a correctly-classed box (yellow) right next to the one we credited it "
                 f"with &mdash; often two overlapping people (e.g. a parent holding a child). Judge for "
                 f"yourself whether this is a genuine misread or a scoring artifact of crowded/overlapping "
                 f"scenes.</p><div class='grid'>")
    for c in ambiguous:
        b64 = case_thumb(c)
        age = c.get("gt_age")
        teen_tag = "<span class='tag teen'>TEEN</span>" if age is not None and TEEN_MIN <= age <= TEEN_MAX else ""
        parts.append(f"<div class='card' data-status='ambiguous'>"
                     f"<img src='data:image/jpeg;base64,{b64}'>"
                     f"<div class='meta'>{teen_tag}<span class='tag artifact'>NEAR-MISS {c['near_miss_iou']:.2f}</span>"
                     f"<b>{html.escape(c['id'])}</b><br>"
                     f"GT: {c['gt_class']} (age {age}, {c.get('gt_gender')}) &rarr; SAM: {c['pred_class']}"
                     f"<br>IoU {c['match_iou']:.2f} &middot; {html.escape(c['image'])}</div></div>")
    parts.append("</div>")

    parts.append(f"<h2 id='wrong'>Wrong classification <span class='count'>"
                 f"({len(wrong_class)} shown, teen-age 13-19 cases first &mdash; excludes the "
                 f"ambiguous cases above)</span></h2><div class='grid'>")
    for c in wrong_class:
        b64 = case_thumb(c)
        age = c.get("gt_age")
        teen_tag = "<span class='tag teen'>TEEN</span>" if age is not None and TEEN_MIN <= age <= TEEN_MAX else ""
        parts.append(f"<div class='card' data-status='wrong_class'>"
                     f"<img src='data:image/jpeg;base64,{b64}'>"
                     f"<div class='meta'>{teen_tag}<b>{html.escape(c['id'])}</b><br>"
                     f"GT: {c['gt_class']} (age {age}, {c.get('gt_gender')}) &rarr; SAM: {c['pred_class']}"
                     f"<br>IoU {c['match_iou']:.2f} &middot; {html.escape(c['image'])}</div></div>")
    parts.append("</div>")

    parts.append(f"<h2 id='missed'>Missed detections <span class='count'>({len(missed)} shown)</span></h2><div class='grid'>")
    for c in missed:
        b64 = case_thumb(c)
        parts.append(f"<div class='card' data-status='missed'>"
                     f"<img src='data:image/jpeg;base64,{b64}'>"
                     f"<div class='meta'><b>{html.escape(c['id'])}</b><br>"
                     f"GT: {c['gt_class']} (age {c.get('gt_age')}, {c.get('gt_gender')}) &rarr; SAM found nothing"
                     f"<br>{html.escape(c['image'])}</div></div>")
    parts.append("</div>")

    parts.append(f"<h2 id='dual'>Dual-class survivors, confirmed (NMS leak on the SAME object) "
                 f"<span class='count'>({len(dual_confirmed)} shown)</span></h2>"
                 f"<p style='color:#8899aa'>The two overlapping-class SAM boxes also overlap EACH OTHER "
                 f"heavily (IoU &ge; 0.5) &mdash; this is the genuine bug: one physical person got written "
                 f"into the training labels under two contradictory classes at once.</p><div class='grid'>")
    for c in dual_confirmed:
        b64 = case_thumb(c)
        parts.append(f"<div class='card' data-status='dual_leak'>"
                     f"<img src='data:image/jpeg;base64,{b64}'>"
                     f"<div class='meta'><span class='tag artifact'>MUTUAL IoU {c['dual_mutual_iou']:.2f}</span>"
                     f"<b>{html.escape(c['id'])}</b><br>"
                     f"GT: {c['gt_class']} &mdash; SAM kept it alive as: {', '.join(c['dual_classes'])}"
                     f"<br>{html.escape(c['image'])}</div></div>")
    parts.append("</div>")

    parts.append(f"<h2 id='dual-distinct'>Dual-class flags, likely two different people "
                 f"<span class='count'>({len(dual_distinct)} shown)</span></h2>"
                 f"<p style='color:#8899aa'>These trip the naive dual-class check (both SAM boxes clip the "
                 f"GT box), but the two SAM boxes barely overlap EACH OTHER (IoU &lt; 0.5) &mdash; plausibly "
                 f"two different real people (e.g. a man sitting behind a child) each legitimately boxed "
                 f"under their own class, not a genuine double-label. Judge for yourself.</p><div class='grid'>")
    for c in dual_distinct:
        b64 = case_thumb(c)
        parts.append(f"<div class='card' data-status='dual_leak_distinct'>"
                     f"<img src='data:image/jpeg;base64,{b64}'>"
                     f"<div class='meta'><span class='tag distinct'>MUTUAL IoU {c['dual_mutual_iou']:.2f}</span>"
                     f"<b>{html.escape(c['id'])}</b><br>"
                     f"GT: {c['gt_class']} &mdash; SAM kept it alive as: {', '.join(c['dual_classes'])}"
                     f"<br>{html.escape(c['image'])}</div></div>")
    parts.append("</div>")

    parts.append(f"<h2 id='unmatched'>Unmatched SAM detections &mdash; boxes on things not in ground truth "
                 f"<span class='count'>({len(unmatched_frames)} images, worst offenders first; "
                 f"{n_crowd_frames} tagged CROWD)</span></h2>"
                 f"<p style='color:#8899aa'>Caveat: LAGENDA only annotates the main subject(s), so a CROWD-tagged "
                 f"image (3+ more SAM boxes than GT boxes) is plausibly a real crowd photo with genuinely uncounted "
                 f"people, not evidence of hallucination. Eyeball the non-crowd ones for the more meaningful signal.</p>"
                 f"<div class='grid'>")
    for f in unmatched_frames:
        b64 = frame_thumb(f)
        crowd_tag = "<span class='tag crowd'>CROWD</span>" if f.get("is_crowd") else ""
        parts.append(f"<div class='card' data-status='unmatched'>"
                     f"<img src='data:image/jpeg;base64,{b64}'>"
                     f"<div class='meta'>{crowd_tag}<b>{html.escape(f['image'])}</b><br>"
                     f"{len(f['gt_boxes'])} GT person(s) &middot; {len(f['unmatched_boxes'])} unmatched SAM box(es)</div></div>")
    parts.append("</div></body></html>")

    out_path.write_text("\n".join(parts))
    return {"ambiguous": len(ambiguous), "wrong_class": len(wrong_class), "missed": len(missed),
            "dual_confirmed": len(dual_confirmed), "dual_distinct": len(dual_distinct),
            "unmatched_frames": len(unmatched_frames), "unmatched_frames_crowd": n_crowd_frames}


# ---------------------------------------------------------------------------
# synthetic self-test — no data or GPU needed
# ---------------------------------------------------------------------------
def _selftest():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for d in ("images", "gt_labels", "pred_labels"):
            (root / d).mkdir()
        for name in ("a", "b", "c", "d", "e"):
            Image.new("RGB", (200, 200), (128,) * 3).save(root / f"images/{name}.jpg")
        (root / "gt_labels/a.txt").write_text(
            "0 0.20 0.50 0.20 0.60\n"    # box_index 0: adult woman
            "0 0.60 0.50 0.20 0.60\n")   # box_index 1: teen (16), wrong-classed
        # image b: mom-holding-child style overlap — Man box wins the match,
        # but SAM's own Child box is right there too (near-miss / ambiguous)
        (root / "gt_labels/b.txt").write_text("0 0.50 0.50 0.20 0.20\n")   # box [40,40,60,60]
        # image c: a real crowd — 1 GT, 4 extra unrelated detections
        (root / "gt_labels/c.txt").write_text("0 0.30 0.30 0.40 0.40\n")   # box [10,10,50,50]
        # image d: a genuine same-object double-label — Man + Woman boxes
        # nearly on top of each other (high mutual IoU) -> confirmed dual leak
        (root / "gt_labels/d.txt").write_text("0 0.75 0.50 0.30 0.80\n")
        # image e: the "man sitting behind a child" case — Child (correctly
        # matched) + Man (a different real person) both clip the wide GT box
        # but barely overlap each other -> NOT a confirmed duplicate
        (root / "gt_labels/e.txt").write_text("0 0.50 0.50 0.80 0.80\n")
        with open(root / "gt.jsonl", "w") as f:
            f.write(json.dumps({"id": "a_0", "image": "a.jpg",
                                "gt_age": 30, "gt_gender": "F"}) + "\n")
            f.write(json.dumps({"id": "a_1", "image": "a.jpg",
                                "gt_age": 16, "gt_gender": "M"}) + "\n")
            f.write(json.dumps({"id": "b_0", "image": "b.jpg",
                                "gt_age": 8, "gt_gender": "M"}) + "\n")
            f.write(json.dumps({"id": "c_0", "image": "c.jpg",
                                "gt_age": 35, "gt_gender": "F"}) + "\n")
            f.write(json.dumps({"id": "d_0", "image": "d.jpg",
                                "gt_age": 8, "gt_gender": "M"}) + "\n")
            f.write(json.dumps({"id": "e_0", "image": "e.jpg",
                                "gt_age": 8, "gt_gender": "M"}) + "\n")
        # a_0 correctly Woman; a_1 (teen) wrongly Child; plus an extra unmatched
        # Man polygon elsewhere in the frame (a false-positive-or-uncounted-person case)
        (root / "pred_labels/a.txt").write_text(
            "0 0.10 0.20 0.30 0.20 0.30 0.80 0.10 0.80\n"
            "2 0.50 0.20 0.70 0.20 0.70 0.80 0.50 0.80\n"
            "1 0.85 0.85 0.99 0.85 0.99 0.99 0.85 0.99\n")
        # image b: Man box exact match to GT (wins) + SAM's own Child box
        # shifted alongside it (IoU ~0.33 -> a near-miss)
        (root / "pred_labels/b.txt").write_text(
            "1 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60\n"
            "2 0.50 0.40 0.70 0.40 0.70 0.60 0.50 0.60\n")
        # image c: Woman box on GT (correct) + 4 tiny unrelated Man boxes
        # far away (a real crowd — excess=4 dets over the 1 labeled GT)
        (root / "pred_labels/c.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n"
            "1 0.60 0.60 0.70 0.60 0.70 0.70 0.60 0.70\n"
            "1 0.71 0.60 0.81 0.60 0.81 0.70 0.71 0.70\n"
            "1 0.60 0.71 0.70 0.71 0.70 0.81 0.60 0.81\n"
            "1 0.71 0.71 0.81 0.71 0.81 0.81 0.71 0.81\n")
        # image d: Man box + a near-duplicate Woman box (mutual IoU ~0.91) ->
        # confirmed dual leak (Man wins the match -> also a wrong_class case)
        (root / "pred_labels/d.txt").write_text(
            "1 0.60 0.10 0.90 0.10 0.90 0.90 0.60 0.90\n"
            "0 0.59 0.09 0.91 0.09 0.91 0.91 0.59 0.91\n")
        # image e: Child box (wins the match, correct) + Man box for a
        # different real person, mutual IoU ~0.18 -> NOT a confirmed duplicate
        (root / "pred_labels/e.txt").write_text(
            "2 0.10 0.10 0.60 0.10 0.60 0.90 0.10 0.90\n"
            "1 0.45 0.10 0.95 0.10 0.95 0.90 0.45 0.90\n")

        from run_autolabel_on_manifest import DEFAULT_CLASS_NAMES
        gt = load_gt(root / "gt.jsonl", 0)
        wc, missed, dual, unmatched = analyze(
            gt, root / "gt_labels", root / "images", root / "pred_labels",
            DEFAULT_CLASS_NAMES, 0.5, 0.5)
        by_id = {c["id"]: c for c in wc}
        assert set(by_id) == {"a_1", "b_0", "d_0"}
        assert by_id["a_1"].get("near_miss_box") is None    # no other Man box near a_1
        assert by_id["b_0"].get("near_miss_box") is not None  # Child box found nearby
        assert by_id["b_0"]["near_miss_iou"] >= 0.3
        assert by_id["d_0"].get("near_miss_box") is None    # no Child det in image d
        assert len(missed) == 0

        dual_by_id = {c["id"]: c for c in dual}
        assert set(dual_by_id) == {"d_0", "e_0"}
        assert dual_by_id["d_0"]["dual_mutual_iou"] >= 0.5   # near-duplicate boxes
        assert dual_by_id["e_0"]["dual_mutual_iou"] < 0.5    # two different people

        # a's extra Man box, b's near-miss Child box (below match threshold,
        # so it never got matched either), and c's 4 crowd boxes -> 3 frames
        by_img = {f["image"]: f for f in unmatched}
        assert len(by_img["a.jpg"]["unmatched_boxes"]) == 1 and not by_img["a.jpg"]["is_crowd"]
        assert len(by_img["b.jpg"]["unmatched_boxes"]) == 1 and not by_img["b.jpg"]["is_crowd"]
        assert len(by_img["c.jpg"]["unmatched_boxes"]) == 4 and by_img["c.jpg"]["is_crowd"]

        out = root / "gallery.html"
        counts = render_gallery(wc, missed, dual, unmatched, out, 60, 40, 0)
        assert counts["wrong_class"] == 2 and counts["ambiguous"] == 1   # a_1, d_0 clear; b_0 ambiguous
        assert counts["dual_confirmed"] == 1 and counts["dual_distinct"] == 1
        # a, b, c, plus d's losing Woman box and e's second-person Man box
        # (neither wins the single-GT match) -> 5 frames; only c is a crowd
        assert counts["unmatched_frames"] == 5 and counts["unmatched_frames_crowd"] == 1
        text = out.read_text()
        assert text.count("data-status='wrong_class'") == 2
        assert text.count("data-status='ambiguous'") == 1
        assert text.count("data-status='dual_leak'") == 1
        assert text.count("data-status='dual_leak_distinct'") == 1
        assert text.count("class='tag crowd'") == 1
        assert text.count("class='tag distinct'") == 1
        assert "base64" in text
    print("build_error_gallery.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="build a visual HTML error gallery for EXP-2026-02")
    ap.add_argument("--gt-manifest")
    ap.add_argument("--gt-labels")
    ap.add_argument("--images")
    ap.add_argument("--pred-labels")
    ap.add_argument("--out", help="output .html path")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--dual-iou", type=float, default=0.5)
    ap.add_argument("--max-per-category", type=int, default=60)
    ap.add_argument("--max-unmatched-frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    for req in ("gt_manifest", "gt_labels", "images", "pred_labels", "out"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required (or use --selftest)")

    from run_autolabel_on_manifest import DEFAULT_CLASS_NAMES
    gt = load_gt(Path(args.gt_manifest), 0)
    wc, missed, dual, unmatched = analyze(
        gt, Path(args.gt_labels), Path(args.images), Path(args.pred_labels),
        DEFAULT_CLASS_NAMES, args.match_iou, args.dual_iou)
    counts = render_gallery(wc, missed, dual, unmatched, Path(args.out),
                            args.max_per_category, args.max_unmatched_frames, args.seed)
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    main()
