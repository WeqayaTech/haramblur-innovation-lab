#!/usr/bin/env python3
"""
EXP-2026-09 — head-to-head: SAM3 vs Gemini 3.5 Flash-Lite on crowd scenes.

Scores TWO models' detections on the SAME images against CrowdHuman ground
truth with the SAME matcher, then compares them per GT person:

  Component 1 (boxes, GT-scored): recall (overall + by occlusion band),
      precision, duplicates, matched-IoU tightness, recall@IoU sweep.
  Labels (3-class {Woman, Man, Child}, NO ground truth in CrowdHuman):
      cross-model agreement on persons BOTH models found, plus a
      disagreement-crop gallery for human adjudication. The gallery — not
      this script — decides who is right; this script only counts.

CPU-only. Reuses (does NOT duplicate): seg_boxes + DEFAULT_CLASS_NAMES +
_rate (Wilson CIs) from run_autolabel_on_manifest.py; match_boxes/iou from
run_model_children.py; load_odgt/_ioa/_occ_band + ignore/occlusion constants
from eval_negatives_crowd.py; select_images from vlm_detect_eval.py (lazy —
only --make-subset needs it, and it needs cv2).

    # 1. materialize the frozen 100-image subset (byte-identical selection to
    #    vlm_detect_eval's --seed shuffle, so both stages agree on the sample)
    python3 crowd_headtohead.py --make-subset \
        --source-images /workspace/datasets/crowdhuman/Images_sample500 \
        --n 100 --seed 42 --subset-out /workspace/exp09/subset100

    # 2. after Stage A-Lite (vlm_detect_eval) + reparse_boxes --write yxyx_1000
    python3 crowd_headtohead.py \
        --images /workspace/exp09/subset100 \
        --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
        --sam-labels /workspace/exp03/sam_labels/crowd \
        --lite-detections /workspace/exp09/lite_crowd/detections.jsonl \
        --out /workspace/exp09/headtohead

    python3 crowd_headtohead.py --selftest     # synthetic, no data, no network

Outputs in --out:
    summary.json       all metrics, per model + head-to-head
    per_person.jsonl   one row per GT person (found-by / labels / IoUs)
    gallery.html       self-contained (base64) adjudication gallery
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from eval_negatives_crowd import (DUP_IOU, IGNORE_IOA, PARTIAL_IOU, _ioa,
                                  _occ_band, list_images, load_odgt)
from run_autolabel_on_manifest import DEFAULT_CLASS_NAMES, _rate, seg_boxes
from run_model_children import iou, match_boxes
from sam_fp_conf_report import parse_line as conf_parse_line

SWEEP_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)   # same sweep as vlm_detect_eval box_tightness
CROP_PAD = 0.25                           # crop padding, same as gate_eval crops
GALLERY_SEED = 7                          # agreement-sample selection only

SAM_COLOR = (40, 90, 220)     # blue-ish
LITE_COLOR = (235, 140, 20)   # orange
GT_COLOR = (0, 190, 0)        # green


def lite_label3(p: dict) -> str:
    """Map a vlm_detect_eval person record into production 3-class space.

    Pre-registered (EXP-2026-09 doc, before running):
      - age_group == "child"  -> Child (regardless of gender)
      - age_group "adult" OR "unknown" -> adult; an unknown age defaults to
        adult because adult = blurred = the acceptable error direction.
      - adult with gender man/woman -> Man/Woman; gender "unknown" -> the
        literal label "Unknown" (counts as a disagreement with any SAM label,
        tallied separately so it can't hide).
    """
    if p.get("age_group") == "child":
        return "Child"
    g = p.get("gender")
    if g == "man":
        return "Man"
    if g == "woman":
        return "Woman"
    return "Unknown"


def load_lite_detections(path: Path) -> dict:
    """detections.jsonl -> {image_stem: record}. Run reparse_boxes.py --write
    yxyx_1000 FIRST for Gemini output — this loader trusts box_xyxy as pixels."""
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[Path(r["image"]).stem] = r
    return recs


def _dets_sam(label_file: Path, w: int, h: int, conf_floor=None):
    """Returns ([(label3, x1, y1, x2, y2)], n_lines_without_conf), or
    (None, 0) if Stage A never produced a file.

    conf_floor: for EXP-2026-09 Part 2 only — labels from the conf-logging
    labeler (autolabel_sam_conf.py, EXP-2026-07 patch) carry a trailing
    confidence per line; parse it (via sam_fp_conf_report.parse_line, the
    same detector EXP-2026-07 used) and keep detections with conf >= floor.
    Lines without a conf column are KEPT and counted by the caller — a silent
    drop would understate SAM3. Plain EXP-2026-03 labels: leave conf_floor
    None; seg_boxes handles them unchanged.
    """
    if conf_floor is None:
        raw = seg_boxes(label_file, w, h)
        if raw is None:
            return None, 0
        return ([(DEFAULT_CLASS_NAMES.get(cls, f"class_{cls}"), x1, y1, x2, y2)
                 for cls, x1, y1, x2, y2 in raw], 0)

    if not label_file.exists():
        return None, 0
    dets, no_conf = [], 0
    for line in label_file.read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 5:
            continue
        cls, conf = conf_parse_line(tokens)
        vals = [float(t) for t in tokens[1:]]
        if conf is not None and len(vals) % 2 == 1:
            vals = vals[:-1]              # strip the trailing conf column
        if len(vals) == 4:                # box: cx cy w h (same as seg_boxes)
            cx, cy, bw, bh = vals
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                   (cx + bw / 2) * w, (cy + bh / 2) * h)
        else:                             # polygon extent (same as seg_boxes)
            xs, ys = vals[0::2], vals[1::2]
            if len(xs) < 3 or len(xs) != len(ys):
                continue
            box = (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)
        if conf is None:
            no_conf += 1
        elif conf < conf_floor:
            continue
        dets.append((DEFAULT_CLASS_NAMES.get(cls, f"class_{cls}"), *box))
    return dets, no_conf


def _dets_lite(rec: dict):
    return [(lite_label3(p), *p["box_xyxy"]) for p in rec.get("people", [])]


def _split_ignores(dets, ignores):
    kept = [d for d in dets
            if not any(_ioa(list(d[1:5]), ig) > IGNORE_IOA for ig in ignores)]
    return kept, len(dets) - len(kept)


class ModelTally:
    """Accumulates one model's GT-scored metrics across images."""

    def __init__(self, name):
        self.name = name
        self.n_gt = self.n_matched = 0
        self.dets_kept = self.dets_ignored = 0
        self.dups = self.partials = self.clear_fps = 0
        self.recall_by_band = {}     # band -> [n_gt, n_found]
        self.sweep = {t: 0 for t in SWEEP_IOUS}
        self.matched_ious = []
        self.label_dist = Counter()

    def add_image(self, persons, vboxes, kept, matched):
        self.dets_kept += len(kept)
        self.n_gt += len(persons)
        self.n_matched += len(matched)
        self.matched_ious += [v for _, v in matched.values()]
        self.label_dist.update(d[0] for d in kept)
        for gi, p in enumerate(persons):
            band = _occ_band(p["occ_ratio"])
            self.recall_by_band.setdefault(band, [0, 0])
            self.recall_by_band[band][0] += 1
            self.recall_by_band[band][1] += (gi in matched)
        for t in SWEEP_IOUS:
            self.sweep[t] += len(match_boxes(vboxes, kept, t))
        matched_ids = {id(d) for d, _ in matched.values()}
        for d in kept:
            if id(d) in matched_ids:
                continue
            best = max((iou(v, d[1:5]) for v in vboxes), default=0.0)
            if best >= DUP_IOU:
                self.dups += 1
            elif best >= PARTIAL_IOU:
                self.partials += 1
            else:
                self.clear_fps += 1

    def summary(self):
        dist = sorted(self.matched_ious)
        n = len(dist)
        return {
            "gt_persons": self.n_gt,
            "recall": _rate(self.n_matched, self.n_gt),
            "recall_by_occlusion": {b: _rate(f, t) for b, (t, f)
                                    in sorted(self.recall_by_band.items())},
            "dets_kept": self.dets_kept,
            "dets_excluded_by_ignore_regions": self.dets_ignored,
            "precision": _rate(self.n_matched, self.dets_kept),
            "duplicates_vs_matched": _rate(self.dups, self.n_matched),
            "partial_overlap_ambiguous": self.partials,
            "clear_false_positives": self.clear_fps,
            "recall_at_iou": {f"{t:.1f}": (round(self.sweep[t] / self.n_gt, 4)
                                           if self.n_gt else None)
                              for t in SWEEP_IOUS},
            "matched_iou_mean": round(sum(dist) / n, 3) if n else None,
            "matched_iou_median": round(dist[n // 2], 3) if n else None,
            "matched_iou_pct_ge_0.75": (round(100 * sum(1 for v in dist
                                                        if v >= 0.75) / n, 1)
                                        if n else None),
            "label_distribution_kept_dets": dict(self.label_dist),
        }


# ---------------------------------------------------------------------------
# gallery helpers — self-contained HTML, base64 thumbs (house style)
# ---------------------------------------------------------------------------

def _b64_jpeg(img: Image.Image, quality=70) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def _annotated(img: Image.Image, boxes, max_dim=520) -> str:
    """boxes = [(xyxy, label, rgb)] -> base64 JPEG of a resized annotated copy."""
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    canvas = img.convert("RGB")
    if scale < 1.0:
        canvas = canvas.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    draw = ImageDraw.Draw(canvas)
    for box, label, color in boxes:
        b = [v * scale for v in box]
        draw.rectangle(b, outline=color, width=2)
        if label:
            draw.text((b[0] + 1, max(0, b[1] - 11)), label, fill=color)
    return _b64_jpeg(canvas)

def _crop(img: Image.Image, box, boxes=(), max_dim=360) -> str:
    """Crop around `box` with padding; draw `boxes` = [(xyxy, rgb, width)]
    translated into crop coordinates, so a crowded crop shows exactly WHICH
    person is being judged (green = GT person) and each model's box."""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    c = img.crop((cx1, cy1,
                  min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph))))
    s = 1.0
    if max(c.size) > max_dim:
        s = max_dim / max(c.size)
        c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))))
    c = c.convert("RGB")
    draw = ImageDraw.Draw(c)
    for b, color, width in boxes:
        draw.rectangle([(b[0] - cx1) * s, (b[1] - cy1) * s,
                        (b[2] - cx1) * s, (b[3] - cy1) * s],
                       outline=color, width=width)
    return _b64_jpeg(c)

def _img_tag(b64, cls=""):
    return f'<img class="{cls}" src="data:image/jpeg;base64,{b64}">'


def build_gallery(out_path: Path, scoreboard: dict, disagreements: list,
                  agreements_sample: list, only_found: dict, pairs: list):
    """All inputs carry pre-rendered base64 images — this only writes HTML."""
    css = """
    body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;
         color:#222;margin:20px;max-width:1200px}
    .card{display:inline-block;vertical-align:top;background:#fff;margin:6px;
          padding:8px;border:1px solid #ddd;border-radius:6px;max-width:360px}
    .card p{margin:6px 0 0;font-size:13px}
    .pair{background:#fff;border:1px solid #ddd;border-radius:6px;
          padding:8px;margin:10px 0}
    .pair img{max-width:48%;margin-right:1%}
    .sam{color:#2858dc;font-weight:600}.lite{color:#e08000;font-weight:600}
    .num{color:#888;font-size:12px}
    h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:34px}
    pre{background:#f0f0f0;padding:10px;border-radius:6px;font-size:12px;
        overflow-x:auto}
    """
    h = [f"<title>EXP-2026-09 head-to-head gallery</title><style>{css}</style>",
         "<h1>EXP-2026-09 — SAM3 vs Gemini 3.5 Flash-Lite, crowd head-to-head</h1>",
         "<p>Box colors everywhere: <span style='color:#00be00'>thick green = "
         "the GT person being judged</span>, ",
         "<span class='sam'>blue = SAM3's box</span>, ",
         "<span class='lite'>orange = Flash-Lite's box</span>. In crowded "
         "crops, judge ONLY the person inside the green box.</p>",
         f"<pre>{json.dumps(scoreboard, indent=2)}</pre>"]

    h.append(f"<h2>1. Label disagreements — ADJUDICATE THESE ({len(disagreements)})</h2>"
             "<p>Both models found this GT person — <b>the one in the thick "
             "green box</b> — and their 3-class labels differ. Tally each crop "
             "as: <b>SAM right</b> / <b>Lite right</b> / <b>both wrong</b> / "
             "<b>can't tell</b>. Count adult&harr;Child direction errors "
             "separately (adult labeled Child = the direction that escapes the "
             "blur). Crops where Lite says 'Unknown' are pre-counted "
             "abstentions — skim them quickly, spend judgment on the real "
             "class conflicts.</p>")
    for i, d in enumerate(disagreements, 1):
        h.append(f"<div class='card'>{_img_tag(d['b64'])}"
                 f"<p><span class='num'>CROP {i} — {d['image']}"
                 f" (occ: {d['occ_band']})</span><br>"
                 f"<span class='sam'>SAM: {d['sam_label']}</span> (IoU {d['sam_iou']:.2f}) · "
                 f"<span class='lite'>Lite: {d['lite_label']}</span> (IoU {d['lite_iou']:.2f})"
                 f"</p></div>")

    h.append(f"<h2>2. Agreement sample ({len(agreements_sample)}) — correlated-error check</h2>"
             "<p>Random sample of persons where both models gave the SAME label. "
             "Skim: are the agreed labels actually right, or do the models share a bias?</p>")
    for i, d in enumerate(agreements_sample, 1):
        h.append(f"<div class='card'>{_img_tag(d['b64'])}"
                 f"<p><span class='num'>AGREE {i} — {d['image']}</span><br>"
                 f"Both say: <b>{d['sam_label']}</b></p></div>")

    for key, title in (("sam_only", "3. Found by SAM3 only (Lite missed)"),
                       ("lite_only", "4. Found by Flash-Lite only (SAM3 missed)")):
        items = only_found[key]
        h.append(f"<h2>{title} — sample of {len(items)}</h2>")
        for d in items:
            who = "sam" if key == "sam_only" else "lite"
            h.append(f"<div class='card'>{_img_tag(d['b64'])}"
                     f"<p><span class='num'>{d['image']} (occ: {d['occ_band']})</span><br>"
                     f"<span class='{who}'>{d['label']}</span></p></div>")

    h.append(f"<h2>5. Side-by-side frames ({len(pairs)})</h2>"
             "<p>Left: SAM3 boxes over GT. Right: Flash-Lite boxes over GT. "
             "Ordered by label-disagreement count.</p>")
    for p in pairs:
        h.append(f"<div class='pair'><p><b>{p['image']}</b> — GT {p['n_gt']} · "
                 f"<span class='sam'>SAM {p['n_sam']}</span> · "
                 f"<span class='lite'>Lite {p['n_lite']}</span> · "
                 f"disagreements {p['n_disagree']}</p>"
                 f"{_img_tag(p['sam_b64'])}{_img_tag(p['lite_b64'])}</div>")

    out_path.write_text("\n".join(h))


# ---------------------------------------------------------------------------
# main scoring
# ---------------------------------------------------------------------------

def run_headtohead(images_dir: Path, odgt_path: Path, sam_dir: Path,
                   lite_jsonl: Path, out_dir: Path, match_iou_thr: float,
                   max_crops: int, max_pairs: int, lite_name: str,
                   sam_conf_floor=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_odgt(odgt_path)
    lite = load_lite_detections(lite_jsonl)

    sam_t, lite_t = ModelTally("sam3"), ModelTally(lite_name)
    n_scored = miss_sam = miss_lite = miss_gt = 0
    lite_parse_failures = 0
    sam_no_conf_lines = 0
    both = sam_only = lite_only = neither = 0
    agree = 0
    agree_matrix = Counter()          # "SAMLABEL|LITELABEL"
    lite_unknown_on_both = 0
    sam_h2h_ious, lite_h2h_ious = [], []

    disagreements, agreements_all, so_all, lo_all = [], [], [], []
    pair_meta = []
    per_person_f = open(out_dir / "per_person.jsonl", "w")

    for img_path in list_images(images_dir):
        stem = img_path.stem
        rec = gt.get(stem)
        if rec is None:
            miss_gt += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        sam_dets, n_no_conf = _dets_sam(sam_dir / f"{stem}.txt", w, h,
                                        sam_conf_floor)
        sam_no_conf_lines += n_no_conf
        lrec = lite.get(stem)
        if sam_dets is None:
            miss_sam += 1
            continue
        if lrec is None:
            miss_lite += 1
            continue
        if not lrec.get("parse_ok", True):
            lite_parse_failures += 1   # scored anyway: an empty/truncated
            # response is Flash-Lite's own failure, not a scoring gap
        lite_dets = _dets_lite(lrec)
        n_scored += 1

        persons = rec["persons"]
        vboxes = [p["vbox"] for p in persons]
        sam_kept, sam_ign = _split_ignores(sam_dets, rec["ignores"])
        lite_kept, lite_ign = _split_ignores(lite_dets, rec["ignores"])
        sam_t.dets_ignored += sam_ign
        lite_t.dets_ignored += lite_ign
        sam_m = match_boxes(vboxes, sam_kept, match_iou_thr)
        lite_m = match_boxes(vboxes, lite_kept, match_iou_thr)
        sam_t.add_image(persons, vboxes, sam_kept, sam_m)
        lite_t.add_image(persons, vboxes, lite_kept, lite_m)

        img = Image.open(img_path).convert("RGB")
        n_disagree_img = 0
        for gi, p in enumerate(persons):
            band = _occ_band(p["occ_ratio"])
            s, l = sam_m.get(gi), lite_m.get(gi)
            row = {"image": img_path.name, "gt_idx": gi, "occ_band": band,
                   "vbox": [round(v, 1) for v in p["vbox"]],
                   "sam": {"matched": s is not None},
                   "lite": {"matched": l is not None}}
            if s:
                row["sam"].update(label=s[0][0], iou=round(s[1], 3))
            if l:
                row["lite"].update(label=l[0][0], iou=round(l[1], 3))
            per_person_f.write(json.dumps(row) + "\n")

            if s and l:
                both += 1
                sam_lab, lite_lab = s[0][0], l[0][0]
                agree_matrix[f"{sam_lab}|{lite_lab}"] += 1
                sam_h2h_ious.append(s[1])
                lite_h2h_ious.append(l[1])
                if lite_lab == "Unknown":
                    lite_unknown_on_both += 1
                item = {"image": img_path.name, "occ_band": band,
                        "sam_label": sam_lab, "lite_label": lite_lab,
                        "sam_iou": s[1], "lite_iou": l[1],
                        "vbox": p["vbox"],
                        "sam_box": list(s[0][1:5]), "lite_box": list(l[0][1:5])}
                if sam_lab == lite_lab:
                    agree += 1
                    agreements_all.append((img_path, item))
                else:
                    n_disagree_img += 1
                    if len(disagreements) < max_crops:
                        item["b64"] = _crop(img, p["vbox"],
                                            [(item["sam_box"], SAM_COLOR, 2),
                                             (item["lite_box"], LITE_COLOR, 2),
                                             (p["vbox"], GT_COLOR, 3)])
                        disagreements.append(item)
            elif s:
                sam_only += 1
                if len(so_all) < 25:
                    so_all.append({"image": img_path.name, "occ_band": band,
                                   "label": s[0][0],
                                   "b64": _crop(img, p["vbox"],
                                                [(list(s[0][1:5]), SAM_COLOR, 2),
                                                 (p["vbox"], GT_COLOR, 3)])})
            elif l:
                lite_only += 1
                if len(lo_all) < 25:
                    lo_all.append({"image": img_path.name, "occ_band": band,
                                   "label": l[0][0],
                                   "b64": _crop(img, p["vbox"],
                                                [(list(l[0][1:5]), LITE_COLOR, 2),
                                                 (p["vbox"], GT_COLOR, 3)])})
            else:
                neither += 1

        pair_meta.append({"path": img_path, "n_gt": len(persons),
                          "n_sam": len(sam_kept), "n_lite": len(lite_kept),
                          "n_disagree": n_disagree_img,
                          "sam_kept": sam_kept, "lite_kept": lite_kept,
                          "vboxes": vboxes})
        img.close()
    per_person_f.close()

    # agreement sample: seeded, drawn AFTER the loop so it spans all images
    rng = random.Random(GALLERY_SEED)
    rng.shuffle(agreements_all)
    agreements_sample = []
    for img_path, item in agreements_all[:30]:
        with Image.open(img_path) as im:
            item["b64"] = _crop(im.convert("RGB"), item["vbox"],
                                [(item["sam_box"], SAM_COLOR, 2),
                                 (item["lite_box"], LITE_COLOR, 2),
                                 (item["vbox"], GT_COLOR, 3)])
        agreements_sample.append(item)

    # side-by-side frames for the images with the most label disagreements
    pair_meta.sort(key=lambda m: (-m["n_disagree"], -m["n_gt"]))
    pairs = []
    for m in pair_meta[:max_pairs]:
        with Image.open(m["path"]) as im:
            im = im.convert("RGB")
            gt_boxes = [(v, "", GT_COLOR) for v in m["vboxes"]]
            sam_b = gt_boxes + [(list(d[1:5]), d[0][0], SAM_COLOR)
                                for d in m["sam_kept"]]
            lite_b = gt_boxes + [(list(d[1:5]), d[0][0], LITE_COLOR)
                                 for d in m["lite_kept"]]
            pairs.append({"image": m["path"].name, "n_gt": m["n_gt"],
                          "n_sam": m["n_sam"], "n_lite": m["n_lite"],
                          "n_disagree": m["n_disagree"],
                          "sam_b64": _annotated(im, sam_b),
                          "lite_b64": _annotated(im, lite_b)})

    n_h2h = len(sam_h2h_ious)
    summary = {
        "mode": "crowd_headtohead", "match_iou": match_iou_thr,
        "models": {"sam3": ("frozen production autolabel_sam.py labels"
                            if sam_conf_floor is None else
                            f"conf-logged labels, floor >= {sam_conf_floor}"),
                   "lite": lite_name},
        "sam_conf_floor": sam_conf_floor,
        "sam_lines_without_conf_kept": sam_no_conf_lines,
        "images_scored": n_scored,
        "images_missing_sam_labels": miss_sam,
        "images_missing_lite_record": miss_lite,
        "images_missing_gt": miss_gt,
        "lite_parse_failures_scored_as_empty": lite_parse_failures,
        "per_model": {"sam3": sam_t.summary(), "flash_lite": lite_t.summary()},
        "head_to_head": {
            "persons_found_by_both": both,
            "sam_only": sam_only, "lite_only": lite_only, "neither": neither,
            "label_agreement_on_both_found": _rate(agree, both),
            "agreement_matrix_sam_x_lite": dict(sorted(agree_matrix.items())),
            "lite_unknown_on_both_found": lite_unknown_on_both,
            "box_quality_on_both_found": {
                "n": n_h2h,
                "sam_mean_iou": round(sum(sam_h2h_ious) / n_h2h, 3) if n_h2h else None,
                "lite_mean_iou": round(sum(lite_h2h_ious) / n_h2h, 3) if n_h2h else None,
                "sam_tighter_pct": (round(100 * sum(
                    1 for a, b in zip(sam_h2h_ious, lite_h2h_ious) if a > b)
                    / n_h2h, 1) if n_h2h else None),
            },
        },
    }
    scoreboard = {k: v for k, v in summary.items() if k != "mode"}
    build_gallery(out_dir / "gallery.html", scoreboard, disagreements,
                  agreements_sample, {"sam_only": so_all, "lite_only": lo_all},
                  pairs)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# subset materialization — MUST select identically to vlm_detect_eval
# ---------------------------------------------------------------------------

def make_subset(source_dir: Path, n: int, seed: int, subset_out: Path):
    """Copy the first n images of vlm_detect_eval.select_images' seeded
    shuffle into subset_out, so the subset is frozen on disk and byte-
    identical to what `vlm_detect_eval.py --seed <seed> --max-images <n>`
    would pick from the same source dir. Lazy import: select_images lives in
    vlm_detect_eval, which needs cv2 (pod has it; not required for scoring)."""
    from vlm_detect_eval import select_images
    subset_out.mkdir(parents=True, exist_ok=True)
    picked = select_images(source_dir, n, seed)
    for p in picked:
        shutil.copy2(p, subset_out / p.name)
    listing = subset_out / "subset_list.txt"
    listing.write_text("\n".join(p.name for p in picked) + "\n")
    print(f"[subset] copied {len(picked)} images from {source_dir} "
          f"(seed {seed}) -> {subset_out}")
    return picked


# ---------------------------------------------------------------------------
# self-test — synthetic, no data, no network, no cv2
# ---------------------------------------------------------------------------

def _selftest():
    import tempfile

    # label mapping edge cases (pre-registered rules)
    assert lite_label3({"age_group": "child", "gender": "man"}) == "Child"
    assert lite_label3({"age_group": "adult", "gender": "woman"}) == "Woman"
    assert lite_label3({"age_group": "unknown", "gender": "man"}) == "Man"
    assert lite_label3({"age_group": "adult", "gender": "unknown"}) == "Unknown"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "imgs").mkdir(); (root / "sam").mkdir()
        for name in ("x", "y"):
            Image.new("RGB", (100, 100), (128,) * 3).save(root / f"imgs/{name}.jpg")

        # image x — GT: A light occ, B partial occ, one ignore region.
        # image y — GT: C light occ.
        odgt = [
            {"ID": "x", "gtboxes": [
                {"tag": "person", "vbox": [10, 10, 30, 80],
                 "fbox": [10, 10, 30, 80], "extra": {}},                 # A
                {"tag": "person", "vbox": [60, 10, 30, 80],
                 "fbox": [50, 10, 45, 80], "extra": {}},                 # B
                {"tag": "mask", "fbox": [0, 92, 20, 7], "extra": {"ignore": 1}},
            ]},
            {"ID": "y", "gtboxes": [
                {"tag": "person", "vbox": [20, 10, 40, 80],
                 "fbox": [20, 10, 40, 80], "extra": {}},                 # C
            ]},
        ]
        (root / "anno.odgt").write_text(
            "\n".join(json.dumps(r) for r in odgt) + "\n")

        # SAM (YOLO cxcywh): x -> matches A as Man, B as Woman, one clear FP,
        # one det inside the ignore region. y -> matches C as Man.
        (root / "sam/x.txt").write_text(
            "1 0.25 0.50 0.30 0.80\n"      # A  (Man)
            "0 0.75 0.50 0.30 0.80\n"      # B  (Woman)
            "2 0.50 0.50 0.06 0.06\n"      # clear FP (Child)
            "0 0.10 0.955 0.16 0.05\n")    # in ignore region -> excluded
        (root / "sam/y.txt").write_text("1 0.40 0.50 0.40 0.80\n")

        # Lite: x -> matches A as Woman (disagrees with SAM), misses B.
        #       y -> matches C as Man (agrees).
        lite = [
            {"image": "x.jpg", "width": 100, "height": 100, "parse_ok": True,
             "people": [{"box_xyxy": [10, 10, 40, 90], "gender": "woman",
                         "age_group": "adult", "estimated_age": 30,
                         "confidence": "high"}]},
            {"image": "y.jpg", "width": 100, "height": 100, "parse_ok": True,
             "people": [{"box_xyxy": [20, 10, 60, 90], "gender": "man",
                         "age_group": "adult", "estimated_age": 40,
                         "confidence": "high"}]},
        ]
        (root / "lite.jsonl").write_text(
            "\n".join(json.dumps(r) for r in lite) + "\n")

        s = run_headtohead(root / "imgs", root / "anno.odgt", root / "sam",
                           root / "lite.jsonl", root / "out", 0.5,
                           max_crops=50, max_pairs=10,
                           lite_name="lite-selftest")

        assert s["images_scored"] == 2, s
        pm = s["per_model"]
        assert pm["sam3"]["recall"]["k"] == 3 and pm["sam3"]["recall"]["n"] == 3, pm
        assert pm["sam3"]["dets_excluded_by_ignore_regions"] == 1, pm
        assert pm["sam3"]["precision"]["k"] == 3 and pm["sam3"]["precision"]["n"] == 4, pm
        assert pm["sam3"]["clear_false_positives"] == 1, pm
        assert pm["flash_lite"]["recall"]["k"] == 2, pm
        assert pm["flash_lite"]["precision"]["n"] == 2, pm
        rb = pm["sam3"]["recall_by_occlusion"]
        assert rb["light"]["n"] == 2 and rb["partial"]["n"] == 1, rb
        h = s["head_to_head"]
        assert h["persons_found_by_both"] == 2, h
        assert h["sam_only"] == 1 and h["lite_only"] == 0 and h["neither"] == 0, h
        assert h["label_agreement_on_both_found"]["k"] == 1, h
        assert h["agreement_matrix_sam_x_lite"] == {"Man|Man": 1, "Man|Woman": 1}, h
        assert h["box_quality_on_both_found"]["n"] == 2, h
        out = root / "out"
        assert (out / "gallery.html").exists()
        assert len((out / "per_person.jsonl").read_text().splitlines()) == 3
        html = (out / "gallery.html").read_text()
        assert "CROP 1" in html and "AGREE 1" in html, "gallery sections missing"

        # ---- conf-floor path (Part 2: conf-logged labels) -------------------
        (root / "sam_conf").mkdir()
        # x: A at conf 0.85 (kept at any floor), B at 0.15 (killed at 0.3),
        #    plus one legacy line WITHOUT a conf column (kept + counted).
        (root / "sam_conf/x.txt").write_text(
            "1 0.25 0.50 0.30 0.80 0.85\n"
            "0 0.75 0.50 0.30 0.80 0.15\n"
            "2 0.50 0.50 0.06 0.06\n")
        (root / "sam_conf/y.txt").write_text("1 0.40 0.50 0.40 0.80 0.55\n")
        s = run_headtohead(root / "imgs", root / "anno.odgt", root / "sam_conf",
                           root / "lite.jsonl", root / "out2", 0.5,
                           max_crops=50, max_pairs=10,
                           lite_name="lite-selftest", sam_conf_floor=0.3)
        assert s["sam_conf_floor"] == 0.3, s
        assert s["sam_lines_without_conf_kept"] == 1, s
        pm = s["per_model"]["sam3"]
        # B's det fell below the floor -> recall 2/3 (A on x, C on y), and the
        # no-conf FP line stayed -> dets_kept 3
        assert pm["recall"]["k"] == 2 and pm["recall"]["n"] == 3, pm
        assert pm["dets_kept"] == 3, pm

    print("crowd_headtohead.py self-test passed")


def main():
    ap = argparse.ArgumentParser(
        description="EXP-2026-09: SAM3 vs Flash-Lite crowd head-to-head")
    ap.add_argument("--images", help="the frozen 100-image subset dir")
    ap.add_argument("--odgt", help="CrowdHuman annotation_val.odgt")
    ap.add_argument("--sam-labels", help="autolabel_sam.py YOLO .txt dir (exp03)")
    ap.add_argument("--lite-detections",
                    help="vlm_detect_eval detections.jsonl AFTER "
                         "reparse_boxes --write yxyx_1000")
    ap.add_argument("--lite-name", default="gemini-3.5-flash-lite")
    ap.add_argument("--sam-conf-floor", type=float, default=None,
                    help="Part 2 only: --sam-labels points at conf-logged "
                         "labels (autolabel_sam_conf.py output); keep "
                         "detections with conf >= this floor. Omit for the "
                         "plain frozen EXP-2026-03 labels.")
    ap.add_argument("--out")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--crops", type=int, default=100,
                    help="max disagreement crops in the gallery")
    ap.add_argument("--pairs", type=int, default=30,
                    help="max side-by-side frames in the gallery")
    # subset mode
    ap.add_argument("--make-subset", action="store_true")
    ap.add_argument("--source-images")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.make_subset:
        if not (args.source_images and args.subset_out):
            ap.error("--make-subset needs --source-images and --subset-out")
        make_subset(Path(args.source_images), args.n, args.seed,
                    Path(args.subset_out))
        return
    for req in ("images", "odgt", "sam_labels", "lite_detections", "out"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required (or --selftest / --make-subset)")
    summary = run_headtohead(Path(args.images), Path(args.odgt),
                             Path(args.sam_labels), Path(args.lite_detections),
                             Path(args.out), args.match_iou,
                             args.crops, args.pairs, args.lite_name,
                             sam_conf_floor=args.sam_conf_floor)
    print(json.dumps({k: v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
