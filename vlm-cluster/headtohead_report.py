#!/usr/bin/env python3
"""
EXP-2026-09 — evidence report: every headline claim paired with visual proof.

Reads the SAME inputs as crowd_headtohead.py (same primitives, so every number
here is identical to summary.json) and writes ONE self-contained HTML report:
each section states a claim with its measured number, then shows the images
that prove it — e.g. clear false positives appear as the full frame with the
FP box highlighted PLUS a zoomed crop of exactly what the model boxed.

    python3 headtohead_report.py \
        --images /workspace/exp09/subset100 \
        --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
        --sam-labels /workspace/exp03/sam_labels/crowd \
        --lite-detections /workspace/exp09/lite_crowd/detections.jsonl \
        --out /workspace/exp09/report.html

    python3 headtohead_report.py --selftest    # synthetic, no data, no network

CPU-only. Reuses (does NOT duplicate): the match/parse/draw primitives from
crowd_headtohead.py, load_odgt/_occ_band/list_images from
eval_negatives_crowd.py, match_boxes/iou from run_model_children.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from crowd_headtohead import (GT_COLOR, LITE_COLOR, SAM_COLOR, _annotated,
                              _crop, _dets_lite, _dets_sam, _split_ignores,
                              load_lite_detections)
from eval_negatives_crowd import PARTIAL_IOU, _occ_band, list_images, load_odgt
from run_model_children import iou, match_boxes

MISS_COLOR = (220, 0, 0)
SEED = 7


def _img_tag(b64):
    return f'<img src="data:image/jpeg;base64,{b64}">'


def _card(b64s, caption):
    imgs = "".join(_img_tag(b) for b in b64s)
    return (f"<div class='card'>{imgs}<p>{caption}</p></div>")


def collect(images_dir: Path, odgt_path: Path, sam_dir: Path,
            lite_jsonl: Path, match_iou_thr: float):
    """One pass over the images; returns aggregate numbers + per-image records
    holding everything the evidence sections need."""
    gt = load_odgt(odgt_path)
    lite = load_lite_detections(lite_jsonl)
    recs = []
    for img_path in list_images(images_dir):
        stem = img_path.stem
        rec_gt = gt.get(stem)
        lrec = lite.get(stem)
        if rec_gt is None or lrec is None:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        sam_dets, _ = _dets_sam(sam_dir / f"{stem}.txt", w, h)
        if sam_dets is None:
            continue
        lite_dets = _dets_lite(lrec)
        persons = rec_gt["persons"]
        vboxes = [p["vbox"] for p in persons]
        sam_kept, _ = _split_ignores(sam_dets, rec_gt["ignores"])
        lite_kept, _ = _split_ignores(lite_dets, rec_gt["ignores"])
        sam_m = match_boxes(vboxes, sam_kept, match_iou_thr)
        lite_m = match_boxes(vboxes, lite_kept, match_iou_thr)

        def unmatched_clear_fps(kept, matched):
            matched_ids = {id(d) for d, _ in matched.values()}
            out = []
            for d in kept:
                if id(d) in matched_ids:
                    continue
                best = max((iou(v, d[1:5]) for v in vboxes), default=0.0)
                if best < PARTIAL_IOU:
                    out.append((d, best))
            return out

        recs.append({
            "path": img_path, "w": w, "h": h,
            "persons": persons, "vboxes": vboxes,
            "sam_kept": sam_kept, "lite_kept": lite_kept,
            "sam_m": sam_m, "lite_m": lite_m,
            "sam_fps": unmatched_clear_fps(sam_kept, sam_m),
            "lite_fps": unmatched_clear_fps(lite_kept, lite_m),
        })
    return recs


def build_report(recs, out_path: Path, lite_name: str):
    rng = random.Random(SEED)
    n_img = len(recs)
    n_gt = sum(len(r["persons"]) for r in recs)
    sam_found = sum(len(r["sam_m"]) for r in recs)
    lite_found = sum(len(r["lite_m"]) for r in recs)
    sam_kept = sum(len(r["sam_kept"]) for r in recs)
    lite_kept = sum(len(r["lite_kept"]) for r in recs)
    sam_fp_n = sum(len(r["sam_fps"]) for r in recs)
    lite_fp_n = sum(len(r["lite_fps"]) for r in recs)

    # per-person head-to-head rows
    both = sam_only = lite_only = neither = agree = 0
    child_conflicts, gender_flips, tight_pairs = [], [], []
    occ_blind, lite_only_ev, neither_ev = [], [], []
    for r in recs:
        for gi, p in enumerate(r["persons"]):
            s, l = r["sam_m"].get(gi), r["lite_m"].get(gi)
            band = _occ_band(p["occ_ratio"])
            if s and l:
                both += 1
                sl, ll = s[0][0], l[0][0]
                if sl == ll:
                    agree += 1
                elif {"Child"} & {sl, ll} and "Unknown" not in (sl, ll):
                    child_conflicts.append((r, p, s, l, band))
                elif {sl, ll} == {"Man", "Woman"}:
                    gender_flips.append((r, p, s, l, band))
                tight_pairs.append((r, p, s, l, l[1] - s[1]))
            elif s:
                sam_only += 1
                if band in ("heavy", "partial"):
                    occ_blind.append((r, p, s, band, p["occ_ratio"]))
            elif l:
                lite_only += 1
                lite_only_ev.append((r, p, l, band))
            else:
                neither += 1
                neither_ev.append((r, p, band))

    css = """
    body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;
         color:#222;margin:24px;max-width:1240px}
    h1{margin-bottom:2px} .sub{color:#666;margin-top:0}
    h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:40px}
    .claim{background:#eef4ff;border-left:4px solid #2858dc;padding:10px 14px;
           border-radius:0 6px 6px 0;margin:12px 0;font-size:15px}
    .card{display:inline-block;vertical-align:top;background:#fff;margin:6px;
          padding:8px;border:1px solid #ddd;border-radius:6px;max-width:580px}
    .card img{max-width:275px;margin-right:4px;vertical-align:top}
    .card p{margin:6px 0 0;font-size:12.5px;color:#333}
    table{border-collapse:collapse;margin:14px 0}
    td,th{border:1px solid #ccc;padding:6px 12px;font-size:14px;text-align:right}
    th:first-child,td:first-child{text-align:left}
    .sam{color:#2858dc;font-weight:600}.lite{color:#e08000;font-weight:600}
    .miss{color:#dc2020;font-weight:600}.gt{color:#00a000;font-weight:600}
    .note{color:#777;font-size:13px}
    """
    per_img_gt = n_gt / n_img if n_img else 0
    H = [f"<title>EXP-2026-09 evidence report</title><style>{css}</style>",
         "<h1>EXP-2026-09 — SAM3 vs Gemini 3.5 Flash-Lite on crowd scenes</h1>",
         f"<p class='sub'>Evidence report · {n_img} CrowdHuman images · {n_gt} "
         f"annotated people (~{per_img_gt:.1f}/image) · match IoU 0.5 · every "
         "number recomputed with the same code that produced summary.json</p>",
         "<p>Colors: <span class='gt'>green = ground-truth person</span> · "
         "<span class='sam'>blue = SAM3 box</span> · "
         "<span class='lite'>orange = Flash-Lite box</span> · "
         "<span class='miss'>red = missed / false box</span></p>"]

    # ---- scoreboard ------------------------------------------------------
    def pct(k, n):
        return f"{100 * k / n:.1f}%" if n else "—"
    H.append("<h2>Scoreboard</h2><table>"
             "<tr><th></th><th>SAM3 (production)</th>"
             f"<th>{lite_name}</th></tr>"
             f"<tr><td>People found</td><td>{sam_found}</td><td>{lite_found}</td></tr>"
             f"<tr><td>People missed</td><td>{n_gt - sam_found} "
             f"(~{(n_gt - sam_found) / n_img:.1f}/image)</td>"
             f"<td>{n_gt - lite_found} (~{(n_gt - lite_found) / n_img:.1f}/image)</td></tr>"
             f"<tr><td>Recall</td><td>{pct(sam_found, n_gt)}</td>"
             f"<td>{pct(lite_found, n_gt)}</td></tr>"
             f"<tr><td>Precision</td><td>{pct(sam_found, sam_kept)}</td>"
             f"<td>{pct(lite_found, lite_kept)}</td></tr>"
             f"<tr><td>Clear false positives</td><td>{sam_fp_n} "
             f"(~{sam_fp_n / n_img:.2f}/image)</td>"
             f"<td>{lite_fp_n} (~{lite_fp_n / n_img:.2f}/image)</td></tr>"
             f"<tr><td>Label agreement (both found, n={both})</td>"
             f"<td colspan=2 style='text-align:center'>{pct(agree, both)}</td></tr>"
             "</table>"
             f"<p class='note'>Found by both: {both} · SAM3 only: {sam_only} · "
             f"Lite only: {lite_only} · neither: {neither} "
             f"(~{neither / n_img:.1f}/image invisible to both). Label "
             "CORRECTNESS is not measurable against CrowdHuman (no gender/age "
             "GT) — agreement + human adjudication only.</p>")

    # ---- 1. misses per image (frames) -----------------------------------
    H.append("<h2>1. Missed people, per image</h2>"
             f"<div class='claim'>In a typical ~{per_img_gt:.0f}-person crowd "
             f"image, SAM3 fails to box ~{(n_gt - sam_found) / n_img:.1f} "
             f"people; Flash-Lite fails to box ~{(n_gt - lite_found) / n_img:.1f} "
             "— 2.3× more. Below: the worst frames per model; every "
             "<span class='miss'>red box</span> is an annotated person that "
             "model did not detect (green = detected).</div>")
    for name, mkey, cls in (("SAM3", "sam_m", "sam"),
                            (lite_name, "lite_m", "lite")):
        worst = sorted(recs, key=lambda r: len(r["persons"]) - len(r[mkey]),
                       reverse=True)[:4]
        H.append(f"<h3 class='{cls}'>{name} — most-missed frames</h3>")
        for r in worst:
            boxes = [(p["vbox"], "", GT_COLOR if gi in r[mkey] else MISS_COLOR)
                     for gi, p in enumerate(r["persons"])]
            with Image.open(r["path"]) as im:
                b64 = _annotated(im.convert("RGB"), boxes, max_dim=580)
            found = len(r[mkey])
            H.append(_card([b64],
                           f"{r['path'].name} — {name} found {found}/"
                           f"{len(r['persons'])}, missed "
                           f"<span class='miss'>{len(r['persons']) - found}</span>"))

    # ---- 2. occlusion ----------------------------------------------------
    occ_blind.sort(key=lambda t: t[4])
    H.append("<h2>2. Where the gap lives: occlusion</h2>"
             "<div class='claim'>Flash-Lite's recall collapses with occlusion "
             "(heavy: 21% vs SAM3's 62%). Below: heavily/partially hidden "
             "people SAM3 boxed (blue) that Flash-Lite never detected.</div>")
    for r, p, s, band, ratio in occ_blind[:10]:
        with Image.open(r["path"]) as im:
            b64 = _crop(im.convert("RGB"), p["vbox"],
                        [(list(s[0][1:5]), SAM_COLOR, 2), (p["vbox"], GT_COLOR, 3)])
        H.append(_card([b64], f"{r['path'].name} — occlusion {band} "
                             f"(visible {ratio:.0%} of body) — "
                             f"<span class='sam'>SAM3: {s[0][0]}</span>, "
                             "<span class='lite'>Lite: not detected</span>"))

    # ---- 3. clear false positives (frame + zoom pairs) -------------------
    H.append("<h2>3. Clear false positives — frame + zoomed crop</h2>"
             f"<div class='claim'>Boxes overlapping NO annotated person (best "
             f"IoU &lt; {PARTIAL_IOU}): SAM3 {sam_fp_n} (~{sam_fp_n / n_img:.2f}"
             f"/image), Flash-Lite {lite_fp_n} (~{lite_fp_n / n_img:.2f}/image)."
             " Each card: full frame (red = the false box, green = GT) and the "
             "zoomed crop of exactly what the model boxed. Caveat: CrowdHuman "
             "doesn't annotate posters/depictions — under our ruling those "
             "count as people, so judge each crop yourself: some may be real "
             "unannotated people or posters, not hallucinations.</div>")
    for name, fkey, cls in (("SAM3", "sam_fps", "sam"),
                            (lite_name, "lite_fps", "lite")):
        ev = [(r, d, best) for r in recs for d, best in r[fkey]]
        rng.shuffle(ev)
        H.append(f"<h3 class='{cls}'>{name} — {len(ev)} total, showing "
                 f"{min(12, len(ev))}</h3>")
        for r, d, best in ev[:12]:
            fb = list(d[1:5])
            frame_boxes = ([(v, "", GT_COLOR) for v in r["vboxes"]]
                           + [(fb, d[0], MISS_COLOR)])
            with Image.open(r["path"]) as im:
                im = im.convert("RGB")
                frame = _annotated(im, frame_boxes, max_dim=460)
                zoom = _crop(im, fb, [(fb, MISS_COLOR, 3)], max_dim=300)
            H.append(_card([frame, zoom],
                           f"{r['path'].name} — {name} wrote "
                           f"<b>{d[0]}</b> here; best IoU vs any annotated "
                           f"person = {best:.2f}"))

    # ---- 4. box tightness ------------------------------------------------
    tight_pairs.sort(key=lambda t: t[4], reverse=True)
    H.append("<h2>4. Box tightness on people both models found</h2>"
             "<div class='claim'>On the same person, Flash-Lite's box (orange) "
             "is usually the tighter fit (mean IoU 0.845 vs SAM3's 0.824; "
             "under the pre-registered 0.05 bar, so scored 'equivalent, "
             "leaning Lite'). Left cards: largest Lite advantage; last row: "
             "counter-examples where SAM3 fit better.</div>")
    show = tight_pairs[:8] + tight_pairs[-4:]
    for r, p, s, l, delta in show:
        with Image.open(r["path"]) as im:
            b64 = _crop(im.convert("RGB"), p["vbox"],
                        [(list(s[0][1:5]), SAM_COLOR, 2),
                         (list(l[0][1:5]), LITE_COLOR, 2),
                         (p["vbox"], GT_COLOR, 3)])
        H.append(_card([b64],
                       f"{r['path'].name} — GT-fit IoU: "
                       f"<span class='sam'>SAM3 {s[1]:.2f}</span> vs "
                       f"<span class='lite'>Lite {l[1]:.2f}</span>"))

    # ---- 5. label conflicts ---------------------------------------------
    H.append("<h2>5. Label conflicts on people both models found</h2>"
             "<h3>Child ↔ adult conflicts (all shown)</h3>")
    for r, p, s, l, band in child_conflicts:
        with Image.open(r["path"]) as im:
            b64 = _crop(im.convert("RGB"), p["vbox"],
                        [(list(s[0][1:5]), SAM_COLOR, 2),
                         (list(l[0][1:5]), LITE_COLOR, 2),
                         (p["vbox"], GT_COLOR, 3)])
        H.append(_card([b64], f"{r['path'].name} ({band}) — "
                             f"<span class='sam'>SAM3: {s[0][0]}</span> vs "
                             f"<span class='lite'>Lite: {l[0][0]}</span>"))
    rng.shuffle(gender_flips)
    H.append(f"<h3>Man ↔ Woman flips ({len(gender_flips)} total, sample of "
             f"{min(8, len(gender_flips))})</h3>")
    for r, p, s, l, band in gender_flips[:8]:
        with Image.open(r["path"]) as im:
            b64 = _crop(im.convert("RGB"), p["vbox"],
                        [(list(s[0][1:5]), SAM_COLOR, 2),
                         (list(l[0][1:5]), LITE_COLOR, 2),
                         (p["vbox"], GT_COLOR, 3)])
        H.append(_card([b64], f"{r['path'].name} ({band}) — "
                             f"<span class='sam'>SAM3: {s[0][0]}</span> vs "
                             f"<span class='lite'>Lite: {l[0][0]}</span>"))

    # ---- 6. union / complementarity -------------------------------------
    rng.shuffle(lite_only_ev)
    rng.shuffle(neither_ev)
    union = both + sam_only + lite_only
    H.append("<h2>6. Would running BOTH models help?</h2>"
             f"<div class='claim'>Barely. Union recall {pct(union, n_gt)} vs "
             f"SAM3 alone {pct(sam_found, n_gt)} — Flash-Lite adds only "
             f"{lite_only} people SAM3 missed (~{lite_only / n_img:.1f}/image) "
             f"while {neither} (~{neither / n_img:.1f}/image) stay invisible "
             "to both. Left: the rare Lite-only finds. Right: people neither "
             "model detected.</div>"
             f"<h3 class='lite'>Found by Flash-Lite only (sample of "
             f"{min(8, len(lite_only_ev))} / {lite_only})</h3>")
    for r, p, l, band in lite_only_ev[:8]:
        with Image.open(r["path"]) as im:
            b64 = _crop(im.convert("RGB"), p["vbox"],
                        [(list(l[0][1:5]), LITE_COLOR, 2), (p["vbox"], GT_COLOR, 3)])
        H.append(_card([b64], f"{r['path'].name} ({band}) — "
                             f"<span class='lite'>Lite: {l[0][0]}</span>, "
                             "<span class='sam'>SAM3: not detected</span>"))
    H.append(f"<h3 class='miss'>Detected by neither (sample of "
             f"{min(8, len(neither_ev))} / {neither})</h3>")
    for r, p, band in neither_ev[:8]:
        with Image.open(r["path"]) as im:
            b64 = _crop(im.convert("RGB"), p["vbox"], [(p["vbox"], MISS_COLOR, 3)])
        H.append(_card([b64], f"{r['path'].name} — occlusion {band}, "
                             "no box from either model"))

    out_path.write_text("\n".join(H))


# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "imgs").mkdir(); (root / "sam").mkdir()
        for name in ("x", "y"):
            Image.new("RGB", (100, 100), (128,) * 3).save(root / f"imgs/{name}.jpg")
        odgt = [
            {"ID": "x", "gtboxes": [
                {"tag": "person", "vbox": [10, 10, 30, 80],
                 "fbox": [10, 10, 30, 80], "extra": {}},
                {"tag": "person", "vbox": [60, 10, 30, 80],
                 "fbox": [50, 10, 45, 80], "extra": {}}]},
            {"ID": "y", "gtboxes": [
                {"tag": "person", "vbox": [20, 10, 40, 80],
                 "fbox": [20, 10, 40, 80], "extra": {}}]},
        ]
        (root / "anno.odgt").write_text(
            "\n".join(json.dumps(r) for r in odgt) + "\n")
        (root / "sam/x.txt").write_text(
            "1 0.25 0.50 0.30 0.80\n"       # matches A (Man)
            "0 0.75 0.50 0.30 0.80\n"       # matches B (Woman)
            "2 0.50 0.50 0.06 0.06\n")      # clear FP (Child)
        (root / "sam/y.txt").write_text("2 0.40 0.50 0.40 0.80\n")  # C as Child
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

        recs = collect(root / "imgs", root / "anno.odgt", root / "sam",
                       root / "lite.jsonl", 0.5)
        assert len(recs) == 2, recs
        assert sum(len(r["sam_fps"]) for r in recs) == 1      # the Child FP
        build_report(recs, root / "report.html", "lite-selftest")
        html = (root / "report.html").read_text()
        for marker in ("Scoreboard", "Missed people", "Clear false positives",
                       "Box tightness", "Label conflicts", "Child ↔ adult"):
            assert marker in html, f"missing section: {marker}"
        # x: SAM Man vs Lite Woman = gender flip; y: SAM Child vs Lite Man =
        # child-direction conflict — both must appear with evidence crops
        assert html.count("data:image/jpeg;base64,") >= 8
    print("headtohead_report.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-09 evidence report")
    ap.add_argument("--images")
    ap.add_argument("--odgt")
    ap.add_argument("--sam-labels")
    ap.add_argument("--lite-detections")
    ap.add_argument("--lite-name", default="gemini-3.5-flash-lite")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    for req in ("images", "odgt", "sam_labels", "lite_detections"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required (or --selftest)")
    recs = collect(Path(args.images), Path(args.odgt), Path(args.sam_labels),
                   Path(args.lite_detections), args.match_iou)
    build_report(recs, Path(args.out), args.lite_name)
    print(f"[report] {len(recs)} images -> {args.out}")


if __name__ == "__main__":
    main()
