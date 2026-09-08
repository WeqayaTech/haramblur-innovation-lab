#!/usr/bin/env python3
"""EXP-2026-04 — visual error galleries from EXISTING Stage A outputs (no model re-run).

Builds ONE self-contained HTML file (base64 thumbnails) per dataset so the errors
can be eyeballed / adjudicated. CPU-only; reads the same label files Stage B
scored, and reuses the SAME parsing/matching functions (seg_boxes, match_boxes,
iou, odgt loader, thresholds) — so the gallery shows exactly what the scorer saw.

  # objects / PASS (negatives: every detection is an error)
  python3 build_gallery_exp04.py --mode negatives \
      --images /workspace/datasets/object_set \
      --pred-labels /workspace/exp04/sam_labels/objects \
      --out /workspace/exp04/gallery_objects.html

  # CrowdHuman (clear FPs + duplicates + missed people, vs odgt GT)
  python3 build_gallery_exp04.py --mode crowd \
      --images /workspace/datasets/crowdhuman/Images_sample500 \
      --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
      --pred-labels /workspace/exp04/sam_labels/crowd \
      --out /workspace/exp04/gallery_crowd.html

  python3 build_gallery_exp04.py --selftest     # synthetic, no data needed

Colors: GT green · clear FP red · duplicate purple · partial-overlap grey ·
missed GT orange. Crowd gallery includes a crop strip of every clear FP for the
poster-vs-statue-vs-hallucination tally (the §2.2 "posters count as people" ruling).
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from eval_negatives_crowd import (DUP_IOU, IGNORE_IOA, PARTIAL_IOU, _ioa,
                                  list_images, load_odgt)
from run_autolabel_on_manifest import seg_boxes
from run_model_children import iou, match_boxes

GREEN, RED, PURPLE, GREY, ORANGE, BLUE = ((0, 190, 0), (220, 0, 0), (160, 0, 200),
                                          (150, 150, 150), (245, 130, 0), (70, 130, 240))


def thumb_b64(img, boxes, max_dim=420, quality=62):
    """boxes = [(xyxy, label, rgb)] drawn on a resized copy -> base64 JPEG."""
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
            draw.text((b[0] + 2, max(0, b[1] - 11)), label, fill=color)
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def crop_b64(img, box, pad=0.15, size=130, quality=62):
    w, h = img.size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    c = img.convert("RGB").crop((max(0, x1 - px), max(0, y1 - py),
                                 min(w, x2 + px), min(h, y2 + py)))
    c.thumbnail((size, size))
    buf = io.BytesIO()
    c.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def tile(b64, caption, name=None):
    name = html_mod.escape(name or caption.split(" ")[0].split(" ·")[0])
    return (f'<div class="tile" data-name="{name}"><img src="data:image/jpeg;base64,{b64}">'
            f'<div class="cap">{html_mod.escape(caption)}</div></div>')


def page(title, sub, sections):
    body = "".join(f'<h2>{html_mod.escape(h)}</h2><p class="sub">{html_mod.escape(s)}</p>'
                   f'<div class="grid">{"".join(tiles)}</div>'
                   for h, s, tiles in sections if tiles)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html_mod.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafbfc;color:#1a1d25}}
h1{{font-size:22px}} h2{{font-size:17px;margin:28px 0 2px}} .sub{{color:#8899aa;font-size:12.5px;margin:2px 0 10px}}
.grid{{display:flex;flex-wrap:wrap;gap:10px}}
.tile{{background:#fff;border:1px solid #e3e7ee;border-radius:6px;padding:6px;max-width:440px}}
.tile img{{display:block;max-width:428px;height:auto;border-radius:3px}}
.cap{{font-size:11px;color:#556;margin-top:4px;max-width:428px}}
.legend span{{display:inline-block;margin-right:14px;font-size:12.5px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:4px}}
.tile{{cursor:pointer}}
.tile.m1{{outline:4px solid #e74c3c}} .tile.m2{{outline:4px solid #4f8ef7}} .tile.m3{{outline:4px solid #f39c12}}
#bar{{position:fixed;top:0;right:0;background:#fff;border:1px solid #e3e7ee;border-radius:0 0 0 8px;
     padding:8px 14px;font-size:12.5px;box-shadow:0 1px 6px rgba(0,0,0,.08);z-index:9}}
#bar b{{margin:0 6px}} #bar button{{margin-left:10px;padding:3px 10px;font-size:12px;cursor:pointer}}
</style>
<script>
// Adjudication: CLICK a tile to cycle none -> RED -> BLUE -> ORANGE -> none.
// Meaning is per-task, e.g. objects rejects pass: RED = image contains a real person / photo of
// one (should not be in the set). Crowd clear-FP tally: RED = genuine hallucination,
// BLUE = real unlabeled person / body part, ORANGE = poster/statue of a person.
document.addEventListener('DOMContentLoaded', () => {{
  const states = ['', 'm1', 'm2', 'm3'];
  const counts = () => {{
    const c = {{m1: 0, m2: 0, m3: 0}};
    document.querySelectorAll('.tile').forEach(t => states.slice(1).forEach(s => {{ if (t.classList.contains(s)) c[s]++; }}));
    document.getElementById('cnt').textContent =
      `RED ${{c.m1}} · BLUE ${{c.m2}} · ORANGE ${{c.m3}}`;
  }};
  document.querySelectorAll('.tile').forEach(t => t.addEventListener('click', () => {{
    const i = states.findIndex(s => s && t.classList.contains(s));
    states.slice(1).forEach(s => t.classList.remove(s));
    const next = states[(i === -1 ? 0 : i) + 1] || '';
    if (next) t.classList.add(next);
    counts();
  }}));
  document.getElementById('exp').addEventListener('click', () => {{
    let out = '';
    const names = {{m1: 'RED', m2: 'BLUE', m3: 'ORANGE'}};
    document.querySelectorAll('.tile').forEach((t, i) => states.slice(1).forEach(s => {{
      if (t.classList.contains(s)) out += `${{t.dataset.name}}\\t${{names[s]}}\\t#${{i + 1}}\\n`;
    }}));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([out], {{type: 'text/plain'}}));
    a.download = 'marked.txt';
    a.click();
  }});
  counts();
}});
</script>
</head><body>
<div id="bar">click tiles to mark: <b style="color:#e74c3c">RED</b><b style="color:#4f8ef7">BLUE</b><b style="color:#f39c12">ORANGE</b>
 <span id="cnt"></span><button id="exp">download marked.txt</button></div>
<h1>{html_mod.escape(title)}</h1><p class="sub">{html_mod.escape(sub)}</p>
<p class="legend">
<span><i class="sw" style="background:rgb(0,190,0)"></i>GT person</span>
<span><i class="sw" style="background:rgb(220,0,0)"></i>false positive</span>
<span><i class="sw" style="background:rgb(160,0,200)"></i>duplicate</span>
<span><i class="sw" style="background:rgb(150,150,150)"></i>partial overlap (ambiguous)</span>
<span><i class="sw" style="background:rgb(245,130,0)"></i>missed GT</span></p>
{body}</body></html>"""


# ---------------------------------------------------------------------------
# negatives gallery — every detection is an error; grouped by filename prefix
# ---------------------------------------------------------------------------
def build_negatives(images_dir: Path, pred_dir: Path, out: Path,
                    max_per_group: int, thumb: int):
    groups = defaultdict(list)   # prefix -> [(n_fp, img_path, dets)]
    n_imgs = n_fp_total = 0
    for img_path in list_images(images_dir):
        dets = seg_boxes(pred_dir / (img_path.stem + ".txt"), 1, 1)
        if dets is None:
            continue
        n_imgs += 1
        if not dets:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        dets = seg_boxes(pred_dir / (img_path.stem + ".txt"), w, h)
        n_fp_total += len(dets)
        prefix = img_path.stem.split("_")[0] if "_" in img_path.stem else "images"
        groups[prefix].append((len(dets), img_path, dets))

    sections = []
    for prefix in sorted(groups, key=lambda p: -len(groups[p])):
        rows = sorted(groups[prefix], key=lambda r: -r[0])
        tiles = []
        for n_fp, img_path, dets in rows[:max_per_group]:
            with Image.open(img_path) as im:
                b64 = thumb_b64(im, [(list(d[1:5]), "FP", RED) for d in dets],
                                max_dim=thumb)
            tiles.append(tile(b64, f"{img_path.name} · {n_fp} false label(s)"))
        extra = f" (showing {min(len(rows), max_per_group)} of {len(rows)})" if len(rows) > max_per_group else ""
        sections.append((f"{prefix} — {len(rows)} images with false person labels{extra}",
                         "every box is a false positive by construction (verified person-free images)",
                         tiles))
    out.write_text(page(
        f"EXP-2026-04 error gallery — negatives ({images_dir.name}, single \"person\" prompt)",
        f"{n_imgs} images scored · {n_fp_total} false person labels total · built from the frozen "
        f"Stage A outputs in {pred_dir} — no model re-run", sections))
    return {"images_scored": n_imgs, "fp_total": n_fp_total,
            "groups": {k: len(v) for k, v in groups.items()}}


# ---------------------------------------------------------------------------
# crowd gallery — clear FPs (frames + crop strip), duplicates, missed people
# ---------------------------------------------------------------------------
def build_crowd(images_dir: Path, odgt_path: Path, pred_dir: Path, out: Path,
                match_iou_thr: float, max_frames: int, max_crops: int, thumb: int):
    gt = load_odgt(odgt_path)
    fp_frames, dup_frames, miss_frames = [], [], []   # (severity, img_path, boxes, caption)
    fp_crops = []                                     # (img_path, box)
    totals = defaultdict(int)

    for img_id, rec in sorted(gt.items()):
        img_path = images_dir / f"{img_id}.jpg"
        lbl = pred_dir / f"{img_id}.txt"
        if not img_path.exists():
            continue
        dets = seg_boxes(lbl, 1, 1)
        if dets is None:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        dets = seg_boxes(lbl, w, h)
        kept = [d for d in dets
                if not any(_ioa(list(d[1:5]), ig) > IGNORE_IOA for ig in rec["ignores"])]
        persons = rec["persons"]
        vboxes = [p["vbox"] for p in persons]
        matched = match_boxes(vboxes, kept, match_iou_thr)
        matched_dets = {id(det) for det, _ in matched.values()}

        dups, partials, clears = [], [], []
        for d in kept:
            if id(d) in matched_dets:
                continue
            best = max((iou(v, d[1:5]) for v in vboxes), default=0.0)
            (dups if best >= DUP_IOU else partials if best >= PARTIAL_IOU else clears).append(d)
        missed = [p["vbox"] for gi, p in enumerate(persons) if gi not in matched]
        totals["clear_fp"] += len(clears); totals["dup"] += len(dups)
        totals["missed"] += len(missed); totals["gt"] += len(persons)

        gt_boxes = [(p["vbox"], "", GREEN) for p in persons]
        if clears:
            boxes = gt_boxes + [(list(d[1:5]), "FP", RED) for d in clears] \
                             + [(list(d[1:5]), "", GREY) for d in partials]
            fp_frames.append((len(clears), img_path,
                              boxes, f"{img_id}.jpg · {len(clears)} clear FP(s)"))
            fp_crops += [(img_path, list(d[1:5])) for d in clears]
        if dups:
            dup_frames.append((len(dups), img_path,
                               gt_boxes + [(list(d[1:5]), "dup", PURPLE) for d in dups],
                               f"{img_id}.jpg · {len(dups)} duplicate(s)"))
        if missed:
            miss_frames.append((len(missed), img_path,
                                [(b, "", ORANGE) for b in missed]
                                + [(p["vbox"], "", GREEN) for gi, p in enumerate(persons) if gi in matched],
                                f"{img_id}.jpg · {len(missed)}/{len(persons)} people missed"))

    def render(frames, cap):
        tiles = []
        for sev, img_path, boxes, caption in sorted(frames, key=lambda r: -r[0])[:cap]:
            with Image.open(img_path) as im:
                tiles.append(tile(thumb_b64(im, boxes, max_dim=thumb), caption))
        return tiles

    crop_tiles = []
    for i, (img_path, box) in enumerate(fp_crops[:max_crops]):
        with Image.open(img_path) as im:
            crop_tiles.append(tile(crop_b64(im, box), f"#{i+1} {img_path.name}",
                                   name=img_path.name))

    sections = [
        (f"Clear false positives — {totals['clear_fp']} boxes "
         f"(frames: top {min(len(fp_frames), max_frames)} of {len(fp_frames)} images)",
         "unmatched detections with <0.1 IoU vs any GT person. HAND-CHECK RULE (§2.2): a box on a "
         "poster/photo of a real person is CORRECT per our ruling; tally real-person / poster-statue "
         "/ genuine hallucination using the crop strip below", render(fp_frames, max_frames)),
        (f"Clear-FP crop strip — first {len(crop_tiles)} of {totals['clear_fp']} for the tally",
         "each crop is one red box above, in order; count the three buckets here",
         crop_tiles),
        (f"Duplicates — {totals['dup']} boxes on already-matched people "
         f"({len(dup_frames)} images)",
         "fragmentation check: same person, extra box (IoU ≥ 0.5 with a matched GT)",
         render(dup_frames, max_frames)),
        (f"Missed people — {totals['missed']} of {totals['gt']} GT persons unmatched "
         f"(worst {min(len(miss_frames), max_frames)} of {len(miss_frames)} images)",
         "orange = GT person SAM never boxed (the recall failure); green = found",
         render(miss_frames, max_frames)),
    ]
    out.write_text(page(
        "EXP-2026-04 error gallery — CrowdHuman (single \"person\" prompt)",
        f"built from frozen Stage A outputs in {pred_dir} · match IoU {match_iou_thr} · "
        "same matching code as the scorer — no model re-run", sections))
    return dict(totals)


# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "imgs").mkdir(); (root / "lbls").mkdir()
        for name in ("doll_a", "statue_b"):
            Image.new("RGB", (100, 100), (128,) * 3).save(root / f"imgs/{name}.jpg")
        (root / "lbls/doll_a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.9 0.1 0.9\n")
        (root / "lbls/statue_b.txt").write_text("")
        s = build_negatives(root / "imgs", root / "lbls", root / "neg.html", 50, 200)
        assert s["images_scored"] == 2 and s["fp_total"] == 1, s
        html = (root / "neg.html").read_text()
        assert "doll — 1 images" in html and "base64" in html, "negatives html wrong"

        (root / "c_imgs").mkdir(); (root / "c_lbls").mkdir()
        Image.new("RGB", (100, 100), (128,) * 3).save(root / "c_imgs/x.jpg")
        odgt = {"ID": "x", "gtboxes": [
            {"tag": "person", "vbox": [10, 10, 30, 80], "fbox": [10, 10, 30, 80], "extra": {}},
            {"tag": "person", "vbox": [60, 10, 30, 80], "fbox": [60, 10, 30, 80], "extra": {}}]}
        (root / "a.odgt").write_text(json.dumps(odgt) + "\n")
        # d0 matches person A; d1 duplicate on A; d2 clear FP; person B missed
        (root / "c_lbls/x.txt").write_text(
            "0 0.10 0.10 0.40 0.10 0.40 0.90 0.10 0.90\n"
            "0 0.12 0.12 0.42 0.12 0.42 0.92 0.12 0.92\n"
            "0 0.46 0.40 0.56 0.40 0.56 0.60 0.46 0.60\n")
        t = build_crowd(root / "c_imgs", root / "a.odgt", root / "c_lbls",
                        root / "crowd.html", 0.5, 40, 40, 200)
        assert t == {"clear_fp": 1, "dup": 1, "missed": 1, "gt": 2}, t
        html = (root / "crowd.html").read_text()
        assert "Clear false positives — 1 boxes" in html and "Missed people — 1 of 2" in html
    print("build_gallery_exp04.py self-test passed (both modes)")


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-04 error galleries from existing outputs")
    ap.add_argument("--mode", choices=["negatives", "crowd"])
    ap.add_argument("--images"); ap.add_argument("--pred-labels")
    ap.add_argument("--gt-odgt"); ap.add_argument("--out")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--max-per-group", type=int, default=100,
                    help="negatives: max tiles per prefix group")
    ap.add_argument("--max-frames", type=int, default=80,
                    help="crowd: max full-frame tiles per section")
    ap.add_argument("--max-crops", type=int, default=400,
                    help="crowd: max clear-FP crops in the tally strip")
    ap.add_argument("--thumb", type=int, default=420, help="frame thumbnail max dimension")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    if not args.mode:
        ap.error("--mode required (or --selftest)")
    for req in ("images", "pred_labels", "out"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required")
    if args.mode == "negatives":
        s = build_negatives(Path(args.images), Path(args.pred_labels), Path(args.out),
                            args.max_per_group, args.thumb)
    else:
        if not args.gt_odgt:
            ap.error("--gt-odgt required in crowd mode")
        s = build_crowd(Path(args.images), Path(args.gt_odgt), Path(args.pred_labels),
                        Path(args.out), args.match_iou, args.max_frames,
                        args.max_crops, args.thumb)
    print(json.dumps(s, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
