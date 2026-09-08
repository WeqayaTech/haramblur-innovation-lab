#!/usr/bin/env python3
"""
Visual proof gallery for the haramblur_holdout benchmark (2026-08-19):
4-panel rows per sampled image -- original / ground truth / finetuned model /
production model -- base64-embedded thumbnails, one self-contained HTML file.
Reuses seg_boxes (run_autolabel_on_manifest.py) and draw_boxes
(eval_negatives_crowd.py) rather than re-implementing GT parsing / drawing.

    python3 build_holdout_gallery.py --out gallery.html

Samples per collection (seeded, reproducible) + a Gulf-dress "bias spotlight"
(shiekhs images where either model calls someone Woman at conf>=0.45) + a
randoms false-positive spotlight (explicitly marked unadjudicated).

`women` / `women_hd` panels are rendered from a heavily Gaussian-blurred copy
of the source image -- ALL FOUR panels (including "original") use the blurred
base, so no unblurred photo of a real woman appears anywhere in the output.
"""
import base64
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_autolabel_on_manifest import seg_boxes  # noqa: E402
from eval_negatives_crowd import draw_boxes      # noqa: E402

H = Path("/workspace/datasets/haramblur_holdout/labeling/full")
S = Path("/workspace/holdout_eval/slices")
MODELS = {
    "warm50": ("y26n_noe2e_warm50-2 (finetuned)", "#3b82f6",
               Path("/workspace/holdout_eval/y26n_warm50/raw")),
    "prod":   ("yolo11N-640 (production)", "#ef4444",
               Path("/workspace/holdout_eval/v11n_shipped/raw")),
}
NAMES = {0: "Woman", 1: "Man", 2: "Child"}
CONF = 0.45
OUT_IMG_DIR = Path("/workspace/holdout_eval/gallery_imgs")
BLUR_COLLECTIONS = {"women", "women_hd"}
PANEL_DIM = 340


def load_raw_dets(raw_dir: Path, stem: str, conf=CONF):
    f = raw_dir / f"{stem}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    out = []
    for det in d["detections"]:
        if det["conf"] < conf:
            continue
        x1, y1, x2, y2 = det["box_xyxy"]
        out.append((det["cls"], x1, y1, x2, y2))
    return out


def b64_img(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def prep_base_image(img_path: Path, blur: bool, tmp_path: Path):
    """Returns the path draw_boxes should read from: the original file, or a
    saved heavily-blurred copy (used for ALL panels, incl. "original")."""
    if not blur:
        return img_path
    from PIL import Image, ImageFilter
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, 900 / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        im = im.filter(ImageFilter.GaussianBlur(radius=28))
        im.save(tmp_path, quality=80)
    return tmp_path


def render_panels(stem: str, img_path: Path, gt_label: Path, collection: str, out_dir: Path):
    from PIL import Image
    with Image.open(img_path) as im:
        w, h = im.size
    gt = seg_boxes(gt_label, w, h) or []
    model_dets = {}
    for key, (_, _, raw_dir) in MODELS.items():
        model_dets[key] = load_raw_dets(raw_dir, stem)

    blur = collection in BLUR_COLLECTIONS
    tmp_path = out_dir / f"{stem}__base.jpg"
    base = prep_base_image(img_path, blur, tmp_path)

    panels = {}
    panels["orig"] = out_dir / f"{stem}__orig.jpg"
    draw_boxes(base, [], panels["orig"], max_dim=PANEL_DIM, quality=72)

    gt_boxes = [((x1, y1, x2, y2), NAMES.get(c, str(c)), (34, 197, 94)) for c, x1, y1, x2, y2 in gt]
    panels["gt"] = out_dir / f"{stem}__gt.jpg"
    draw_boxes(base, gt_boxes, panels["gt"], max_dim=PANEL_DIM, quality=72)

    for key, (_, color_hex, _) in MODELS.items():
        rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
        boxes = [((x1, y1, x2, y2), NAMES.get(c, str(c)), rgb) for c, x1, y1, x2, y2 in model_dets[key]]
        panels[key] = out_dir / f"{stem}__{key}.jpg"
        draw_boxes(base, boxes, panels[key], max_dim=PANEL_DIM, quality=72)

    if blur and tmp_path.exists():
        tmp_path.unlink()
    return gt, model_dets, panels


def pick_samples(collection: str, n: int, seed: int, filter_fn=None):
    img_dir = S / collection / "images"
    lbl_dir = S / collection / "labels"
    stems = sorted(p.stem for p in img_dir.iterdir())
    if filter_fn:
        stems = [s for s in stems if filter_fn(s)]
    random.Random(seed).shuffle(stems)
    return stems[:n], img_dir, lbl_dir


def build_section(title: str, collection: str, stems, img_dir, lbl_dir, note=""):
    rows = []
    blurred = collection in BLUR_COLLECTIONS
    for stem in stems:
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            cands = list(img_dir.glob(f"{stem}.*"))
            if not cands:
                continue
            img_path = cands[0]
        gt_label = lbl_dir / f"{stem}.txt"
        try:
            gt, model_dets, panels = render_panels(stem, img_path, gt_label, collection, OUT_IMG_DIR)
        except Exception as e:
            print(f"  skip {stem}: {e}")
            continue
        gt_summary = ", ".join(NAMES.get(c, str(c)) for c, *_ in gt) or "none"
        warm_s = ", ".join(NAMES.get(c, str(c)) for c, *_ in model_dets.get("warm50", [])) or "none"
        prod_s = ", ".join(NAMES.get(c, str(c)) for c, *_ in model_dets.get("prod", [])) or "none"

        def cell(key, label, caption):
            return f"""<div class="cell">
              <img src="data:image/jpeg;base64,{b64_img(panels[key])}" loading="lazy">
              <p class="plabel">{label}</p></div>"""

        rows.append(f"""
        <div class="row">
          <p class="rowid">{stem}{' &middot; <span class="blurtag">blurred</span>' if blurred else ''}</p>
          <div class="quad">
            {cell('orig', 'original', '')}
            {cell('gt', f'ground truth &middot; {gt_summary}', '')}
            {cell('warm50', f'finetuned &middot; {warm_s}', '')}
            {cell('prod', f'production &middot; {prod_s}', '')}
          </div>
        </div>""")
    return f"""
    <section>
      <h2>{title} <span class="n">({len(rows)} images)</span></h2>
      {f'<p class="note">{note}</p>' if note else ''}
      {''.join(rows)}
    </section>"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/holdout_eval/gallery.html")
    ap.add_argument("--n-per-collection", type=int, default=3)
    ap.add_argument("--n-shiekhs-spotlight", type=int, default=5)
    ap.add_argument("--n-randoms-fp", type=int, default=4)
    args = ap.parse_args()

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    sections = []

    warm_raw = MODELS["warm50"][2]
    prod_raw = MODELS["prod"][2]

    def shiekhs_has_woman(stem):
        for raw_dir in (warm_raw, prod_raw):
            for cls, *_ in load_raw_dets(raw_dir, stem):
                if cls == 0:
                    return True
        return False

    stems, img_dir, lbl_dir = pick_samples("shiekhs", args.n_shiekhs_spotlight,
                                            seed=1, filter_fn=shiekhs_has_woman)
    print(f"shiekhs Woman-spotlight candidates: {len(stems)}")
    sections.append(build_section(
        "Gulf-dress bias spotlight (shiekhs)", "shiekhs", stems, img_dir, lbl_dir,
        note="Images where at least one model predicted \"Woman\" at conf&ge;0.45 "
             "in the Gulf/traditional-dress slice — the historically biased case."))

    for c in ("child", "men", "women", "women_hd", "shiekhs"):
        stems, img_dir, lbl_dir = pick_samples(c, args.n_per_collection, seed=42)
        note = ("Faces/bodies heavily blurred in every panel below, including "
                "\"original\" — no unblurred photo of a real person in this section."
                if c in BLUR_COLLECTIONS else "")
        sections.append(build_section(f"{c} — general sample", c, stems, img_dir, lbl_dir, note))

    def randoms_has_any_det(stem):
        return bool(load_raw_dets(warm_raw, stem) or load_raw_dets(prod_raw, stem))

    stems, img_dir, lbl_dir = pick_samples("randoms", args.n_randoms_fp, seed=7,
                                            filter_fn=randoms_has_any_det)
    sections.append(build_section(
        "randoms — detections on person-free images (UNADJUDICATED)",
        "randoms", stems, img_dir, lbl_dir,
        note="These images are labeled person-free, but the ground truth "
             "itself includes 254 unadjudicated gate-survivors project-wide — a box "
             "here may be a real false positive OR a real person the labeler correctly "
             "found. Do not read this section as a confirmed FP rate."))

    html = f"""<!doctype html><meta charset="utf-8">
<title>haramblur_holdout visual proof gallery</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background:#0b0f14; color:#e5e7eb; margin:0; padding:24px; }}
  h1 {{ font-size:1.4rem; }} h2 {{ font-size:1.1rem; margin-top:2rem; border-bottom:1px solid #333; padding-bottom:6px; }}
  .n {{ color:#888; font-weight:normal; font-size:0.9rem; }}
  .note {{ color:#fbbf24; font-size:0.9rem; }}
  .row {{ margin-bottom: 18px; }}
  .rowid {{ font-size:0.8rem; color:#9aa3b2; margin: 0 0 6px; }}
  .blurtag {{ color:#f0b429; }}
  .quad {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }}
  .cell {{ background:#161b22; border-radius:6px; overflow:hidden; border:1px solid #262d38; }}
  .cell img {{ width:100%; display:block; }}
  .plabel {{ font-size:0.7rem; padding:5px 7px; margin:0; color:#c8ccd4; }}
</style>
<h1>haramblur_holdout visual proof gallery</h1>
<p>Each row: original image, ground truth (Gemini 3.5 Flash-Lite via Spotlight),
<code>y26n_noe2e_warm50-2</code> (finetuned) at conf&ge;0.45,
<code>yolo11N-640</code> (production) at conf&ge;0.45.
Full numeric results: <code>docs/MODEL_COMPARISON.md</code> → "Holdout benchmark".</p>
{''.join(sections)}
"""
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
