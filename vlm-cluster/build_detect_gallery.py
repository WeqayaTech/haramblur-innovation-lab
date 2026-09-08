#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-06: report-style HTML gallery of every model's
whole-image detections, with a model dropdown filter.

Per image, TWO panes side by side:
    LEFT  — ground truth: green boxes, captioned "F/A/41" =
            (Gender M/F, Adult-vs-Child at the <=12 cutoff, Age) where the
            dataset has person labels (LAGENDA); CrowdHuman GT is boxes-only;
            PASS/objects have no GT boxes.
    RIGHT — prediction: gender-colored boxes captioned "M/A/34/H" =
            (Gender M/F/?, Adult-vs-Child A/C/?, estimated age, Confidence H/L).

A dropdown at the top filters to one model (or all). Base64-embedded JPEGs
(same style as build_error_gallery.py) so the single HTML file works anywhere
— scp it off the pod and open locally.

    python3 build_detect_gallery.py --out /workspace/exp06/gallery.html
    python3 build_detect_gallery.py --selftest      # no data needed
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2

# dataset name (dir under each engine) -> images dir + GT source, pod defaults
DATASETS = {
    "lagenda": {"images": "/workspace/lagenda_eval/lagenda_yolo/images/val",
                "gt": "yolo",
                "labels": "/workspace/lagenda_eval/lagenda_yolo/labels/val",
                "manifest": "/workspace/lagenda_eval/lagenda_yolo/gt.jsonl"},
    "crowd":   {"images": "/workspace/datasets/crowdhuman/Images_sample500",
                "gt": "odgt",
                "odgt": "/workspace/datasets/crowdhuman/annotation_val.odgt"},
    "pass":    {"images": "/workspace/datasets/pass_3k", "gt": None},
    "objects": {"images": "/workspace/datasets/object_set", "gt": None},
}

GENDER_COLORS = {"man": (255, 140, 0),      # BGR: blue-ish
                 "woman": (200, 0, 200),    # magenta
                 "unknown": (140, 140, 140)}
GT_COLOR = (0, 170, 0)
PANE_W = 380
JPEG_Q = 78
CHILD_CUTOFF = 12


def find_image(images_dir: Path, name: str, cache: dict) -> Path | None:
    if not cache:
        for p in images_dir.rglob("*"):
            if p.is_file():
                cache[p.name] = p
    return cache.get(name)


def _load_lagenda_manifest(ds_cfg: dict) -> dict:
    """{(stem, box_idx): ("F/A/41", "woman"|"man")} from gt.jsonl."""
    cache = ds_cfg.get("_manifest_cache")
    if cache is None:
        cache = {}
        mf = ds_cfg.get("manifest")
        if mf and Path(mf).exists():
            with open(mf) as fh:
                for line in fh:
                    rec = json.loads(line)
                    stem, _, idx = rec["id"].rpartition("_")
                    ac = "C" if rec["gt_age"] <= CHILD_CUTOFF else "A"
                    gender = "woman" if rec["gt_gender"] == "F" else "man"
                    cache[(stem, int(idx))] = (
                        f"{rec['gt_gender']}/{ac}/{rec['gt_age']}", gender)
        ds_cfg["_manifest_cache"] = cache
    return cache


def gt_entries_for(ds_cfg: dict, image_name: str, w: int, h: int) -> list:
    """[(box_xyxy, caption_or_None), ...] for the image's ground truth."""
    stem = Path(image_name).stem
    if ds_cfg.get("gt") == "yolo":
        lf = Path(ds_cfg["labels"]) / f"{stem}.txt"
        if not lf.exists():
            return []
        meta = _load_lagenda_manifest(ds_cfg)
        out = []
        for i, line in enumerate(lf.read_text().splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            cap, gender = meta.get((stem, i), (None, None))
            out.append(([(cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h], cap, gender))
        return out
    if ds_cfg.get("gt") == "odgt":
        odgt = ds_cfg.setdefault("_odgt_cache", {})
        if not odgt:
            with open(ds_cfg["odgt"]) as fh:
                for line in fh:
                    rec = json.loads(line)
                    odgt[rec["ID"]] = [
                        [g["vbox"][0], g["vbox"][1],
                         g["vbox"][0] + g["vbox"][2], g["vbox"][1] + g["vbox"][3]]
                        for g in rec.get("gtboxes", []) if g.get("tag") == "person"
                        and not g.get("extra", {}).get("ignore", 0)]
        return [(b, None, None) for b in odgt.get(stem, [])]
    return []


def draw_label(img, text: str, x: int, y: int, color, scale=0.5):
    """Readable label: filled color chip + white text, kept inside the frame."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    x = max(0, min(x, img.shape[1] - tw - 6))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base), color, -1)
    cv2.putText(img, text, (x + 3, y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_gt_pane(img, gts):
    ann = img.copy()
    for box, caption, gender in gts:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = GENDER_COLORS.get(gender, GT_COLOR)   # green when GT has no gender
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        if caption:
            draw_label(ann, caption, x1, y1 - 2, color)
    return ann


def _letter(v: str, table: dict) -> str:
    return table.get(v, "?")


def draw_pred_pane(img, people):
    ann = img.copy()
    for p in people:
        x1, y1, x2, y2 = [int(v) for v in p["box_xyxy"]]
        color = GENDER_COLORS.get(p["gender"], GENDER_COLORS["unknown"])
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        g = _letter(p["gender"], {"man": "M", "woman": "F"})
        ac = _letter(p["age_group"], {"adult": "A", "child": "C"})
        age = p.get("estimated_age")
        age_s = str(int(age)) if isinstance(age, (int, float)) else "?"
        conf = _letter(p.get("confidence", ""), {"high": "H", "low": "L"})
        draw_label(ann, f"{g}/{ac}/{age_s}/{conf}", x1, y1 - 2, color)
    return ann


def to_b64_jpeg(img) -> str:
    h, w = img.shape[:2]
    if w > PANE_W:
        s = PANE_W / w
        img = cv2.resize(img, (PANE_W, max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


CSS = """
body{font-family:sans-serif;background:#fff;color:#222;margin:20px}
h1{font-size:20px} h2{font-size:17px;margin:26px 0 4px;border-bottom:2px solid #444}
h3{font-size:14px;margin:14px 0 6px;color:#555}
.card{border:1px solid #ccc;border-radius:6px;padding:8px;margin:0 0 12px 0;
  max-width:820px;overflow-x:auto}
.panes{display:flex;gap:8px;flex-wrap:nowrap}
.pane{flex:0 0 auto;max-width:380px}
.pane img{max-width:380px;display:block;border-radius:3px}
.pane .ph{font-size:11px;font-weight:bold;color:#444;margin:0 0 3px 1px}
.cap{font-size:11.5px;color:#333;margin-top:5px;white-space:pre-line}
.legend{font-size:13px;margin:8px 0 14px;line-height:1.5}
.sw{display:inline-block;width:12px;height:12px;margin:0 4px -1px 10px;border-radius:2px}
.warn{color:#b00}
.picker{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #ddd;
  z-index:5;font-size:14px}
.picker select{font-size:14px;padding:4px 8px;margin-left:8px}
"""

LEGEND = (
    '<div class="legend">'
    'Caption format on every box: <code>Gender(M/F) / Adult-vs-Child(A/C) / Age / '
    'Confidence(H/L, predictions only)</code>. Both panes use the same colors: '
    '<span class="sw" style="background:#008cff"></span>man '
    '<span class="sw" style="background:#c800c8"></span>woman '
    '<span class="sw" style="background:#8c8c8c"></span>unknown '
    '<span class="sw" style="background:#00aa00"></span>GT without gender labels '
    '(CrowdHuman is boxes-only)'
    '</div>')

FILTER_JS = """
<script>
function filterModel(v){
  document.querySelectorAll('.engine').forEach(function(e){
    e.style.display = (v === 'all' || e.dataset.engine === v) ? '' : 'none';
  });
}
</script>
"""


def build(exp_root: Path, out_path: Path, dataset_cfg: dict, only=None):
    engines = sorted(d for d in exp_root.iterdir() if d.is_dir())
    if only:
        engines = [e for e in engines if e.name in only]
    options = "".join(f"<option value='{html.escape(e.name)}'>"
                      f"{html.escape(e.name)}</option>" for e in engines)
    parts = [f"<style>{CSS}</style>", FILTER_JS,
             "<h1>EXP-2026-06 — VLM detections vs ground truth (smoke test)</h1>",
             LEGEND,
             "<div class='picker'><b>Model:</b>"
             f"<select onchange='filterModel(this.value)'>"
             f"<option value='all'>All models</option>{options}</select></div>"]
    n_cards = 0
    for eng_dir in engines:
        parts.append(f"<div class='engine' data-engine='{html.escape(eng_dir.name)}'>")
        parts.append(f"<h2>{html.escape(eng_dir.name)}</h2>")
        for ds_name, ds_cfg in dataset_cfg.items():
            dj = eng_dir / ds_name / "detections.jsonl"
            if not dj.exists():
                continue
            parts.append(f"<h3>{html.escape(ds_name)}</h3>")
            img_cache: dict = {}
            for line in dj.open():
                rec = json.loads(line)
                p = find_image(Path(ds_cfg["images"]), rec["image"], img_cache)
                cap = (f"{rec['image']} — {rec['n_people']} predicted"
                       f"{', PARSE FAIL' if not rec['parse_ok'] else ''}"
                       f"{', API ERROR' if rec.get('api_error') else ''}"
                       f"{', boxes reparsed as ' + rec['box_format'] if rec.get('box_format') and rec['box_format'] != 'xyxy_px' else ''}")
                if p is None:
                    parts.append(f"<div class='card'><div class='cap warn'>"
                                 f"{html.escape(cap)} — image not found under "
                                 f"{html.escape(ds_cfg['images'])}</div></div>")
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                gts = gt_entries_for(ds_cfg, rec["image"], rec["width"], rec["height"])
                gt_b64 = to_b64_jpeg(draw_gt_pane(img, gts))
                pred_b64 = to_b64_jpeg(draw_pred_pane(img, rec["people"]))
                gt_head = (f"Ground truth — {len(gts)} boxes" if gts
                           else "Ground truth — no persons")
                parts.append(
                    "<div class='card'><div class='panes'>"
                    f"<div class='pane'><div class='ph'>{gt_head}</div>"
                    f"<img src='data:image/jpeg;base64,{gt_b64}'></div>"
                    f"<div class='pane'><div class='ph'>Prediction — "
                    f"{rec['n_people']} people</div>"
                    f"<img src='data:image/jpeg;base64,{pred_b64}'></div>"
                    f"</div><div class='cap'>{html.escape(cap)}</div></div>")
                n_cards += 1
        parts.append("</div>")
    out_path.write_text("\n".join(parts))
    print(f"[gallery] {n_cards} image pairs -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-06 detection gallery")
    ap.add_argument("--exp-root", default="/workspace/exp06")
    ap.add_argument("--out", default="/workspace/exp06/gallery.html")
    ap.add_argument("--engines", nargs="+", default=None,
                    help="only include these engine dirs (default: all)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    build(Path(args.exp_root), Path(args.out), DATASETS, only=args.engines)


def selftest():
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_dir = td / "imgs"
        lbl_dir = td / "lbls"
        img_dir.mkdir()
        lbl_dir.mkdir()
        cv2.imwrite(str(img_dir / "a.jpg"),
                    np.full((200, 300, 3), 200, dtype="uint8"))
        (lbl_dir / "a.txt").write_text("0 0.4 0.5 0.4 0.6\n")
        (td / "gt.jsonl").write_text(json.dumps(
            {"id": "a_0", "image": "a.jpg", "gt_age": 8, "gt_gender": "F"}) + "\n")
        for eng in ("modelx", "modely"):
            run_dir = td / eng / "fake"
            run_dir.mkdir(parents=True)
            (run_dir / "detections.jsonl").write_text(json.dumps({
                "image": "a.jpg", "width": 300, "height": 200, "n_people": 1,
                "parse_ok": True, "rescaled": False, "api_error": False,
                "people": [{"box_xyxy": [50, 40, 150, 180], "gender": "woman",
                            "age_group": "child", "estimated_age": 9,
                            "confidence": "high"}]}) + "\n")
        out = td / "g.html"
        cfg = {"fake": {"images": str(img_dir), "gt": "yolo",
                        "labels": str(lbl_dir), "manifest": str(td / "gt.jsonl")}}
        build(td, out, cfg)
        s = out.read_text()
        assert s.count("data:image/jpeg;base64,") == 4, "2 engines x 2 panes"
        assert "filterModel" in s and "<option value='modelx'>" in s
        assert "data-engine='modely'" in s
        # GT child at age 8 -> caption F/C/8 rendered into the image (can't
        # assert pixels; assert the manifest formatting path directly)
        cfg2 = {"manifest": str(td / "gt.jsonl")}
        m = _load_lagenda_manifest(cfg2)
        assert m[("a", 0)] == ("F/C/8", "woman"), m
    print("[build_detect_gallery] selftest OK")


if __name__ == "__main__":
    main()
