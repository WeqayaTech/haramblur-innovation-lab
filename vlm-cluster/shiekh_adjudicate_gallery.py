#!/usr/bin/env python3
"""
Adjudication sheet for the shiekh-collection GT-"woman" rows.

Owner ruling (2026-08-25): the shiekhs collection contains NO women — every
class-0 ground-truth row there is a labeling-pipeline error. This tool
renders each such row as a numbered card (zoomed crop + full image with the
box drawn + the Gemini verdict fields that produced the label) so a human
can classify each case: mislabeled man / poster or depiction / statue-doll /
actually a woman (would partially overturn the ruling) / unclear.

INTERNAL ADJUDICATION SHEET — cases render UNREDACTED by design, because
judging them is the point. Do not circulate as a report.

    python3 shiekh_adjudicate_gallery.py \\
        --cases /workspace/deploycmp/shiekh_gt_women_adjudicate.jsonl \\
        --images .../labeling/full/images \\
        --verdicts .../run/verdicts_batch.jsonl \\
        --out /workspace/deploycmp/SHIEKH_ADJUDICATION.html

    python3 shiekh_adjudicate_gallery.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_verdict_women(verdicts):
    """stem -> [(box, verdict-dict)] for woman-gendered real_person rows."""
    out = defaultdict(list)
    if not verdicts:
        return out
    for line in open(verdicts):
        d = json.loads(line)
        v = d.get("v") or {}
        if v.get("verdict") == "real_person" and v.get("gender") == "woman":
            out[d["image_stem"]].append((tuple(d["box"]), v))
    return out


def b64(img, q=80):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def render_case(img_path, box, crop_max=320, full_max=400):
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = box
    pad = 0.25 * max(x2 - x1, y2 - y1)
    cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    cx2, cy2 = min(img.width, int(x2 + pad)), min(img.height, int(y2 + pad))
    crop = img.crop((cx1, cy1, cx2, cy2))
    dr = ImageDraw.Draw(crop)
    dr.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
                 outline="#CC00CC", width=max(2, crop.width // 150))
    crop.thumbnail((crop_max, crop_max))
    full = img.copy()
    dr = ImageDraw.Draw(full)
    dr.rectangle([x1, y1, x2, y2], outline="#CC00CC",
                 width=max(2, full.width // 200))
    full.thumbnail((full_max, full_max))
    return b64(crop), b64(full)


def build(cases_file, images_dir, verdicts, out):
    cases = [json.loads(x) for x in
             Path(cases_file).read_text().splitlines() if x.strip()]
    vw = load_verdict_women(verdicts)
    img_index = {}
    for p in Path(images_dir).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            img_index.setdefault(p.stem, p)
    cards, missing = "", 0
    for i, c in enumerate(sorted(cases, key=lambda r: r["stem"]), 1):
        p = img_index.get(c["stem"])
        if not p:
            missing += 1
            continue
        box = c["box"]
        try:
            crop64, full64 = render_case(p, box)
        except OSError:
            missing += 1
            continue
        best, bv = 0.0, None
        for vb, v in vw.get(c["stem"], ()):
            j = iou(box, vb)
            if j > best:
                best, bv = j, v
        meta = ""
        if bv and best >= 0.5:
            parts = ",".join(bv.get("exposed_body_parts") or []) or "-"
            meta = (f"Gemini said: age≈{bv.get('estimated_age')}, "
                    f"head_covering={bv.get('head_covering')}, "
                    f"confidence={bv.get('confidence')}<br>"
                    f"exposed: {html.escape(parts)}")
        else:
            meta = "(no joinable woman verdict — relabeled row)"
        w, h = box[2] - box[0], box[3] - box[1]
        cards += f"""<div class=card><div class=num>#{i}</div>
<div class=imgs><img src='data:image/jpeg;base64,{crop64}'>
<img src='data:image/jpeg;base64,{full64}'></div>
<div class=cap><b>{html.escape(c["stem"][:70])}</b><br>
box {w:.0f}×{h:.0f}px · {meta}<br>
<span class=call>your call: ☐ mislabeled man &nbsp; ☐ poster/depiction
&nbsp; ☐ statue/doll &nbsp; ☐ actually a woman &nbsp; ☐ unclear</span>
</div></div>"""
    Path(out).write_text(f"""<!doctype html><meta charset=utf-8>
<title>Shiekh GT-woman adjudication</title><style>
body{{font-family:system-ui;background:#fcfcfb;color:#222;max-width:1000px;
margin:20px auto;padding:0 14px}}
.warn{{background:#fdecec;border:1px solid #d99;padding:10px 14px;
border-radius:6px}}
.card{{display:flex;gap:12px;border-bottom:1px solid #ddd;padding:14px 0}}
.num{{font-size:18px;font-weight:700;color:#888;min-width:44px}}
.imgs img{{max-height:280px;margin-right:8px;border:1px solid #ccc;
vertical-align:top}}
.cap{{font-size:13px;line-height:1.5}} .call{{color:#8a5a00}}</style>
<h1>Shiekh-collection GT-"woman" rows — hand adjudication</h1>
<p class=warn><b>Internal adjudication sheet — unredacted by design</b>
(judging these cases is the point). Owner ruling 2026-08-25: the shiekhs
collection contains no women, so each of these {len(cases)} ground-truth
"woman" rows is presumed to be a labeling error. Magenta box = the GT row
under review; left = zoomed crop, right = full image. The Gemini fields
show what the labeling pipeline believed when it wrote the row.</p>
{cards}
<p>{missing} case(s) could not be rendered (missing/unreadable image).</p>""")
    print(f"[adjudicate] {len(cases)} cases, {missing} unrenderable "
          f"-> {out}")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "imgs").mkdir()
        img = Image.new("RGB", (300, 200), "#889")
        img.save(td / "imgs" / "shiekhs__a.jpg")
        (td / "cases.jsonl").write_text(json.dumps(
            {"stem": "shiekhs__a", "box": [50, 40, 150, 180]}) + "\n")
        (td / "verd.jsonl").write_text(json.dumps(
            {"image_stem": "shiekhs__a", "box": [51, 41, 149, 178],
             "v": {"verdict": "real_person", "gender": "woman",
                   "estimated_age": 30, "head_covering": "none",
                   "confidence": "high",
                   "exposed_body_parts": ["face"]}}) + "\n")
        out = td / "g.html"
        build(td / "cases.jsonl", td / "imgs", td / "verd.jsonl", out)
        t = out.read_text()
        assert "unredacted by design" in t and "your call" in t
        assert "age≈30" in t, "verdict join failed"
        assert t.count("data:image/jpeg") == 2, "crop + full rendered"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--images")
    ap.add_argument("--verdicts")
    ap.add_argument("--out", default="SHIEKH_ADJUDICATION.html")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    build(args.cases, args.images, args.verdicts, args.out)


if __name__ == "__main__":
    main()
