#!/usr/bin/env python3
"""HTML trace for the CrowdHuman GT-box classification run.

Per sampled image, shows exactly what the pipeline saw and said:
  Stage 1  full frame with every GT vbox drawn, colored by Gemini's verdict
  Stage 2  per person: the EXACT crop sent (rebuilt with the same
           build_crop box-outline fallback), the parsed verdict fields,
           and the raw response text verbatim
No API calls -- it audits the run that happened.

    python3 crowd_trace.py --gt .../gt_boxes.jsonl --images .../Images \
        --verdicts .../verdicts_train.jsonl --n 20 --seed 5 \
        --out /workspace/spotlight/crowd_1k/trace_pilot.html

    python3 crowd_trace.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import html as H
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from spotlight_run import build_crop

COLOR = {"man": (40, 90, 220), "woman": (200, 40, 160),
         "child": (0, 170, 200), "unknown": (130, 130, 130)}
RED = (220, 30, 30)


def b64(img: Image.Image, max_dim=760, q=74):
    im = img.convert("RGB")
    if max(im.size) > max_dim:
        s = max_dim / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def box_color(v):
    if not v:
        return RED
    if v.get("verdict") == "not_person":
        return RED
    if v.get("age_group") == "child":
        return COLOR["child"]
    return COLOR.get(v.get("gender", "unknown"), COLOR["unknown"])


def load_parts(masks_dir, stem, box_index):
    if not masks_dir:
        return []
    f = Path(masks_dir) / f"{stem}.json"
    if not f.exists():
        return []
    for p in json.loads(f.read_text()).get("persons", []):
        if p["box_index"] == box_index:
            return [[(x, y) for x, y in part] for part in p.get("parts", [])]
    return []


def build(gt_path: Path, images: Path, verdicts_path: Path, n, seed, out: Path,
          masks_dir=None):
    verdicts = {}
    for line in verdicts_path.read_text().splitlines():
        r = json.loads(line)
        verdicts.setdefault(r["stem"], {})[r["box_index"]] = r
    gt = {}
    for line in gt_path.read_text().splitlines():
        r = json.loads(line)
        gt[r["stem"]] = r["boxes"]
    avail = sorted(s for s in verdicts if s in gt and (images / f"{s}.jpg").exists())
    if not avail:
        raise SystemExit("no traced images available")
    rng = random.Random(seed)
    picked = sorted(rng.sample(avail, min(n, len(avail))))
    print(f"[trace] {len(picked)} of {len(avail)} verdict-bearing images (seed {seed})")

    parts = ["""<meta charset='utf-8'><title>Crowd GT-box classification trace</title>
<style>body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;
color:#222;margin:24px;max-width:1280px}
h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:44px}
.stage{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;
border-top:1px solid #eee;padding:10px 0}
.crop img{border:1px solid #bbb;max-height:300px}
table{border-collapse:collapse;font-size:13px}
td{border:1px solid #ddd;padding:2px 8px}
pre{background:#f4f4f4;font-size:11px;padding:6px;max-width:520px;
white-space:pre-wrap;word-break:break-all}
.legend span{display:inline-block;padding:2px 10px;margin-right:8px;color:#fff;
border-radius:4px;font-size:13px}</style>
<h1>Crowd GT-box classification &mdash; stage trace</h1>
<p class='legend'>
<span style='background:rgb(40,90,220)'>man</span>
<span style='background:rgb(200,40,160)'>woman</span>
<span style='background:rgb(0,170,200)'>child</span>
<span style='background:rgb(130,130,130)'>unknown</span>
<span style='background:rgb(220,30,30)'>not_person / unparsed</span></p>"""]

    for stem in picked:
        img = Image.open(images / f"{stem}.jpg").convert("RGB")
        vd = verdicts[stem]
        frame = img.copy()
        d = ImageDraw.Draw(frame)
        for i, bx in enumerate(gt[stem]):
            x, y, w, h = bx["vbox"]
            r = vd.get(i)
            d.rectangle([x, y, x + w, y + h],
                        outline=box_color((r or {}).get("v")), width=3)
        parts.append(f"<h2>{H.escape(stem)} &mdash; {len(gt[stem])} GT persons, "
                     f"{len(vd)} verdicts</h2>")
        parts.append(f"<div class='stage'><b>Stage 1 &mdash; frame with GT boxes "
                     f"(color = Gemini verdict)</b><br>"
                     f"<img src='data:image/jpeg;base64,{b64(frame)}'></div>")
        parts.append("<div class='stage'><b>Stage 2 &mdash; exact crops and "
                     "verdicts</b>")
        for i, bx in enumerate(gt[stem]):
            r = vd.get(i)
            x, y, w, h = bx["vbox"]
            crop, _ = build_crop(img, [x, y, x + w, y + h],
                                 load_parts(masks_dir, stem, i))
            v = (r or {}).get("v") or {}
            rows = "".join(f"<tr><td>{H.escape(str(k))}</td>"
                           f"<td>{H.escape(str(v[k]))}</td></tr>" for k in v)
            raw = H.escape((r or {}).get("raw_text", "-- NO VERDICT --"))
            parts.append(
                f"<div class='row'><div class='crop'>"
                f"<img src='data:image/jpeg;base64,{b64(crop, 380)}'><br>"
                f"det {i} &middot; vbox {[int(t) for t in bx['vbox']]}</div>"
                f"<table>{rows or '<tr><td>no parsed fields</td></tr>'}</table>"
                f"<pre>{raw}</pre></div>")
        parts.append("</div>")

    out.write_text("\n".join(parts))
    print(f"[trace] -> {out}  ({out.stat().st_size/1e6:.1f} MB)")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "Images").mkdir()
        img = Image.new("RGB", (500, 400), (100, 110, 100))
        ImageDraw.Draw(img).rectangle([120, 60, 220, 340], fill=(210, 190, 170))
        img.save(td / "Images" / "x,1.jpg")
        (td / "gt.jsonl").write_text(json.dumps(
            {"stem": "x,1", "boxes": [{"vbox": [120, 60, 100, 280]}]}) + "\n")
        (td / "v.jsonl").write_text(json.dumps(
            {"stem": "x,1", "box_index": 0, "parse_ok": True,
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult"},
             "raw_text": '{"verdict": "real_person"}'}) + "\n")
        build(td / "gt.jsonl", td / "Images", td / "v.jsonl", 5, 1,
              td / "t.html")
        html = (td / "t.html").read_text()
        assert "Stage 1" in html and "Stage 2" in html and "real_person" in html
    print("crowd_trace.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt"); ap.add_argument("--images"); ap.add_argument("--verdicts")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="crowd_trace.html")
    ap.add_argument("--masks", default=None,
                    help="crowd_sam_masks.py output dir (mask-outline crops)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.gt and a.images and a.verdicts):
        ap.error("--gt --images --verdicts are required")
    build(Path(a.gt), Path(a.images), Path(a.verdicts), a.n, a.seed,
          Path(a.out), a.masks)


if __name__ == "__main__":
    main()
