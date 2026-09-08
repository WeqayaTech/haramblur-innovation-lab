#!/usr/bin/env python3
"""
Visual inspection gallery for one exposure tier of one gender — "what does
the data actually look like, and what did the models do with it".

For every person of the chosen gender+tier (from the Gemini verdicts),
computes the blur outcome per model at its selected threshold (covered /
partial / called_<other> / called_child / subthreshold / no_detection from
the logged sidecars), then renders sampled cases grouped by outcome: a
zoomed crop and the full image with GT (green) and the model's predictions
(Woman magenta / Man blue / Child orange) drawn.

REDACTION: every GT-woman box, every predicted-Woman box and every ignore
region is pixelated, regardless of which gender is being inspected.

    python3 tier_gallery.py --gender man --tier t3 \\
        --model y26n_warm50=0.20 --model v11n_shipped=0.20 \\
        --verdicts .../run/verdicts_batch.jsonl --images .../images \\
        --gt-labels .../labels_eval --ignore .../ignore \\
        --raw-root /workspace/holdout_eval \\
        --out /workspace/deploycmp/T3_MEN_GALLERY.html

    python3 tier_gallery.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from deploy_compare import covered_frac, seg_boxes
from subset_verify import pixelate
from women_report_final import TIER_ORDER, parallel_jsons

CLASS_COLOR = {0: "#CC00CC", 1: "#0072B2", 2: "#E69F00"}
T0 = {"face", "hands", "eyes"}
T1 = T0 | {"hair", "neck", "ears"}
T2 = T1 | {"arms", "shoulders", "feet", "knees", "wrists", "head"}
T4P = {"midriff", "stomach", "torso", "thighs"}


def tier_of(parts):
    core = {p for p in (parts or []) if p != "none"}
    if core <= T0:
        return "t0"
    if core <= T1:
        return "t1"
    if core <= T2:
        return "t2"
    if core & T4P or ("chest" in core and "legs" in core):
        return "t4"
    return "t3"


def ioa(pred, gt):
    ix = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
    iy = max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    pa = (pred[2] - pred[0]) * (pred[3] - pred[1])
    return ix * iy / pa if pa > 0 else 0.0


def outcome(g, dets, target, conf):
    wrong = 1 - target
    blur = [b for c, b, cf in dets if c == target and cf >= conf]
    cov = covered_frac(g, blur)
    if cov >= 0.9:
        return "covered"
    if cov > 0.1:
        return "partial"
    if any(c == wrong and cf >= conf and ioa(b, g) >= 0.3
           for c, b, cf in dets):
        return "called_woman" if wrong == 0 else "called_man"
    if any(c == 2 and cf >= conf and ioa(b, g) >= 0.3 for c, b, cf in dets):
        return "called_child"
    if any(c == target and cf < conf and ioa(b, g) >= 0.3
           for c, b, cf in dets):
        return "subthreshold"
    return "no_detection"


def render(img_path, g, dets, conf, redact, thumb=340):
    img = Image.open(img_path).convert("RGB")
    for b in redact:
        pixelate(img, b)
    for c, b, cf in dets:
        if c == 0 and cf >= 0.05:
            pixelate(img, b)
    x1, y1, x2, y2 = g
    pad = 0.3 * max(x2 - x1, y2 - y1)
    cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    cx2 = min(img.width, int(x2 + pad))
    cy2 = min(img.height, int(y2 + pad))
    lw = max(2, img.width // 250)
    dr = ImageDraw.Draw(img)
    dr.rectangle([x1, y1, x2, y2], outline="#007a3d", width=lw + 1)
    for c, b, cf in dets:
        if cf >= conf and ioa(b, g) >= 0.10:
            dr.rectangle(list(b), outline=CLASS_COLOR.get(c, "#666"),
                         width=lw)
    crop = img.crop((cx1, cy1, cx2, cy2))
    crop.thumbnail((thumb, thumb))
    full = img.copy()
    full.thumbnail((thumb + 60, thumb + 60))
    outs = []
    for im in (crop, full):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        outs.append(base64.b64encode(buf.getvalue()).decode())
    return outs


def build(args):
    target = 1 if args.gender == "man" else 0
    models = {}
    for spec in args.model:
        m, _, c = spec.partition("=")
        models[m] = float(c)

    people = defaultdict(list)          # stem -> [(box, parts)]
    for line in open(args.verdicts):
        d = json.loads(line)
        v = d.get("v") or {}
        if (v.get("verdict") == "real_person"
                and v.get("gender") == args.gender
                and v.get("age_group") != "child"
                and tier_of(v.get("exposed_body_parts")) == args.tier):
            people[d["image_stem"]].append(
                (tuple(d["box"]), v.get("exposed_body_parts") or []))
    n_people = sum(len(v) for v in people.values())
    print(f"[gallery] {n_people} {args.gender} in {args.tier} "
          f"across {len(people)} images")

    sidecars = {}
    for m in models:
        pairs = parallel_jsons(
            [Path(args.raw_root) / m / "raw" / f"{s}.json" for s in people])
        sidecars[m] = {
            p.stem: (d["width"], d["height"],
                     [(r["cls"], tuple(r["box_xyxy"]), r["conf"])
                      for r in d.get("detections", [])
                      if not r.get("excluded")])
            for p, d in pairs if d is not None}

    img_index = {}
    for p in Path(args.images).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            img_index.setdefault(p.stem, p)

    cases = defaultdict(list)           # (model, outcome) -> [case]
    for stem, plist in people.items():
        for m, conf in models.items():
            if stem not in sidecars[m]:
                continue
            w, h, dets = sidecars[m][stem]
            for g, parts in plist:
                o = outcome(g, dets, target, conf)
                cases[(m, o)].append({"stem": stem, "g": g, "parts": parts,
                                      "conf": conf})
    rng = random.Random(args.seed)
    order_out = ["called_woman" if target == 1 else "called_man",
                 "called_child", "partial", "subthreshold", "no_detection",
                 "covered"]
    sections = ""
    counts_html = "<table><tr><th>model</th>" + "".join(
        f"<th>{o}</th>" for o in order_out) + "</tr>"
    for m in models:
        tot = sum(len(cases[(m, o)]) for o in order_out)
        counts_html += f"<tr><td><b>{m}</b></td>" + "".join(
            f"<td>{len(cases[(m, o)])} "
            f"({100 * len(cases[(m, o)]) / max(1, tot):.1f}%)</td>"
            for o in order_out) + "</tr>"
    counts_html += "</table>"

    for m in models:
        for o in order_out:
            pool = cases[(m, o)]
            if not pool or o == "covered" and not args.show_covered:
                continue
            k = min(args.per_cell if o != "covered" else 6, len(pool))
            cards = ""
            for case in rng.sample(pool, k):
                p = img_index.get(case["stem"])
                if not p:
                    continue
                w, h, dets = sidecars[m][case["stem"]]
                gt = seg_boxes(Path(args.gt_labels) /
                               f"{case['stem']}.txt", w, h) or []
                redact = [(x1, y1, x2, y2)
                          for c, x1, y1, x2, y2 in gt if c == 0]
                ign = seg_boxes(Path(args.ignore) /
                                f"{case['stem']}.txt", w, h) if args.ignore \
                    else None
                redact += [(x1, y1, x2, y2)
                           for _c, x1, y1, x2, y2 in ign or []]
                try:
                    crop64, full64 = render(p, case["g"], dets,
                                            case["conf"], redact)
                except OSError:
                    continue
                parts = ",".join(case["parts"])
                cards += (f"<div class=card>"
                          f"<img src='data:image/jpeg;base64,{crop64}'>"
                          f"<img src='data:image/jpeg;base64,{full64}'>"
                          f"<div class=cap>{html.escape(case['stem'][:58])}"
                          f"<br>exposed: {html.escape(parts)}</div></div>")
            if cards:
                sections += (f"<h3>{m} — {o} "
                             f"({len(pool)} cases, {k} sampled)</h3>"
                             f"<div class=row>{cards}</div>")

    Path(args.out).write_text(f"""<!doctype html><meta charset=utf-8>
<title>{args.tier} {args.gender} — data inspection</title><style>
body{{font-family:system-ui;background:#fcfcfb;color:#222;max-width:1150px;
margin:20px auto;padding:0 14px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid
#ccc;padding:3px 8px;text-align:right}} td:first-child{{text-align:left}}
.row{{display:flex;flex-wrap:wrap;gap:10px}} .card{{width:100%;max-width:
760px;font-size:11.5px;border-bottom:1px solid #eee;padding:6px 0}}
.card img{{max-height:250px;margin-right:8px;border:1px solid #ddd;
vertical-align:top}} .cap{{color:#555}}
.legend span{{padding:0 8px}}</style>
<h1>What {args.tier} {args.gender} look like, and what the models did</h1>
<p>{n_people} {args.gender} in tier {args.tier}. Green box = the person
under inspection (from the answer key); prediction boxes at the model's
selected threshold: <span style="color:#CC00CC">■ Woman</span>
<span style="color:#0072B2">■ Man</span>
<span style="color:#E69F00">■ Child</span>. All women (labeled or
predicted) and unknown-gender people are pixelated.</p>
<h2>Outcome counts per model</h2>{counts_html}
<h2>Sampled cases by outcome</h2>{sections}""")
    print(f"[gallery] wrote {args.out}")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "imgs").mkdir()
        (td / "labels").mkdir()
        (td / "m1" / "raw").mkdir(parents=True)
        Image.new("RGB", (200, 160), "#7a8a7a").save(td / "imgs" / "men__a.jpg")
        (td / "labels" / "men__a.txt").write_text("1 0.5 0.5 0.5 0.8\n")
        (td / "verd.jsonl").write_text(json.dumps(
            {"image_stem": "men__a", "box": [50, 16, 150, 144],
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult",
                   "exposed_body_parts": ["face", "chest"]}}) + "\n")
        (td / "m1" / "raw" / "men__a.json").write_text(json.dumps(
            {"image": "men__a.jpg", "width": 200, "height": 160,
             "detections": [{"det_index": 0, "cls": 0,
                             "box_xyxy": [50, 16, 150, 144], "conf": 0.9,
                             "kept": True, "excluded": None}]}))
        assert tier_of(["face", "chest"]) == "t3"
        out = td / "g.html"
        build(argparse.Namespace(
            gender="man", tier="t3", model=["m1=0.20"],
            verdicts=td / "verd.jsonl", images=td / "imgs",
            gt_labels=td / "labels", ignore=None,
            raw_root=td, out=out, per_cell=4, seed=1, show_covered=False))
        t = out.read_text()
        assert "called_woman" in t and "1 (100.0%)" in t, \
            "man with woman-pred must classify as called_woman"
        assert t.count("data:image/jpeg") == 2
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", choices=["man", "woman"], default="man")
    ap.add_argument("--tier", choices=TIER_ORDER + ["t3"], default="t3")
    ap.add_argument("--model", action="append", help="name=conf")
    ap.add_argument("--verdicts")
    ap.add_argument("--images")
    ap.add_argument("--gt-labels")
    ap.add_argument("--ignore")
    ap.add_argument("--raw-root")
    ap.add_argument("--out", default="TIER_GALLERY.html")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--show-covered", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    build(args)


if __name__ == "__main__":
    main()
