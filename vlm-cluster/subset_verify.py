#!/usr/bin/env python3
"""
Verify the materialized deploy_compare subset folders — hard counts + a visual
gallery a human can eyeball.

Two halves, same as verify_eval_ready.py / verify_report.py for the holdout:

  1. HARD CHECKS (printed + embedded in the HTML header): per subset folder —
     image-link and label-link counts match the slices.json census; image and
     label stems pair up; symlinks resolve to real files; sampled labels parse
     through seg_boxes (segment polygons, the 5-field-reader trap).
  2. GALLERY: N seeded-sample images per subset with GT boxes drawn
     (Woman magenta, Man blue, Child orange). For exposure slices the caption
     shows the Gemini `exposed_body_parts` that put the image in the bucket,
     so bucket assignment is verifiable at a glance.

GT Woman regions are PIXELATED by default (the EXP-2026-19 report convention);
--show-women opts out. Redaction happens before boxes are drawn, so the box
and caption stay readable.

    python3 subset_verify.py \\
        --subsets /workspace/deploycmp/subsets \\
        --slices /workspace/deploycmp/holdout_slices.json \\
        --verdicts .../run/verdicts_batch.jsonl \\
        --out /workspace/deploycmp/subsets_VERIFY.html

    python3 subset_verify.py --selftest      # no data, no GPU, no network

This certifies the SUBSET FOLDERS (membership, integrity, bucket assignment).
It does not certify label accuracy — the labels are machine-generated and
inherit the Spotlight pipeline's errors; see EVAL_VERIFICATION.html for that
evidence chain.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from run_autolabel_on_manifest import seg_boxes
except ImportError:                     # cv2-less dev machine (same fallback
    from deploy_compare import seg_boxes  # as deploy_compare.py uses)

CLASS_COLOR = {0: "#CC00CC", 1: "#0072B2", 2: "#E69F00"}
CLASS_NAME = {0: "Woman", 1: "Man", 2: "Child"}
THUMB = 380


def pixelate(img, box, factor=14):
    """Redact a region by heavy down/up-sampling (nearest)."""
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    crop = img.crop((x1, y1, x2, y2))
    small = crop.resize((max(1, crop.width // factor),
                         max(1, crop.height // factor)))
    img.paste(small.resize(crop.size, Image.NEAREST), (x1, y1))


def render_thumb(img_path, label_path, show_women=False):
    img = Image.open(img_path).convert("RGB")
    boxes = seg_boxes(label_path, img.width, img.height) or []
    if not show_women:
        for cls, x1, y1, x2, y2 in boxes:
            if cls == 0:
                pixelate(img, (x1, y1, x2, y2))
    dr = ImageDraw.Draw(img)
    lw = max(2, img.width // 250)
    for cls, x1, y1, x2, y2 in boxes:
        dr.rectangle([x1, y1, x2, y2],
                     outline=CLASS_COLOR.get(cls, "#666666"), width=lw)
    img.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, len(boxes)


def load_verdict_captions(path):
    """stem -> ['woman: chest,shoulders', ...] for adult persons."""
    caps = defaultdict(list)
    if not path:
        return caps
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("v") or {}
            if (v.get("verdict") == "real_person"
                    and v.get("age_group") != "child"
                    and v.get("gender") in ("woman", "man")):
                parts = ",".join(v.get("exposed_body_parts") or []) or "none"
                caps[d["image_stem"]].append(f"{v['gender']}: {parts}")
    return caps


def verify(args):
    sl = json.loads(Path(args.slices).read_text())
    census = sl["_meta"]["census"]
    subsets = Path(args.subsets)
    caps = load_verdict_captions(args.verdicts)
    rng = random.Random(args.seed)
    checks, sections = [], []

    for tag in sorted(census):
        d = subsets / tag.replace(":", "_")
        row = {"tag": tag, "expected": census[tag]}
        if not d.exists():
            row["status"] = "FAIL missing folder"
            checks.append(row)
            continue
        imgs = sorted((d / "images").iterdir())
        lbls = sorted((d / "labels").glob("*.txt"))
        row["images"], row["labels"] = len(imgs), len(lbls)
        istems = {p.stem for p in imgs}
        lstems = {p.stem for p in lbls}
        problems = []
        if len(imgs) != census[tag]:
            problems.append(f"img count {len(imgs)} != census {census[tag]}")
        if len(lbls) != census[tag]:
            problems.append(f"label count {len(lbls)} != census {census[tag]}")
        if istems != lstems:
            problems.append(f"{len(istems ^ lstems)} unpaired stems")
        sample = rng.sample(imgs, min(len(imgs), args.check_links))
        broken = [p.name for p in sample if not p.resolve().exists()]
        if broken:
            problems.append(f"{len(broken)} broken links (e.g. {broken[0]})")
        parse_fail = 0
        for p in rng.sample(lbls, min(len(lbls), 25)):
            if seg_boxes(p, 100, 100) is None:
                parse_fail += 1
        if parse_fail:
            problems.append(f"{parse_fail} labels unreadable")
        row["status"] = "PASS" if not problems else "FAIL " + "; ".join(problems)
        checks.append(row)

        cards = []
        for p in rng.sample(imgs, min(len(imgs), args.per_slice)):
            lp = d / "labels" / f"{p.stem}.txt"
            try:
                b64, nb = render_thumb(p, lp, args.show_women)
            except Exception as e:                       # noqa: BLE001
                cards.append(f"<div class=card><b>{p.stem}</b>"
                             f"<br>RENDER FAIL: {e}</div>")
                continue
            cap = ""
            if tag.startswith(("w_exposure", "m_exposure")) and caps.get(p.stem):
                cap = "<br><span class=parts>" + \
                      "<br>".join(caps[p.stem][:4]) + "</span>"
            cards.append(
                f"<div class=card><img src='data:image/jpeg;base64,{b64}'>"
                f"<div class=cap>{p.stem[:60]}<br>{nb} GT boxes{cap}</div></div>")
        sections.append(f"<h2>{tag} <small>(n={census[tag]}, "
                        f"{row['status'].split()[0]})</small></h2>"
                        f"<div class=row>{''.join(cards)}</div>")

    n_fail = sum(1 for c in checks if c["status"] != "PASS")
    check_rows = "".join(
        f"<tr class={'ok' if c['status'] == 'PASS' else 'bad'}>"
        f"<td>{c['tag']}</td><td>{c.get('expected', '')}</td>"
        f"<td>{c.get('images', '-')}</td><td>{c.get('labels', '-')}</td>"
        f"<td>{c['status']}</td></tr>" for c in checks)
    redact = ("GT Woman regions pixelated (--show-women to opt out)."
              if not args.show_women else "WOMEN SHOWN (--show-women).")
    html = f"""<!doctype html><meta charset=utf-8>
<title>Subset verification — deploy_compare</title><style>
body{{font-family:system-ui;background:#fcfcfb;color:#222;margin:20px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #ccc;
padding:3px 8px}} tr.ok td:last-child{{color:#007a3d}}
tr.bad td:last-child{{color:#b00020;font-weight:bold}}
.row{{display:flex;flex-wrap:wrap;gap:10px}} .card{{width:{THUMB}px;
font-size:11px}} .card img{{max-width:100%;border:1px solid #ddd}}
.cap{{color:#555}} .parts{{color:#8a5a00}} h2 small{{color:#777}}</style>
<h1>Subset folders — verification</h1>
<p>{len(checks)} subsets · <b>{len(checks) - n_fail} PASS / {n_fail} FAIL</b>
 · symlink folders under <code>{subsets}</code> · {redact}<br>
Certifies membership + integrity of the subset folders, NOT label accuracy
(labels are machine-generated; see EVAL_VERIFICATION.html).</p>
<table><tr><th>subset</th><th>census</th><th>img links</th><th>label links</th>
<th>status</th></tr>{check_rows}</table>
{''.join(sections)}"""
    Path(args.out).write_text(html)
    print(f"[verify] {len(checks) - n_fail} PASS / {n_fail} FAIL "
          f"-> {args.out}")
    for c in checks:
        print(f"  {c['tag']:28s} {c['status']}")
    return n_fail


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src_i, src_l = td / "src_img", td / "src_lbl"
        src_i.mkdir(), src_l.mkdir()
        img = Image.new("RGB", (200, 160), "#88aa88")
        img.paste(Image.new("RGB", (60, 100), "#dd4444"), (20, 20))
        img.save(src_i / "women__a.jpg")
        (src_l / "women__a.txt").write_text("0 0.25 0.44 0.30 0.63\n")
        sub = td / "subsets" / "collection_women"
        (sub / "images").mkdir(parents=True), (sub / "labels").mkdir()
        (sub / "images" / "women__a.jpg").symlink_to(src_i / "women__a.jpg")
        (sub / "labels" / "women__a.txt").symlink_to(src_l / "women__a.txt")
        slf = td / "slices.json"
        slf.write_text(json.dumps(
            {"_meta": {"census": {"collection:women": 1}},
             "slices": {"women__a": ["collection:women"]}}))

        # redaction actually changes the woman's pixels
        before = Image.open(src_i / "women__a.jpg").convert("RGB")
        red = before.copy()
        pixelate(red, (10, 6, 110, 156))
        assert list(before.getdata()) != list(red.getdata()), "pixelate no-op"

        out = td / "v.html"
        n_fail = verify(argparse.Namespace(
            subsets=td / "subsets", slices=slf, verdicts=None, out=out,
            per_slice=2, seed=1, check_links=10, show_women=False))
        assert n_fail == 0, "clean subset must PASS"
        assert "collection:women" in out.read_text(), "gallery section missing"

        # a broken link must FAIL
        (sub / "images" / "women__a.jpg").unlink()
        (sub / "images" / "women__a.jpg").symlink_to(td / "nope.jpg")
        n_fail = verify(argparse.Namespace(
            subsets=td / "subsets", slices=slf, verdicts=None,
            out=td / "v2.html", per_slice=1, seed=1, check_links=10,
            show_women=False))
        assert n_fail == 1, "broken symlink must be caught"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--subsets")
    ap.add_argument("--slices")
    ap.add_argument("--verdicts")
    ap.add_argument("--out", default="subsets_VERIFY.html")
    ap.add_argument("--per-slice", type=int, default=6)
    ap.add_argument("--check-links", type=int, default=200,
                    help="symlinks resolution-checked per subset (sampled)")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--show-women", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.subsets and args.slices):
        ap.error("--subsets and --slices required (or --selftest)")
    verify(args)


if __name__ == "__main__":
    main()
