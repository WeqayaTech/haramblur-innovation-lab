#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-08: adjudication gallery for gate SURVIVORS.

Renders every crop the gate KEPT on an FP arm (verdict real_person/depiction
where the dataset says no person exists) as an HTML grid, so a human can
sort them into: (a) genuine gate failure — statue/object kept, (b) leaked
real person — dataset contamination, gate was RIGHT, (c) genuine depiction
(poster/photo-in-photo) — gate right under the pre-registered ruling.

    python3 survivor_gallery.py \
        --runs /workspace/exp08/lite/objects /workspace/exp08/lite/pass \
        --images /workspace/datasets/object_set /workspace/datasets/pass_3k \
        --out /workspace/exp08/survivors.html

    python3 survivor_gallery.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2

KEEP = {"real_person", "depiction"}
THUMB = 240
PAD = 0.25   # same padding the gate saw

CSS = """
body{font-family:sans-serif;background:#fff;color:#222;margin:20px}
h1{font-size:19px} h2{font-size:16px;margin:22px 0 6px;border-bottom:2px solid #444}
.grid{display:flex;flex-wrap:wrap;gap:10px}
.card{border:2px solid #ccc;border-radius:6px;padding:6px;width:252px}
.card.depiction{border-color:#c98a00}
.card.real_person{border-color:#b02020}
.card.not_person{border-color:#2e8b57}
.card.parse_fail{border-color:#888}
h3{font-size:14px;margin:14px 0 6px;color:#555}
.card img{max-width:240px;display:block;border-radius:3px}
.cap{font-size:11px;color:#333;margin-top:4px;line-height:1.4}
.legend{font-size:13px;margin:6px 0 14px;line-height:1.5}
"""


def find_image(dirs, name, cache):
    if not cache:
        for d in dirs:
            for p in Path(d).rglob("*"):
                if p.is_file():
                    cache.setdefault(p.name, p)
    return cache.get(name)


def crop_b64(img, box, pad=PAD):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return None
    ch, cw = c.shape[:2]
    if max(ch, cw) > THUMB:
        s = THUMB / max(ch, cw)
        c = cv2.resize(c, (max(1, int(cw * s)), max(1, int(ch * s))))
    ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


VERDICT_ORDER = ["real_person", "depiction", "not_person", "parse_fail"]


def build(runs, image_dirs, out_path: Path, show_all=False):
    parts = [f"<style>{CSS}</style>",
             "<h1>EXP-2026-08 — gate survivors (kept crops on FP arms) for adjudication</h1>",
             '<div class="legend">Sort each crop: <b>genuine gate failure</b> '
             '(statue/object kept) · <b>leaked real person</b> (dataset '
             'contamination — gate was right) · <b>genuine depiction</b> '
             '(poster/photo-in-photo — gate right under the posters-count ruling).<br>'
             'Border: <span style="color:#b02020"><b>red</b></span> = kept as '
             'real_person · <span style="color:#c98a00"><b>orange</b></span> = '
             'kept as depiction. Caption: gate verdict / gender / est. age / '
             'gate confidence | SAM3 class + conf.</div>']
    cache: dict = {}
    total = 0
    for run in runs:
        vj = Path(run) / "verdicts.jsonl"
        records = [json.loads(l) for l in vj.open()]
        if not show_all:
            records = [r for r in records if r["verdict"] in KEEP]
        parts.append(f"<h2>{html.escape(str(run))} — {len(records)} crops shown</h2>")
        groups = {v: [r for r in records if r["verdict"] == v]
                  for v in VERDICT_ORDER}
        for verdict in VERDICT_ORDER:
            grp = groups.get(verdict) or []
            if not grp:
                continue
            label = {"not_person": "REJECTED (gate says: not a person)",
                     "real_person": "KEPT as real person",
                     "depiction": "KEPT as depiction (poster/photo-in-photo)",
                     "parse_fail": "PARSE FAILURES"}.get(verdict, verdict)
            parts.append(f"<h3>{label} — {len(grp)}</h3><div class='grid'>")
            for r in grp:
                p = find_image(image_dirs, r["image"], cache)
                if p is None:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                b64 = crop_b64(img, r["box_px"])
                if not b64:
                    continue
                conf = f"{r['sam_conf']:.2f}" if r.get("sam_conf") is not None else "?"
                cap = (f"{r['verdict']} / {r['gender']} / {r.get('estimated_age','?')} / "
                       f"{r['confidence']}<br>SAM3: {r['sam_class']} @ {conf} · {r['image']}")
                parts.append(f"<div class='card {r['verdict']}'>"
                             f"<img src='data:image/jpeg;base64,{b64}'>"
                             f"<div class='cap'>{cap}</div></div>")
                total += 1
            parts.append("</div>")
    out_path.write_text("\n".join(parts))
    print(f"[survivors] {total} crops -> {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="gate-survivor adjudication gallery")
    ap.add_argument("--runs", nargs="+", help="gate run dirs with verdicts.jsonl")
    ap.add_argument("--images", nargs="+", help="image dirs to search (any order)")
    ap.add_argument("--out", default="/workspace/exp08/survivors.html")
    ap.add_argument("--all", action="store_true",
                    help="show EVERY crop grouped by verdict (rejected too), "
                         "not just the survivors")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.runs and args.images):
        raise SystemExit("--runs and --images required (or --selftest)")
    build(args.runs, args.images, Path(args.out), show_all=args.all)


def selftest():
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "img").mkdir()
        cv2.imwrite(str(td / "img" / "a.jpg"),
                    np.full((300, 400, 3), 190, dtype="uint8"))
        run = td / "run"
        run.mkdir()
        (run / "verdicts.jsonl").write_text("\n".join([
            json.dumps({"crop_id": "a_0", "image": "a.jpg", "sam_class": "Woman",
                        "sam_conf": 0.83, "box_px": [50, 40, 200, 280],
                        "verdict": "real_person", "gender": "woman",
                        "estimated_age": 30, "confidence": "high"}),
            json.dumps({"crop_id": "a_1", "image": "a.jpg", "sam_class": "Man",
                        "sam_conf": 0.6, "box_px": [210, 40, 380, 280],
                        "verdict": "not_person", "gender": "unknown",
                        "estimated_age": 0, "confidence": "high"}),
        ]))
        out = td / "g.html"
        build([str(run)], [str(td / "img")], out)
        s = out.read_text()
        assert s.count("base64,") == 1, "only the survivor should render"
        out2 = td / "g2.html"
        build([str(run)], [str(td / "img")], out2, show_all=True)
        s2 = out2.read_text()
        assert s2.count("base64,") == 2, "all mode should render both"
        assert "REJECTED" in s2
    print("[survivor_gallery] selftest OK")


if __name__ == "__main__":
    main()
