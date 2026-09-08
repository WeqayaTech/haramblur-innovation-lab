#!/usr/bin/env python3
"""
SPOTLIGHT — stage-by-stage trace of a REAL production run.

Reads what the pipeline already produced (raw SAM3 sidecars + verdicts.jsonl +
emitted labels) and renders one self-contained HTML showing, per image:

  Stage 1  raw image + SAM3's detections (class, confidence, mask parts)
  Stage 2  the EXACT crop sent to Gemini, rebuilt with the same build_crop()
  Stage 3  Gemini's raw response text, verbatim
  Stage 4  the merge decision (keep/delete, final class, and why)
  Stage 5  the emitted YOLO line, and the final image with surviving labels

No API calls — it verifies the run that happened, not a fresh one.

    python3 trace_report.py --images <images dir> --raw-labels <raw dir> \\
        --run <run dir> --n 5 --out /workspace/prod_trace.html

    python3 trace_report.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import html as H
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from spotlight_run import CLASS_ID, CLASS_NAME, build_crop, load_raw, merge

KEEP_RGB = {"Woman": (200, 40, 160), "Man": (40, 90, 220),
            "Child": (0, 170, 200), "Unknown": (130, 130, 130)}
DEL_RGB = (220, 0, 0)


def b64(img: Image.Image, max_dim=560, q=72):
    im = img.convert("RGB")
    if max(im.size) > max_dim:
        s = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def annotated(img, boxes, max_dim=560):
    """boxes = [(xyxy, label, rgb)] in image pixels."""
    im = img.convert("RGB")
    s = min(1.0, max_dim / max(im.size))
    if s < 1.0:
        im = im.resize((int(im.width * s), int(im.height * s)))
    d = ImageDraw.Draw(im)
    for box, lab, col in boxes:
        b = [v * s for v in box]
        d.rectangle(b, outline=col, width=3)
        if lab:
            d.text((b[0] + 2, max(0, b[1] - 11)), lab, fill=col)
    return b64(im, max_dim)


def build(images_dir: Path, raw_dir: Path, run_dir: Path, n: int, out: Path,
          seed: int | None = None):
    verdicts = {}
    for f in sorted(run_dir.glob("verdicts*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                verdicts.setdefault(r["image_stem"], {})[r["det_index"]] = r
    if not verdicts:
        raise SystemExit(f"no verdicts found in {run_dir}")
    meta = {}
    mf = run_dir / "run_meta.json"
    if mf.exists():
        meta = json.loads(mf.read_text())

    avail = [s for s in sorted(verdicts) if (raw_dir / f"{s}.txt").exists()]
    # Alphabetical-first is a biased slice for an accuracy check -- Open Images
    # stems are hex ids, so the first n are an arbitrary but FIXED corner of the
    # corpus. --seed draws a reproducible random sample instead.
    if seed is None:
        picked = avail[:n]
    else:
        import random
        picked = sorted(random.Random(seed).sample(avail, min(n, len(avail))))
        print(f"[trace] random sample of {len(picked)} from {len(avail):,} "
              f"verified images (seed {seed})")

    css = """body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;
    color:#222;margin:24px;max-width:1240px}
    h1{margin-bottom:2px}.sub{color:#666;margin-top:0}
    h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:40px}
    h3{margin:16px 0 6px;color:#333;font-size:15px}
    .stage{background:#fff;border:1px solid #ddd;border-radius:6px;
           padding:12px;margin:10px 0}
    .det{display:inline-block;vertical-align:top;background:#fdfdfd;
         border:1px solid #e2e2e2;border-radius:6px;padding:10px;margin:6px;
         max-width:400px}
    .det img{max-width:370px}
    pre{background:#f0f0f0;padding:9px;border-radius:5px;font-size:11px;
        overflow-x:auto;white-space:pre-wrap;margin:4px 0}
    .keep{color:#0a0;font-weight:700}.del{color:#d00;font-weight:700}
    .tag{color:#777;font-size:12px}
    table{border-collapse:collapse;margin:6px 0}
    td,th{border:1px solid #ccc;padding:4px 9px;font-size:12px;text-align:left}"""

    doc = [f"<title>Spotlight production trace</title><style>{css}</style>",
           "<h1>Spotlight pipeline — stage-by-stage trace</h1>",
           f"<p class='sub'>{len(picked)} images from a real run in "
           f"<code>{H.escape(str(run_dir))}</code> · no API calls were made to "
           "build this page — it shows what the pipeline actually produced.</p>"]
    if meta:
        doc.append("<div class='stage'><h3>Frozen run configuration</h3><table>"
                   + "".join(f"<tr><th>{H.escape(k)}</th><td>{H.escape(str(v))[:200]}</td></tr>"
                             for k, v in meta.items() if k != "prompt_text")
                   + "</table><details><summary>full prompt sent with every "
                     "crop</summary><pre>"
                   + H.escape(meta.get("prompt_text", "")) + "</pre></details></div>")

    for stem in picked:
        img_path = next((images_dir / f"{stem}{e}" for e in
                         (".jpg", ".jpeg", ".png", ".bmp")
                         if (images_dir / f"{stem}{e}").exists()), None)
        if img_path is None:
            continue
        rec = load_raw(raw_dir, stem)
        img = Image.open(img_path).convert("RGB")
        dets = rec.get("detections", [])
        vs = verdicts.get(stem, {})
        doc.append(f"<h2>{H.escape(img_path.name)} "
                   f"<span class='tag'>({img.width}x{img.height} · "
                   f"{len(dets)} SAM3 detections · {len(vs)} verdicts)</span></h2>")

        # Stage 1
        s1 = [(d["box"], f"{CLASS_NAME.get(d['cls'], d['cls'])} {d.get('conf', 0):.2f}",
               (40, 90, 220)) for d in dets]
        doc.append("<div class='stage'><h3>Stage 1 — SAM3 detections "
                   "(class, confidence, mask parts)</h3>"
                   f"<img src='data:image/jpeg;base64,{annotated(img, s1)}'>"
                   f"<pre>{H.escape(json.dumps([{k: v for k, v in d.items() if k != 'parts'} | {'n_parts': len(d.get('parts', []))} for d in dets], indent=1)[:1400])}</pre>"
                   f"<p class='tag'>{len(rec.get('suppressed', []))} additional "
                   f"detections were suppressed by NMS and logged.</p></div>")

        # Stages 2-4
        doc.append("<div class='stage'><h3>Stages 2–4 — per detection: the exact "
                   "crop sent, Gemini's raw reply, and the merge decision</h3>")
        for i, d in enumerate(dets):
            r = vs.get(i)
            crop, scale = build_crop(img, d["box"],
                                     [[(x, y) for x, y in p] for p in d.get("parts", [])])
            if r is None:
                doc.append(f"<div class='det'><img src='data:image/jpeg;base64,"
                           f"{b64(crop, 370)}'><p class='tag'>det {i} — "
                           f"not verified yet</p></div>")
                continue
            m = merge(r.get("v"))
            if not m["keep"]:
                status = "<span class='del'>DELETE — not a person</span>"
            elif m["final_class"] == "Unknown":
                # kept as a person, but no gender committed -> excluded from
                # training by default (emit's dropped_unknown bucket)
                status = ("<span class='del'>DROPPED from training</span> "
                          "<span class='tag'>(real person, but gender "
                          "undetermined)</span>")
            else:
                status = f"<span class='keep'>KEEP → {m['final_class']}</span>"
            v = r.get("v") or {}
            why = (f"verdict={v.get('verdict')}, gender={v.get('gender')}, "
                   f"age_group={v.get('age_group')}, est_age={v.get('estimated_age')}")
            doc.append(
                f"<div class='det'><b>det {i}</b> — SAM3 said "
                f"{CLASS_NAME.get(d['cls'], d['cls'])} (conf {d.get('conf', 0):.2f})<br>"
                f"<span class='tag'>crop {crop.width}x{crop.height}px"
                + (f", upscaled {scale:.1f}x" if scale > 1 else "")
                + f", {len(d.get('parts', []))} mask part(s)</span>"
                f"<img src='data:image/jpeg;base64,{b64(crop, 370)}'>"
                f"<b>Gemini raw reply:</b><pre>{H.escape((r.get('raw_text') or '(empty)')[:700])}</pre>"
                f"<b>Merge:</b> {status}<br><span class='tag'>{H.escape(why)}</span>"
                f"<br><span class='tag'>blur {r.get('blurriness')} · "
                f"height {r.get('person_px_height')}px</span></div>")
        doc.append("</div>")

        # Stage 5
        lab = run_dir / "labels" / f"{stem}.txt"
        if lab.exists():
            lines = [l for l in lab.read_text().splitlines() if l.strip()]
            final_boxes = []
            for i, d in enumerate(dets):
                r = vs.get(i)
                m = merge(r.get("v")) if r else {"keep": True, "final_class": None}
                if r and not m["keep"]:
                    final_boxes.append((d["box"], "DELETED", DEL_RGB))
                elif m.get("final_class"):
                    final_boxes.append((d["box"], m["final_class"],
                                        KEEP_RGB.get(m["final_class"], (90,) * 3)))
            doc.append("<div class='stage'><h3>Stage 5 — emitted training label</h3>"
                       f"<img src='data:image/jpeg;base64,{annotated(img, final_boxes)}'>"
                       f"<pre>{H.escape(chr(10).join(l[:110] + (' …' if len(l) > 110 else '') for l in lines)) or '(empty — all detections removed)'}</pre>"
                       f"<p class='tag'>{len(lines)} of {len(dets)} detections "
                       f"survived. Geometry is byte-identical to SAM3's; only the "
                       f"class id changes.</p></div>")
        else:
            doc.append("<div class='stage'><h3>Stage 5</h3><p class='tag'>no "
                       "emitted label yet (image not fully verified)</p></div>")
        img.close()

    out.write_text("\n".join(doc))
    print(f"[trace] {len(picked)} images -> {out}")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        (r / "img").mkdir(); (r / "raw").mkdir(); (r / "run" / "labels").mkdir(parents=True)
        Image.new("RGB", (200, 200), (110,) * 3).save(r / "img/a.jpg")
        (r / "raw/a.txt").write_text("1 0.10 0.10 0.40 0.10 0.40 0.90\n")
        json.dump({"image": "a.jpg", "width": 200, "height": 200,
                   "detections": [{"cls": 1, "conf": 0.91, "box": [20, 20, 80, 180],
                                   "parts": [[[20, 20], [80, 20], [80, 180]]]}],
                   "suppressed": []}, open(r / "raw/a.json", "w"))
        (r / "run/verdicts.jsonl").write_text(json.dumps({
            "image_stem": "a", "det_index": 0, "sam_class_id": 1,
            "raw_text": '{"verdict":"real_person","gender":"woman"}',
            "blurriness": 12.3, "person_px_height": 160.0,
            "v": {"verdict": "real_person", "gender": "woman",
                  "age_group": "adult", "estimated_age": 31,
                  "highlight_quality": "good", "confidence": "high"}}) + "\n")
        (r / "run/labels/a.txt").write_text("0 0.10 0.10 0.40 0.10 0.40 0.90")
        (r / "run/run_meta.json").write_text(json.dumps(
            {"prompt_version": "spotlight-e1", "prompt_sha": "abc123",
             "prompt_text": "TEST PROMPT"}))
        build(r / "img", r / "raw", r / "run", 5, r / "t.html")
        html = (r / "t.html").read_text()
        for marker in ("Stage 1", "Stages 2–4", "Stage 5", "Gemini raw reply",
                       "KEEP", "spotlight-e1", "TEST PROMPT"):
            assert marker in html, f"missing {marker}"
        assert html.count("data:image/jpeg;base64,") >= 3
    print("trace_report.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="Stage-by-stage production trace")
    ap.add_argument("--images"); ap.add_argument("--raw-labels")
    ap.add_argument("--run"); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="trace.html")
    ap.add_argument("--seed", type=int, default=None,
                    help="draw a reproducible RANDOM sample instead of the "
                         "alphabetically-first n (use this for accuracy checks)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.images and a.raw_labels and a.run):
        ap.error("--images --raw-labels --run are required")
    build(Path(a.images), Path(a.raw_labels), Path(a.run), a.n, Path(a.out),
          a.seed)


if __name__ == "__main__":
    main()
