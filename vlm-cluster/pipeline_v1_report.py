#!/usr/bin/env python3
"""
EXP-2026-10 — team accuracy report: pilot/full-run results + visual evidence.

Reads the pipeline_v1_eval.py run dirs (verdicts.jsonl + score_*.json +
cost_report.json) and writes ONE self-contained HTML report: per-arm
scorecards against the pre-registered bars, plus evidence crops re-rendered
exactly as Gemini saw them — deletions, label corrections, box corrections.

    python3 pipeline_v1_report.py \
      --arm "lagenda:/workspace/exp10/lagenda:/workspace/lagenda_eval/lagenda_yolo/images/val:/workspace/lagenda_eval/sam_autolabel/labels" \
      --arm "crowd:/workspace/exp10/crowd:/workspace/datasets/crowdhuman/Images_sample500:/workspace/exp03/sam_labels/crowd" \
      --arm "pass:/workspace/exp10/pass:/workspace/datasets/pass_3k:/workspace/exp03/sam_labels/pass" \
      --out /workspace/exp10/pilot_report.html

    python3 pipeline_v1_report.py --selftest    # synthetic, no network

Each --arm is name:run_dir:images_dir:sam_labels_dir (image + label dirs are
needed to re-render the exact highlighted crops for the evidence cards).
CPU-only, no API calls. Reuses pipeline_v1_eval's build_crop/seg_polys/merge
and crowd_headtohead's base64 helpers.
"""
from __future__ import annotations

import argparse
import html as H
import json
from pathlib import Path

from PIL import Image, ImageDraw

from crowd_headtohead import _b64_jpeg
from pipeline_v1_eval import (build_crop, load_raw_dets, load_verdicts,
                              merge, seg_polys)

MAX_CARDS = 12


def _crop_for(rec, images_dir: Path, sam_dir: Path, extra_boxes=()):
    """Re-render the exact highlighted crop Gemini saw for this verdict
    record (same build_crop, same params), optionally drawing extra boxes
    (crop coords) on top. Returns base64 JPEG or None."""
    stem, _, li = rec["id"].rpartition("_")
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        cand = images_dir / (stem + ext)
        if cand.exists():
            img_path = cand
            break
    if img_path is None:
        return None
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        # prefer raw sidecars (exact per-part masks — what Gemini saw)
        raw_dets, parts_list = load_raw_dets(sam_dir, stem)
        parts = None
        if raw_dets is not None:
            dets = raw_dets
            try:
                parts = parts_list[int(li)]
            except (IndexError, ValueError):
                return None
        else:
            dets = seg_polys(sam_dir / f"{stem}.txt", im.width, im.height)
        try:
            cls, poly, box = dets[int(li)]
        except (TypeError, IndexError, ValueError):
            return None
        crop, _, _, _ = build_crop(im, box, poly,
                                   highlight=rec.get("highlighted", True),
                                   style=rec.get("highlight_style") or "tint",
                                   parts=parts)
    if extra_boxes:
        d = ImageDraw.Draw(crop)
        for b, color in extra_boxes:
            d.rectangle(b, outline=color, width=3)
    if max(crop.size) > 340:
        s = 340 / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * s)),
                            max(1, int(crop.height * s))))
        # note: extra boxes drawn pre-resize scale with the image
    return _b64_jpeg(crop)


def _card(b64, caption):
    img = (f"<img src='data:image/jpeg;base64,{b64}'>" if b64
           else "<p class='tag'>(image unavailable)</p>")
    return f"<div class='card'>{img}<p>{caption}</p></div>"


def _score_table(score: dict) -> str:
    skip = {"survivors", "mode"}
    rows = []
    for k, v in score.items():
        if k in skip:
            continue
        if isinstance(v, dict) and {"k", "n"} <= set(v):
            ci = v.get("ci95")
            val = (f"{100 * v['rate']:.1f}% ({v['k']}/{v['n']})"
                   + (f" [CI {100*ci[0]:.1f}–{100*ci[1]:.1f}]" if ci else ""))
        elif isinstance(v, dict):
            val = H.escape(json.dumps(v))
        else:
            val = H.escape(str(v))
        rows.append(f"<tr><td>{H.escape(k)}</td><td>{val}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def build_report(arms, out_path: Path, title_note: str):
    css = """body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;
    color:#222;margin:24px;max-width:1240px}
    h1{margin-bottom:2px}.sub{color:#666;margin-top:0}
    h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:42px}
    h3{margin:16px 0 6px}
    .card{display:inline-block;vertical-align:top;background:#fff;margin:6px;
          padding:8px;border:1px solid #ddd;border-radius:6px;max-width:380px}
    .card img{max-width:340px}.card p{margin:6px 0 0;font-size:12.5px}
    table{border-collapse:collapse;margin:10px 0}
    td{border:1px solid #ccc;padding:5px 10px;font-size:13px}
    td:first-child{font-weight:600;background:#f7f7f7}
    .keep{color:#0a0;font-weight:700}.del{color:#d00;font-weight:700}
    .tag{color:#777;font-size:12px}
    .bars{background:#eef4ff;border-left:4px solid #2858dc;
          padding:10px 14px;border-radius:0 6px 6px 0;font-size:13.5px}"""
    doc = [f"<title>EXP-2026-10 pipeline accuracy report</title><style>{css}</style>",
           "<h1>Pipeline v1 — accuracy report</h1>",
           f"<p class='sub'>SAM3 detects (frozen labels) → mask-highlighted "
           f"25%-padded crop per detection → Gemini 3.5 Flash-Lite verdict → "
           f"merge (rejects deleted, age &gt; 12 → adult, sanity-checked box "
           f"corrections). {H.escape(title_note)}</p>",
           "<div class='bars'><b>Pre-registered bars</b> "
           "(EXP-2026-10 doc, written before running): TP-keep ≥ 97% on "
           "GT-verified people (blocker) · PASS FP-kill ≥ 65% raw (+ survivor "
           "skim) · crowd clear-FP kill ≥ 60% raw (+ skim) · LAGENDA final "
           "3-class ≥ 93%, adult→Child leak ≤ 2%, committed-adult gender "
           "≥ 99% · box corrections &gt; 55% improving · highlight-vs-plain "
           "QC flips &lt; 2% / &lt; 3%.</div>"]

    total_cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for name, run_dir, images_dir, sam_dir in arms:
        recs = load_verdicts(run_dir)
        merged = [(r, merge(r)) for r in recs]
        scores = sorted(run_dir.glob("score_*.json"))
        cost = (json.loads((run_dir / "cost_report.json").read_text())
                if (run_dir / "cost_report.json").exists() else {})
        for k in total_cost:
            total_cost[k] += cost.get(k, 0)

        kept = [(r, m) for r, m in merged if m["keep"]]
        deleted = [(r, m) for r, m in merged if not m["keep"]]
        relabeled = [(r, m) for r, m in kept
                     if m["final_class"] not in (r["sam_class"], "Unknown")]
        boxfix = [(r, m) for r, m in kept if m["box_corrected"]]

        doc.append(f"<h2>Arm: {H.escape(name)} "
                   f"<span class='tag'>({len(recs)} detections · "
                   f"{len(kept)} kept · {len(deleted)} deleted · "
                   f"{len(relabeled)} relabeled · {len(boxfix)} boxes "
                   f"corrected)</span></h2>")
        for sp in scores:
            doc.append(f"<h3>Scores — {H.escape(sp.name)}</h3>")
            doc.append(_score_table(json.loads(sp.read_text())))
        if cost:
            doc.append(f"<p class='tag'>Cost: {cost.get('calls', '?')} calls · "
                       f"{cost.get('input_tokens', '?')} in / "
                       f"{cost.get('output_tokens', '?')} out tokens · "
                       f"USD {cost.get('cost_usd')}</p>")

        for title, rows, render in (
            (f"Deleted detections (gate kills) — showing "
             f"{min(MAX_CARDS, len(deleted))}/{len(deleted)}", deleted,
             lambda r, m: (f"SAM3 said <b>{r['sam_class']}</b> — Lite: "
                           f"<span class='del'>{r['v']['verdict'] if r['v'] else 'parse-fail'}</span>")),
            (f"Label corrections (kept, final ≠ SAM3) — showing "
             f"{min(MAX_CARDS, len(relabeled))}/{len(relabeled)}", relabeled,
             lambda r, m: (f"SAM3: <b>{r['sam_class']}</b> → final: "
                           f"<span class='keep'>{m['final_class']}</span> "
                           f"(Lite: {r['v']['gender']}/{r['v']['age_group']}, "
                           f"age {r['v']['estimated_age']}, conf "
                           f"{r['v'].get('verdict_confidence') or r['v'].get('confidence')})")),
        ):
            doc.append(f"<h3>{title}</h3>" if rows else
                       f"<h3>{title}</h3><p class='tag'>none</p>")
            for r, m in rows[:MAX_CARDS]:
                doc.append(_card(_crop_for(r, images_dir, sam_dir),
                                 f"{H.escape(r['image'])} — {render(r, m)}"))

        doc.append(f"<h3>Box corrections accepted — showing "
                   f"{min(8, len(boxfix))}/{len(boxfix)} "
                   f"<span class='tag'>(gray = SAM3 box, green = corrected)"
                   f"</span></h3>" if boxfix else
                   "<h3>Box corrections accepted</h3><p class='tag'>none</p>")
        for r, m in boxfix[:8]:
            ox, oy = r["crop_origin"]
            fb = m["box_final_img"]
            extra = [(r["sam_box_crop"], (150, 150, 150)),
                     ([fb[0] - ox, fb[1] - oy, fb[2] - ox, fb[3] - oy],
                      (0, 190, 0))]
            doc.append(_card(_crop_for(r, images_dir, sam_dir, extra),
                             f"{H.escape(r['image'])} — final class "
                             f"{m['final_class']}"))

    doc.append("<h2>Total cost</h2>"
               f"<p>{total_cost['calls']} calls · "
               f"{total_cost['input_tokens']:,} input / "
               f"{total_cost['output_tokens']:,} output tokens "
               "<span class='tag'>(USD null until Flash-Lite rates are "
               "screenshot-verified in model_pricing.json — token counts are "
               "measured either way)</span></p>")
    out_path.write_text("\n".join(doc))
    print(f"[report] -> {out_path}")


def _selftest():
    import tempfile

    from pipeline_v1_eval import _StubEngine, run_stage_b, score_pass
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "imgs").mkdir(); (root / "sam").mkdir()
        Image.new("RGB", (100, 100), (120,) * 3).save(root / "imgs/x.jpg")
        (root / "sam/x.txt").write_text(
            "1 0.10 0.10 0.40 0.10 0.40 0.90 0.10 0.90\n"
            "0 0.70 0.70 0.10 0.10\n")
        answers = [
            json.dumps({"verdict": "real_person", "gender": "woman",
                        "age_group": "adult", "estimated_age": 30,
                        "box_correction": [8, 8, 38, 78],
                        "highlight_quality": "good", "confidence": "high"}),
            json.dumps({"verdict": "not_person", "gender": "unknown",
                        "age_group": "unknown", "estimated_age": 5,
                        "box_correction": "ok",
                        "highlight_quality": "good", "confidence": "high"}),
        ]
        run_stage_b(_StubEngine(answers), root / "imgs", root / "sam",
                    root / "run", 0, 42, highlight=True)
        s = score_pass(load_verdicts(root / "run"))
        (root / "run/score_none.json").write_text(json.dumps(s))
        build_report([("pass-selftest", root / "run", root / "imgs",
                       root / "sam")],
                     root / "report.html", "selftest")
        html_text = (root / "report.html").read_text()
        for marker in ("Deleted detections", "Label corrections",
                       "Box corrections", "score_none.json", "Total cost"):
            assert marker in html_text, f"missing: {marker}"
        assert html_text.count("data:image/jpeg;base64,") >= 2
    print("pipeline_v1_report.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-10 team report")
    ap.add_argument("--arm", action="append", default=[],
                    help="name:run_dir:images_dir:sam_labels_dir (repeatable)")
    ap.add_argument("--note", default="",
                    help="subtitle note, e.g. 'PILOT — 25 images/arm'")
    ap.add_argument("--out", default="pipeline_report.html")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.arm:
        ap.error("at least one --arm required (or --selftest)")
    arms = []
    for spec in args.arm:
        parts = spec.split(":")
        if len(parts) != 4:
            ap.error(f"bad --arm (need name:run:images:labels): {spec}")
        arms.append((parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])))
    build_report(arms, Path(args.out), args.note)


if __name__ == "__main__":
    main()
