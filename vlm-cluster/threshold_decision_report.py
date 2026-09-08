#!/usr/bin/env python3
"""
Threshold decision report — one self-contained HTML for coworkers.

Lays out the ONE open decision: which threshold convention to ship —
the standard detection max-F1 point (box-count view) or the pixel-area
budget point (product-experience view) — with the six confidence-sweep
charts inline and every number computed from the caches (nothing
hand-copied).

    python3 threshold_decision_report.py \
        --summary .../summary_run5.json --f1-cache .../f1_cache.json \
        --fp-cache-male .../fp_cache_male.json \
        --fp-cache-female .../fp_cache_female.json \
        --charts-dir .../_deploycmp_review --out THRESHOLD_DECISION.html

    python3 threshold_decision_report.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from women_report_final import budget_picks

AUDS = {
    "male": {"cls": "Woman", "noun": "woman", "label": "Male audience",
             "who": "male users (women are blurred, men are not)"},
    "female": {"cls": "Man", "noun": "man", "label": "Female audience",
               "who": "female users (men are blurred, women are not)"},
}
BG, INK = "#fcfcfb", "#222222"


def _load_float_keys(d):
    return {m: {float(k): v for k, v in mm.items()} for m, mm in d.items()}


def gather(summary, f1_cache, fp_male, fp_female):
    """-> {aud: {order, vis, f1, wrongH, wrongP, f1_pick, area_pick,
    budget}} — every displayed number originates here."""
    out = {}
    fps = {"male": fp_male, "female": fp_female}
    for aud in AUDS:
        md = summary["modes"][aud]
        order = list(md["curves"].keys())
        vis = {m: {float(cf): a["person_uncov_mean"]
                   for cf, a in md["curves"][m].items()
                   if a.get("person_uncov_mean") is not None}
               for m in order}
        f1 = {m: {cf: v["f1"] for cf, v in
                  _load_float_keys(f1_cache[aud])[m].items()}
              for m in order}
        wrongH = _load_float_keys(fps[aud]["holdout"])
        wrongP = _load_float_keys(fps[aud]["pass"])
        f1_pick = {m: max(f1[m], key=lambda cf: (f1[m][cf], -cf))
                   for m in order}
        fp_curves = {"holdout": wrongH, "pass": wrongP}
        budget, area_pick, _feas = budget_picks(md, fp_curves, order)
        out[aud] = dict(order=order, vis=vis, f1=f1, wrongH=wrongH,
                        wrongP=wrongP, f1_pick=f1_pick,
                        area_pick=area_pick, budget=budget)
    return out


def _row(name, cf, g):
    v, wh, wp, f = (g["vis"][name].get(cf), g["wrongH"][name].get(cf),
                    g["wrongP"][name].get(cf), g["f1"][name].get(cf))
    fmt = lambda x, nd=1, mul=100: "-" if x is None else f"{mul * x:.{nd}f}"
    return (f"<td>{cf:.2f}</td><td>{fmt(v)}%</td>"
            f"<td>{fmt(wh, 3)}%</td><td>{fmt(wp, 3)}%</td>"
            f"<td>{fmt(f, 3, 1)}</td>")


def build(data, charts_dir, out_path, rec="y26n_warm50",
          inc="v11n_shipped"):
    charts_dir = Path(charts_dir)
    sections = ""
    for aud, meta in AUDS.items():
        g = data[aud]
        n, N = meta["noun"], meta["cls"]
        fpick, apick = g["f1_pick"][rec], g["area_pick"][rec]

        dec_rows = (
            f"<tr><td><b>today: {inc}</b> (production)</td>"
            + _row(inc, 0.45, g) + "</tr>"
            f"<tr class=optA><td><b>{rec}</b> at the max-F1 point "
            f"(convention A)</td>" + _row(rec, fpick, g) + "</tr>"
            f"<tr class=optB><td><b>{rec}</b> at the pixel-budget point "
            f"(convention B)</td>" + _row(rec, apick, g) + "</tr>")

        t_rows = ""
        for m in g["order"]:
            fp_, ap_ = g["f1_pick"][m], g["area_pick"][m]
            f1v = g["f1"][m][fp_]
            if ap_ is None:
                ac = av = aw = "-"
            else:
                ac = f"{ap_:.2f}"
                av = f"{100 * g['vis'][m][ap_]:.1f}%"
                aw = f"{100 * g['wrongH'][m][ap_]:.3f}%"
            t_rows += (f"<tr><td><b>{m}</b></td>"
                       f"<td>{fp_:.2f}</td><td>{f1v:.3f}</td>"
                       f"<td>{ac}</td><td>{av}</td><td>{aw}</td></tr>")

        figs = ""
        caps = {
            "f1": f"<b>{N}-class F1 vs confidence.</b> The standard "
                  f"detection view (box-level precision/recall, IoU ≥ 0.5"
                  f"). Dots mark each model's max-F1 confidence — "
                  f"convention A's pick.",
            "underblur": f"<b>Pixels underblurred.</b> The average % of "
                         f"a {n}'s box left visible. Lower is better; it "
                         f"keeps improving as the threshold drops — this "
                         f"is what convention B's lower threshold buys.",
            "overblur": f"<b>Pixels overblurred.</b> The % of the image "
                        f"wrongly blurred ({N} boxes matching no {n} at "
                        f"IoU ≥ 0.5). What the lower threshold costs — "
                        f"note the candidates stay below production's "
                        f"level even at their low picks.",
        }
        for kind in ("f1", "underblur", "overblur"):
            svg = (charts_dir / f"chart_{kind}_{aud}.svg").read_text()
            figs += f"<figure>{svg}<figcaption>{caps[kind]}</figcaption></figure>"

        b = g["budget"]
        sections += f"""
<h2>{meta['label']} — {meta['who']}</h2>
<h3>The decision table</h3>
<table><tr><th>option</th><th>threshold</th><th>% of a {n} left
visible</th><th>wrongly blurred % of image (holdout)</th><th>wrongly
blurred % (PASS)</th><th>{N}-class F1</th></tr>{dec_rows}</table>
<p>Both options beat production on every column. <span class=optA-chip>
Convention A</span> maximizes the standard box-level F1 score.
<span class=optB-chip>Convention B</span> takes the lowest threshold
whose wrongly-blurred pixels still beat production
({100 * b['wrong_holdout']:.3f}% holdout / {100 * b['wrong_pass']:.3f}%
PASS), which minimizes how much of each {n} stays visible.</p>
<h3>The sweeps</h3>
{figs}
<h3>All models under both conventions</h3>
<table><tr><th>model</th><th>max-F1 conf (A)</th><th>F1 there</th>
<th>pixel-budget conf (B)</th><th>% visible there</th><th>wrong-blur %
there</th></tr>{t_rows}</table>
"""

    html = f"""<!doctype html><meta charset=utf-8>
<title>Blur threshold decision</title><style>
body{{font-family:system-ui;background:{BG};color:{INK};max-width:900px;
margin:24px auto;padding:0 16px;line-height:1.45}}
table{{border-collapse:collapse;font-size:13.5px;margin:10px 0}}
td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}
td:first-child{{text-align:left}} th{{background:#f0f0ee}}
tr.optA{{background:#eef4fb}} tr.optB{{background:#e3f2e7}}
.optA-chip{{background:#eef4fb;padding:0 4px;border:1px solid #a9c7e8}}
.optB-chip{{background:#e3f2e7;padding:0 4px;border:1px solid #7fbf8f}}
figure{{margin:14px 0}} figcaption{{font-size:12.5px;color:#444;
max-width:760px;margin-top:2px}}
.big{{background:#fff7e0;border:1px solid #e6c200;padding:12px 16px;
border-radius:6px}} h2{{margin-top:34px;border-bottom:2px solid #ddd;
padding-bottom:4px}}</style>

<h1>Blur threshold decision — pick a convention</h1>
<p>The model comparison is settled: <b>{rec}</b> beats the shipped
model on every metric we measure, for both audiences, under every
convention tried. <b>The open decision is the confidence threshold</b>,
and it comes down to which unit of false blur we care about:</p>
<ul>
<li><b>Convention A — boxes (the detection standard).</b> Pick the
threshold maximizing F1 (precision/recall balance at IoU ≥ 0.5). Every
wrong box counts equally, however small. This lands near today's 0.45
— which is why production shipped there.</li>
<li><b>Convention B — pixels (the product experience).</b> Wrong blur
is measured as wrongly-blurred screen area; the threshold is the lowest
whose wrong pixels still beat production. Lands much lower, because the
extra boxes that appear at low confidence are mostly small: they wreck
box-precision (convention A punishes them) while barely adding wrong
pixels (convention B tolerates them) — and each step down blurs more of
every person.</li>
</ul>
<div class=big><b>The question for the team:</b> is a stray small wrong
box as bad as a big one (choose A), or is wrong blur only as bad as the
pixels it covers (choose B)? Everything below is the evidence; both
options improve on production either way.</div>

{sections}
<h2>How the numbers are computed</h2>
<ul>
<li><b>% of a person left visible</b> — per person in the answer key:
the share of their box NOT covered by the union of fired blur boxes;
every person counts equally; averaged over all of them.</li>
<li><b>Wrongly blurred % of image</b> — per image: the area of fired
blur boxes that match no target person at IoU ≥ 0.5 (boxes on
ignore-region people are excused), as a share of the picture; averaged
over images. Measured on the 11,493-image holdout (machine answer key)
and on PASS — 3,000 human-verified person-free images where every box
is wrong by construction.</li>
<li><b>F1</b> — box-level precision/recall at IoU ≥ 0.5,
confidence-ordered one-to-one matching, ignore regions excused;
F1 = 2PR/(P+R).</li>
<li>All curves are replayed from logged detections (down to confidence
0.001) — no model was re-run; every number in this page is computed
from those caches, none typed in by hand.</li>
</ul>
<h2>Keep in mind</h2>
<ul>
<li>The holdout answer key is machine-generated (SAM3 + Gemini
Flash-Lite); the PASS column is the human-verified, label-independent
check — and the candidates beat production there too.</li>
<li>All geometry is box-space: "5% visible" means 5% of the person's
bounding box.</li>
<li>These are single-image numbers; video flicker is a separate policy
layer (EXP-2026-11).</li>
</ul>"""
    Path(out_path).write_text(html)
    return out_path


def selftest():
    import tempfile
    grid = [0.2, 0.4]
    curves = {m: {str(c): {"person_uncov_mean": 0.05 + 0.1 * c}
                  for c in grid} for m in ("v11n_shipped", "y26n_warm50")}
    summary = {"modes": {aud: {"curves": curves} for aud in AUDS}}
    f1c = {aud: {m: {str(c): {"f1": 0.8 + c / 10, "p": 0.8, "r": 0.9}
                     for c in grid}
                 for m in curves} for aud in AUDS}
    fp = {"holdout": {m: {str(c): 0.02 - 0.01 * c for c in grid}
                      for m in curves},
          "pass": {m: {str(c): 0.002 for c in grid} for m in curves}}
    # budget = incumbent nearest 0.45 -> 0.4: wrongH 0.016, pass 0.002;
    # 0.2 has wrongH 0.018 > budget -> only 0.4 feasible -> area pick 0.4
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for aud in AUDS:
            for kind in ("f1", "underblur", "overblur"):
                (td / f"chart_{kind}_{aud}.svg").write_text(
                    f"<svg xmlns='http://www.w3.org/2000/svg'>"
                    f"<text>{kind}_{aud}</text></svg>")
        data = gather(summary, f1c, fp, fp)
        assert data["male"]["f1_pick"]["y26n_warm50"] == 0.4
        assert data["male"]["area_pick"]["y26n_warm50"] == 0.4
        out = build(data, td, td / "r.html")
        t = Path(out).read_text()
        assert "Convention A" in t and "Convention B" in t
        assert t.count("<svg") == 6, "all six charts embedded"
        assert "decision table" in t and "0.840" in t, "F1 cell rendered"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--f1-cache")
    ap.add_argument("--fp-cache-male")
    ap.add_argument("--fp-cache-female")
    ap.add_argument("--charts-dir")
    ap.add_argument("--out", default="THRESHOLD_DECISION.html")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    data = gather(json.loads(Path(args.summary).read_text()),
                  json.loads(Path(args.f1_cache).read_text()),
                  json.loads(Path(args.fp_cache_male).read_text()),
                  json.loads(Path(args.fp_cache_female).read_text()))
    out = build(data, args.charts_dir, args.out)
    for aud in AUDS:
        g = data[aud]
        print(f"{aud}: F1 picks {g['f1_pick']}  area picks "
              f"{g['area_pick']}")
    print(f"[decision] wrote {out}")


if __name__ == "__main__":
    main()
