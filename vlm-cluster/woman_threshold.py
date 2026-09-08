#!/usr/bin/env python3
"""
Woman threshold selection — ONE deployable confidence per model.

Production ships a single confidence threshold, so per-slice / per-dataset
confs are diagnostics, never a config. This tool reads a deploy_compare
score summary (the male-mode curves ARE the Woman class: exposure = woman
TP side, false blur = woman FP side, both logged at every conf) and, per
model, sweeps the grid to find the balance point. No pod, no GPU, no
re-run — pure post-processing.

Two pre-registered selection rules, both reported so the owner can pick:

  J     = woman cov90 rate − false-blur image rate   (Youden-style balance;
          both are rates in [0,1]; argmax)
  Udist = sqrt( (E-person/E-person_ref)^2 + (FB-area/FB-area_ref)^2 )
          normalized distance to the utopia point (0 exposure, 0 false
          blur), ref = the incumbent at its anchor conf so units cancel;
          argmin.

Outputs: printed per-model tables, one HTML report, one SVG (J vs conf,
all models). Women appear nowhere in this report — it is tables and curves
only.

    python3 woman_threshold.py --summary .../holdout_run4/summary.json \\
        --out-html WOMAN_THRESHOLD_REPORT.html --out-svg chart_woman_J.svg

    python3 woman_threshold.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
BG, INK, MUTED, GRID = "#fcfcfb", "#222222", "#555555", "#e8e8e6"
FONT = "font-family='system-ui,Helvetica,Arial' "


def sweep_rows(curves, ref):
    """[(conf, dict)] sorted by conf, with J and Udist computed."""
    rows = []
    for cf_s, a in curves.items():
        if a.get("person_covered90_rate") is None:
            continue
        j = a["person_covered90_rate"] - a["falseblur_img_rate"]
        ud = ((a["person_uncov_mean"] / ref["person_uncov_mean"]) ** 2
              + (a["fb_area_mean"] / ref["fb_area_mean"]) ** 2) ** 0.5
        rows.append((float(cf_s), {**a, "J": j, "Udist": ud}))
    return sorted(rows)


def select(rows):
    best_j = max(rows, key=lambda r: (r[1]["J"], r[0]))
    best_u = min(rows, key=lambda r: (r[1]["Udist"], -r[0]))
    return best_j, best_u


def j_svg(model_rows, order):
    W, H = 760, 460
    ML, MR, MT, MB = 64, 175, 52, 52
    pw, ph = W - ML - MR, H - MT - MB
    all_j = [r[1]["J"] for n in order for r in model_rows[n]]
    ymin, ymax = min(all_j) - 0.02, max(all_j) + 0.02
    xmin, xmax = 0.0, max(r[0] for n in order for r in model_rows[n]) + 0.05

    def X(v):
        return ML + pw * (v - xmin) / (xmax - xmin)

    def Y(v):
        return MT + ph * (1 - (v - ymin) / (ymax - ymin))

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='{BG}'/>",
         f"<text x='{ML}' y='24' {FONT}font-size='15' fill='{INK}' "
         f"font-weight='600'>Woman balance J = cov90 − false-blur rate, "
         f"per confidence</text>",
         f"<text x='{ML}' y='40' {FONT}font-size='11' fill='{MUTED}'>"
         f"higher is better; ● marks each model's argmax — the deployable "
         f"single threshold</text>"]
    for i in range(6):
        gy = ymin + (ymax - ymin) * i / 5
        p.append(f"<line x1='{ML}' y1='{Y(gy)}' x2='{ML + pw}' y2='{Y(gy)}' "
                 f"stroke='{GRID}'/>")
        p.append(f"<text x='{ML - 8}' y='{Y(gy) + 4}' {FONT}font-size='10' "
                 f"fill='{MUTED}' text-anchor='end'>{gy:.2f}</text>")
    for gx in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p.append(f"<line x1='{X(gx)}' y1='{MT}' x2='{X(gx)}' y2='{MT + ph}' "
                 f"stroke='{GRID}'/>")
        p.append(f"<text x='{X(gx)}' y='{MT + ph + 16}' {FONT}font-size='10' "
                 f"fill='{MUTED}' text-anchor='middle'>{gx:.1f}</text>")
    p.append(f"<text x='{ML + pw / 2}' y='{H - 12}' {FONT}font-size='12' "
             f"fill='{INK}' text-anchor='middle'>confidence threshold</text>")
    for k, name in enumerate(order):
        col = PALETTE[k % len(PALETTE)]
        rows = model_rows[name]
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f},"
                        f"{Y(r['J']):.1f}" for i, (c, r) in enumerate(rows))
        p.append(f"<path d='{path}' fill='none' stroke='{col}' "
                 f"stroke-width='2'/>")
        bj, _ = select(rows)
        p.append(f"<circle cx='{X(bj[0])}' cy='{Y(bj[1]['J'])}' r='6' "
                 f"fill='{col}' stroke='{BG}' stroke-width='2'/>")
        p.append(f"<circle cx='{ML + pw + 14}' "
                 f"cy='{MT + 14 + 22 * k}' r='5' fill='{col}'/>")
        p.append(f"<text x='{ML + pw + 24}' y='{MT + 18 + 22 * k}' {FONT}"
                 f"font-size='12' fill='{INK}'>{name} @{bj[0]:.2f}</text>")
    p.append("</svg>")
    return "\n".join(p)


def build(summary, out_html, out_svg):
    male = summary["modes"]["male"]
    order = list(male["curves"].keys())
    inc = order[0]
    anchor = str(summary["protocol"]["anchor_conf"])
    ref = male["curves"][inc][anchor]
    model_rows = {n: sweep_rows(male["curves"][n], ref) for n in order}

    svg = j_svg(model_rows, order)
    if out_svg:
        Path(out_svg).write_text(svg)

    def pct(v, nd=1):
        return f"{100 * v:.{nd}f}"

    sections, summary_rows, printed = "", "", []
    for n in order:
        rows = model_rows[n]
        bj, bu = select(rows)
        printed.append((n, bj, bu))
        body = ""
        for cf, a in rows:
            mark = (" class=pick" if cf == bj[0] else
                    " class=pick2" if cf == bu[0] else "")
            body += (f"<tr{mark}><td>{cf:.2f}</td>"
                     f"<td>{pct(a['person_covered90_rate'])}</td>"
                     f"<td>{pct(a['person_uncov_mean'])}</td>"
                     f"<td>{pct(a['mean_exposure'], 2)}</td>"
                     f"<td>{pct(a['falseblur_img_rate'])}</td>"
                     f"<td>{pct(a['fb_area_mean'], 3)}</td>"
                     f"<td>{a['fp_boxes_per_100']:.2f}</td>"
                     f"<td><b>{a['J']:.4f}</b></td>"
                     f"<td>{a['Udist']:.3f}</td></tr>")
        sections += (
            f"<h3>{n} — pick J: <b>conf {bj[0]:.2f}</b> "
            f"(J={bj[1]['J']:.4f}) · pick Udist: conf {bu[0]:.2f}</h3>"
            f"<table><tr><th>conf</th><th>women ≥90% covered %</th>"
            f"<th>E-person %</th><th>E-img %</th><th>FB img %</th>"
            f"<th>FB area %</th><th>FP boxes/100</th><th>J</th>"
            f"<th>Udist</th></tr>{body}</table>")
        a = bj[1]
        summary_rows += (f"<tr><td><b>{n}</b></td><td>{bj[0]:.2f}</td>"
                         f"<td>{pct(a['person_covered90_rate'])}</td>"
                         f"<td>{pct(a['person_uncov_mean'])}</td>"
                         f"<td>{pct(a['falseblur_img_rate'])}</td>"
                         f"<td>{a['fp_boxes_per_100']:.2f}</td>"
                         f"<td>{a['J']:.4f}</td><td>{bu[0]:.2f}</td></tr>")

    html = f"""<!doctype html><meta charset=utf-8>
<title>Woman threshold selection</title><style>
body{{font-family:system-ui;background:{BG};color:{INK};max-width:1050px;
margin:24px auto;padding:0 16px}} table{{border-collapse:collapse;
font-size:12.5px;margin:8px 0}} td,th{{border:1px solid #ccc;
padding:2px 8px;text-align:right}} td:first-child{{text-align:left}}
tr.pick{{background:#e3f2e7;font-weight:600}}
tr.pick2{{background:#fff4dd}}
.note{{background:#eef4fb;border:1px solid #a9c7e8;padding:8px 12px;
border-radius:4px}}</style>
<h1>Woman threshold selection — one deployable confidence per model</h1>
<p class=note>Production ships a SINGLE confidence per model; varying it
per dataset/slice is diagnostics, not a config. This sweep balances the
Woman class's two sides — TP (women covered) vs FP (false Woman blur) —
from run4's logged curves ({summary['protocol']['n_images']} images,
ignore regions honored). Green row = argmax J (cov90 − false-blur rate);
amber row = min normalized utopia distance. Both rules are stated before
reading results; the owner selects.</p>
<h2>Selection summary</h2>
<table><tr><th>model</th><th>pick (J)</th><th>women ≥90% covered %</th>
<th>E-person %</th><th>FB img %</th><th>FP boxes/100</th><th>J</th>
<th>alt pick (Udist)</th></tr>{summary_rows}</table>
{svg}
<h2>Full sweeps</h2>{sections}
<p>Caveats: GT is machine labels; the incumbent's own optimum may differ
from its shipped 0.45 — compare models at each one's own pick, which this
table does. Slice behavior at the picked conf (t4, small) should be
re-checked before freezing; per-slice numbers exist in the run4 summary.</p>"""
    if out_html:
        Path(out_html).write_text(html)
    return printed


def selftest():
    curves = {}
    for i, cf in enumerate([0.1, 0.3, 0.5, 0.7]):
        curves[str(cf)] = {
            "person_covered90_rate": 0.95 - 0.1 * i,
            "person_uncov_mean": 0.05 + 0.05 * i,
            "mean_exposure": 0.02 + 0.01 * i,
            "falseblur_img_rate": 0.20 - 0.06 * i,
            "fb_area_mean": 0.04 - 0.01 * i,
            "fp_boxes_per_100": 8 - 2 * i}
    summary = {"protocol": {"anchor_conf": 0.5, "n_images": 4},
               "modes": {"male": {"curves": {"m": curves}}}}
    rows = sweep_rows(curves, curves["0.5"])
    bj, bu = select(rows)
    js = {c: r["J"] for c, r in rows}
    assert bj[0] == max(js, key=lambda c: js[c]), (bj, js)
    assert abs(js[0.1] - (0.95 - 0.20)) < 1e-9
    out = Path("/tmp/_wt_selftest.html")
    printed = build(summary, out, None)
    assert printed[0][1][0] == bj[0]
    t = out.read_text()
    assert "Selection summary" in t and "class=pick" in t
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--out-html", default="WOMAN_THRESHOLD_REPORT.html")
    ap.add_argument("--out-svg", default="chart_woman_J.svg")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    summary = json.loads(Path(args.summary).read_text())
    printed = build(summary, args.out_html, args.out_svg)
    print(f"{'model':16s} {'pickJ':>6s} {'cov90%':>7s} {'Epers%':>7s} "
          f"{'FBimg%':>7s} {'FP/100':>7s} {'J':>7s} {'pickU':>6s}")
    for n, bj, bu in printed:
        a = bj[1]
        print(f"{n:16s} {bj[0]:>6.2f} "
              f"{100 * a['person_covered90_rate']:>7.1f} "
              f"{100 * a['person_uncov_mean']:>7.1f} "
              f"{100 * a['falseblur_img_rate']:>7.1f} "
              f"{a['fp_boxes_per_100']:>7.2f} {a['J']:>7.4f} {bu[0]:>6.2f}")
    print(f"[woman_threshold] wrote {args.out_html} + {args.out_svg}")


if __name__ == "__main__":
    main()
