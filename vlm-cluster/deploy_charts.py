#!/usr/bin/env python3
"""
Standalone SVG charts from a deploy_compare score summary.json — one graph per
file (owner convention: .svg files, never a bundled webpage).

  chart_tradeoff_<mode>.svg   M-1 exposure pass rate vs false-blur image rate,
                              one curve per model across the conf grid; the
                              incumbent's anchor point and each model's
                              matched operating point are marked. Up-and-left
                              is better; a curve strictly above another
                              dominates it at every threshold.
  chart_slices_<mode>.svg     M-1 per slice at the matched operating points,
                              grouped horizontal bars.

    python3 deploy_charts.py --summary summary.json --out _deploycmp_review/

Colors are the Okabe-Ito subset validated for CVD (validate_palette.js:
adjacent-pair floor met with direct labels as secondary encoding). Self-
contained: explicit colors, own background, no external anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BG = "#fcfcfb"
INK, MUTED, GRID = "#222222", "#555555", "#e8e8e6"
MODEL_COLOR = {}          # filled in model order from the summary
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
FONT = "font-family='system-ui,Helvetica,Arial' "


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")


def tradeoff_svg(mode, mdata, order):
    W, H = 760, 520
    ML, MR, MT, MB = 64, 170, 56, 56
    pw, ph = W - ML - MR, H - MT - MB
    sample = next(iter(mdata["curves"][order[0]].values()))
    v2 = "fb_area_mean" in sample
    xkey = "fb_area_mean" if v2 else "falseblur_img_rate"
    ykey = "mean_exposure" if v2 else "m1_pass_rate"
    xmax = max(100 * r[xkey]
               for n in order for r in mdata["curves"][n].values()) * 1.08
    xmax = max(xmax, 6 if not v2 else 2.5)
    ymax = (max(100 * r[ykey] for n in order
                for r in mdata["curves"][n].values()) * 1.1 if v2 else 100.0)

    def X(v):
        return ML + pw * v / xmax

    def Y(v):
        # v2: exposure, lower is better -> plot low = good (near baseline)
        return MT + ph * (v / ymax if v2 else 1 - v / 100.0)

    title = ("Exposed area (E-img) vs false-blur area" if v2
             else "M-1 exposure pass rate vs false blur")
    ylab = ("unblurred-woman screen area (mean %)" if v2
            else "M-1 exposure pass rate (%)")

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='{BG}'/>",
         f"<text x='{ML}' y='24' {FONT}font-size='15' fill='{INK}' "
         f"font-weight='600'>{title} — {esc(mode)} audience</text>",
         f"<text x='{ML}' y='42' {FONT}font-size='11' fill='{MUTED}'>"
         f"each point = one confidence cut; "
         f"{'down' if v2 else 'up'}-and-left is better; "
         f"◆ = matched operating point, ○ = incumbent anchor 0.45</text>"]
    gys = ([round(ymax * i / 5, 2) for i in range(6)] if v2
           else list(range(0, 101, 20)))
    for gy in gys:
        p.append(f"<line x1='{ML}' y1='{Y(gy)}' x2='{ML + pw}' y2='{Y(gy)}' "
                 f"stroke='{GRID}'/>")
        p.append(f"<text x='{ML - 8}' y='{Y(gy) + 4}' {FONT}font-size='11' "
                 f"fill='{MUTED}' text-anchor='end'>{gy}</text>")
    nx = 6
    for i in range(nx + 1):
        gv = xmax * i / nx
        p.append(f"<line x1='{X(gv)}' y1='{MT}' x2='{X(gv)}' y2='{MT + ph}' "
                 f"stroke='{GRID}'/>")
        p.append(f"<text x='{X(gv)}' y='{MT + ph + 18}' {FONT}font-size='11' "
                 f"fill='{MUTED}' text-anchor='middle'>{gv:.1f}</text>")
    xlab = ("falsely-blurred screen area (mean %), lower is better" if v2
            else "false-blur image rate (% of images), lower is better")
    p.append(f"<text x='{ML + pw / 2}' y='{H - 14}' {FONT}font-size='12' "
             f"fill='{INK}' text-anchor='middle'>{xlab}</text>")
    p.append(f"<text x='18' y='{MT + ph / 2}' {FONT}font-size='12' "
             f"fill='{INK}' text-anchor='middle' "
             f"transform='rotate(-90 18 {MT + ph / 2})'>{ylab}</text>")

    for name in order:
        col = MODEL_COLOR[name]
        curve = mdata["curves"][name]
        pts = sorted(((100 * r[xkey], 100 * r[ykey],
                       float(cf)) for cf, r in curve.items()),
                     key=lambda t: t[2])
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}"
                        for i, (x, y, _) in enumerate(pts))
        p.append(f"<path d='{path}' fill='none' stroke='{col}' "
                 f"stroke-width='2'/>")
        mcf = mdata["matched_conf"][name]
        mr = curve[[k for k in curve if float(k) == mcf][0]]
        mx, my = X(100 * mr[xkey]), Y(100 * mr[ykey])
        p.append(f"<path d='M{mx},{my - 6} L{mx + 6},{my} L{mx},{my + 6} "
                 f"L{mx - 6},{my} Z' fill='{col}' stroke='{BG}' "
                 f"stroke-width='2'/>")
        if "0.45" in curve:
            ar = curve["0.45"]
            ax, ay = X(100 * ar[xkey]), Y(100 * ar[ykey])
            p.append(f"<circle cx='{ax}' cy='{ay}' r='5' fill='none' "
                     f"stroke='{col}' stroke-width='2'/>")
        lx, ly = pts[0][0], pts[0][1]           # lowest-conf end = rightmost?
        ex, ey = X(lx) + 8, Y(ly)
        p.append(f"<circle cx='{ML + pw + 14}' cy='{MT + 14 + 22 * order.index(name)}' "
                 f"r='5' fill='{col}'/>")
        p.append(f"<text x='{ML + pw + 24}' y='{MT + 18 + 22 * order.index(name)}' "
                 f"{FONT}font-size='12' fill='{INK}'>{esc(name)} "
                 f"@{mcf:.2f}</text>")
    p.append("</svg>")
    return "\n".join(p)


def slices_svg(mode, mdata, order, slice_order):
    sample = mdata["models"][order[0]]["slices"]["all"]
    v2 = "person_uncov_mean" in sample
    vkey = "person_uncov_mean" if v2 else "m1_pass_rate"
    stitle = ("E-person (mean % of pixels exposed, lower is better)"
              if v2 else "M-1")
    rows = [t for t in slice_order
            if any(mdata["models"][n]["slices"].get(t, {}).get(vkey)
                   is not None for n in order)]
    bar, gap, group_gap = 13, 2, 16
    gh = len(order) * (bar + gap) + group_gap
    W = 760
    ML, MR, MT, MB = 190, 60, 76, 30
    H = MT + MB + len(rows) * gh
    pw = W - ML - MR

    def X(v):
        return ML + pw * v / 100.0

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' "
         f"viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='{BG}'/>",
         f"<text x='{ML}' y='24' {FONT}font-size='15' fill='{INK}' "
         f"font-weight='600'>{stitle} per slice — {esc(mode)} audience, "
         f"matched operating points</text>"]
    for i, name in enumerate(order):
        p.append(f"<circle cx='{ML + 4 + 178 * (i % 2)}' "
                 f"cy='{40 + 16 * (i // 2)}' r='5' "
                 f"fill='{MODEL_COLOR[name]}'/>")
        p.append(f"<text x='{ML + 14 + 178 * (i % 2)}' "
                 f"y='{44 + 16 * (i // 2)}' {FONT}font-size='11' "
                 f"fill='{INK}'>{esc(name)}</text>")
    for gv in range(0, 101, 25):
        p.append(f"<line x1='{X(gv)}' y1='{MT}' x2='{X(gv)}' "
                 f"y2='{H - MB}' stroke='{GRID}'/>")
        p.append(f"<text x='{X(gv)}' y='{MT - 6}' {FONT}font-size='10' "
                 f"fill='{MUTED}' text-anchor='middle'>{gv}</text>")
    y = MT + 6
    for tag in rows:
        p.append(f"<text x='{ML - 8}' y='{y + (gh - group_gap) / 2 + 4}' "
                 f"{FONT}font-size='11' fill='{INK}' text-anchor='end'>"
                 f"{esc(tag)}</text>")
        for name in order:
            v = mdata["models"][name]["slices"].get(tag, {}).get(vkey)
            col = MODEL_COLOR[name]
            if v is None:
                p.append(f"<text x='{ML + 4}' y='{y + bar - 3}' {FONT}"
                         f"font-size='9' fill='{MUTED}'>n/a</text>")
            else:
                wpx = max(0.1, (X(100 * v) - ML))
                r = min(3, wpx)
                p.append(
                    f"<path d='M{ML},{y} h{wpx - r:.1f} q{r},0 {r},{r} "
                    f"v{bar - 2 * r} q0,{r} -{r},{r} h-{wpx - r:.1f} Z' "
                    f"fill='{col}'/>")
                p.append(f"<text x='{ML + wpx + 5:.1f}' y='{y + bar - 3}' "
                         f"{FONT}font-size='9' fill='{MUTED}'>"
                         f"{100 * v:.0f}</text>")
            y += bar + gap
        y += group_gap
    p.append("</svg>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.summary).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode, mdata in s["modes"].items():
        order = list(mdata["curves"].keys())
        for i, n in enumerate(order):
            MODEL_COLOR.setdefault(n, PALETTE[i % len(PALETTE)])
        pref = "w_" if mode == "male" else "m_"
        all_tags = sorted(mdata["models"][order[0]]["slices"])
        slice_order = (
            [t for t in all_tags if t.startswith("collection:")]
            + [t for t in all_tags if t.startswith(f"{pref}exposure:")]
            + [f"{pref}size:large", f"{pref}size:med", f"{pref}size:small"])
        (out / f"chart_tradeoff_{mode}.svg").write_text(
            tradeoff_svg(mode, mdata, order))
        (out / f"chart_slices_{mode}.svg").write_text(
            slices_svg(mode, mdata, order, slice_order))
        print(f"[charts] {mode}: chart_tradeoff_{mode}.svg + "
              f"chart_slices_{mode}.svg -> {out}")


if __name__ == "__main__":
    main()
