#!/usr/bin/env python3
"""
Confidence threshold vs recall / precision for `y26n_humanshaped_v2`, fp32 `.pt`
against two INT8 exports (plain W8A8, and W8A8 + float decode ops), replayed
offline from the floor-0.001 raw dumps with `vlm-cluster/pr_curve.py`
(`--grid-step 0.01 --agnostic`, IoU>=0.5, real FP-based precision).

Input : models/pr_quant_20260914/y26n_humanshaped_v2_<tag>_sz<SZ>_<ds>.json
Output: experiments/assets/pr_quant/<ds>_sz<SZ>_<target>.svg   (one figure per file)
        experiments/assets/pr_quant/SUMMARY.md                  (tables for the doc)

    python3 experiments/make_charts_pr_quant.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
IN = ROOT / "models" / "pr_quant_20260914"
OUT = HERE / "assets" / "pr_quant"
RUN = "y26n_humanshaped_v2"

INK, MUTE, GRID = "#1a1a2e", "#7a7a8c", "#e4e4ea"
SERIES = [  # tag, label, colour (validated: blue / orange / aqua, dataviz palette)
    ("pt",   "fp32 .pt (reference)",          "#2a78d6"),
    ("int8", "INT8 W8A8 TFLite",              "#eb6834"),
    ("fdec", "INT8 + float decode ops (fix)", "#1baf7a"),
]
EXTRA = [("fp32", "fp32 TFLite"), ("fp16", "FP16 TFLite")]   # tables only
SHIP_THR = 0.45
DS_LABEL = {"holdout": "haramblur_holdout (QA set, 11,494 images)",
            "spotval": "Spotlight-val (4,232 images)"}
TARGETS = [("agnostic", "any person (class-merged)"),
           ("Woman", "Woman"), ("Man", "Man"), ("Child", "Child")]


def load(tag, sz, ds):
    f = IN / f"{RUN}_{tag}_sz{sz}_{ds}.json"
    return json.loads(f.read_text()) if f.exists() else None


def series_at(d, target):
    """{thr(float): (recall, precision)} from at_thresholds."""
    node = d["agnostic"] if target == "agnostic" else d["classes"][target]
    return {float(k): (v["recall"], v["precision"]) for k, v in node["at_thresholds"].items()}, node["n_gt"]


def fmt(v):
    return f"{v:.3f}"


def ceiling(d, target):
    """Highest confidence any detection reached (the curve is conf-descending)."""
    node = d["agnostic"] if target == "agnostic" else d["classes"][target]
    return node["curve"][0]["conf"] if node["curve"] else 0.0


# ---------------------------------------------------------------- svg helpers
def axes(s, x0, y0, w, h, ymin, ymax, ylabel, ytick, yfmt):
    s.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="{GRID}"/>')
    n = int(round((ymax - ymin) / ytick))
    for i in range(n + 1):
        v = ymin + i * ytick
        y = y0 + h - (v - ymin) / (ymax - ymin) * h
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-6}" y="{y+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTE}">{yfmt(v)}</text>')
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = x0 + t * w
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}" stroke="{GRID}"/>')
        s.append(f'<text x="{x:.1f}" y="{y0+h+15}" text-anchor="middle" font-size="10.5" fill="{MUTE}">{t:.1f}</text>')
    s.append(f'<text x="{x0+w/2}" y="{y0+h+32}" text-anchor="middle" font-size="11.5" fill="{INK}">confidence threshold</text>')
    s.append(f'<text transform="translate({x0-40},{y0+h/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="{INK}">{ylabel}</text>')
    # shipped threshold
    xs = x0 + SHIP_THR * w
    s.append(f'<line x1="{xs:.1f}" y1="{y0}" x2="{xs:.1f}" y2="{y0+h}" stroke="{MUTE}" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{xs+4:.1f}" y="{y0+12}" font-size="10" fill="{MUTE}">shipped 0.45</text>')


def polyline(s, pts, x0, y0, w, h, ymin, ymax, colour, dash=""):
    d = " ".join(f"{x0 + t*w:.1f},{y0 + h - (v-ymin)/(ymax-ymin)*h:.1f}" for t, v in pts)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"{extra}/>')


def marker(s, t, v, x0, y0, w, h, ymin, ymax, colour, label=None):
    x = x0 + t * w
    y = y0 + h - (v - ymin) / (ymax - ymin) * h
    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}" stroke="white" stroke-width="2"/>')
    if label:
        s.append(f'<text x="{x+7:.1f}" y="{y+4:.1f}" font-size="10.5" font-weight="600" fill="{INK}">{label}</text>')


def legend(s, x, y, rows):
    for i, (label, colour, dash) in enumerate(rows):
        yy = y + i * 17
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{x}" y1="{yy}" x2="{x+22}" y2="{yy}" stroke="{colour}" stroke-width="2.5"{extra}/>')
        s.append(f'<text x="{x+28}" y="{yy+4}" font-size="11.5" fill="{INK}">{label}</text>')


# ---------------------------------------------------------------- one figure
def figure(sz, ds, target, tlabel, data, ceil):
    """data: {tag: ({thr: (rec, prec)}, n_gt)}; ceil: {tag: max conf} for the three main series."""
    W, H = 1180, 520
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    n_gt = data["pt"][1]
    s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">'
             f'{RUN} @{sz} — threshold vs recall &amp; precision, fp32 vs two INT8 exports · {tlabel}</text>')
    s.append(f'<text x="{W/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">'
             f'{DS_LABEL[ds]} · {n_gt:,} GT boxes · IoU ≥ 0.5 · real FP-based precision · '
             f'offline replay of the floor-0.001 raw dumps (vlm-cluster/pr_curve.py, 0.01 grid)</text>')

    pw, ph, top = 300, 300, 95
    x1, x2, x3 = 70, 440, 810
    thr = sorted(t for t in data["pt"][0] if 0.01 <= t <= 0.99)

    # recall panel
    axes(s, x1, top, pw, ph, 0, 1, "recall", 0.2, lambda v: f"{v:.1f}")
    s.append(f'<text x="{x1}" y="{top-8}" font-size="13" font-weight="700" fill="{INK}">Recall</text>')
    for tag, label, col in SERIES:
        d = data[tag][0]
        polyline(s, [(t, d[t][0]) for t in thr], x1, top, pw, ph, 0, 1, col)
    for tag, label, col in SERIES:
        r = data[tag][0][SHIP_THR][0]
        marker(s, SHIP_THR, r, x1, top, pw, ph, 0, 1, col)
    s.append(f'<text x="{x1+SHIP_THR*pw+8:.1f}" y="{top+ph-(data["pt"][0][SHIP_THR][0])*ph+22:.1f}" font-size="10.5" fill="{INK}">'
             f'@0.45: ' + " / ".join(fmt(data[t][0][SHIP_THR][0]) for t, _, _ in SERIES) + '</text>')
    for px in (x1, x2):
        for i, (tag, label, col) in enumerate(SERIES[1:]):
            c = ceil[tag]
            if c < 0.99:
                xc = px + c * pw
                s.append(f'<line x1="{xc:.1f}" y1="{top}" x2="{xc:.1f}" y2="{top+ph}" stroke="{col}" stroke-width="1" stroke-dasharray="2 3"/>')
                s.append(f'<text x="{xc-3:.1f}" y="{top+ph-8-i*13:.1f}" text-anchor="end" font-size="9.5" fill="{col}">no boxes ≥ {c:.3f}</text>')

    # precision panel
    axes(s, x2, top, pw, ph, 0, 1, "precision", 0.2, lambda v: f"{v:.1f}")
    s.append(f'<text x="{x2}" y="{top-8}" font-size="13" font-weight="700" fill="{INK}">Precision</text>')
    for tag, label, col in SERIES:
        d = data[tag][0]
        polyline(s, [(t, d[t][1]) for t in thr], x2, top, pw, ph, 0, 1, col)
    for tag, label, col in SERIES:
        p = data[tag][0][SHIP_THR][1]
        marker(s, SHIP_THR, p, x2, top, pw, ph, 0, 1, col)
    s.append(f'<text x="{x2+SHIP_THR*pw+8:.1f}" y="{top+ph-(data["pt"][0][SHIP_THR][1])*ph+26:.1f}" font-size="10.5" fill="{INK}">'
             f'@0.45: ' + " / ".join(fmt(data[t][0][SHIP_THR][1]) for t, _, _ in SERIES) + '</text>')

    # delta panel (INT8 − fp32 .pt), magnified
    dmax = 0.0
    deltas = {}
    for tag, label, col in SERIES[1:]:
        ok = [t for t in thr if t <= ceil[tag]]          # above the ceiling the export returns nothing
        dr = [(t, data[tag][0][t][0] - data["pt"][0][t][0]) for t in ok]
        dp = [(t, data[tag][0][t][1] - data["pt"][0][t][1]) for t in ok]
        deltas[tag] = (dr, dp)
        dmax = max(dmax, max(abs(v) for _, v in dr + dp))
    lim = next(l for l in (0.05, 0.10, 0.20, 0.50, 1.0) if dmax <= l)
    axes(s, x3, top, pw, ph, -lim, lim, "difference vs fp32 .pt (points)", lim / 2.5,
         lambda v: f"{v*100:+.0f}")
    y0 = top + ph / 2
    s.append(f'<line x1="{x3}" y1="{y0}" x2="{x3+pw}" y2="{y0}" stroke="{MUTE}" stroke-width="1.5"/>')
    s.append(f'<text x="{x3}" y="{top-8}" font-size="13" font-weight="700" fill="{INK}">Quantization effect (INT8 − fp32 .pt)</text>')
    for i, (tag, label, col) in enumerate(SERIES[1:]):
        dr, dp = deltas[tag]
        polyline(s, dr, x3, top, pw, ph, -lim, lim, col)
        polyline(s, dp, x3, top, pw, ph, -lim, lim, col, dash="5 4")
        if ceil[tag] < 0.99:
            xc = x3 + ceil[tag] * pw
            s.append(f'<line x1="{xc:.1f}" y1="{top}" x2="{xc:.1f}" y2="{top+ph}" stroke="{col}" stroke-width="1" stroke-dasharray="2 3"/>')
            s.append(f'<text x="{xc-3:.1f}" y="{top+ph-8-i*13:.1f}" text-anchor="end" font-size="9.5" fill="{col}">ceiling {ceil[tag]:.3f}</text>')
    legend(s, x3 + 8, top + 22, [("solid = recall", INK, ""), ("dashed = precision", INK, "5 4")])

    legend(s, x1, top + ph + 52, [(label, col, "") for _, label, col in SERIES])
    # summary line
    for i, (tag, label, _) in enumerate(SERIES[1:]):
        lo, hi = 0.05, min(0.95, ceil[tag])
        s.append(f'<text x="{x2}" y="{top+ph+56+i*16}" font-size="11" fill="{MUTE}">{label}: '
                 f'max |Δrecall| {max(abs(v) for t, v in deltas[tag][0] if lo <= t <= hi)*100:.1f} pts, '
                 f'max |Δprecision| {max(abs(v) for t, v in deltas[tag][1] if lo <= t <= hi)*100:.1f} pts '
                 f'over thr {lo:.2f}–{hi:.2f}; '
                 f'@0.45 Δrecall {(data[tag][0][SHIP_THR][0]-data["pt"][0][SHIP_THR][0])*100:+.1f}, '
                 f'Δprecision {(data[tag][0][SHIP_THR][1]-data["pt"][0][SHIP_THR][1])*100:+.1f}</text>')
    s.append(f'<text x="{x2}" y="{top+ph+92}" font-size="11" fill="{MUTE}">'
             f'INT8 confidences sit on a coarse ladder (int8 class-logit tensor, ≈0.3 logits/step) and stop at a ceiling; above it the export returns no boxes.</text>')
    s.append('</svg>')
    return "\n".join(s)


def table(sz, ds, target, tlabel, all_data):
    tags = [t for t, _, _ in SERIES] + [t for t, _ in EXTRA]
    tags = [t for t in tags if t in all_data]
    lab = dict([(t, l) for t, l, _ in SERIES] + EXTRA)
    lines = [f"**{RUN} @{sz} · {ds} · {tlabel}** (n_gt = {all_data['pt'][1]:,}) — recall / precision",
             "", "| thr | " + " | ".join(f"`{lab[t]}`" for t in tags) + " |",
             "|---|" + "---|" * len(tags)]
    for thr in [round(0.05 * i, 2) for i in range(1, 20)]:
        row = [f"{thr:.2f}"] + [f"{all_data[t][0][thr][0]:.3f} / {all_data[t][0][thr][1]:.3f}" for t in tags]
        if thr == SHIP_THR:
            row = [f"**{c}**" for c in row]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# Threshold vs recall/precision — `{RUN}` across export precisions (2026-09-14)", "",
          "Generated by `experiments/make_charts_pr_quant.py` from `models/pr_quant_20260914/*.json`.", ""]
    made = 0
    for ds in ("holdout", "spotval"):
        for sz in (640, 416, 320):
            raw = {t: load(t, sz, ds) for t in [x for x, _, _ in SERIES] + [x for x, _ in EXTRA]}
            raw = {t: d for t, d in raw.items() if d}
            if not all(t in raw for t, _, _ in SERIES):
                continue
            for target, tlabel in TARGETS:
                all_data = {t: series_at(d, target) for t, d in raw.items()}
                svg = figure(sz, ds, target, tlabel, {t: all_data[t] for t, _, _ in SERIES},
                             {t: ceiling(raw[t], target) for t, _, _ in SERIES})
                f = OUT / f"{ds}_sz{sz}_{target}.svg"
                f.write_text(svg)
                made += 1
                md.append(table(sz, ds, target, tlabel, all_data))
    (OUT / "SUMMARY.md").write_text("\n".join(md))
    print(f"wrote {made} SVGs + SUMMARY.md to {OUT}")


if __name__ == "__main__":
    main()
