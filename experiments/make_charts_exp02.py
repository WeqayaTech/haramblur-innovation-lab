#!/usr/bin/env python3
"""Generate SVG charts for EXP-2026-02 from the final n=5,000 SAM3 auto-labeler eval."""
import pathlib

OUT = pathlib.Path("/Users/mostafaelkabir/Desktop/HaramBlur/Innovation-lab/experiments/assets")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTE, GRID = "#1a1d25", "#8899aa", "#dfe3ea"
GREEN, AMBER, RED, BLUE = "#27ae60", "#f39c12", "#e74c3c", "#4f8ef7"

# ---------------------------------------------------------------------------
# 1. SAM3 "child" call rate by TRUE age (its implicit, unlabeled boundary)
# ---------------------------------------------------------------------------
age_buckets = [
    ("0", 99.5, 420), ("5", 99.3, 440), ("10", 88.8, 587), ("15", 36.3, 705),
    ("20", 5.8, 411), ("25", 1.5, 400), ("30", 0.0, 222), ("35", 0.0, 249),
    ("40", 0.0, 250), ("45", 0.0, 266), ("50", 0.0, 200), ("55", 0.0, 220),
    ("60", 0.0, 172), ("65", 0.0, 181), ("70", 0.0, 124), ("75", 0.0, 51),
    ("80", 0.0, 28), ("85", 0.0, 9),
]
W, H = 940, 440
x0, x1, y0, y1 = 64, 904, 70, 370
n = len(age_buckets); slot = (x1 - x0) / n; bw = slot * 0.72
def yy(p): return y1 - (p / 100) * (y1 - y0)
def color(p): return GREEN if p >= 90 else (AMBER if p >= 20 else RED)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">SAM3 auto-labeler: "child" call rate by TRUE age</text>')
s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="13" fill="{MUTE}">SAM3 has no numeric age cutoff — this is its implicit boundary from the "child" text prompt alone. n=4,936 detected</text>')
for g in (0, 25, 50, 75, 100):
    yg = yy(g)
    s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}" stroke-width="1"/>')
    s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
for i, (lab, p, nn) in enumerate(age_buckets):
    bx = x0 + i * slot + (slot - bw) / 2; by = yy(p); bh = y1 - by
    s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{color(p)}"/>')
    s.append(f'<text x="{bx+bw/2:.1f}" y="{by-4:.1f}" text-anchor="middle" font-size="9.5" fill="{INK}">{p:.0f}</text>')
    s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+14:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{lab}</text>')
s.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+34:.1f}" text-anchor="middle" font-size="12" fill="{INK}">true age (5-year buckets, start age shown)</text>')
vx = x0 + 3 * slot
s.append(f'<rect x="{vx-4:.1f}" y="{yy(36.3)-2:.1f}" width="{slot:.1f}" height="{y1-yy(36.3)+2:.1f}" fill="none" stroke="{AMBER}" stroke-width="2" stroke-dasharray="4 3"/>')
s.append(f'<text x="{x0+5.4*slot:.1f}" y="{yy(60):.1f}" text-anchor="start" font-size="12" font-weight="700" fill="{AMBER}">teen gray zone (15–19):</text>')
s.append(f'<text x="{x0+5.4*slot:.1f}" y="{yy(60)+16:.1f}" text-anchor="start" font-size="12" fill="{MUTE}">SAM3 is a coin-flip here</text>')
lx = x0
for c, t in ((GREEN, "≥90% called Child"), (AMBER, "20–90% (gray zone)"), (RED, "under 20%")):
    s.append(f'<rect x="{lx}" y="{H-24}" width="12" height="12" rx="2" fill="{c}"/>')
    s.append(f'<text x="{lx+16}" y="{H-14}" font-size="11" fill="{MUTE}">{t}</text>'); lx += 165
s.append('</svg>')
(OUT / "sam_child_call_rate_by_age.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# helper: confusion-matrix heatmap (same pattern as EXP-2026-01)
# ---------------------------------------------------------------------------
def conf_svg(title, subtitle, rows, cols, M, fname, w=680):
    nR, nC = len(rows), len(cols)
    cell = 100; lx = 150; ty = 96; H = ty + nR * cell + 70
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{H}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">{title}</text>')
    s.append(f'<text x="{w/2}" y="50" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>')
    s.append(f'<text x="{lx+nC*cell/2}" y="80" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">SAM3 predicted →</text>')
    for ci, c in enumerate(cols):
        s.append(f'<text x="{lx+ci*cell+cell/2}" y="{ty-6}" text-anchor="middle" font-size="12" fill="{INK}">{c}</text>')
    for ri, r in enumerate(rows):
        rowtot = sum(M[ri]) or 1
        s.append(f'<text x="{lx-10}" y="{ty+ri*cell+cell/2+4}" text-anchor="end" font-size="12" fill="{INK}">{r}</text>')
        for ci in range(nC):
            frac = M[ri][ci] / rowtot
            base = GREEN if rows[ri] == cols[ci] else RED
            op = 0.12 + 0.83 * frac
            cx = lx + ci * cell; cy = ty + ri * cell
            s.append(f'<rect x="{cx}" y="{cy}" width="{cell-4}" height="{cell-4}" rx="4" fill="{base}" fill-opacity="{op:.2f}" stroke="{GRID}"/>')
            txt = "#ffffff" if op > 0.55 else INK
            s.append(f'<text x="{cx+(cell-4)/2}" y="{cy+(cell-4)/2-2}" text-anchor="middle" font-size="15" font-weight="700" fill="{txt}">{M[ri][ci]}</text>')
            s.append(f'<text x="{cx+(cell-4)/2}" y="{cy+(cell-4)/2+14}" text-anchor="middle" font-size="10.5" fill="{txt}">{100*frac:.0f}%</text>')
    s.append(f'<text x="14" y="{ty+nR*cell/2}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}" transform="rotate(-90 14 {ty+nR*cell/2})">Ground truth →</text>')
    s.append(f'<text x="{lx}" y="{ty+nR*cell+34}" font-size="11" fill="{MUTE}">Green = correct (diagonal) · Red = error · shade = share of that GT row · % is row-normalized</text>')
    s.append('</svg>')
    (OUT / fname).write_text("\n".join(s))

conf_rows = ["Woman", "Man", "Child"]
conf_M = [[1647, 21, 252], [17, 1566, 246], [13, 10, 1164]]
conf_svg("SAM3 auto-labeler: 3-class confusion matrix",
         "Gender swaps (Woman↔Man) are only 1.2% — nearly all error is adults labeled Child. n=4,936",
         conf_rows, conf_rows, conf_M, "sam_confusion_matrix.svg", w=680)


# ---------------------------------------------------------------------------
# 2. Accuracy at each candidate "child" cutoff (9 / 12 / 17)
# ---------------------------------------------------------------------------
def threshold_sweep():
    groups = [("under 10", "child = 9 and under", 83.6, 99.4, 80.2),
              ("puberty (current production)", "child = 12 and under", 89.4, 98.1, 86.7),
              ("minor", "child = 17 and under", 91.4, 81.2, 98.2)]
    series = [("overall accuracy", INK), ("child recall", GREEN), ("adult recall", BLUE)]
    W, H = 780, 470; x0, x1, y0, y1 = 72, 708, 92, 384
    ng = len(groups); gslot = (x1 - x0) / ng; nb = 3; bw = gslot * 0.22
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">SAM3 accuracy at each candidate "child" cutoff</text>')
    s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12" fill="{MUTE}">Raising the cutoff to 17 buys accuracy by trading away real-teen recall (81%) — the current 12 cutoff keeps child recall highest.</text>')
    for g in (0, 25, 50, 75, 100):
        yg = yy(g); s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    for gi, (name, sub, acc, ck, ca) in enumerate(groups):
        vals = [acc, ck, ca]; gc = x0 + gi * gslot + gslot / 2; start = gc - (nb * bw + (nb - 1) * 7) / 2
        for bi, ((lab, col), v) in enumerate(zip(series, vals)):
            bx = start + bi * (bw + 7); by = yy(v)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="2" fill="{col}"/>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="10.5" font-weight="700" fill="{INK}">{v:.0f}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+18:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{INK}">{name}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+34:.1f}" text-anchor="middle" font-size="10.5" fill="{MUTE}">{sub}</text>')
    lx = x0
    for lab, col in series:
        s.append(f'<rect x="{lx}" y="{H-24}" width="12" height="12" rx="2" fill="{col}"/><text x="{lx+16}" y="{H-14}" font-size="11" fill="{MUTE}">{lab}</text>'); lx += 185
    s.append('</svg>')
    (OUT / "sam_child_threshold_sweep.svg").write_text("\n".join(s))
threshold_sweep()

for f in ("sam_child_call_rate_by_age.svg", "sam_confusion_matrix.svg", "sam_child_threshold_sweep.svg"):
    print("wrote", OUT / f, (OUT / f).stat().st_size, "bytes")
