#!/usr/bin/env python3
"""Charts for EXP-2026-10 (Spotlight pipeline v1) -> SVG + PNG (Chrome, 2x).

All numbers are the 2026-07-26 full run (prompt sha b30e8ec997f6, fresh raw
SAM3 labels, outline+upscale-320 crops, child cutoff <=12):
  crowd   150 imgs / 3,853 dets  — CrowdHuman vbox GT
  lagenda 150 imgs / 1,331 dets  — human apparent-age + gender GT (151 scored)
  PASS    500 imgs /    52 dets  — verified person-free scenes
SAM3-alone baselines: EXP-2026-02 (88.7% 3-class, ~10.5% adult->Child).
"""
import pathlib
import subprocess

OUT = pathlib.Path("/Users/mostafaelkabir/Desktop/HaramBlur/Innovation-lab/experiments/assets")
PNG = OUT / "png" / "EXP-2026-10"
OUT.mkdir(parents=True, exist_ok=True)
PNG.mkdir(parents=True, exist_ok=True)

INK, MUTE, GRID = "#1a1d25", "#8899aa", "#dfe3ea"
GREEN, RED, BLUE, AMBER = "#27ae60", "#e74c3c", "#2858dc", "#f39c12"
GREY = "#9aa7b5"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def svg_open(w, h, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         f'<rect width="{w}" height="{h}" fill="white"/>',
         f'<text x="{w/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{title}</text>',
         f'<text x="{w/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>']
    return s


# ---------------------------------------------------------------------------
# 1. The age gradient — the product-critical result
# ---------------------------------------------------------------------------
def age_gradient():
    W, H = 900, 520
    x0, x1, y0, y1 = 70, 860, 90, 380
    bands = [("0-12", 32, 100.0), ("13-15", 17, 76.5), ("16-17", 7, 14.3),
             ("18-19", 5, 0.0), ("20-29", 27, 0.0), ("30+", 63, 0.0)]
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    s = svg_open(W, H, "Who gets labeled &quot;Child&quot; — and nobody over 17 does",
                 "LAGENDA arm, 151 people with human apparent-age ground truth. "
                 "Bars = share of that age band the pipeline labeled Child.")
    for g in (0, 25, 50, 75, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    n = len(bands); slot = (x1 - x0) / n; bw = min(78.0, slot * 0.52)
    for i, (lab, cnt, pct) in enumerate(bands):
        cx = x0 + i * slot + slot / 2
        col = GREEN if (lab == "0-12" or pct == 0.0) else AMBER
        by = yy(pct) if pct > 0 else y1
        s.append(f'<rect x="{cx-bw/2:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{max(2,y1-by):.1f}" rx="3" fill="{col}"/>')
        s.append(f'<text x="{cx:.1f}" y="{by-8:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="{INK}">{pct:.1f}%</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+18:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{lab}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+33:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">n={cnt}</text>')
    xa = x0 + 3 * slot
    s.append(f'<line x1="{xa:.1f}" y1="{y0-8}" x2="{xa:.1f}" y2="{y1+40}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{xa+8:.1f}" y="{y0+4:.1f}" font-size="12.5" font-weight="700" fill="{RED}">18+ : 0 of 95 labeled Child</text>')
    s.append(f'<text x="{xa+8:.1f}" y="{y0+22:.1f}" font-size="11.5" fill="{MUTE}">no adult escapes the blur</text>')
    s.append(f'<text x="{x0}" y="{y1+72:.1f}" font-size="12" fill="{INK}">'
             f'<tspan font-weight="700" fill="{GREEN}">Green</tspan> = correct outcome: every real child caught (32/32), every adult 18+ kept out of the Child class.</text>')
    s.append(f'<text x="{x0}" y="{y1+90:.1f}" font-size="12" fill="{INK}">'
             f'<tspan font-weight="700" fill="{AMBER}">Amber</tspan> = the teen band (13-17). LAGENDA ages are human <tspan font-style="italic">apparent-age</tspan> guesses (&#177;2-3 yrs), so these are annotation-noise disagreements.</text>')
    s.append('</svg>')
    (OUT / "exp10_age_gradient.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 2. Pipeline vs SAM3 alone
# ---------------------------------------------------------------------------
def vs_sam3():
    W, H = 900, 500
    x0, x1, y0, y1 = 70, 860, 90, 370
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    s = svg_open(W, H, "Spotlight pipeline vs raw SAM3 labels",
                 "SAM3-alone figures from EXP-2026-02 (different sample — a like-for-like "
                 "re-measurement is still owed). Higher is better in all four.")
    for g in (0, 25, 50, 75, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    groups = [("Label accuracy", "3-class, teen-tolerant", 88.7, 99.3),
              ("Adults NOT leaking", "18+ kept out of Child", 89.5, 100.0),
              ("Gender on adults", "when committed", 98.8, 99.05),
              ("Junk removed", "genuine non-people", 0.0, 92.0)]
    n = len(groups); slot = (x1 - x0) / n; bw = min(62.0, slot * 0.28)
    for i, (lab, sub, base, mine) in enumerate(groups):
        cx = x0 + i * slot + slot / 2
        for val, col, dx, name in ((base, GREY, -bw - 5, "SAM3"), (mine, GREEN, 5, "Spotlight")):
            by = yy(val)
            s.append(f'<rect x="{cx+dx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{max(2,y1-by):.1f}" rx="3" fill="{col}"/>')
            txt = f"{val:.1f}%" if val > 0 else "none"
            s.append(f'<text x="{cx+dx+bw/2:.1f}" y="{by-7:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{INK}">{txt}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+18:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{lab}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+33:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{sub}</text>')
    s.append(f'<rect x="{x0}" y="{H-42}" width="14" height="14" rx="2" fill="{GREY}"/>')
    s.append(f'<text x="{x0+20}" y="{H-31}" font-size="11.5" fill="{INK}">SAM3 auto-labels today (EXP-2026-02)</text>')
    s.append(f'<rect x="{x0+300}" y="{H-42}" width="14" height="14" rx="2" fill="{GREEN}"/>')
    s.append(f'<text x="{x0+320}" y="{H-31}" font-size="11.5" fill="{INK}">Spotlight v1 (EXP-2026-10, 5,236 detections)</text>')
    s.append('</svg>')
    (OUT / "exp10_vs_sam3.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 3. Per-dataset results
# ---------------------------------------------------------------------------
def per_dataset():
    W, H = 900, 470
    x0, x1, y0, y1 = 70, 860, 92, 350
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    s = svg_open(W, H, "Results per dataset — never pooled",
                 "Each dataset answers a different question. Blue = keeping real people, "
                 "red = removing non-people.")
    for g in (0, 25, 50, 75, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    bars = [("LAGENDA", "keep real people", "151/152", 99.34, BLUE),
            ("CrowdHuman", "keep real people", "2534/2573", 98.48, BLUE),
            ("LAGENDA", "gender correct", "104/105", 99.05, BLUE),
            ("PASS", "remove genuine junk", "24/26*", 92.3, RED),
            ("Object set", "reject dolls/statues", "not run", 0.0, RED)]
    n = len(bars); slot = (x1 - x0) / n; bw = min(74.0, slot * 0.5)
    for i, (ds, what, frac, val, col) in enumerate(bars):
        cx = x0 + i * slot + slot / 2
        if val > 0:
            by = yy(val)
            s.append(f'<rect x="{cx-bw/2:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="3" fill="{col}"/>')
            s.append(f'<text x="{cx:.1f}" y="{by-8:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="{INK}">{val:.1f}%</text>')
        else:
            s.append(f'<rect x="{cx-bw/2:.1f}" y="{y1-60:.1f}" width="{bw:.1f}" height="60" rx="3" fill="none" stroke="{MUTE}" stroke-width="2" stroke-dasharray="5 4"/>')
            s.append(f'<text x="{cx:.1f}" y="{y1-34:.1f}" text-anchor="middle" font-size="11" fill="{MUTE}">not</text>')
            s.append(f'<text x="{cx:.1f}" y="{y1-20:.1f}" text-anchor="middle" font-size="11" fill="{MUTE}">measured</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+18:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{ds}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+33:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{what}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+47:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{frac}</text>')
    s.append(f'<text x="{x0}" y="{y1+80:.1f}" font-size="11.5" fill="{MUTE}">'
             '* PASS raw kill rate is 46% (24/52), but owner adjudication found all but ~2 survivors were real people or depictions the benchmark missed;</text>')
    s.append(f'<text x="{x0}" y="{y1+96:.1f}" font-size="11.5" fill="{MUTE}">'
             'on genuine junk the gate scores ~92%. The CrowdHuman false-positive metric was retired for the same reason (unannotated real people).</text>')
    s.append('</svg>')
    (OUT / "exp10_per_dataset.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 4. Cost
# ---------------------------------------------------------------------------
def cost():
    W, H = 940, 470
    x0, x1, y0, y1 = 200, 900, 95, 330
    rows = [("Ordinary photos", "~2.5 people/image = 1.25M", 1245),
            ("Multi-person photos", "~6.5 people/image = 3.25M", 3237),
            ("Crowd-heavy", "~26 people/image = 13M", 12950)]
    vmax = 13500
    def xx(v): return x0 + (v / vmax) * (x1 - x0)
    s = svg_open(W, H, "Cost to label 500,000 images",
                 "Measured: ~$1.00 per 1,000 people labeled (1,838 input + 178 output tokens each). "
                 "Cost scales with PEOPLE, not images.")
    for g in (0, 2500, 5000, 7500, 10000, 12500):
        xg = xx(g)
        s.append(f'<line x1="{xg:.1f}" y1="{y0-6}" x2="{xg:.1f}" y2="{y1+6}" stroke="{GRID}"/>')
        s.append(f'<text x="{xg:.1f}" y="{y1+22:.1f}" text-anchor="middle" font-size="11" fill="{MUTE}">${g:,}</text>')
    bh = 34
    for i, (lab, sub, seq) in enumerate(rows):
        yb = y0 + i * 78
        s.append(f'<rect x="{x0}" y="{yb:.1f}" width="{xx(seq)-x0:.1f}" height="{bh}" rx="3" fill="{GREY}"/>')
        s.append(f'<rect x="{x0}" y="{yb+bh+4:.1f}" width="{xx(seq/2)-x0:.1f}" height="{bh}" rx="3" fill="{GREEN}"/>')
        s.append(f'<text x="{xx(seq)+8:.1f}" y="{yb+23:.1f}" font-size="12.5" font-weight="700" fill="{INK}">${seq:,}</text>')
        s.append(f'<text x="{xx(seq/2)+8:.1f}" y="{yb+bh+27:.1f}" font-size="12.5" font-weight="700" fill="{GREEN}">${seq//2:,} (batch)</text>')
        s.append(f'<text x="{x0-10}" y="{yb+22:.1f}" text-anchor="end" font-size="12" font-weight="600" fill="{INK}">{lab}</text>')
        s.append(f'<text x="{x0-10}" y="{yb+38:.1f}" text-anchor="end" font-size="10" fill="{MUTE}">{sub}</text>')
    s.append(f'<text x="{x0}" y="{H-58}" font-size="12" fill="{INK}">For comparison: human annotation of the same 500k images &#8776; <tspan font-weight="700">$175,000</tspan>. This validation experiment cost <tspan font-weight="700">$5.50</tspan>.</text>')
    s.append(f'<text x="{x0}" y="{H-38}" font-size="11.5" fill="{MUTE}">Dollar figures use press-reported Flash-Lite rates ($0.30/$2.50 per 1M tokens) — token counts are measured, rates await vendor confirmation.</text>')
    s.append('</svg>')
    (OUT / "exp10_cost.svg").write_text("\n".join(s))


SVGS = {"exp10_age_gradient.svg": (900, 520), "exp10_vs_sam3.svg": (900, 500),
        "exp10_per_dataset.svg": (900, 470), "exp10_cost.svg": (940, 470)}


def to_png():
    for name, (w, h) in SVGS.items():
        png = PNG / name.replace(".svg", ".png")
        subprocess.run([CHROME, "--headless", "--disable-gpu", f"--screenshot={png}",
                        f"--window-size={w},{h}", "--force-device-scale-factor=2",
                        "--hide-scrollbars", str((OUT / name).resolve())],
                       check=True, capture_output=True)
        print("wrote", png, png.stat().st_size, "bytes")


if __name__ == "__main__":
    age_gradient(); vs_sam3(); per_dataset(); cost()
    for f in SVGS:
        print("wrote", OUT / f, (OUT / f).stat().st_size, "bytes")
    to_png()
