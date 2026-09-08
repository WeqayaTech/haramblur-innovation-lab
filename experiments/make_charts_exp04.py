#!/usr/bin/env python3
"""Generate SVG charts for EXP-2026-04 (person-prompt vs 3-prompt detection) and
convert them to PNG (Chrome headless, 2x) into assets/png/EXP-2026-04/.

All numbers are the final run results (2026-07-14, L4 pod):
  - 3-prompt baseline = EXP-2026-03 summaries
  - "person" prompt   = /workspace/exp04/eval/*/summary.json
Crowd person-prompt numbers are over 499/500 images (one 310-person image
produced no label file — flagged in the experiment doc, resolution pending).
"""
import pathlib
import subprocess

OUT = pathlib.Path("/Users/mostafaelkabir/Desktop/HaramBlur/Innovation-lab/experiments/assets")
PNG = OUT / "png" / "EXP-2026-04"
OUT.mkdir(parents=True, exist_ok=True)
PNG.mkdir(parents=True, exist_ok=True)

INK, MUTE, GRID = "#1a1d25", "#8899aa", "#dfe3ea"
GREEN, AMBER, RED, BLUE = "#27ae60", "#f39c12", "#e74c3c", "#4f8ef7"
GREY = "#9aa7b5"   # 3-prompt baseline bars
PURP = "#8e44ad"   # person-prompt bars

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def svg_open(w, h, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{title}</text>')
    s.append(f'<text x="{w/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>')
    return s


def legend(s, x, y):
    s.append(f'<rect x="{x}" y="{y}" width="14" height="14" rx="2" fill="{GREY}"/>')
    s.append(f'<text x="{x+20}" y="{y+11}" font-size="11.5" fill="{INK}">production 3 prompts (woman/man/child)</text>')
    s.append(f'<rect x="{x+270}" y="{y}" width="14" height="14" rx="2" fill="{PURP}"/>')
    s.append(f'<text x="{x+290}" y="{y+11}" font-size="11.5" fill="{INK}">single &quot;person&quot; prompt</text>')


def paired_bars(s, x0, x1, y0, y1, vmax, groups, yy, fmt="{:.1f}%"):
    """groups = [(label, sub, baseline, person)]"""
    n = len(groups); gslot = (x1 - x0) / n; bw = min(64.0, gslot * 0.30)
    for gi, (lab, sub, base, pers) in enumerate(groups):
        gc = x0 + gi * gslot + gslot / 2
        for v, col, dx in ((base, GREY, -bw - 5), (pers, PURP, 5)):
            bx = gc + dx; by = yy(v)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="3" fill="{col}"/>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{INK}">{fmt.format(v)}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+17:.1f}" text-anchor="middle" font-size="11.5" font-weight="600" fill="{INK}">{lab}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+32:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{sub}</text>')


# ---------------------------------------------------------------------------
# 1. Hallucination rate on person-free images: the prompt made it WORSE
# ---------------------------------------------------------------------------
def neg_comparison():
    W, H = 900, 480
    x0, x1, y0, y1 = 64, 860, 82, 380
    vmax = 80.0
    def yy(p): return y1 - (p / vmax) * (y1 - y0)
    s = svg_open(W, H, "Asking for &quot;person&quot; makes SAM3 hallucinate MORE, not less",
                 "% of verified person-free images that gained at least one person label, by prompt configuration.")
    for g in (0, 20, 40, 60, 80):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    groups = [
        ("Human-shaped objects", "dolls/statues/toys · n=259 · 144 vs 193 FPs/100", 62.2, 68.7),
        ("Empty scenes (PASS)", "n=3,000 · 8.4 vs 27.4 FPs/100 (~3x)", 5.4, 12.7),
    ]
    paired_bars(s, x0, x1, y0, y1, vmax, groups, yy)
    yb = yy(25)
    s.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{x1-4}" y="{yb-7:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{RED}">objects bar: above 25% = needs a person-verifier gate</text>')
    legend(s, x0, H - 34)
    s.append('</svg>')
    (OUT / "exp04_neg_fp_prompt_comparison.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 2. Crowd recall by occlusion: +2 points where it worked, nothing where it fails
# ---------------------------------------------------------------------------
def crowd_recall_comparison():
    W, H = 900, 500
    x0, x1, y0, y1 = 64, 860, 82, 390
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    s = svg_open(W, H, "Crowd recall: the prompt helps a little where SAM already worked",
                 "CrowdHuman detection recall by occlusion band. Person-prompt numbers over 499/500 images (one pending).")
    for g in (0, 25, 50, 75, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    groups = [
        ("Light occlusion", "n=6,246 / 6,030", 82.7, 84.7),
        ("Partial occlusion", "n=3,699 / 3,613", 73.5, 74.2),
        ("Heavy occlusion", "n=1,314 / 1,306", 61.0, 61.0),
        ("All crowd people", "n=11,259 / 10,949", 77.1, 78.5),
    ]
    paired_bars(s, x0, x1, y0, y1, 100, groups, yy)
    yb = yy(90)
    s.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{x1-4}" y="{yb-7:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{RED}">pre-registered bar (light occlusion): 90% — both fail</text>')
    legend(s, x0, H - 34)
    s.append('</svg>')
    (OUT / "exp04_crowd_recall_prompt_comparison.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 3. The trade: +2.1 recall bought with -2.3 precision (and worse negatives)
# ---------------------------------------------------------------------------
def crowd_tradeoff():
    W, H = 900, 480
    x0, x1, y0, y1 = 64, 860, 82, 380
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    s = svg_open(W, H, "The crowd trade: a small recall gain bought with a precision loss",
                 "CrowdHuman, same matching (IoU 0.5, visible boxes, ignore regions excluded). Duplicates stay negligible.")
    for g in (0, 25, 50, 75, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    groups = [
        ("Recall (light occl.)", "bar: 90%", 82.7, 84.7),
        ("Detection precision", "bar: 85% — person prompt slips under", 86.6, 84.3),
        ("Duplicates / matched", "bar: under 5%", 1.1, 1.1),
    ]
    paired_bars(s, x0, x1, y0, y1, 100, groups, yy)
    yb = yy(85)
    s.append(f'<line x1="{x0+270}" y1="{yb:.1f}" x2="{x1-250}" y2="{yb:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    legend(s, x0, H - 34)
    s.append('</svg>')
    (OUT / "exp04_crowd_precision_tradeoff.svg").write_text("\n".join(s))


SVGS = {
    "exp04_neg_fp_prompt_comparison.svg": (900, 480),
    "exp04_crowd_recall_prompt_comparison.svg": (900, 500),
    "exp04_crowd_precision_tradeoff.svg": (900, 480),
}


def to_png():
    for name, (w, h) in SVGS.items():
        png = PNG / name.replace(".svg", ".png")
        subprocess.run([CHROME, "--headless", "--disable-gpu",
                        f"--screenshot={png}", f"--window-size={w},{h}",
                        "--force-device-scale-factor=2", "--hide-scrollbars",
                        str((OUT / name).resolve())],
                       check=True, capture_output=True)
        print("wrote", png, png.stat().st_size, "bytes")


if __name__ == "__main__":
    neg_comparison()
    crowd_recall_comparison()
    crowd_tradeoff()
    for f in SVGS:
        print("wrote", OUT / f, (OUT / f).stat().st_size, "bytes")
    to_png()
