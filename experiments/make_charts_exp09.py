#!/usr/bin/env python3
"""Generate SVG charts for EXP-2026-09 (SAM3 vs Gemini 3.5 Flash-Lite crowd
head-to-head) and convert to PNG (Chrome headless, 2x) into
assets/png/EXP-2026-09/.

All numbers are the Part 1 final run (2026-07-23, 100 CrowdHuman images,
2,378 GT persons): /workspace/exp09/headtohead/summary.json ->
per_model.*.recall_at_iou (the scorer re-matches both models at each
threshold with the same greedy matcher).
"""
import pathlib
import subprocess

OUT = pathlib.Path("/Users/mostafaelkabir/Desktop/HaramBlur/Innovation-lab/experiments/assets")
PNG = OUT / "png" / "EXP-2026-09"
OUT.mkdir(parents=True, exist_ok=True)
PNG.mkdir(parents=True, exist_ok=True)

INK, MUTE, GRID = "#1a1d25", "#8899aa", "#dfe3ea"
SAM_BLUE = "#2858dc"   # matches the evidence report / gallery colors
LITE_ORANGE = "#e08000"
RED = "#e74c3c"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# recall (%) re-matched at each IoU threshold — summary.json recall_at_iou
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
SAM_RECALL = (82.97, 80.95, 78.22, 72.04, 60.56)
LITE_RECALL = (51.05, 50.38, 49.20, 47.48, 43.73)


def svg_open(w, h, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{title}</text>')
    s.append(f'<text x="{w/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>')
    return s


def recall_vs_iou():
    W, H = 900, 520
    x0, x1, y0, y1 = 70, 850, 84, 420

    def xx(t):
        return x0 + (t - 0.3) / 0.4 * (x1 - x0)

    def yy(p):
        return y1 - (p / 100) * (y1 - y0)

    s = svg_open(W, H,
                 "Recall vs box-match strictness: SAM3 sees more, boxes looser",
                 "CrowdHuman head-to-head, 100 images / 2,378 people. Recall re-matched at each IoU threshold "
                 "(same greedy matcher for both models).")
    # grid + axes
    for g in (0, 20, 40, 60, 80, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    for t in IOUS:
        xt = xx(t)
        s.append(f'<line x1="{xt:.1f}" y1="{y1}" x2="{xt:.1f}" y2="{y1+5}" stroke="{MUTE}"/>')
        lab = f"{t:.1f}" + (" (headline)" if t == 0.5 else "")
        w = "700" if t == 0.5 else "400"
        s.append(f'<text x="{xt:.1f}" y="{y1+20}" text-anchor="middle" font-size="11.5" font-weight="{w}" fill="{INK}">{lab}</text>')
    s.append(f'<text x="{(x0+x1)/2}" y="{y1+42}" text-anchor="middle" font-size="12" fill="{MUTE}">IoU threshold required to count a detection as a match (stricter &#8594;)</text>')

    # headline-threshold marker
    xh = xx(0.5)
    s.append(f'<line x1="{xh:.1f}" y1="{y0}" x2="{xh:.1f}" y2="{y1}" stroke="{GRID}" stroke-width="2" stroke-dasharray="5 4"/>')

    # lines + points + value labels
    for vals, col, name in ((SAM_RECALL, SAM_BLUE, "SAM3 (production)"),
                            (LITE_RECALL, LITE_ORANGE, "Gemini 3.5 Flash-Lite")):
        pts = " ".join(f"{xx(t):.1f},{yy(v):.1f}" for t, v in zip(IOUS, vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3"/>')
        for t, v in zip(IOUS, vals):
            s.append(f'<circle cx="{xx(t):.1f}" cy="{yy(v):.1f}" r="4.5" fill="{col}"/>')
            s.append(f'<text x="{xx(t):.1f}" y="{yy(v)-11:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{col}">{v:.1f}%</text>')
        s.append(f'<text x="{xx(0.7)+10:.1f}" y="{yy(vals[-1])+4:.1f}" font-size="12.5" font-weight="700" fill="{col}"></text>')

    # slope annotations: what the shape of each curve MEANS
    s.append(f'<text x="{xx(0.42):.1f}" y="{yy(64)-14:.1f}" text-anchor="middle" font-size="12" fill="{SAM_BLUE}">'
             f'drops 17.7 pts from 0.5&#8594;0.7:</text>')
    s.append(f'<text x="{xx(0.42):.1f}" y="{yy(64):.1f}" text-anchor="middle" font-size="12" fill="{SAM_BLUE}">'
             f'finds people but boxes them loosely</text>')
    s.append(f'<text x="{xx(0.55):.1f}" y="{yy(33)-14:.1f}" text-anchor="middle" font-size="12" fill="{LITE_ORANGE}">'
             f'nearly flat (-5.5 pts): boxes are tight,</text>')
    s.append(f'<text x="{xx(0.55):.1f}" y="{yy(33):.1f}" text-anchor="middle" font-size="12" fill="{LITE_ORANGE}">'
             f'the missing half was never detected at ANY strictness</text>')

    # legend
    lx, ly = x0 + 6, H - 32
    s.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" rx="2" fill="{SAM_BLUE}"/>')
    s.append(f'<text x="{lx+20}" y="{ly+11}" font-size="11.5" fill="{INK}">SAM3 (frozen production labels) &#183; matched-IoU median 0.81</text>')
    s.append(f'<rect x="{lx+400}" y="{ly}" width="14" height="14" rx="2" fill="{LITE_ORANGE}"/>')
    s.append(f'<text x="{lx+420}" y="{ly+11}" font-size="11.5" fill="{INK}">Gemini 3.5 Flash-Lite &#183; matched-IoU median 0.869</text>')
    s.append('</svg>')
    (OUT / "exp09_recall_vs_iou.svg").write_text("\n".join(s))


SVGS = {
    "exp09_recall_vs_iou.svg": (900, 520),
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
    recall_vs_iou()
    for f in SVGS:
        print("wrote", OUT / f, (OUT / f).stat().st_size, "bytes")
    to_png()
