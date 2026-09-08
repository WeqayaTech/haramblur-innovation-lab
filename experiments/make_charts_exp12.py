#!/usr/bin/env python3
"""
EXP-2026-12 charts: production YOLO-MIT v9 checkpoint vs YOLO26n fine-tuned on
Spotlight labels. Numbers hardcoded from the 2026-08-04 Phase C + anchor runs
(see the experiment doc's scorecard). Writes 3 SVGs to experiments/assets/ and
2x PNGs to experiments/assets/png/EXP-2026-12/ via Chrome headless.
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
PNG = ASSETS / "png" / "EXP-2026-12"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, MUTE = "#1a1a2e", "#7a7a8c"
GREY, TEAL = "#9aa0a6", "#0f9d8f"   # grey = production MIT, teal = YOLO26n
RED = "#d93025"


def svg_open(w, h, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{title}</text>')
    s.append(f'<text x="{w/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>')
    return s


def legend(s, x, y):
    s.append(f'<rect x="{x}" y="{y}" width="14" height="14" rx="2" fill="{GREY}"/>')
    s.append(f'<text x="{x+20}" y="{y+11}" font-size="11.5" fill="{INK}">production YOLO-MIT v9 (8.2M params)</text>')
    s.append(f'<rect x="{x+280}" y="{y}" width="14" height="14" rx="2" fill="{TEAL}"/>')
    s.append(f'<text x="{x+300}" y="{y+11}" font-size="11.5" fill="{INK}">YOLO26n + Spotlight labels (2.4M params)</text>')


def paired(s, groups, x0, y_base, bar_w=42, gap=14, group_gap=72, vmax=100.0,
           scale_h=210, fmt="{:.1f}"):
    x = x0
    for label, a, b, better in groups:
        for val, col in ((a, GREY), (b, TEAL)):
            h = val / vmax * scale_h
            s.append(f'<rect x="{x}" y="{y_base-h}" width="{bar_w}" height="{h}" rx="3" fill="{col}"/>')
            s.append(f'<text x="{x+bar_w/2}" y="{y_base-h-6}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{fmt.format(val)}</text>')
            x += bar_w + gap
        cx = x - gap - bar_w - (bar_w + gap) / 2 + bar_w / 2
        for i, line in enumerate(label.split("\n")):
            s.append(f'<text x="{cx}" y="{y_base+18+i*14}" text-anchor="middle" font-size="11.5" fill="{INK}">{line}</text>')
        if better:
            s.append(f'<text x="{cx}" y="{y_base+18+2*14}" text-anchor="middle" font-size="11" font-weight="700" fill="{TEAL if better=="new" else RED}">{"✓ new model" if better=="new" else "✓ production"}</text>')
        x += group_gap
    return s


def chart_fp():
    s = svg_open(760, 360, "False people on things that are not people",
                 "lower is better · object set = 259 hand-verified doll/statue/toy images · PASS = 3,000 verified-empty scenes")
    legend(s, 80, 66)
    s += paired(s and [] or [], [], 0, 0)  # no-op keeps type checkers quiet
    body = []
    paired(body, [
        ("object set:\n% images w/ false person", 53.7, 12.7, "new"),
        ("object set:\nFPs per 100 images", 89.6, 23.9, "new"),
        ("PASS:\nFPs per 100 images", 1.43, 0.23, "new"),
    ], x0=100, y_base=310, vmax=100, scale_h=190)
    s += body
    s.append("</svg>")
    return "exp12_false_people.svg", "\n".join(s)


def chart_lagenda():
    s = svg_open(760, 360, "LAGENDA (5,000 people, human ground truth)",
                 "classification quality at conf 0.45 · adult→Child leak = adults 20+ misread as Child (escape the blur)")
    legend(s, 80, 66)
    body = []
    paired(body, [
        ("gender accuracy\non adults (%)", 87.2, 90.7, "new"),
        ("detection\nrecall (%)", 96.0, 94.1, "prod"),
        ("teens 15–19\ncalled Child (%)", 26.6, 8.4, "new"),
        ("adults 20+ → Child\n(leak, %)", 0.77, 0.11, "new"),
    ], x0=90, y_base=310, vmax=100, scale_h=190, fmt="{:.2f}")
    s += body
    s.append("</svg>")
    return "exp12_lagenda.svg", "\n".join(s)


def chart_agecurve():
    W, H = 760, 400
    s = svg_open(W, H, "How often each model calls a person “Child”, by their real age",
                 "the production cutoff is 12 · a sharp drop after 12 is ideal · gray zone = teens")
    x0, y0, x1, y1 = 90, 330, 700, 90
    ages = [0, 5, 10, 15, 20, 25, 30]
    mit = [98.8, 94.2, 72.5, 26.6, 4.3, 0.26, 0.46]
    y26 = [96.0, 81.4, 42.5, 8.4, 0.51, 0.0, 0.47]

    def X(i):
        return x0 + i * (x1 - x0) / (len(ages) - 1)

    def Y(v):
        return y0 - v / 100 * (y0 - y1)

    s.append(f'<rect x="{X(2.6)}" y="{y1}" width="{X(3.8)-X(2.6)}" height="{y0-y1}" fill="#f3f0fa"/>')
    s.append(f'<text x="{(X(2.6)+X(3.8))/2}" y="{y1+16}" text-anchor="middle" font-size="11" fill="{MUTE}">teen gray zone</text>')
    for v in (0, 25, 50, 75, 100):
        s.append(f'<line x1="{x0}" y1="{Y(v)}" x2="{x1}" y2="{Y(v)}" stroke="#e8e8ee"/>')
        s.append(f'<text x="{x0-8}" y="{Y(v)+4}" text-anchor="end" font-size="11" fill="{MUTE}">{v}%</text>')
    for i, a in enumerate(ages):
        s.append(f'<text x="{X(i)}" y="{y0+18}" text-anchor="middle" font-size="11.5" fill="{INK}">{a}–{a+4}</text>')
    s.append(f'<text x="{(x0+x1)/2}" y="{y0+38}" text-anchor="middle" font-size="12" fill="{MUTE}">actual age (years)</text>')
    for vals, col, name in ((mit, GREY, "production YOLO-MIT v9"), (y26, TEAL, "YOLO26n + Spotlight")):
        pts = " ".join(f"{X(i)},{Y(v)}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{X(i)}" cy="{Y(v)}" r="4" fill="{col}"/>')
    s.append(f'<line x1="{X(2.4)}" y1="{y1}" x2="{X(2.4)}" y2="{y0}" stroke="{RED}" stroke-dasharray="5,4"/>')
    s.append(f'<text x="{X(2.4)-8}" y="{y0-14}" text-anchor="end" font-size="11" fill="{RED}">cutoff: 12</text>')
    legend(s, 120, 66)
    s.append("</svg>")
    return "exp12_age_curve.svg", "\n".join(s)


def main():
    ASSETS.mkdir(exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    for fn in (chart_fp, chart_lagenda, chart_agecurve):
        name, svg = fn()
        (ASSETS / name).write_text(svg)
        out = PNG / name.replace(".svg", ".png")
        subprocess.run([CHROME, "--headless", "--disable-gpu",
                        f"--screenshot={out}", "--window-size=1520,800",
                        "--force-device-scale-factor=2",
                        "--default-background-color=FFFFFFFF",
                        str(ASSETS / name)], check=True, capture_output=True)
        print("wrote", ASSETS / name, "and", out)


if __name__ == "__main__":
    main()
