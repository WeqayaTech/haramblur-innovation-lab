#!/usr/bin/env python3
"""Generate SVG charts for EXP-2026-03 (SAM3 negatives + crowds) from the final run numbers."""
import pathlib

OUT = pathlib.Path("/Users/mostafaelkabir/Desktop/HaramBlur/Innovation-lab/experiments/assets")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTE, GRID = "#1a1d25", "#8899aa", "#dfe3ea"
GREEN, AMBER, RED, BLUE = "#27ae60", "#f39c12", "#e74c3c", "#4f8ef7"


def svg_open(w, h, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{title}</text>')
    s.append(f'<text x="{w/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>')
    return s


def grid_and_axis(s, x0, x1, y0, y1, yy):
    for g in (0, 25, 50, 75, 100):
        yg = yy(g)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')


# ---------------------------------------------------------------------------
# 1. False person-labels on images with NO people — the human-likeness gradient
# ---------------------------------------------------------------------------
def fp_gradient():
    data = [  # (label, sub, pct images with >=1 FP, color override or None)
        ("Doll", "n=80", 88.8, None),
        ("Bronze sculpture", "n=66", 65.2, None),
        ("Sculpture", "n=80", 52.5, None),
        ("Teddy bear", "n=33", 15.2, None),
        ("Empty scenes (PASS)", "n=3,000", 5.4, None),
    ]
    W, H = 900, 470
    x0, x1, y0, y1 = 64, 860, 78, 380
    n = len(data); slot = (x1 - x0) / n; bw = slot * 0.62
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    def color(p): return RED if p >= 50 else (AMBER if p >= 10 else GREEN)
    s = svg_open(W, H, "SAM3 writes false &quot;person&quot; labels on images with NO people",
                 "% of person-free images that gained a person label — scales with human-likeness. Overlay check pending.")
    grid_and_axis(s, x0, x1, y0, y1, yy)
    for i, (lab, sub, p, co) in enumerate(data):
        bx = x0 + i * slot + (slot - bw) / 2; by = yy(p)
        s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="3" fill="{co or color(p)}"/>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="{INK}">{p:.0f}%</text>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+16:.1f}" text-anchor="middle" font-size="11.5" font-weight="600" fill="{INK}">{lab}</text>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+31:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{sub}</text>')
    # pre-registered bar at 25%
    yb = yy(25)
    s.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{x1-4}" y="{yb-7:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{RED}">pre-registered bar: above 25% = needs a person-verifier gate</text>')
    s.append('</svg>')
    (OUT / "sam_fp_by_object_type.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 2. Detection recall: LAGENDA framing vs real crowds (by occlusion)
# ---------------------------------------------------------------------------
def crowd_recall():
    data = [  # (label, sub, recall, color)
        ("Prominent subjects", "LAGENDA · EXP-2026-02", 98.7, BLUE),
        ("Light occlusion", "n=6,246", 82.7, None),
        ("Partial occlusion", "n=3,699", 73.5, None),
        ("Heavy occlusion", "n=1,314", 61.0, None),
        ("All crowd people", "n=11,259", 77.1, None),
    ]
    W, H = 900, 470
    x0, x1, y0, y1 = 64, 860, 78, 380
    n = len(data); slot = (x1 - x0) / n; bw = slot * 0.62
    def yy(p): return y1 - (p / 100) * (y1 - y0)
    def color(p): return GREEN if p >= 90 else (AMBER if p >= 70 else RED)
    s = svg_open(W, H, "Detection recall: the crowd reality vs the LAGENDA framing",
                 "Same frozen SAM3 labeler. In real crowds (CrowdHuman) it misses 1 in 4 people, worse with occlusion.")
    grid_and_axis(s, x0, x1, y0, y1, yy)
    for i, (lab, sub, p, co) in enumerate(data):
        bx = x0 + i * slot + (slot - bw) / 2; by = yy(p)
        s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="3" fill="{co or color(p)}"/>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="{INK}">{p:.1f}%</text>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+16:.1f}" text-anchor="middle" font-size="11.5" font-weight="600" fill="{INK}">{lab}</text>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+31:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{sub}</text>')
    yb = yy(90)
    s.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{x1-4}" y="{yb-7:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="{RED}">pre-registered bar (light occlusion): 90%</text>')
    s.append('</svg>')
    (OUT / "sam_crowd_recall_by_occlusion.svg").write_text("\n".join(s))


# ---------------------------------------------------------------------------
# 3. Which class prompt fires falsely, per dataset
# ---------------------------------------------------------------------------
def fp_class_split():
    groups = [  # (group label, n, [(class, count, color)])
        ("Human-shaped objects", "373 false labels on 259 images",
         [("Woman", 180, RED), ("Man", 140, BLUE), ("Child", 53, AMBER)]),
        ("Empty scenes (PASS)", "251 false labels on 3,000 images",
         [("Man", 173, BLUE), ("Woman", 51, RED), ("Child", 27, AMBER)]),
    ]
    W, H = 860, 440
    x0, x1, y0, y1 = 64, 820, 78, 350
    def yy(v, vmax): return y1 - (v / vmax) * (y1 - y0)
    vmax = 200.0
    s = svg_open(W, H, "Which prompt writes the false label?",
                 "Counts of spurious person-labels by class prompt. Dolls/statues mostly become "
                 "&quot;Woman&quot;; empty-scene hallucinations are mostly &quot;Man&quot;.")
    for g in (0, 50, 100, 150, 200):
        yg = yy(g, vmax)
        s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}</text>')
    gslot = (x1 - x0) / len(groups); bw = 74
    for gi, (glab, gsub, bars) in enumerate(groups):
        gc = x0 + gi * gslot + gslot / 2
        start = gc - (len(bars) * bw + (len(bars) - 1) * 10) / 2
        for bi, (cls, v, col) in enumerate(bars):
            bx = start + bi * (bw + 10); by = yy(v, vmax)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{y1-by:.1f}" rx="3" fill="{col}"/>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{INK}">{v}</text>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+15:.1f}" text-anchor="middle" font-size="11" fill="{INK}">{cls}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+38:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{INK}">{glab}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+54:.1f}" text-anchor="middle" font-size="10.5" fill="{MUTE}">{gsub}</text>')
    s.append('</svg>')
    (OUT / "sam_fp_class_split.svg").write_text("\n".join(s))


fp_gradient()
crowd_recall()
fp_class_split()
for f in ("sam_fp_by_object_type.svg", "sam_crowd_recall_by_occlusion.svg", "sam_fp_class_split.svg"):
    print("wrote", OUT / f, (OUT / f).stat().st_size, "bytes")
