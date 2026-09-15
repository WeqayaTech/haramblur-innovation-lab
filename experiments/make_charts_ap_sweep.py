#!/usr/bin/env python3
"""
Per-class AP vs deployment confidence threshold for `y26n_humanshaped_v2` across export
precisions (fp32 .pt vs INT8 W8A8 vs INT8 + float decode), from `vlm-cluster/ap_sweep.py`
JSONs in models/ap_sweep_20260915/. One figure per dataset × size: three panels
(Woman / Man / Child), AP50-95 solid and AP50 dashed, x = threshold, ceilings marked.

Output: experiments/assets/ap_sweep/<ds>_sz<SZ>.svg + SUMMARY.md
    python3 experiments/make_charts_ap_sweep.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
IN = ROOT / "models" / "ap_sweep_20260915"
OUT = HERE / "assets" / "ap_sweep"
RUN = "y26n_humanshaped_v2"
INK, MUTE, GRID = "#1a1a2e", "#7a7a8c", "#e4e4ea"
SERIES = [("pt", "fp32 .pt (reference)", "#2a78d6"), ("int8", "INT8 W8A8 TFLite", "#eb6834"),
          ("fdec", "INT8 + float decode ops (fix)", "#1baf7a")]
EXTRA = [("fp32", "fp32 TFLite"), ("fp16", "FP16 TFLite"), ("fheadall", "INT8 + whole head float")]
CLASSES = ["Woman", "Man", "Child"]
SHIP = 0.45
DS_LABEL = {"holdout": "haramblur_holdout (QA set, 11,494 images)", "spotval": "Spotlight-val (4,232 images)"}


def load(tag, sz, ds):
    f = IN / f"{RUN}_{tag}_sz{sz}_{ds}.json"
    return json.loads(f.read_text()) if f.exists() else None


def series(d, cls, key):
    return sorted((float(t), v[cls][key]) for t, v in d["at_thresholds"].items() if cls in v)


def axes(s, x0, y0, w, h, ymin, ymax, ylabel):
    s.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="{GRID}"/>')
    for i in range(6):
        v = ymin + i * (ymax - ymin) / 5
        y = y0 + h - (v - ymin) / (ymax - ymin) * h
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-6}" y="{y+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTE}">{v:.1f}</text>')
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = x0 + t * w
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}" stroke="{GRID}"/>')
        s.append(f'<text x="{x:.1f}" y="{y0+h+15}" text-anchor="middle" font-size="10.5" fill="{MUTE}">{t:.1f}</text>')
    s.append(f'<text x="{x0+w/2}" y="{y0+h+32}" text-anchor="middle" font-size="11.5" fill="{INK}">confidence threshold (detections below it deleted)</text>')
    s.append(f'<text transform="translate({x0-40},{y0+h/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="{INK}">{ylabel}</text>')
    xs = x0 + SHIP * w
    s.append(f'<line x1="{xs:.1f}" y1="{y0}" x2="{xs:.1f}" y2="{y0+h}" stroke="{MUTE}" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{xs+4:.1f}" y="{y0+12}" font-size="10" fill="{MUTE}">shipped 0.45</text>')


def polyline(s, pts, x0, y0, w, h, ymin, ymax, colour, dash=""):
    d = " ".join(f"{x0 + t*w:.1f},{y0 + h - (v-ymin)/(ymax-ymin)*h:.1f}" for t, v in pts)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"{extra}/>')


def figure(sz, ds, data):
    W, H = 1180, 520
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">'
         f'{RUN} @{sz} — per-class AP vs confidence threshold, fp32 vs two INT8 exports</text>',
         f'<text x="{W/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{DS_LABEL[ds]} · '
         f'AP at threshold t = map_eval.py AP with every detection below t deleted · solid = AP50-95, dashed = AP50 · '
         f'vlm-cluster/ap_sweep.py, 0.05 grid</text>']
    pw, ph, top = 300, 300, 95
    xs = [70, 440, 810]
    for i, cls in enumerate(CLASSES):
        x0 = xs[i]
        axes(s, x0, top, pw, ph, 0, 1, "AP")
        n_gt = data["pt"]["n_gt"][cls]
        s.append(f'<text x="{x0}" y="{top-8}" font-size="13" font-weight="700" fill="{INK}">{cls} <tspan font-weight="400" fill="{MUTE}">· {n_gt:,} GT</tspan></text>')
        for tag, label, col in SERIES:
            d = data[tag]
            polyline(s, [(t, v) for t, v in series(d, cls, "ap50_95") if t > 0], x0, top, pw, ph, 0, 1, col)
            polyline(s, [(t, v) for t, v in series(d, cls, "ap50") if t > 0], x0, top, pw, ph, 0, 1, col, dash="5 4")
            c = d["max_conf"][cls]
            if c < 0.98:
                xc = x0 + c * pw
                s.append(f'<line x1="{xc:.1f}" y1="{top}" x2="{xc:.1f}" y2="{top+ph}" stroke="{col}" stroke-width="1" stroke-dasharray="2 3"/>')
        # @0.45 annotation
        vals = " / ".join(f"{data[t]['at_thresholds']['0.450'][cls]['ap50_95']:.3f}" for t, _, _ in SERIES)
        full = " / ".join(f"{data[t]['at_thresholds']['0.000'][cls]['ap50_95']:.3f}" for t, _, _ in SERIES)
        s.append(f'<text x="{x0+4}" y="{top+ph-22}" font-size="10.5" fill="{INK}">AP50-95 @0.45: {vals}</text>')
        s.append(f'<text x="{x0+4}" y="{top+ph-8}" font-size="10.5" fill="{MUTE}">no threshold (map_eval): {full}</text>')
    lx = 70
    for tag, label, col in SERIES:
        s.append(f'<line x1="{lx}" y1="{top+ph+56}" x2="{lx+22}" y2="{top+ph+56}" stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<text x="{lx+28}" y="{top+ph+60}" font-size="11.5" fill="{INK}">{label}</text>')
        lx += 250
    s.append(f'<line x1="{lx}" y1="{top+ph+56}" x2="{lx+22}" y2="{top+ph+56}" stroke="{INK}" stroke-width="2"/><text x="{lx+28}" y="{top+ph+60}" font-size="11.5" fill="{INK}">AP50-95</text>')
    s.append(f'<line x1="{lx+95}" y1="{top+ph+56}" x2="{lx+117}" y2="{top+ph+56}" stroke="{INK}" stroke-width="2" stroke-dasharray="5 4"/><text x="{lx+123}" y="{top+ph+60}" font-size="11.5" fill="{INK}">AP50</text>')
    s.append(f'<text x="70" y="{top+ph+80}" font-size="11" fill="{MUTE}">Dotted verticals: the highest confidence any box of that class reaches in the INT8 export — above it the class has no detections and AP is 0.</text>')
    s.append('</svg>')
    return "\n".join(s)


def table(sz, ds, data):
    tags = [t for t, _, _ in SERIES] + [t for t, _ in EXTRA if t in data]
    lab = dict([(t, l) for t, l, _ in SERIES] + EXTRA)
    L = [f"**{RUN} @{sz} · {ds}** — AP50-95 per class (Woman / Man / Child) at each confidence threshold; mAP50-95 in brackets", "",
         "| thr | " + " | ".join(f"`{lab[t]}`" for t in tags) + " |", "|---|" + "---|" * len(tags)]
    for t in ["0.000"] + [f"{0.05*i:.3f}" for i in range(1, 20)]:
        row = [("none" if t == "0.000" else t[:4])]
        for tag in tags:
            c = data[tag]["at_thresholds"][t]
            row.append(" / ".join(f"{c[k]['ap50_95']:.3f}" for k in CLASSES) + f" [{c['map50_95']:.3f}]")
        if t == "0.450":
            row = [f"**{x}**" for x in row]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# Per-class AP vs confidence threshold — `{RUN}` across export precisions (2026-09-15)", "",
          "Generated by `experiments/make_charts_ap_sweep.py` from `models/ap_sweep_20260915/*.json` (`vlm-cluster/ap_sweep.py`).", ""]
    n = 0
    for ds in ("holdout", "spotval"):
        for sz in (640, 416, 320):
            data = {t: load(t, sz, ds) for t in [x for x, _, _ in SERIES] + [x for x, _ in EXTRA]}
            data = {t: d for t, d in data.items() if d}
            if not all(t in data for t, _, _ in SERIES):
                continue
            (OUT / f"{ds}_sz{sz}.svg").write_text(figure(sz, ds, data)); n += 1
            md.append(table(sz, ds, data))
    (OUT / "SUMMARY.md").write_text("\n".join(md))
    print(f"wrote {n} SVGs + SUMMARY.md to {OUT}")


if __name__ == "__main__":
    main()
