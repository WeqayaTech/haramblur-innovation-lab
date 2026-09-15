#!/usr/bin/env python3
"""
Apple-silicon (local Mac) TFLite latency vs the pod's EPYC numbers, from two `bench_matrix.py`
JSONs: models/bench_mac_20260915/bench_mac_m2.json (this Mac) and
models/exp22_20260914/bench_matrix.json (pod, EXP-2026-22). Writes:
  experiments/assets/bench_mac/m2_latency_4threads.svg       grouped bars, every local file
  experiments/assets/bench_mac/int8_speedup_m2_vs_epyc.svg   fp32/INT8 ratio, Mac vs pod
  experiments/assets/bench_mac/SUMMARY.md                    full table for the docs

    python3 experiments/make_charts_bench_mac.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
MAC = ROOT / "models" / "bench_mac_20260915" / "bench_mac_m2.json"
POD = ROOT / "models" / "exp22_20260914" / "bench_matrix.json"
OUT = HERE / "assets" / "bench_mac"

INK, MUTE, GRID = "#1a1a2e", "#7a7a8c", "#e4e4ea"
TAGS = [("fp32", "fp32 TFLite", "#2a78d6"), ("fp16", "FP16 TFLite", "#86b6ef"),
        ("int8", "INT8 W8A8", "#eb6834"), ("fdec", "INT8 + float decode", "#1baf7a")]
MODELS = ["y26n_humanshaped_v2", "y26n_noe2e_warm50-2", "y26s_humanshaped_smallpatch_v1"]
SIZES = [640, 416, 320]


def med(d, run, tag, sz, th):
    r = d["results"].get(f"{run}|{tag}|{sz}")
    if not r or str(th) not in r["threads"]:
        return None
    return r["threads"][str(th)]


def svg_open(w, h, title, subtitle):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         f'<rect width="{w}" height="{h}" fill="white"/>',
         f'<text x="{w/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{title}</text>',
         f'<text x="{w/2}" y="51" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>']
    return s


def chart_latency(mac, th):
    W, H = 1180, 520
    env = mac["env"]
    s = svg_open(W, H, f"TFLite latency on this Mac — {env['cpu_model'].split(',')[0]}, {th} threads",
                 f"LiteRT {env['ai_edge_litert']} + XNNPACK · batch 1 · 20 warm-up + 100 timed invokes × 3 interleaved repeats · median of medians (bar), p90 (tick) · vlm-cluster/bench_matrix.py --root")
    x0, y0, pw, ph = 70, 90, 1060, 330
    vals = [med(mac, m, t, sz, th)["median_ms"] for m in MODELS for sz in SIZES for t, _, _ in TAGS if med(mac, m, t, sz, th)]
    vmax = max(vals) * 1.12
    step = 10 if vmax <= 60 else (20 if vmax <= 150 else 50)
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="{GRID}"/>')
    v = 0
    while v <= vmax:
        y = y0 + ph - v / vmax * ph
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-6}" y="{y+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTE}">{v}</text>')
        v += step
    s.append(f'<text transform="translate({x0-42},{y0+ph/2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="{INK}">ms per frame</text>')
    ngroups = len(MODELS) * len(SIZES)
    gw = pw / ngroups
    bw = gw * 0.8 / len(TAGS)
    gi = 0
    for m in MODELS:
        gx0 = x0 + gi * gw
        s.append(f'<text x="{gx0 + 1.5*gw:.1f}" y="{y0+ph+42}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{m}</text>')
        for sz in SIZES:
            gx = x0 + gi * gw + gw * 0.1
            s.append(f'<text x="{gx + gw*0.4:.1f}" y="{y0+ph+18}" text-anchor="middle" font-size="11" fill="{MUTE}">{sz}</text>')
            for j, (t, label, col) in enumerate(TAGS):
                r = med(mac, m, t, sz, th)
                if not r:
                    continue
                h = r["median_ms"] / vmax * ph
                x = gx + j * bw
                s.append(f'<rect x="{x:.1f}" y="{y0+ph-h:.1f}" width="{bw-2:.1f}" height="{h:.1f}" rx="2" fill="{col}"/>')
                yp = y0 + ph - r["p90_ms"] / vmax * ph
                s.append(f'<line x1="{x:.1f}" y1="{yp:.1f}" x2="{x+bw-2:.1f}" y2="{yp:.1f}" stroke="{INK}" stroke-width="1.2"/>')
                s.append(f'<text x="{x+(bw-2)/2:.1f}" y="{y0+ph-h-4:.1f}" text-anchor="middle" font-size="9" fill="{INK}">{r["median_ms"]:.0f}</text>')
            gi += 1
        if gi < ngroups:
            xs = x0 + gi * gw
            s.append(f'<line x1="{xs:.1f}" y1="{y0}" x2="{xs:.1f}" y2="{y0+ph}" stroke="{GRID}" stroke-dasharray="3 3"/>')
    lx = x0
    for t, label, col in TAGS:
        s.append(f'<rect x="{lx}" y="{y0+ph+62}" width="12" height="12" rx="2" fill="{col}"/>')
        s.append(f'<text x="{lx+17}" y="{y0+ph+72}" font-size="11.5" fill="{INK}">{label}</text>')
        lx += 170
    s.append(f'<text x="{x0+pw}" y="{y0+ph+72}" text-anchor="end" font-size="11" fill="{MUTE}">missing bars = file not on this Mac (pod unreachable when this ran)</text>')
    s.append('</svg>')
    return "\n".join(s)


def chart_speedup(mac, pod):
    W, H = 1180, 460
    s = svg_open(W, H, "INT8 speed-up over fp32 — this Mac (M2) vs the pod (EPYC 9655P), 1 thread",
                 "ratio = fp32 median ms ÷ INT8 W8A8 median ms, same file, same method on both machines · above 1 = INT8 faster")
    x0, y0, pw, ph = 70, 90, 1060, 280
    pairs = [(m, sz) for m in MODELS for sz in SIZES]
    vals = []
    for m, sz in pairs:
        for d in (mac, pod):
            a, b = med(d, m, "fp32", sz, 1), med(d, m, "int8", sz, 1)
            if a and b:
                vals.append(a["median_ms"] / b["median_ms"])
    vmax = max(vals) * 1.15
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="{GRID}"/>')
    v = 0.0
    while v <= vmax:
        y = y0 + ph - v / vmax * ph
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-6}" y="{y+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTE}">{v:.1f}×</text>')
        v += 0.5
    y1 = y0 + ph - 1.0 / vmax * ph
    s.append(f'<line x1="{x0}" y1="{y1:.1f}" x2="{x0+pw}" y2="{y1:.1f}" stroke="{MUTE}" stroke-dasharray="4 3"/>')
    gw = pw / len(pairs)
    bw = gw * 0.34
    cols = {"mac": "#2a78d6", "pod": "#9aa0a6"}
    for i, (m, sz) in enumerate(pairs):
        gx = x0 + i * gw + gw * 0.16
        for j, (name, d) in enumerate((("mac", mac), ("pod", pod))):
            a, b = med(d, m, "fp32", sz, 1), med(d, m, "int8", sz, 1)
            if not (a and b):
                continue
            r = a["median_ms"] / b["median_ms"]
            h = r / vmax * ph
            x = gx + j * bw
            s.append(f'<rect x="{x:.1f}" y="{y0+ph-h:.1f}" width="{bw-2:.1f}" height="{h:.1f}" rx="2" fill="{cols[name]}"/>')
            s.append(f'<text x="{x+(bw-2)/2:.1f}" y="{y0+ph-h-4:.1f}" text-anchor="middle" font-size="9.5" fill="{INK}">{r:.1f}×</text>')
        s.append(f'<text x="{x0 + i*gw + gw/2:.1f}" y="{y0+ph+18}" text-anchor="middle" font-size="11" fill="{MUTE}">{sz}</text>')
        if sz == 416:
            s.append(f'<text x="{x0 + i*gw + gw/2:.1f}" y="{y0+ph+40}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">{m}</text>')
    s.append(f'<rect x="{x0}" y="{y0+ph+58}" width="12" height="12" rx="2" fill="{cols["mac"]}"/><text x="{x0+17}" y="{y0+ph+68}" font-size="11.5" fill="{INK}">this Mac · Apple M2</text>')
    s.append(f'<rect x="{x0+200}" y="{y0+ph+58}" width="12" height="12" rx="2" fill="{cols["pod"]}"/><text x="{x0+217}" y="{y0+ph+68}" font-size="11.5" fill="{INK}">pod · AMD EPYC 9655P (EXP-2026-22 bench)</text>')
    s.append('</svg>')
    return "\n".join(s)


def summary(mac, pod):
    env = mac["env"]
    L = [f"# TFLite latency on this Mac — {env['cpu_model']} (2026-09-15)", "",
         f"LiteRT {env['ai_edge_litert']} + XNNPACK, Python {env['python']}, batch 1, fixed random input, 20 warm-up + 100 timed invokes, "
         f"3 interleaved repeats, each measurement in a fresh process (`vlm-cluster/bench_matrix.py --root`). Load average at start: {env['loadavg_at_start']}. "
         "Pod columns: the EXP-2026-22 bench on AMD EPYC 9655P (`models/exp22_20260914/bench_matrix.json`), same method. "
         "`spread` = max−min of the 3 repeat medians in % (⚠ > 10 %).", "",
         "| model | px | precision | MB | **M2 1 thr** median / p90 | M2 4 thr | M2 8 thr | spread 1/4/8 | EPYC 1 thr | EPYC 4 thr | M2 ÷ EPYC (1 thr) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for sz in SIZES:
            for t, label, _ in TAGS:
                r = mac["results"].get(f"{m}|{t}|{sz}")
                if not r:
                    continue
                th = r["threads"]
                def cell(k):
                    return f"{th[k]['median_ms']:.1f} / {th[k]['p90_ms']:.1f}" if k in th else "—"
                spread = " / ".join((f"{th[k]['spread_pct']:.0f}%" + (" ⚠" if th[k]["spread_pct"] > 10 else "")) if k in th else "—" for k in ("1", "4", "8"))
                p1, p4 = med(pod, m, t, sz, 1), med(pod, m, t, sz, 4)
                ratio = f"{th['1']['median_ms']/p1['median_ms']:.2f}×" if p1 and "1" in th else "—"
                L.append(f"| `{m}` | {sz} | {label} | {r['bytes']/1e6:.1f} | **{cell('1')}** | {cell('4')} | {cell('8')} | {spread} | "
                         f"{p1['median_ms']:.1f} | " if p1 else f"| `{m}` | {sz} | {label} | {r['bytes']/1e6:.1f} | **{cell('1')}** | {cell('4')} | {cell('8')} | {spread} | — | ")
                L[-1] += (f"{p4['median_ms']:.1f} | " if p4 else "— | ") + f"{ratio} |"
    L.append("")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mac = json.loads(MAC.read_text())
    pod = json.loads(POD.read_text())
    (OUT / "m2_latency_4threads.svg").write_text(chart_latency(mac, 4))
    (OUT / "m2_latency_1thread.svg").write_text(chart_latency(mac, 1))
    (OUT / "int8_speedup_m2_vs_epyc.svg").write_text(chart_speedup(mac, pod))
    (OUT / "SUMMARY.md").write_text(summary(mac, pod))
    print("wrote 3 SVGs + SUMMARY.md to", OUT)


if __name__ == "__main__":
    main()
