#!/usr/bin/env python3
"""Render the EXP-2026-22 precision × resolution × dataset report from map_eval JSONs.

Reads <eval_dir>/<run>_<tag>_sz<SZ>_<ds>_map.json (+ _map_<collection>.json and _randoms.json for
holdout) and writes a self-contained HTML page. Missing cells render as "—" so the page can be
regenerated as the matrix fills in.

usage: build_matrix_report.py --eval <dir> --out report.html [--bench bench.json] [--date 2026-09-14]
"""
import argparse, glob, html, json, os, re

MODELS = [
    ("yolo11N-640", "production (shipped) · YOLO11n · old labels", "prod"),
    ("y26n_humanshaped_v2", "YOLO26n · humanshaped policy", "cand"),
    ("y26n_noe2e_warm50-2", "YOLO26n · previous holdout champion", "cand"),
    ("y26s_humanshaped_smallpatch_v1", "YOLO26s · best accuracy measured", "cand"),
    ("y26n_humanshaped_v2_distill_v1", "YOLO26n distilled from y26s · epoch 61/100 snapshot", "cand"),
]
TAGS = [("pt", "fp32 .pt"), ("fp32", "fp32 TFLite"), ("fp16", "FP16 TFLite"), ("int8", "INT8 W8A8"), ("fdec", "INT8 + float decode")]
SIZES = [640, 416, 320]
COLS = ["women_hd", "women", "men", "child", "shiekhs"]
DOM = {"women_hd": "Woman", "women": "Woman", "men": "Man", "child": "Child", "shiekhs": "Man"}   # dominant class per collection


def load(d):
    cells = {}
    for p in glob.glob(os.path.join(d, "*_map.json")):
        b = os.path.basename(p)[:-len("_map.json")]
        m = re.match(r"(.+)_(pt|fp32|fp16|int8|fdec)_sz(\d+)_(spotval|holdout)$", b)
        if not m: continue
        run, tag, sz, ds = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        try: j = json.load(open(p))
        except Exception: continue
        c = {"map50": j["map50"], "map50_95": j["map50_95"], "map75": j.get("map75"), "pc": j.get("per_class", {}), "n_gt": j.get("n_gt")}
        if ds == "holdout":
            for col in COLS:
                q = os.path.join(d, f"{b}_map_{col}.json")
                if os.path.exists(q):
                    jj = json.load(open(q)); dc = (jj.get("per_class") or {}).get(DOM[col], {})
                    c[col] = {"map50": jj["map50"], "map50_95": jj["map50_95"], "n_gt": jj.get("n_gt"), "ap50_95": dc.get("ap50_95"), "ap50": dc.get("ap50")}
            q = os.path.join(d, f"{b}_randoms.json")
            if os.path.exists(q): c["randoms"] = json.load(open(q))
        cells[(ds, run, tag, sz)] = c
    return cells


def f4(v): return "—" if v is None else f"{v:.4f}"


def dcell(v, ref):
    if v is None: return '<td class="d"></td>'
    if ref is None: return '<td class="d"></td>'
    x = v - ref; cls = "zero" if abs(x) < 0.0005 else ("pos" if x > 0 else "neg")
    return f'<td class="d {cls}">{x:+.4f}</td>'


def matrix_table(cells, ds, metric, label):
    h = [f'<div class="scroll"><table><tr><th>model</th><th>px</th>']
    for t, tl in TAGS: h.append(f'<th>{tl}</th><th>Δ</th>')
    h.append('</tr>')
    for run, desc, kind in MODELS:
        first = True
        for sz in SIZES:
            ref = cells.get((ds, run, "pt", sz), {}).get(metric)
            row = f'<tr class="{kind}">'
            if first: row += f'<th class="model" rowspan="3">{run}<br><span class="desc">{html.escape(desc)}</span></th>'; first = False
            row += f'<td class="sz">{sz}</td>'
            for t, tl in TAGS:
                v = cells.get((ds, run, t, sz), {}).get(metric)
                bold = t == "fdec" and v is not None and ref is not None and v >= ref - 0.002
                row += f'<td class="{"best" if bold else ""}">{f4(v)}</td>' + (dcell(v, ref) if t != "pt" else '<td class="d ref">ref</td>')
            h.append(row + '</tr>')
    h.append('</table></div>')
    return "".join(h)


def per_class_table(cells, ds, sz):
    h = ['<div class="scroll"><table><tr><th>model</th><th>precision</th><th>Woman AP50-95</th><th>Man</th><th>Child</th><th>Woman AP50</th><th>Man</th><th>Child</th></tr>']
    for run, desc, kind in MODELS:
        first = True
        for t, tl in TAGS:
            c = cells.get((ds, run, t, sz))
            if not c: continue
            pc = c["pc"]; row = f'<tr class="{kind}">'
            if first: row += f'<th class="model" rowspan="{sum(1 for tt,_ in TAGS if (ds,run,tt,sz) in cells)}">{run}</th>'; first = False
            row += f'<td class="prec">{tl}</td>'
            for k in ("ap50_95", "ap50"):
                for cl in ("Woman", "Man", "Child"): row += f'<td>{f4(pc.get(cl, {}).get(k))}</td>'
            h.append(row + '</tr>')
    h.append('</table></div>'); return "".join(h)


def holdout_collections(cells, sz):
    h = ['<div class="scroll"><table><tr><th>model</th><th>precision</th><th>pooled mAP</th>']
    for col in COLS: h.append(f'<th>{col} · {DOM[col]} AP</th>')
    h.append('<th>randoms · img FP rate @0.25 / @0.45</th></tr>')
    for run, desc, kind in MODELS:
        first = True; n = sum(1 for tt, _ in TAGS if ("holdout", run, tt, sz) in cells)
        for t, tl in TAGS:
            c = cells.get(("holdout", run, t, sz))
            if not c: continue
            row = f'<tr class="{kind}">'
            if first: row += f'<th class="model" rowspan="{n}">{run}</th>'; first = False
            row += f'<td class="prec">{tl}</td><td>{f4(c["map50_95"])} <span class="sub2">{f4(c["map50"])}</span></td>'
            for col in COLS:
                cc = c.get(col); row += f'<td>{f4(cc["ap50_95"]) if cc else "—"} <span class="sub2">{f4(cc["ap50"]) if cc else ""}</span></td>'
            r = c.get("randoms"); row += f'<td>{r["img_fp_rate@0.25"]:.3f} / {r["img_fp_rate@0.45"]:.3f}</td>' if r else '<td>—</td>'
            h.append(row + '</tr>')
    h.append('</table></div>'); return "".join(h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bench"); ap.add_argument("--bench-matrix"); ap.add_argument("--date", default="2026-09-14"); a = ap.parse_args()
    cells = load(a.eval)
    n_done = len(cells); n_total = len(MODELS) * len(TAGS) * len(SIZES) * 2
    prod640 = cells.get(("spotval", "yolo11N-640", "pt", 640), {}); best = max(((cells.get(("spotval", r, "fdec", 640), {}).get("map50_95") or 0, r) for r, _, _ in MODELS))
    bench_html = ""
    if a.bench_matrix and os.path.exists(a.bench_matrix):
        bm = json.load(open(a.bench_matrix)); env = bm["env"]; me = bm["method"]; res = bm["results"]
        ths = [t for t in me["threads"].split(",")]
        bench_html = (f'<h2>CPU latency — one pod, one method, every file</h2><p class="cap">{html.escape(env.get("cpu_model",""))}, {env.get("nproc")} vCPU (shared host; load at start {", ".join(f"{x:.1f}" for x in env.get("loadavg_at_start", []))}), '
                      f'LiteRT {env.get("ai_edge_litert")} + XNNPACK, batch 1, fixed random input, {me["warmup"]} warm-up + {me["runs"]} timed invokes, {me["repeats"]} interleaved repeats, each measurement in a fresh process. '
                      f'Cell = median of the repeat medians in ms; <span class="sub2">p90</span>; ⚠ marks a spread &gt; 10 % between repeats (noisy neighbour — read with care). '
                      f'These rank sizes and precisions and compare models on equal footing; they are not the extension\'s in-browser (TF.js) latency.</p>')
        for th in ths:
            bench_html += f'<h3>{th} thread{"s" if th != "1" else ""}</h3><div class="scroll"><table><tr><th>model</th><th>px</th>' + "".join(f"<th>{tl}</th>" for t, tl in TAGS if t != "pt") + "<th>MB (fp32 → INT8)</th></tr>"
            for run, desc, kind in MODELS:
                first = True
                for sz in SIZES:
                    row = f'<tr class="{kind}">'
                    if first: row += f'<th class="model" rowspan="3">{run}</th>'; first = False
                    row += f'<td class="sz">{sz}</td>'; mb = ["—", "—"]
                    for t, tl in TAGS:
                        if t == "pt": continue
                        r = res.get(f"{run}|{t}|{sz}"); v = r["threads"].get(th) if r else None
                        if r and t == "fp32": mb[0] = f'{r["bytes"] / 1e6:.1f}'
                        if r and t == "int8": mb[1] = f'{r["bytes"] / 1e6:.1f}'
                        if not v: row += "<td>—</td>"; continue
                        flag = " ⚠" if v["spread_pct"] > 10 else ""
                        row += f'<td class="{"best" if t == "fdec" else ""}">{v["median_ms"]:.1f}{flag} <span class="sub2">{v["p90_ms"]:.1f}</span></td>'
                    row += f'<td class="sz">{mb[0]} → {mb[1]}</td>'
                    bench_html += row + "</tr>"
            bench_html += "</table></div>"
    elif a.bench and os.path.exists(a.bench):
        b = json.load(open(a.bench)); bench_html = '<h2>Latency @640 (CPU, 4 threads, median · p90 ms)</h2><div class="scroll"><table><tr><th>model</th>' + "".join(f"<th>{tl}</th>" for t, tl in TAGS if t != "pt") + "</tr>"
        for run, _, kind in MODELS:
            bench_html += f'<tr class="{kind}"><td class="model-inline">{run}</td>' + "".join(f'<td>{b.get(run,{}).get(t,"—")}</td>' for t, _ in TAGS if t != "pt") + "</tr>"
        bench_html += "</table></div>"
    css = """
:root{color-scheme:light;--bg:#F5F6F8;--surface:#FFF;--ink:#1B1F26;--ink-2:#4B5563;--muted:#6B7280;--rule:#D9DEE5;--rule-soft:#E9ECF1;--accent:#0F766E;--accent-soft:#E6F4F1;--gain:#1A7F37;--gain-bg:#E6F4EA;--loss:#B42318;--loss-bg:#FCE8E6;--zero:#6B7280;--prod-bg:#FFF7E6;--warn:#8A5A00;--best:#0F766E}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#14171C;--surface:#1B1F26;--ink:#E8EAEE;--ink-2:#B4BAC4;--muted:#8B93A1;--rule:#2E343D;--rule-soft:#262B33;--accent:#2DD4BF;--accent-soft:#12302D;--gain:#3FB950;--gain-bg:#12301C;--loss:#F0716B;--loss-bg:#3A1A18;--zero:#8B93A1;--prod-bg:#332A12;--warn:#F2C56B;--best:#2DD4BF}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#14171C;--surface:#1B1F26;--ink:#E8EAEE;--ink-2:#B4BAC4;--muted:#8B93A1;--rule:#2E343D;--rule-soft:#262B33;--accent:#2DD4BF;--accent-soft:#12302D;--gain:#3FB950;--gain-bg:#12301C;--loss:#F0716B;--loss-bg:#3A1A18;--zero:#8B93A1;--prod-bg:#332A12;--warn:#F2C56B;--best:#2DD4BF}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 72px}.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 10px}
h1{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:36px;line-height:1.15;margin:0 0 8px;text-wrap:balance}h2{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:23px;margin:44px 0 6px}h3{font-size:15px;font-weight:600;margin:22px 0 8px}
p{max-width:74ch;margin:0 0 12px}.sub{color:var(--ink-2);max-width:70ch;font-size:16px;margin:0 0 24px}.lede{border-left:3px solid var(--accent);padding:4px 0 4px 16px;margin:0 0 20px;max-width:76ch}.lede p{margin:0 0 8px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:22px 0 8px}.kpi{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 18px}.kpi .lab{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}.kpi .val{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:22px;font-weight:500;font-variant-numeric:tabular-nums}.kpi .exp{font-size:13px;color:var(--ink-2);margin-top:6px}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface);margin:12px 0 6px}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule-soft);white-space:nowrap;font-variant-numeric:tabular-nums}td{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px}
th{font-weight:600;color:var(--ink-2);font-size:12px;letter-spacing:.03em;background:var(--bg);position:sticky;top:0}th.model,td.prec,td.model-inline{text-align:left}
th.model{vertical-align:top;font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-size:12.5px;color:var(--ink);background:var(--surface);border-right:1px solid var(--rule-soft)}.desc{font-family:"IBM Plex Sans",sans-serif;font-size:11.5px;color:var(--muted);white-space:normal;display:block;max-width:220px}
td.prec{font-family:"IBM Plex Sans",sans-serif;color:var(--ink-2)}td.sz{color:var(--muted)}tr.prod td{background:var(--prod-bg)}td.best{color:var(--best);font-weight:600}
td.d{font-weight:500;font-size:12px}td.d.pos{color:var(--gain);background:var(--gain-bg)}td.d.neg{color:var(--loss);background:var(--loss-bg)}td.d.zero{color:var(--zero)}td.d.ref{color:var(--muted);font-family:"IBM Plex Sans",sans-serif}
.sub2{color:var(--muted);font-size:11px}.cap{font-size:12.5px;color:var(--muted);margin:0 0 4px;max-width:84ch}.callout{background:var(--prod-bg);border-radius:8px;padding:12px 16px;margin:16px 0;max-width:84ch}.callout b{color:var(--warn)}
ul{max-width:78ch;padding-left:20px}li{margin:0 0 6px}code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;background:var(--accent-soft);padding:1px 5px;border-radius:4px}footer{margin-top:48px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--rule);padding-top:14px}
"""
    out = [f'<title>Precision × Resolution Matrix</title>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n<style>{css}</style>\n<div class="wrap">']
    out.append(f'<p class="eyebrow">HaramBlur · Innovation Lab · {a.date} · {n_done}/{n_total} cells scored</p>')
    out.append('<h1>Every candidate, every precision, every resolution — on the validation set and on the QA holdout</h1>')
    out.append('<p class="sub">Five models (the shipped production model plus four YOLO26 candidates) × five precisions × three input sizes, scored with one scorer on two datasets: Spotlight-val (4,232 images, 8,035 people, model-derived labels) and the <code>haramblur_holdout</code> QA set (11,494 images, six collections, Gemini-labelled, never trained on).</p>')
    out.append('<div class="lede"><p><b>How to read it.</b> Every row is one model at one input size; every pair of columns is a precision with its Δ vs the fp32 <code>.pt</code> checkpoint at the same size. Production rows are tinted. mAP50-95 rewards tight boxes; mAP50 only asks whether the person was found and classified. Cells still running show "—".</p>'
               '<p><b>Two equivalences you can rely on:</b> FP16 TFLite reproduced fp32 within ±0.0004 in every measured cell (weights are cast, nothing is calibrated), and fp32 TFLite reproduces the <code>.pt</code> within ±0.005. Where a holdout FP16 / fp32-TFLite cell is still blank, read the <code>.pt</code> column.</p></div>')
    out.append('<div class="kpis">')
    out.append(f'<div class="kpi"><div class="lab">production · fp32 @640 · Spotlight-val</div><div class="val">{f4(prod640.get("map50_95"))} <span class="sub2">mAP50-95</span></div><div class="exp">mAP50 {f4(prod640.get("map50"))}. The bar every candidate has to clear.</div></div>')
    out.append(f'<div class="kpi"><div class="lab">best INT8-class file @640 · Spotlight-val</div><div class="val">{best[0]:.4f}</div><div class="exp">{best[1]} with the float-decode fix (2.9–10 MB, INT8 speed).</div></div>')
    out.append(f'<div class="kpi"><div class="lab">cells scored</div><div class="val">{n_done} / {n_total}</div><div class="exp">5 models × 5 precisions × 3 sizes × 2 datasets. Reused Spotlight-val cells carry the 9–11 Sep JSONs unchanged.</div></div>')
    out.append('</div>')
    out.append('<h2>Spotlight-val — mAP50-95</h2><p class="cap">Δ vs fp32 <code>.pt</code> at the same size. The float-decode column is highlighted when it matches or beats the checkpoint.</p>' + matrix_table(cells, "spotval", "map50_95", "mAP50-95"))
    out.append('<h3>Spotlight-val — mAP50</h3>' + matrix_table(cells, "spotval", "map50", "mAP50"))
    out.append('<h3>Per-class @640, Spotlight-val</h3>' + per_class_table(cells, "spotval", 640))
    out.append('<h2>QA holdout (<code>haramblur_holdout</code>) — pooled mAP50-95</h2><p class="cap">Pooled over all six collections for the matrix view; the handoff rule is "score per collection, never pooled" for any claim — the per-collection table follows. GT is machine-labelled (Gemini 3.5 Flash-Lite), 1,619 unknown-gender people are ignore regions.</p>' + matrix_table(cells, "holdout", "map50_95", "mAP50-95"))
    out.append('<h3>QA holdout — pooled mAP50</h3>' + matrix_table(cells, "holdout", "map50", "mAP50"))
    for sz in SIZES:
        out.append(f'<h3>QA holdout per collection @{sz} — AP50-95 of the collection\'s dominant class <span class="sub2">(AP50 small)</span> · randoms = person-free images, fraction with ≥1 detection</h3><p class="cap">Per collection the 3-class mean is meaningless (a men-only collection has almost no Woman/Child GT, so two of the three class APs collapse); the dominant-class AP is the honest per-collection number, as in the 2026-08 holdout tables. Pooled mAP over all collections is shown for the matrix view only.</p>' + holdout_collections(cells, sz))
    out.append(bench_html)
    out.append('<h2>What this does not show</h2><ul>'
               '<li><b>Neither label set is human ground truth.</b> Spotlight-val labels come from the pipeline that trained the candidates (within-model precision deltas are valid; cross-model gaps favour the candidates). Holdout labels are Gemini 3.5 Flash-Lite\'s; <code>shiekhs</code> carries ~16% known label noise and the 254 <code>randoms</code> gate-survivors are unadjudicated.</li>'
               '<li><b>The INT8-class Child gain is unexplained.</b> INT8 and the float-decode fix raise Child AP on the nanos (int8 head convs); whether that is real or adult→Child drift needs the LAGENDA classification sweep before any INT8 file ships.</li>'
               '<li><b><code>y26n_humanshaped_v2_distill_v1</code> is an epoch-61/100 snapshot</b> — its rows will move when training finishes.</li>'
               '<li><b>Thresholds:</b> every cell is scored at conf floor 0.001 (threshold-free AP). INT8 redistributes confidence; deployment thresholds must be re-swept per precision.</li>'
               '<li><b>Latency</b> is CPU/XNNPACK on a pod, 4 threads — relative, not device numbers.</li></ul>')
    out.append('<footer>Scorer: <code>map_eval.py</code> (pycocotools 101-pt, IoU .50:.95, floor 0.001, ignore regions). Exports: ultralytics 8.4.146 / ai-edge-litert 2.2.0; INT8 calibrated on the fixed 500-image train subset (sha 55d0a78e…); float-decode fix = <code>float_head_quant.py</code>. Pipeline: <code>vlm-cluster/full_matrix.sh</code>; results: <code>/workspace/exp22/eval/</code>, local <code>models/exp22_20260914/</code>. Full write-up: <code>experiments/EXP-2026-22-precision-matrix.md</code>.</footer></div>')
    open(a.out, "w").write("\n".join(out)); print(f"wrote {a.out}: {n_done}/{n_total} cells")


if __name__ == "__main__":
    main()
