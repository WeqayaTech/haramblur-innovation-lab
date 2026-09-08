#!/usr/bin/env python3
"""
Audience threshold charts — standalone SVGs, one graph per file (house
rule: SVG files, never a bundled webpage).

For each audience (male: blur target Woman; female: blur target Man):
  chart_f1_<aud>.svg        box-level F1 vs confidence (IoU>=0.5 greedy
                            matching, ignore regions excused) — the
                            standard detection view; the dot marks each
                            model's max-F1 confidence
  chart_overblur_<aud>.svg  pixels overblurred: wrongly blurred % of the
                            image vs confidence (holdout, IoU-0.5-judged,
                            from the report's fp_cache); dot = selected
                            threshold
  chart_underblur_<aud>.svg pixels underblurred: % of a person left
                            visible vs confidence (per-person mean, from
                            the run summary curves); dot = selected
                            threshold

F1 curves are computed from the log-raw sidecars and cached
(--f1-cache), so re-rendering after the first run is instant.

    python3 audience_charts.py \
        --summary /workspace/deploycmp/holdout_run5/summary.json \
        --raw-root /workspace/holdout_eval \
        --gt-labels $H/labels_eval --ignore $H/ignore \
        --fp-cache-male /workspace/deploycmp/fp_cache_male.json \
        --fp-cache-female /workspace/deploycmp/fp_cache_female.json \
        --f1-cache /workspace/deploycmp/f1_cache.json \
        --selected-male "v11n_shipped=0.45,..." \
        --selected-female "v11n_shipped=0.45,..." \
        --out-dir /workspace/deploycmp

    python3 audience_charts.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from women_report_final import parallel_jsons, _iou, _GT_CACHE

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
BG, INK, MUTED, GRID = "#ffffff", "#222222", "#555555", "#e8e8e6"
FONT = "font-family='system-ui,Helvetica,Arial' "

AUDS = {
    "male": {"target": 0, "cls": "Woman", "label": "Male audience"},
    "female": {"target": 1, "cls": "Man", "label": "Female audience"},
}


def f1_curves(raw_dir, gt_dir, ignore_dir, grid, target, workers=16):
    """Standard box-level P/R/F1 per confidence: IoU>=0.5 one-to-one
    greedy matching in confidence order; unmatched detections that sit
    on an ignore-region person (IoU>=0.5) are excused, not FPs."""
    from deploy_compare import seg_boxes, load_ignores
    pairs = parallel_jsons(sorted(Path(raw_dir).glob("*.json")), workers)
    imgs = []
    for p, d in pairs:
        if d is None:
            continue
        w, h = d["width"], d["height"]
        key = (str(gt_dir), str(ignore_dir), p.stem)
        if key in _GT_CACHE:
            gt, ign = _GT_CACHE[key]
        else:
            gt = seg_boxes(Path(gt_dir) / f"{p.stem}.txt", w, h)
            ign = (load_ignores(ignore_dir, p.stem, w, h)
                   if gt is not None else [])
            _GT_CACHE[key] = (gt, ign)
        if gt is None:
            continue
        gts = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in gt if c == target]
        dets = sorted(((r["conf"], tuple(r["box_xyxy"]))
                       for r in d.get("detections", [])
                       if not r.get("excluded") and r["cls"] == target),
                      reverse=True)
        imgs.append((gts, ign, dets))
    out = {}
    for cf in grid:
        tp = fp = fn = 0
        for gts, ign, dets in imgs:
            used = [False] * len(gts)
            for c2, b in dets:
                if c2 < cf:
                    break                       # dets sorted descending
                best, bi = 0.0, -1
                for i, g in enumerate(gts):
                    if not used[i]:
                        v = _iou(b, g)
                        if v > best:
                            best, bi = v, i
                if best >= 0.5:
                    used[bi] = True
                    tp += 1
                elif any(_iou(b, g) >= 0.5 for g in ign):
                    pass                        # ignore-region: excused
                else:
                    fp += 1
            fn += used.count(False)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        out[cf] = {"p": p_, "r": r_,
                   "f1": (2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)}
    return out


def line_chart(title, subtitle, series, y_label, y_pct=True,
               y_from_zero=False):
    """series: [(name, {conf: value}, mark_conf)] -> svg string.
    One graph per file; explicit colors; white background."""
    W, H = 760, 460
    ML, MR, MT, MB = 64, 195, 52, 52
    pw, ph = W - ML - MR, H - MT - MB
    ys = [v for _n, c, _m in series for v in c.values() if v is not None]
    ymin = 0.0 if y_from_zero else min(ys)
    ymax = max(ys)
    pad = (ymax - ymin) * 0.06 or 0.01
    ymin = max(0.0, ymin - (0 if y_from_zero else pad))
    ymax += pad
    xs = [cf for _n, c, _m in series for cf in c]
    xmin, xmax = min(xs) - 0.02, max(xs) + 0.02

    def X(v):
        return ML + pw * (v - xmin) / (xmax - xmin)

    def Y(v):
        return MT + ph * (1 - (v - ymin) / (ymax - ymin))

    def fmt(v):
        return f"{100 * v:.1f}" if y_pct else f"{v:.2f}"

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' "
         f"height='{H}' viewBox='0 0 {W} {H}'>",
         f"<rect width='{W}' height='{H}' fill='{BG}'/>",
         f"<text x='{ML}' y='24' {FONT}font-size='15' fill='{INK}' "
         f"font-weight='600'>{title}</text>",
         f"<text x='{ML}' y='40' {FONT}font-size='11' fill='{MUTED}'>"
         f"{subtitle}</text>"]
    for i in range(6):
        gy = ymin + (ymax - ymin) * i / 5
        p.append(f"<line x1='{ML}' y1='{Y(gy):.1f}' x2='{ML + pw}' "
                 f"y2='{Y(gy):.1f}' stroke='{GRID}'/>")
        p.append(f"<text x='{ML - 8}' y='{Y(gy) + 4:.1f}' {FONT}"
                 f"font-size='10' fill='{MUTED}' text-anchor='end'>"
                 f"{fmt(gy)}</text>")
    for gx in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p.append(f"<line x1='{X(gx):.1f}' y1='{MT}' x2='{X(gx):.1f}' "
                 f"y2='{MT + ph}' stroke='{GRID}'/>")
        p.append(f"<text x='{X(gx):.1f}' y='{MT + ph + 16}' {FONT}"
                 f"font-size='10' fill='{MUTED}' text-anchor='middle'>"
                 f"{gx:.1f}</text>")
    p.append(f"<text x='{ML + pw / 2}' y='{H - 12}' {FONT}font-size='12' "
             f"fill='{INK}' text-anchor='middle'>confidence "
             f"threshold</text>")
    p.append(f"<text x='16' y='{MT + ph / 2:.0f}' {FONT}font-size='11' "
             f"fill='{INK}' transform='rotate(-90 16 {MT + ph / 2:.0f})' "
             f"text-anchor='middle'>{y_label}</text>")
    for k, (name, curve, mark) in enumerate(series):
        col = PALETTE[k % len(PALETTE)]
        pts = sorted(curve.items())
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f},"
                        f"{Y(v):.1f}" for i, (c, v) in enumerate(pts))
        p.append(f"<path d='{path}' fill='none' stroke='{col}' "
                 f"stroke-width='2'/>")
        lab = name
        if mark is not None and mark in curve:
            p.append(f"<circle cx='{X(mark):.1f}' "
                     f"cy='{Y(curve[mark]):.1f}' r='6' fill='{col}' "
                     f"stroke='{BG}' stroke-width='2'/>")
            lab += f" @{mark:.2f}"
        p.append(f"<circle cx='{ML + pw + 14}' "
                 f"cy='{MT + 14 + 22 * k}' r='5' fill='{col}'/>")
        p.append(f"<text x='{ML + pw + 24}' y='{MT + 18 + 22 * k}' {FONT}"
                 f"font-size='12' fill='{INK}'>{lab}</text>")
    p.append("</svg>")
    return "\n".join(p)


def build_charts(summary, f1, fp_curves, selected, out_dir):
    """f1/fp_curves: {aud: {model: {conf: ...}}}; selected: {aud: {m: cf}}.
    Writes six SVGs into out_dir; returns their paths."""
    out_dir = Path(out_dir)
    written = []
    for aud, meta in AUDS.items():
        md = summary["modes"][aud]
        order = list(md["curves"].keys())
        sel = selected.get(aud, {})

        s = []
        for m in order:
            c = {cf: v["f1"] for cf, v in f1[aud][m].items()}
            best = max(c, key=lambda cf: (c[cf], -cf))
            s.append((m, c, best))
        f = out_dir / f"chart_f1_{aud}.svg"
        f.write_text(line_chart(
            f"{meta['label']} — {meta['cls']}-class F1 vs confidence",
            "box-level precision/recall at IoU≥0.5, ignore regions "
            "excused; ● = each model's max-F1 confidence",
            s, f"{meta['cls']}-class F1", y_pct=False))
        written.append(f)

        s = []
        for m in order:
            c = {cf: v for cf, v in fp_curves[aud][m].items()
                 if v is not None}
            s.append((m, c, sel.get(m)))
        f = out_dir / f"chart_overblur_{aud}.svg"
        f.write_text(line_chart(
            f"{meta['label']} — pixels overblurred vs confidence",
            f"wrongly blurred % of the image (holdout; fired "
            f"{meta['cls']} boxes matching no {meta['cls'].lower()} at "
            f"IoU≥0.5); ● = selected threshold",
            s, "wrongly blurred % of image", y_from_zero=True))
        written.append(f)

        s = []
        for m in order:
            c = {float(cf): a["person_uncov_mean"]
                 for cf, a in md["curves"][m].items()
                 if a.get("person_uncov_mean") is not None}
            s.append((m, c, sel.get(m)))
        f = out_dir / f"chart_underblur_{aud}.svg"
        f.write_text(line_chart(
            f"{meta['label']} — pixels underblurred vs confidence",
            f"% of a {meta['cls'].lower()} left visible (per-person "
            f"mean, every person equal); ● = selected threshold",
            s, f"% of a {meta['cls'].lower()} left visible",
            y_from_zero=True))
        written.append(f)
    return written


def _parse_sel(spec):
    out = {}
    for part in (spec or "").split(","):
        m, _, c = part.partition("=")
        if c:
            out[m.strip()] = float(c)
    return out


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m1" / "raw").mkdir(parents=True)
        (td / "gt").mkdir()
        # GT: one woman [10,10,30,50] in 100x100 + one ignore person
        (td / "gt" / "a.txt").write_text("0 0.2 0.3 0.2 0.4\n")
        (td / "ign").mkdir()
        (td / "ign" / "a.txt").write_text("3 0.7 0.7 0.2 0.2\n")
        dets = [
            {"det_index": 0, "cls": 0, "box_xyxy": [10, 10, 30, 50],
             "conf": 0.9, "kept": True, "excluded": None},     # TP
            {"det_index": 1, "cls": 0, "box_xyxy": [11, 11, 30, 50],
             "conf": 0.8, "kept": True, "excluded": None},     # dup -> FP
            {"det_index": 2, "cls": 0, "box_xyxy": [60, 60, 80, 80],
             "conf": 0.7, "kept": True, "excluded": None},     # ignore
            {"det_index": 3, "cls": 0, "box_xyxy": [0, 80, 10, 95],
             "conf": 0.2, "kept": True, "excluded": None},     # low conf
        ]
        (td / "m1" / "raw" / "a.json").write_text(json.dumps(
            {"image": "a.jpg", "width": 100, "height": 100,
             "detections": dets}))
        c = f1_curves(td / "m1" / "raw", td / "gt", td / "ign",
                      [0.5, 0.85], 0)
        # at 0.5: TP=1 (0.9 box), dup fails one-to-one -> FP, ignore-box
        # det excused, 0.2 det gated out => P=0.5 R=1 F1=2/3
        assert abs(c[0.5]["f1"] - 2 / 3) < 1e-9, c[0.5]
        # at 0.85 only the perfect det survives => F1=1
        assert c[0.85]["f1"] == 1.0, c[0.85]
        best = max(c, key=lambda cf: (c[cf]["f1"], -cf))
        assert best == 0.85, "max-F1 pick"

        summary = {"modes": {
            aud: {"curves": {"m1": {"0.5": {"person_uncov_mean": 0.1},
                                    "0.85": {"person_uncov_mean": 0.2}}}}
            for aud in ("male", "female")}}
        f1 = {aud: {"m1": c} for aud in ("male", "female")}
        fpc = {aud: {"m1": {0.5: 0.01, 0.85: 0.002}}
               for aud in ("male", "female")}
        sel = {aud: {"m1": 0.5} for aud in ("male", "female")}
        files = build_charts(summary, f1, fpc, sel, td)
        assert len(files) == 6 and all(f.exists() for f in files)
        t = (td / "chart_f1_male.svg").read_text()
        assert "<svg" in t and "max-F1" in t and "@0.85" in t
        t = (td / "chart_underblur_female.svg").read_text()
        assert "man left visible" in t and "@0.50" in t
        t = (td / "chart_overblur_male.svg").read_text()
        assert "overblurred" in t and "Woman boxes" in t
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--raw-root")
    ap.add_argument("--gt-labels")
    ap.add_argument("--ignore")
    ap.add_argument("--fp-cache-male")
    ap.add_argument("--fp-cache-female")
    ap.add_argument("--f1-cache")
    ap.add_argument("--selected-male")
    ap.add_argument("--selected-female")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    summary = json.loads(Path(args.summary).read_text())
    order = list(summary["modes"]["male"]["curves"].keys())
    grid = sorted(float(k) for k in
                  summary["modes"]["male"]["curves"][order[0]])

    f1 = None
    if args.f1_cache and Path(args.f1_cache).exists():
        c = json.loads(Path(args.f1_cache).read_text())
        f1 = {aud: {m: {float(k): v for k, v in mm.items()}
                    for m, mm in c[aud].items()} for aud in AUDS}
    else:
        f1 = {aud: {} for aud in AUDS}
        for aud, meta in AUDS.items():
            for m in order:
                f1[aud][m] = f1_curves(
                    Path(args.raw_root) / m / "raw", args.gt_labels,
                    args.ignore, grid, meta["target"])
                print(f"[f1] {aud}/{m} done", flush=True)
        if args.f1_cache:
            Path(args.f1_cache).write_text(json.dumps(
                {aud: {m: {str(k): v for k, v in mm.items()}
                       for m, mm in f1[aud].items()} for aud in AUDS}))

    fp_curves = {}
    for aud, path in (("male", args.fp_cache_male),
                      ("female", args.fp_cache_female)):
        c = json.loads(Path(path).read_text())
        fp_curves[aud] = {m: {float(k): v for k, v in mm.items()}
                          for m, mm in c["holdout"].items()}

    selected = {"male": _parse_sel(args.selected_male),
                "female": _parse_sel(args.selected_female)}
    for f in build_charts(summary, f1, fp_curves, selected, args.out_dir):
        print(f"[chart] wrote {f}")
    for aud in AUDS:
        for m in order:
            c = {cf: v["f1"] for cf, v in f1[aud][m].items()}
            best = max(c, key=lambda cf: (c[cf], -cf))
            print(f"{aud:6s} {m:16s} max-F1 {c[best]:.4f} @ {best:.2f}")


if __name__ == "__main__":
    main()
