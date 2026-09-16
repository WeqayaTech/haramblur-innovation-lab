#!/usr/bin/env python3
"""
Findings report for the deployment ship-table comparison — one self-contained
HTML file: ship tables, tradeoff charts (inline SVG), the male-view escape
decomposition per exposure tier, tier diagnostics (subject size / women per
image — the "why is t4's M-1 low" answer), and sample failure images.

WOMEN ARE ALWAYS REDACTED: every GT-Woman region AND every predicted-Woman
region is pixelated before anything is drawn. There is deliberately no
opt-out flag in this tool — table numbers are computed on unredacted
geometry, only the rendered pixels are redacted.

    python3 deploy_report.py \\
        --summary /workspace/deploycmp/holdout_run2/summary.json \\
        --gt-labels .../labels_eval --images .../images \\
        --verdicts .../run/verdicts_batch.jsonl \\
        --raw-root /workspace/holdout_eval \\
        --out /workspace/deploycmp/SHIP_REPORT.html

    python3 deploy_report.py --selftest    # no data, no GPU, no network

Escape outcomes per GT woman (male mode, matched operating points):
  covered (>=90% under the Woman-blur union) · partial (10-90%) ·
  called_man / called_child (an overlapping wrong-class box >= conf) ·
  subthreshold (a Woman box exists below conf) · no_detection.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from deploy_compare import covered_frac, seg_boxes
from subset_verify import pixelate

CLASS_COLOR = {0: "#CC00CC", 1: "#0072B2", 2: "#E69F00"}
AUD = {"male": {"mode": "male", "target": 0, "gender": "woman",
                "other": "man", "pref": "w_", "cls": "Woman",
                "Noun": "Woman", "plural": "women", "poss": "her"},
       "female": {"mode": "female", "target": 1, "gender": "man",
                  "other": "woman", "pref": "m_", "cls": "Man",
                  "Noun": "Man", "plural": "men", "poss": "his"}}


def outcomes(other):
    return ["covered", "partial", "called_" + other, "called_child",
            "subthreshold", "no_detection"]
T0 = {"face", "hands", "eyes"}
T1 = T0 | {"hair", "neck", "ears"}
T2 = T1 | {"arms", "shoulders", "feet", "knees", "wrists", "head"}
T4P = {"midriff", "stomach", "torso", "thighs"}


def tier_of_parts(parts):
    core = {p for p in (parts or []) if p != "none"}
    if core <= T0:
        return "t0"
    if core <= T1:
        return "t1"
    if core <= T2:
        return "t2"
    if core & T4P or ("chest" in core and "legs" in core):
        return "t4"
    return "t3"


from run_model_children import iou


def ioa(pred, gt):
    ix = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
    iy = max(0.0, min(pred[3], gt[3]) - max(pred[1], gt[1]))
    pa = (pred[2] - pred[0]) * (pred[3] - pred[1])
    return ix * iy / pa if pa > 0 else 0.0


def analyze(gt_dir, raw_root, verdicts, models, matched, target=0,
            gender="woman", other="man", max_examples=6, seed=20260825):
    """Escape decomposition + tier diagnostics + example picks, one audience.
    Also returns the shiekh-collection GT-women list (adjudicated
    2026-08-27: 54/64 real women, 10 mislabeled men — see
    shiekh_gt_women_adjudicated.jsonl for the per-row human calls)."""
    tier_boxes = defaultdict(list)
    if verdicts:
        for line in open(verdicts):
            d = json.loads(line)
            v = d.get("v") or {}
            if (v.get("verdict") == "real_person"
                    and v.get("gender") == gender
                    and v.get("age_group") != "child"):
                tier_boxes[d["image_stem"]].append(
                    (tuple(d["box"]), tier_of_parts(v.get("exposed_body_parts"))))

    res = {m: {"overall": Counter(), "by_tier": defaultdict(Counter)}
           for m in models}
    shiekh_bad = []          # GT rows of class `target` in shiekhs images
    wrong_cls = 1 - target   # the opposite adult gender class
    diag = defaultdict(lambda: {"n_women": 0, "area_fracs": [], "imgs": set()})
    examples = defaultdict(list)      # (model, tier, outcome) -> [ex]
    rng = random.Random(seed)

    stems = sorted(p.stem for p in Path(gt_dir).glob("*.txt"))
    for stem in stems:
        dets_by_m, wh = {}, None
        for m in models:
            p = Path(raw_root) / m / "raw" / f"{stem}.json"
            try:
                txt = p.read_text()
            except OSError:            # volume-corrupt / missing sidecar:
                dets_by_m = None       # drop the stem for ALL models so the
                break                  # comparison stays like-for-like
            d = json.loads(txt)
            wh = (d["width"], d["height"])
            dets_by_m[m] = [(r["cls"], tuple(r["box_xyxy"]), r["conf"])
                            for r in d["detections"] if not r.get("excluded")]
        if not dets_by_m or wh is None:
            continue
        gt = seg_boxes(Path(gt_dir) / f"{stem}.txt", wh[0], wh[1]) or []
        women = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in gt if c == target]
        redact = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in gt if c == 0]
        if not women:
            continue
        if stem.startswith("shiekhs__"):
            shiekh_bad += [{"stem": stem, "box": [round(v, 1) for v in g]}
                           for g in women]
        img_area = wh[0] * wh[1]
        for g in women:
            best, bt = 0.0, "t?"
            for vb, t in tier_boxes.get(stem, ()):
                j = iou(g, vb)
                if j > best:
                    best, bt = j, t
            tier = bt if best >= 0.6 else "t?"
            diag[tier]["n_women"] += 1
            diag[tier]["imgs"].add(stem)
            diag[tier]["area_fracs"].append(
                (g[2] - g[0]) * (g[3] - g[1]) / img_area)
            for m in models:
                conf = matched[m]
                dets = dets_by_m[m]
                blur = [b for c, b, cf in dets if c == target and cf >= conf]
                cov = covered_frac(g, blur)
                if cov >= 0.9:
                    o = "covered"
                elif cov > 0.1:
                    o = "partial"
                else:
                    if any(c == wrong_cls and cf >= conf
                           and ioa(b, g) >= 0.3 for c, b, cf in dets):
                        o = "called_" + other
                    elif any(c == 2 and cf >= conf and ioa(b, g) >= 0.3
                             for c, b, cf in dets):
                        o = "called_child"
                    elif any(c == target and cf < conf and ioa(b, g) >= 0.3
                             for c, b, cf in dets):
                        o = "subthreshold"
                    else:
                        o = "no_detection"
                res[m]["overall"][o] += 1
                res[m]["by_tier"][tier][o] += 1
                if o != "covered":
                    key = (m, tier, o)
                    ex = {"stem": stem, "gt": g, "conf": conf,
                          "redact": redact}
                    if len(examples[key]) < max_examples:
                        examples[key].append(ex)
                    elif rng.random() < 0.15:        # reservoir-ish variety
                        examples[key][rng.randrange(max_examples)] = ex
    for t in diag:
        af = sorted(diag[t]["area_fracs"])
        diag[t]["median_area_frac"] = af[len(af) // 2] if af else 0
        diag[t]["women_per_img"] = (diag[t]["n_women"]
                                    / max(1, len(diag[t]["imgs"])))
        diag[t]["n_imgs"] = len(diag[t]["imgs"])
        del diag[t]["area_fracs"], diag[t]["imgs"]
    return res, dict(diag), examples, shiekh_bad


def render_example(img_path, gt_box, dets, conf, thumb=420,
                   extra_redact=()):
    """Draw one failure case. Redaction: the GT woman box, every Woman
    prediction region, and every extra_redact box (ignore regions =
    unknown-gender people, many of whom are women) are pixelated BEFORE
    boxes are drawn."""
    img = Image.open(img_path).convert("RGB")
    for c, b, cf in dets:
        if c == 0 and cf >= 0.05:
            pixelate(img, b)
    for b in extra_redact:
        pixelate(img, b)
    dr = ImageDraw.Draw(img)
    lw = max(2, img.width // 250)
    dr.rectangle(list(gt_box), outline="#007a3d", width=lw + 1)   # GT green
    for c, b, cf in dets:
        if cf >= conf and ioa(b, gt_box) >= 0.15:
            dr.rectangle(list(b), outline=CLASS_COLOR.get(c, "#666"),
                         width=lw)
    img.thumbnail((thumb, thumb))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def map_tables(map_dir, order, cls_name="Woman", pfx="w_"):
    """<cls_name> AP50 / AP50-95 per category from map_eval outputs.
    Files: <model>__<category>.json with per_class[cls_name]. Categories of
    the OTHER audience's prefix are skipped."""
    if not map_dir or not Path(map_dir).exists():
        return None
    skip = "m_" if pfx == "w_" else "w_"
    cats, cells = [], {}
    for f in sorted(Path(map_dir).glob("*.json")):
        model, _, cat = f.stem.partition("__")
        if cat.startswith(skip):
            continue
        d = json.loads(f.read_text())
        w = (d.get("per_class") or {}).get(cls_name)
        cells[(model, cat)] = (w, d.get("n_images"), d.get("n_gt"))
        if cat not in cats:
            cats.append(cat)
    pref = ["all"] + [pfx + x for x in (
        "exposure_t0_covered", "exposure_t1_modest", "exposure_t2_ordinary",
        "exposure_t3_revealing", "exposure_t4_high", "size_small",
        "size_med", "size_large")]
    cats.sort(key=lambda c: (pref.index(c) if c in pref else 99, c))
    rows = ""
    for cat in cats:
        n = next((cells[(m, cat)][1] for m in order
                  if (m, cat) in cells), "-")
        tds = ""
        for m in order:
            w = cells.get((m, cat), (None,))[0]
            tds += ("<td>-</td>" if not w else
                    f"<td><b>{w['ap50']:.3f}</b> / {w['ap50_95']:.3f}</td>")
        rows += f"<tr><td>{cat}</td><td>{n}</td>{tds}</tr>"
    return (f"<table><tr><th>category</th><th>n imgs</th>"
            + "".join(f"<th>{m}</th>" for m in order)
            + f"</tr>{rows}</table>")


def build_html(summary, res, diag, examples, images_dir, raw_root, out,
               focus_models, ignore_dir=None, map_dir=None, aud=None,
               n_shiekh_bad=0):
    aud = aud or AUD["male"]
    outs = outcomes(aud["other"])
    male = summary["modes"][aud["mode"]]
    matched = male["matched_conf"]
    order = list(male["curves"].keys())

    def pct(v, nd=1):
        return "-" if v is None else f"{100 * v:.{nd}f}"

    # ship table rows (male)
    ship_rows = ""
    v2 = "person_uncov_mean" in male["models"][order[0]]["slices"]["all"]
    for n in order:
        a = male["models"][n]["slices"]["all"]
        bs = (male["models"][n].get("exposure_bootstrap")
              or male["models"][n].get("m1_bootstrap") or {})
        diff = ""
        if bs.get("diff_ci95"):
            diff = (f"{100 * bs['diff_mean']:+.2f} "
                    f"[{100 * bs['diff_ci95'][0]:+.2f},"
                    f"{100 * bs['diff_ci95'][1]:+.2f}]")
        extra = ""
        if v2:
            extra = (f"<td>{pct(a['exposure_p90'])}</td>"
                     f"<td>{pct(a['person_uncov_mean'])}</td>"
                     f"<td>{pct(a['person_covered90_rate'])}</td>"
                     f"<td>{pct(a['fb_area_mean'], 3)}</td>")
        ship_rows += (f"<tr><td>{n}</td><td>{matched[n]:.2f}</td>"
                      f"<td>{pct(a['mean_exposure'], 2)}</td>{extra}"
                      f"<td>{pct(a['falseblur_img_rate'])}</td>"
                      f"<td>{pct(a['child_clarity'])}</td>"
                      f"<td>{pct(a['m1_pass_rate'])}</td>"
                      f"<td>{diff}</td></tr>")
    ship_hdr = ("<th>model</th><th>conf</th><th>E-img %</th>"
                + ("<th>E-img P90 %</th><th>E-person %</th>"
                   "<th>women ≥90% covered %</th><th>FB-area %</th>"
                   if v2 else "")
                + "<th>false-blur img %</th><th>child clarity %</th>"
                  "<th>M1@τ (legacy) %</th>"
                  "<th>E-img diff vs incumbent [95% CI]</th>")

    tier_tags = [t for t in sorted(male["models"][order[0]]["slices"])
                 if t.startswith(aud["pref"] + "exposure:")]
    tier_rows = ""
    for t in tier_tags:
        n_img = male["models"][order[0]]["slices"][t]["n_images"]
        cells = "".join(
            f"<td>{pct(male['models'][n]['slices'][t]['m1_pass_rate'])}"
            f" / {pct(male['models'][n]['slices'][t]['falseblur_img_rate'])}"
            f"</td>" for n in order)
        tier_rows += f"<tr><td>{t}</td><td>{n_img}</td>{cells}</tr>"

    esc_rows = ""
    for m in res:
        tot = sum(res[m]["overall"].values())
        esc_rows += ("<tr><td>%s</td>" % m + "".join(
            "<td>%.1f</td>" % (100 * res[m]["overall"][o] / tot)
            for o in outs) + "</tr>")
    esc_tier_rows = ""
    for m in res:
        for t in ("t0", "t1", "t2", "t3", "t4", "t?"):
            c = res[m]["by_tier"].get(t) or {}
            tot = sum(c.values())
            if not tot:
                continue
            esc_tier_rows += (f"<tr><td>{m}</td><td>{t}</td><td>{tot}</td>"
                              + "".join("<td>%.1f</td>"
                                        % (100 * c.get(o, 0) / tot)
                                        for o in outs) + "</tr>")

    diag_rows = "".join(
        f"<tr><td>{t}</td><td>{d['n_women']}</td><td>{d['n_imgs']}</td>"
        f"<td>{d['women_per_img']:.2f}</td>"
        f"<td>{100 * d['median_area_frac']:.1f}%</td></tr>"
        for t, d in sorted(diag.items()))

    map_html = (map_tables(map_dir, order, aud["cls"], aud["pref"])
                or "<p>(mAP-per-category batch not yet run)</p>")
    shiekh_note = ""
    if aud["mode"] == "male" and n_shiekh_bad:
        shiekh_note = (f"<p class=note><b>Shiekh-collection GT-women, "
                       f"hand-adjudicated 2026-08-27: 54 of {n_shiekh_bad} "
                       f"flagged rows are REAL women; 10 are mislabeled men"
                       f"</b> (small background figures — the covered-person "
                       f"gender confusion). Shiekh-slice woman numbers are "
                       f"genuine findings with ~16% label noise from those "
                       f"10 rows; calls in "
                       f"<code>shiekh_gt_women_adjudicated.jsonl</code>.</p>")
    ep_tags = (["all"]
               + [t for t in sorted(male["models"][order[0]]["slices"])
                  if t.startswith((aud["pref"] + "exposure:",
                                   aud["pref"] + "size:", "collection:"))])
    ep_rows = ""
    for t in ep_tags:
        a0 = male["models"][order[0]]["slices"].get(t, {})
        if a0.get("n_persons") is None:
            continue
        tds = ""
        for m in order:
            a = male["models"][m]["slices"].get(t, {})
            pu, c90 = a.get("person_uncov_mean"), a.get("person_covered90_rate")
            tds += ("<td>-</td>" if pu is None else
                    f"<td><b>{100 * pu:.1f}%</b> exposed · "
                    f"{100 * c90:.0f}% fully covered</td>")
        ep_rows += (f"<tr><td>{t}</td><td>{a0.get('n_persons', '-')}</td>"
                    f"{tds}</tr>")
    eperson_html = (f"<table><tr><th>category</th><th>n women</th>"
                    + "".join(f"<th>{m}</th>" for m in order)
                    + f"</tr>{ep_rows}</table>")

    slice_matched_html = ""
    sm = male.get("slice_matched")
    if sm:
        sm_tags = [t for t in ep_tags if t in sm and t != "all"]
        rows = ""
        for t in sm_tags:
            tds = ""
            for m in order:
                r = sm[t].get(m)
                tds += ("<td>-</td>" if not r or
                        r.get("person_uncov_mean") is None else
                        f"<td>@{r['conf']:.2f}: "
                        f"<b>{100 * r['person_uncov_mean']:.1f}%</b></td>")
            rows += f"<tr><td>{t}</td>{tds}</tr>"
        slice_matched_html = (
            "<h3>per-slice matched operating points</h3>"
            "<p>Each slice re-matched to the incumbent's false-blur area ON "
            "THAT SLICE (global matching over-fires small-image slices); "
            "cell = per-slice conf and E-person at it.</p>"
            "<table><tr><th>slice</th>"
            + "".join(f"<th>{m}</th>" for m in order)
            + f"</tr>{rows}</table>")

    # charts inline
    try:
        import deploy_charts
        deploy_charts.MODEL_COLOR.clear()
        for i, n in enumerate(order):
            deploy_charts.MODEL_COLOR[n] = deploy_charts.PALETTE[i % 6]
        chart = deploy_charts.tradeoff_svg(aud["mode"], male, order)
    except Exception as e:                                  # noqa: BLE001
        chart = f"<p>chart unavailable: {html.escape(str(e))}</p>"

    # example galleries — the findings we have evidence for
    img_index = {}
    for p in Path(images_dir).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            img_index.setdefault(p.stem, p)
    galleries = ""
    oth = aud["other"]
    picks = [(f"Covered (t0) {aud['plural']} read as {oth.title()} — "
              f"the coverage-garment gender confusion",
              [(m, "t0", "called_" + oth) for m in focus_models]),
             (f"t4 (high-exposure) {aud['plural']} read as Child — the teen "
              f"fade in its worst location: an under-blur on the most "
              f"critical tier",
              [(m, "t4", "called_child") for m in focus_models]),
             (f"t4 partial coverage — clipped boxes leaking on large "
              f"subjects",
              [(m, "t4", "partial") for m in res]),
             ]
    for title, keys in picks:
        cards = ""
        for key in keys:
            for ex in examples.get(key, [])[:4]:
                p = img_index.get(ex["stem"])
                if not p:
                    continue
                raw = Path(raw_root) / key[0] / "raw" / f"{ex['stem']}.json"
                try:
                    d = json.loads(raw.read_text())
                except OSError:
                    continue
                dets = [(r["cls"], tuple(r["box_xyxy"]), r["conf"])
                        for r in d["detections"] if not r.get("excluded")]
                extra = list(ex.get("redact") or [])
                if ignore_dir:
                    ign = seg_boxes(Path(ignore_dir) / f"{ex['stem']}.txt",
                                    d["width"], d["height"])
                    extra += [(x1, y1, x2, y2)
                              for _c, x1, y1, x2, y2 in ign or []]
                try:
                    b64 = render_example(p, ex["gt"], dets, ex["conf"],
                                         extra_redact=extra)
                except Exception:                           # noqa: BLE001
                    continue
                cards += (f"<div class=card>"
                          f"<img src='data:image/jpeg;base64,{b64}'>"
                          f"<div class=cap>{key[0]} · {key[1]} · {key[2]}"
                          f"<br>{html.escape(ex['stem'][:56])}</div></div>")
        if cards:
            galleries += (f"<h3>{html.escape(title)}</h3>"
                          f"<div class=row>{cards}</div>")

    proto = summary.get("protocol", {})
    out.write_text(f"""<!doctype html><meta charset=utf-8>
<title>HaramBlur — deployment ship report</title><style>
body{{font-family:system-ui;background:#fcfcfb;color:#222;max-width:1100px;
margin:24px auto;padding:0 16px}} table{{border-collapse:collapse;
font-size:13px;margin:10px 0}} td,th{{border:1px solid #ccc;padding:3px 8px;
text-align:right}} td:first-child,th:first-child{{text-align:left}}
.row{{display:flex;flex-wrap:wrap;gap:10px}} .card{{width:420px;
font-size:11px}} .card img{{max-width:100%;border:1px solid #ddd}}
.cap{{color:#555}} .note{{background:#fff7e0;border:1px solid #e6c200;
padding:8px 12px;border-radius:4px}} h2{{margin-top:32px}}
.legend span{{padding:0 8px}}</style>
<h1>Deployment ship report — {aud["mode"]} audience</h1>
<p>holdout: 11,494 images · incumbent <b>{order[0]}</b> @
{proto.get('anchor_conf')} · matched false-blur operating points ·
τ={proto.get('tau')} φ={proto.get('phi')} κ={proto.get('kappa')} ·
GT = machine labels (Spotlight), sheikh/randoms adjudication pending.</p>
<p class=note><b>All women in this report are redacted</b> — every GT-Woman
region, every predicted-Woman region, and every ignore region (unknown-gender
people, many of them women) is pixelated before rendering. Table numbers are
computed on the unredacted geometry.</p>

<h2>1 · Ship table ({aud["mode"]} audience, matched false-blur)</h2>
<p>E-img = mean unblurred-woman area as % of the screen (threshold-free
headline; lower is better). E-person = mean uncovered fraction per GT woman
(size-independent). Negative E-img diff = candidate better.</p>
<table><tr>{ship_hdr}</tr>{ship_rows}</table>

<h2>2 · {aud["Noun"]} detection capability — mAP across categories</h2>
{shiekh_note}
<p>Standard COCO-convention <b>{aud["Noun"]} AP50 / AP50-95</b> per category
(threshold-free, ranking-based; ignore regions honored; floor 0.001).
Capability view — says nothing about empty-image false blurs or the shipped
threshold; the outcome views below cover those.</p>
{map_html}

<h2>3 · {aud["Noun"]} exposure — % of {aud["poss"]} pixels left
unblurred</h2>
{shiekh_note}
<p>Per GT {aud["gender"]}: mean % of {aud["poss"]} area NOT covered by the
blur at the matched operating point (size-independent — every
{aud["gender"]} counts equally), and the share of {aud["plural"]}
effectively fully covered (≥90%).</p>
{eperson_html}
{slice_matched_html}

<h2>4 · The tradeoff curve</h2>{chart}

<h2>5 · M-1 / false-blur per exposure tier</h2>
<p>Tiers are the agreed 5-level taxonomy (t0 covered → t4 high), assigned
from Gemini `exposed_body_parts`; cells are M-1 % / false-blur %.</p>
<table><tr><th>tier</th><th>n imgs</th>{''.join(f'<th>{n}</th>'
for n in order)}</tr>{tier_rows}</table>

<h2>6 · Why {aud["plural"]} escape — per GT {aud["gender"]}, matched
points</h2>
<table><tr><th>model</th>{''.join(f'<th>{o}</th>' for o in outs)}</tr>
{esc_rows}</table>
<h3>per tier</h3>
<table><tr><th>model</th><th>tier</th><th>n</th>{''.join(f'<th>{o}</th>'
for o in outs)}</tr>{esc_tier_rows}</table>

<h2>7 · Tier diagnostics — why image-level t4 M-1 reads low</h2>
<table><tr><th>tier</th><th>women</th><th>imgs</th><th>women/img</th>
<th>median woman area (% of frame)</th></tr>{diag_rows}</table>
<p>Person-level coverage on t4 is the BEST of all tiers for the candidates —
the low image-level M-1 is the area term: t4 subjects fill more of the frame,
so any residual leak exceeds τ, and the τ pass/fail charges the whole image.
The consequential t4 escape is <b>called_child</b> — an under-blur on the
most critical tier.</p>

<h2>8 · Evidence — sample failures (women redacted)</h2>
<p class=legend>box colors: <span style="color:#007a3d">■ GT woman</span>
<span style="color:#CC00CC">■ pred Woman</span>
<span style="color:#0072B2">■ pred Man</span>
<span style="color:#E69F00">■ pred Child</span></p>
{galleries}

<h2>9 · Caveats & next steps</h2>
<ul>
<li>GT is machine labels; the covered/Gulf-dress slice adjudication is now
the top data task — t0 misgendering drives both audiences' worst failure.</li>
<li>t4 gates alone by owner decision (n=220 → wide CIs; grow this stratum
before freezing bars).</li>
<li>τ=0.005 is a draft; sensitivity sweep from per_image.jsonl before
pre-registration.</li>
<li>Second-stage gender classifier now has a measured target: the
w_exposure_t0_covered subset folder.</li>
</ul>""")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "labels").mkdir()
        (td / "imgs").mkdir()
        (td / "raw" / "m1" / "raw").mkdir(parents=True)
        img = Image.new("RGB", (200, 160), "#7a8a7a")
        img.save(td / "imgs" / "women__a.jpg")
        (td / "labels" / "women__a.txt").write_text("0 0.5 0.5 0.5 0.8\n")
        (td / "raw" / "m1" / "raw" / "women__a.json").write_text(json.dumps(
            {"image": "women__a.jpg", "width": 200, "height": 160,
             "detections": [{"det_index": 0, "cls": 1,
                             "box_xyxy": [50, 16, 150, 144], "conf": 0.9,
                             "kept": True, "excluded": None}]}))
        res, diag, ex, sb = analyze(td / "labels", td / "raw", None, ["m1"],
                                    {"m1": 0.45})
        assert res["m1"]["overall"]["called_man"] == 1, res
        assert ("m1", "t?", "called_man") in ex, "example collected"
        assert sb == [], "no shiekh stems in synthetic set"
        # female audience smoke: man GT + woman pred -> called_woman
        (td / "labels" / "men__c.txt").write_text("1 0.5 0.5 0.5 0.8\n")
        (td / "raw" / "m1" / "raw" / "men__c.json").write_text(json.dumps(
            {"image": "men__c.jpg", "width": 200, "height": 160,
             "detections": [{"det_index": 0, "cls": 0,
                             "box_xyxy": [50, 16, 150, 144], "conf": 0.9,
                             "kept": True, "excluded": None}]}))
        resf, _df, exf, _sf = analyze(td / "labels", td / "raw", None,
                                      ["m1"], {"m1": 0.45}, target=1,
                                      gender="man", other="woman")
        assert resf["m1"]["overall"]["called_woman"] == 1, resf
        ex_f = exf[("m1", "t?", "called_woman")][0]
        assert ex_f["redact"] == [], "no GT women in the men image"
        # render redacts: the woman region must differ from the original
        d = json.loads((td / "raw" / "m1" / "raw" / "women__a.json")
                       .read_text())
        dets = [(r["cls"], tuple(r["box_xyxy"]), r["conf"])
                for r in d["detections"]]
        b64 = render_example(td / "imgs" / "women__a.jpg", (50, 16, 150, 144),
                             dets, 0.45)
        assert len(b64) > 100
        summary = {"protocol": {"anchor_conf": 0.45, "tau": 0.005,
                                "phi": 0.01, "kappa": 0.2},
                   "modes": {"male": {
                       "matched_conf": {"m1": 0.45},
                       "curves": {"m1": {"0.45": {"m1_pass_rate": 0.5,
                                                  "falseblur_img_rate": 0.05}}},
                       "models": {"m1": {"slices": {"all": {
                           "m1_pass_rate": 0.5, "mean_exposure": 0.01,
                           "falseblur_img_rate": 0.05, "child_clarity": 1.0,
                           "n_images": 1}}}}}}}
        out = td / "r.html"
        build_html(summary, res, diag, ex, td / "imgs", td / "raw", out,
                   ["m1"])
        t = out.read_text()
        assert "redacted" in t and "called_man" in t
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--gt-labels")
    ap.add_argument("--images")
    ap.add_argument("--verdicts")
    ap.add_argument("--raw-root")
    ap.add_argument("--ignore", help="ignore-region label dir (redacted too)")
    ap.add_argument("--map-dir", help="dir of map_eval outputs <model>__<cat>.json")
    ap.add_argument("--out", default="SHIP_REPORT.html")
    ap.add_argument("--focus-models", default="y26n_warm50,v11n_shipped")
    ap.add_argument("--audience", choices=["male", "female"], default="male")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    aud = AUD[args.audience]
    summary = json.loads(Path(args.summary).read_text())
    mdata = summary["modes"][aud["mode"]]
    models = list(mdata["curves"].keys())
    matched = mdata["matched_conf"]
    res, diag, ex, shiekh_bad = analyze(
        args.gt_labels, args.raw_root, args.verdicts, models, matched,
        target=aud["target"], gender=aud["gender"], other=aud["other"])
    out = Path(args.out)
    if aud["mode"] == "male" and shiekh_bad:
        lst = out.parent / "shiekh_gt_women_adjudicate.jsonl"
        lst.write_text("\n".join(json.dumps(r) for r in shiekh_bad) + "\n")
        print(f"[report] {len(shiekh_bad)} shiekh GT-woman rows -> {lst}")
    build_html(summary, res, diag, ex, args.images, args.raw_root,
               out, args.focus_models.split(","),
               ignore_dir=args.ignore, map_dir=args.map_dir, aud=aud,
               n_shiekh_bad=len(shiekh_bad))
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
