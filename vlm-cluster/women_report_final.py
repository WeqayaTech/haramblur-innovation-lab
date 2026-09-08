#!/usr/bin/env python3
"""
FINAL audience report — each model at its ONE selected deployable
threshold, written for coworkers (plain language), with the false-positive
verification against the PASS empty-scenes set.

Two audiences, one tool:
  --audience male    (default) blur target = Woman -> WOMEN_REPORT_FINAL
  --audience female  blur target = Man            -> MEN_REPORT_FINAL

    python3 women_report_final.py --summary .../holdout_run4/summary.json \\
        --selected "v11n_shipped=0.30,y26n_sop50=0.15,..." \\
        --pass-dir /workspace/deploycmp/pass_eval \\
        --gt-labels ... --ignore ... --raw-root ... --slices ... \\
        --map-dir ... --tier-cache ... --tier-cache-045 ... --out ...

    python3 women_report_final.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BG, INK = "#fcfcfb", "#222222"

from concurrent.futures import ThreadPoolExecutor

_PASS_CACHE = {}
_GT_CACHE = {}


def _read_json(p):
    try:
        return p, json.loads(p.read_text())
    except OSError as e:
        print(f"[load] WARNING unreadable, skipped: {p} ({e})")
        return p, None


def parallel_jsons(paths, workers=16):
    """Network-volume small-file reads are latency-bound — thread them."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(_read_json, paths))


def _c(v):
    return "-" if v is None else f"{v:.2f}"

AUD = {
    "male": dict(
        mode="male", target=0, cls="Woman", noun="woman", nouns="women",
        Noun="Woman", other="man", others="men", pref="w_exposure:",
        poss="her", user="male user", other_user="female user",
        approx_n="~5,700", title="Women blur — final model comparison",
        t_desc={
            "t0_covered": "only face and hands visible (hijab-level)",
            "t1_modest": "hair/neck visible, body covered",
            "t2_ordinary": "arms/shoulders visible (t-shirt level)",
            "t3_revealing": "neckline, shorts/skirt, or bare back",
            "t4_high": "midriff visible or swimwear-level"},
        pattern="""
<p>The pattern to notice: every model is at its <b>worst on fully covered
(t0) women</b> — both by AP50 and at-threshold coverage — and at its best
on the revealing tiers. The failure is not "the model can't see them"; it
is that covered women are often classified as men, so no woman-blur
fires. That is a training-data problem, not a threshold problem.</p>""",
        extra_caveats="""
<li><b>Shiekh-collection GT-women, adjudicated by hand (2026-08-27):
54 of the 64 flagged "woman" rows are REAL women; 10 are mislabeled men
</b> (all small background figures — the covered-person gender confusion).
So the shiekh-context woman numbers are genuine product findings carrying
~16% label noise from those 10 rows (listed with the human calls in
shiekh_gt_women_adjudicated.jsonl); the earlier "no women in this
collection" assumption is retired.</li>
<li>Covered women (hijab-level) are every model's weakest group — they get
misread as men several times more often than any other group. That is the
next data/labeling task, not a threshold problem.</li>"""),
    "female": dict(
        mode="female", target=1, cls="Man", noun="man", nouns="men",
        Noun="Man", other="woman", others="women", pref="m_exposure:",
        poss="his", user="female user", other_user="male user",
        approx_n="~6,700", title="Men blur — final model comparison",
        t_desc={
            "t0_covered": "only face and hands visible (thobe/ghutra or "
                          "full traditional dress)",
            "t1_modest": "hair/neck visible, body covered",
            "t2_ordinary": "arms/shoulders visible (t-shirt level)",
            "t3_revealing": "chest, shorts, or bare back visible",
            "t4_high": "midriff visible / shirtless-level"},
        pattern="""
<p>The pattern to notice — and it is the OPPOSITE of the women's report:
<b>fully covered (t0) men are every model's BEST male tier</b> (the
training data is rich in thobe/traditional dress), and accuracy falls as
exposure rises, bottoming at t2–t3. The cause table behind this shows the
escapes are dominated by <b>men misgendered as women</b>, at a rate that
climbs with skin exposure (≈1% of t0 men → 5–13% of t3 men, model
dependent). Together with the women's report (covered women misread as
men) this is one learned shortcut seen from both sides: <b>the models use
clothing coverage as a gender proxy</b> — covered ⇒ man, exposed skin ⇒
woman. Note: the men's test collection deliberately includes
gender-confusion probes (e.g. long-haired men), which likely concentrate
in t2–t3 and may amplify the measured rate.</p>""",
        extra_caveats="""
<li>The shiekhs collection (2,897 images of thobe-clad men) is the CORE
slice for this audience — a female user's blur must work exactly there.
Covered (t0) men misread as women is the same sheikh gender confusion
seen from the escape side: no man-blur fires on him.</li>
<li>t4 (shirtless-level) contains only ~31 men in this test set — treat
that row as anecdotal, not statistical.</li>"""),
}

TIER_ORDER = ["t0_covered", "t1_modest", "t2_ordinary", "t3_revealing",
              "t4_high"]
TIER_TITLE = {"t0_covered": "t0 — fully covered", "t1_modest": "t1 — modest",
              "t2_ordinary": "t2 — ordinary", "t3_revealing": "t3 — revealing",
              "t4_high": "t4 — high exposure"}


def tier_tags(aud):
    return [aud["pref"] + t for t in TIER_ORDER]


def all_ap(map_dir, models, cls="Woman"):
    out = {}
    for m in models:
        f = Path(map_dir or "") / f"{m}__all.json"
        if f.exists():
            w = (json.loads(f.read_text()).get("per_class") or {}).get(cls)
            if w:
                out[m] = (w["ap50"], w["ap50_95"])
    return out


def pass_fp(raw_dir: Path, conf: float, target_cls: int = 0):
    """PASS replay: false persons per 100 empty images at `conf`.
    Files are read ONCE per dir (parallel), then memoized — later calls at
    other thresholds replay from memory."""
    key = str(raw_dir)
    if key not in _PASS_CACHE:
        pairs = parallel_jsons(sorted(Path(raw_dir).glob("*.json")))
        _PASS_CACHE[key] = [d for _p, d in pairs if d is not None]
    n_img = tot = tgt = imgs_hit = 0
    for d in _PASS_CACHE[key]:
        n_img += 1
        hit = 0
        for r in d.get("detections", []):
            if r.get("excluded") or r["conf"] < conf:
                continue
            hit += 1
            tot += 1
            if r["cls"] == target_cls:
                tgt += 1
        imgs_hit += bool(hit)
    if not n_img:
        return None
    return {"n_images": n_img, "fp_per_100": 100.0 * tot / n_img,
            "cls_fp_per_100": 100.0 * tgt / n_img,
            "img_rate": 100.0 * imgs_hit / n_img}


def tier_metrics(selected, gt_dir, ignore_dir, raw_root, slices_file,
                 target=0, tags=None):
    """Per-tier coverage at each model's SELECTED threshold (one pass)."""
    from deploy_compare import (load_sidecars, image_metrics, aggregate,
                                load_ignores, seg_boxes)
    sl = json.loads(Path(slices_file).read_text())
    tag_of = {st: set(t) for st, t in sl["slices"].items()}
    tiers = tags or [("w_exposure:" + t) for t in TIER_ORDER]
    out = {}
    for m, cf in selected.items():
        sc = load_sidecars(Path(raw_root) / m / "raw")
        thr = {"tau": 0.005, "phi": 0.01, "kappa": 0.2, "dilate": 0.1,
               "conf": cf}
        recs = {}
        for stem, (w, h, dets) in sc.items():
            gt = seg_boxes(Path(gt_dir) / f"{stem}.txt", w, h)
            if gt is None:
                continue
            ign = load_ignores(ignore_dir, stem, w, h)
            recs[stem] = image_metrics(gt, ign, dets, target, w, h, thr)
        out[m] = aggregate(recs, tag_of, tags=tiers)
    return tiers, out


def tier_metrics_multi(sel_list, gt_dir, ignore_dir, raw_root, slices_file,
                       target=0, tags=None, workers=16):
    """N threshold-sets, ONE read pass per model (parallel IO). Returns
    (tiers, [out_per_sel...]). The fused/threaded version of tier_metrics —
    the serial two-pass form took ~1h on a cold network volume."""
    from deploy_compare import (image_metrics, aggregate, load_ignores,
                                seg_boxes)
    sl = json.loads(Path(slices_file).read_text())
    tag_of = {st: set(t) for st, t in sl["slices"].items()}
    tiers = tags or [("w_exposure:" + t) for t in TIER_ORDER]
    models = sorted(set().union(*[set(s) for s in sel_list]))
    outs = [dict() for _ in sel_list]
    for m in models:
        pairs = parallel_jsons(
            sorted((Path(raw_root) / m / "raw").glob("*.json")), workers)
        sc = {}
        for p, d in pairs:
            if d is None:
                continue
            dets = [(r["cls"], *r["box_xyxy"], r["conf"])
                    for r in d.get("detections", []) if not r.get("excluded")]
            sc[p.stem] = (d["width"], d["height"], dets)
        gt_dir = Path(gt_dir)

        def _load_gt(stem):
            w, h, _ = sc[stem]
            return (stem, seg_boxes(gt_dir / f"{stem}.txt", w, h),
                    load_ignores(ignore_dir, stem, w, h))
        with ThreadPoolExecutor(workers) as ex:
            gtmap = {s: (g, i) for s, g, i in ex.map(_load_gt, sc)}
        for si, sel in enumerate(sel_list):
            thr = {"tau": 0.005, "phi": 0.01, "kappa": 0.2, "dilate": 0.1,
                   "conf": sel[m]}
            recs = {}
            for stem, (w, h, dets) in sc.items():
                gt, ign = gtmap[stem]
                if gt is None:
                    continue
                recs[stem] = image_metrics(gt, ign, dets, target, w, h, thr)
            outs[si][m] = aggregate(recs, tag_of, tags=tiers)
        print(f"[tiers] {m}: {len(sc)} images x {len(sel_list)} "
              f"threshold-sets done")
    return tiers, outs


def tier_ap50(map_dir, models, aud):
    out = {}
    if not map_dir:
        return out
    for t in tier_tags(aud):
        cat = t.replace(":", "_")
        for m in models:
            f = Path(map_dir) / f"{m}__{cat}.json"
            if not f.exists():
                continue
            w = (json.loads(f.read_text()).get("per_class")
                 or {}).get(aud["cls"])
            if w:
                out.setdefault(t, {})[m] = (w["ap50"], w["ap50_95"])
    return out


def nearest_conf(curves, conf):
    return min(curves, key=lambda k: abs(float(k) - conf))


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def wrong_area_curves(raw_dir, gt_dir, ignore_dir, grid, target,
                      workers=16):
    """The FP metric, per threshold: mean over images of the area of
    fired target-class boxes that match NO GT target (and no ignore
    region) at IoU >= 0.5, as a fraction of the image. A box that
    matches a target at IoU >= 0.5 is a legitimate blur and contributes
    zero. gt_dir None = person-free arm (every fired box is wrong).
    A box's wrongness is threshold-independent, so it is judged once
    and only the union is redone per threshold."""
    from deploy_compare import seg_boxes, _union_area_within, load_ignores
    pairs = parallel_jsons(sorted(Path(raw_dir).glob("*.json")), workers)
    sums = {cf: 0.0 for cf in grid}
    n = 0
    for p, d in pairs:
        if d is None:
            continue
        w, h = d["width"], d["height"]
        legit = []
        if gt_dir is not None:
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
            legit = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in gt
                     if c == target]
            legit += ign
        n += 1
        wrong = []
        for r in d.get("detections", []):
            if r.get("excluded") or r["cls"] != target:
                continue
            b = tuple(r["box_xyxy"])
            if not any(_iou(b, g) >= 0.5 for g in legit):
                wrong.append((r["conf"], b))
        if not wrong:
            continue
        win = (0.0, 0.0, float(w), float(h))
        area = float(w * h)
        for cf in grid:
            boxes = [b for c2, b in wrong if c2 >= cf]
            if boxes:
                sums[cf] += _union_area_within(win, boxes) / area
    return {cf: (sums[cf] / n if n else None) for cf in grid}


def budget_picks(md, fp_curves, order, budget_conf=0.45):
    """The ONE selection rule, two metrics total: minimize the %-of-a-
    person-left-visible (TP side) subject to the wrongly-blurred area
    (FP side, IoU-0.5-judged) being no worse than the incumbent (first
    model) at `budget_conf`, on BOTH datasets: holdout and the
    person-free PASS set. Both metrics move monotonically with the
    threshold, so this is the standard fixed-FP operating point
    (Neyman-Pearson): the lowest threshold whose wrong-blur area still
    beats production. Returns (budget, picks, feasible_sets)."""
    inc = order[0]
    fh = (fp_curves or {}).get("holdout", {})
    fp = (fp_curves or {}).get("pass", {})

    def at(m, cf):
        return md["curves"][m][nearest_conf(md["curves"][m], cf)]

    grid = sorted(float(k) for k in md["curves"][inc])
    bcf = min(grid, key=lambda c: abs(c - budget_conf))
    budget = {"conf": budget_conf,
              "wrong_holdout": fh.get(inc, {}).get(bcf),
              "wrong_pass": fp.get(inc, {}).get(bcf)}
    feas, picks = {}, {}
    for m in order:
        feas[m] = set()
        for cf in grid:
            vh = fh.get(m, {}).get(cf)
            vp = fp.get(m, {}).get(cf)
            if ((budget["wrong_holdout"] is None or vh is None
                 or vh <= budget["wrong_holdout"] + 1e-12)
                    and (budget["wrong_pass"] is None or vp is None
                         or vp <= budget["wrong_pass"] + 1e-12)):
                feas[m].add(cf)
        cand = [(cf, at(m, cf)) for cf in feas[m]
                if at(m, cf).get("person_uncov_mean") is not None]
        picks[m] = (min(cand, key=lambda r:
                        (r[1]["person_uncov_mean"], r[0]))[0]
                    if cand else None)
    return budget, picks, feas


def build(summary, selected, pass_dir, out, aud=None, tiers_data=None,
          ap_data=None, tiers045=None, ap_all=None, fp_curves=None):
    aud = aud or AUD["male"]
    N, n, ns, oth = aud["Noun"], aud["noun"], aud["nouns"], aud["other"]
    md = summary["modes"][aud["mode"]]
    order = [m for m in md["curves"] if m in selected]

    # one-rule pick provenance for §1: min visibility within the FP budget
    budget, rule_picks, feas = budget_picks(md, fp_curves, order)
    fh = (fp_curves or {}).get("holdout", {})
    fpp = (fp_curves or {}).get("pass", {})

    def _wa(m, cf):
        v = fh.get(m, {}).get(cf)
        return "-" if v is None else f"{100 * v:.3f}%"

    def _wp(m, cf):
        v = fpp.get(m, {}).get(cf)
        return "-" if v is None else f"{100 * v:.3f}%"

    rows, oprows, passrows, chat = "", "", "", []
    for m in order:
        cf = selected[m]
        a = md["curves"][m][nearest_conf(md["curves"][m], cf)]
        pf = (pass_fp(Path(pass_dir) / m / "raw", cf, aud["target"])
              if pass_dir else None)
        chat.append((m, cf, a, pf))
        oprows += (f"<tr><td><b>{m}</b></td><td><b>{cf:.2f}</b></td>"
                   f"<td>{_c(rule_picks.get(m))}</td>"
                   f"<td>{100 * a['person_uncov_mean']:.1f}%</td></tr>")
        rows += (f"<tr><td><b>{m}</b> @ {cf:.2f}</td>"
                 f"<td>{100 * a['person_uncov_mean']:.1f}%</td>"
                 f"<td>{_wa(m, cf)}</td>"
                 f"<td>{_wp(m, cf)}</td></tr>")
        if pf:
            passrows += (f"<tr><td><b>{m}</b></td><td>{cf:.2f}</td>"
                         f"<td>{pf['fp_per_100']:.2f}</td>"
                         f"<td>{pf['cls_fp_per_100']:.2f}</td>"
                         f"<td>{pf['img_rate']:.2f}%</td>"
                         f"<td>{pf['n_images']}</td></tr>")

    # every selected point sits inside the same FP budget, so the model
    # with the LEAST of the person left visible is the recommendation
    best = min(chat, key=lambda t: t[2]["person_uncov_mean"])
    bud_hold = ("-" if budget["wrong_holdout"] is None
                else f"{100 * budget['wrong_holdout']:.3f}%")
    bud_pass = ("-" if budget["wrong_pass"] is None
                else f"{100 * budget['wrong_pass']:.3f}%")

    tier_html = ""
    if tiers_data:
        tiers, tdata = tiers_data
        defs = "".join(
            f"<li><b>{TIER_TITLE[t.split(':')[1]]}</b>: "
            f"{aud['t_desc'][t.split(':')[1]]}</li>" for t in tiers)
        trs = ""
        for t in tiers:
            nn = next((tdata[m][t]["n_persons"] for m in order
                       if t in tdata.get(m, {})), "-")
            tds = ""
            for m in order:
                a = tdata.get(m, {}).get(t)
                ap = (ap_data or {}).get(t, {}).get(m)
                vis = ("-" if not a or a.get("person_uncov_mean") is None
                       else f"{100 * a['person_uncov_mean']:.0f}%")
                aps = "-" if not ap else f"{ap[0]:.2f}"
                aps95 = "-" if not ap else f"{ap[1]:.2f}"
                tds += f"<td>{vis} · AP50 {aps} · AP50-95 {aps95}</td>"
            trs += (f"<tr><td>{TIER_TITLE[t.split(':')[1]]}</td>"
                    f"<td>{nn}</td>{tds}</tr>")
        heads = "".join(f"<th>{m}</th>" for m in order)
        tier_html = f"""
<h2>4 · Accuracy by how covered the {n} is</h2>
<p>Question this section answers: <i>does accuracy depend on how covered
the {n} is?</i> We grouped every {n} in the test set by how much of the
body is visible, using five tiers:</p><ul>{defs}</ul>
<p>Each cell below shows three numbers. <b>% left visible</b> — the
average share of a {n}'s body box left unblurred for {ns} in that
group, <i>at the model's selected
threshold from §1</i> (lower is better). <b>AP50</b> — the standard
detection metric (average precision at 50% box overlap, exactly as in
COCO/mAP benchmarks), computed only on the images of that group;
threshold-free, it scores whether the model finds and correctly classes
these {ns} at all. <b>AP50-95</b> — the same metric averaged over overlap
requirements from 50% up to 95% (the COCO headline number); because a
detection only counts at 95% if its box matches the person almost
exactly, this is the strict-box version — a big gap between AP50 and
AP50-95 means the model finds the {n} but draws a loose box.</p>
<table><tr><th>group</th><th>{ns}</th>{heads}</tr>{trs}</table>
{aud["pattern"]}"""

    inc = order[0]
    anchor = str(summary.get("protocol", {}).get("anchor_conf", 0.45))
    sweep_html = ""
    if anchor in md["curves"][inc]:
        for m in order:
            ap_line = ""
            if ap_all and m in ap_all:
                ap_line = (f" · threshold-free capability: {N} AP50 "
                           f"{ap_all[m][0]:.2f}, AP50-95 {ap_all[m][1]:.2f}")
            body = ""
            for cf in sorted(float(k) for k in md["curves"][m]):
                a = md["curves"][m][nearest_conf(md["curves"][m], cf)]
                mark = (" class=pick"
                        if abs(cf - selected.get(m, -1)) < 1e-9 else
                        " class=cur" if abs(cf - 0.45) < 1e-9 else "")
                body += (f"<tr{mark}><td>{cf:.2f}</td>"
                         f"<td>{100 * a['person_uncov_mean']:.1f}</td>"
                         f"<td>{_wa(m, cf)}</td>"
                         f"<td>{_wp(m, cf)}</td>"
                         f"<td>{'✓' if cf in feas.get(m, ()) else '–'}"
                         f"</td></tr>")
            sweep_html += (
                f"<h3>{m} — selected {selected.get(m, float(anchor)):.2f}"
                f"{ap_line}</h3>"
                f"<table><tr><th>conf</th>"
                f"<th>% of a {n} left visible</th>"
                f"<th>wrongly blurred % of image (holdout)</th>"
                f"<th>wrongly blurred % (PASS)</th><th>in budget?</th>"
                f"</tr>{body}</table>")
        sweep_html = f"""
<h2>5 · The confidence sweeps — how each threshold was found</h2>
<p><b>Selection-only:</b> these tables exist to pick each model's own
threshold; they are never used to compare one model against another
(§2 does that, at the picked thresholds). One table per model, one row
per threshold tried, showing only the two decision metrics. The
<b>in budget?</b> column applies the rule from §1: ✓ means the
wrongly-blurred area is at or below the production baseline
({inc} @ {budget['conf']:.2f}) on both datasets; among ✓ rows the one
leaving the least of a {n} visible is the pick — visibility rises and
wrong blur falls with the threshold, so that is simply the lowest
threshold whose wrong-blur area still beats production. The
<span class=sw-g>green row</span> is the selected threshold; the
<span class=sw-c>grey row</span> is 0.45, today's production setting.
Each model's header also shows its {N} AP50 and
AP50-95: those are <i>threshold-free by definition</i>, so they are the
fixed capability ceiling the sweep moves beneath — a model cannot sweep
its way above its own AP.</p>
{sweep_html}"""

    s045_html = ""
    if anchor in md["curves"][inc]:
        rows045 = ""
        for m in order:
            a = md["curves"][m].get("0.45")
            if not a:
                continue
            rows045 += (f"<tr><td><b>{m}</b></td>"
                        f"<td>{100 * a['person_uncov_mean']:.1f}%</td>"
                        f"<td>{_wa(m, 0.45)}</td>"
                        f"<td>{_wp(m, 0.45)}</td></tr>")
        t045 = ""
        if tiers045:
            tiers2, tdata2 = tiers045
            trs2 = ""
            for t in tiers2:
                n2 = next((tdata2[m][t]["n_persons"] for m in order
                           if t in tdata2.get(m, {})), "-")
                tds2 = "".join(
                    ("<td>-</td>" if t not in tdata2.get(m, {}) or
                     tdata2[m][t].get("person_uncov_mean") is None else
                     f"<td>{100 * tdata2[m][t]['person_uncov_mean']:.0f}%"
                     f"</td>") for m in order)
                trs2 += (f"<tr><td>{TIER_TITLE[t.split(':')[1]]}</td>"
                         f"<td>{n2}</td>{tds2}</tr>")
            heads2 = "".join(f"<th>{m}</th>" for m in order)
            t045 = (f"<h3>% of a {n} left visible by tier — everyone at "
                    f"0.45</h3>"
                    f"<p>Each cell is the average share of a {n}'s box "
                    f"left unblurred in that group, with the model at "
                    f"threshold 0.45 — the same measure as in "
                    f"§2 and §4, NOT an AP number. (AP50 / AP50-95 are "
                    f"threshold-free and therefore identical to §4's values "
                    f"— they are not repeated here.)</p>"
                    f"<table><tr><th>group</th><th>{ns}</th>{heads2}</tr>"
                    f"{trs2}</table>")
        s045_html = f"""
<h2>6 · The same comparison if every model ran at 0.45 (today's setting)</h2>
<p>Question this section answers: <i>how much of each model's win in §2
is calibration rather than capability?</i> The identical measurements
with every model at the current production threshold instead of its own
selected one.</p>
<table><tr><th>model</th><th>% of a {n} left visible</th>
<th>wrongly blurred % of image (holdout)</th>
<th>wrongly blurred % (PASS)</th></tr>
{rows045}</table>
{t045}"""

    pass_html = "<p>(PASS verification pending — sidecars not provided)</p>"
    if passrows:
        pass_html = f"""
<p>Question this section answers: <i>do the false-positive numbers hold
up without trusting our own machine labels?</i> PASS is a dataset of
3,000 images verified by humans to contain <b>no people at all</b>, so
every box a model draws there is a mistake by construction, with no
reliance on our answer key. §2 already shows the wrongly-blurred
<i>area</i> on PASS; the table below adds the box <i>counts</i> (FPPI —
false positives per image, shown ×100) as supporting detail.</p>
<table><tr><th>model</th><th>threshold</th><th>FPPI ×100 — all
classes</th><th>FPPI ×100 — {N} boxes</th><th>% images with
any false box</th><th>images checked</th></tr>{passrows}</table>
<p>For a {aud['user']} only the {N}-box column produces a visible wrong
blur; false "{aud['other'].title()}" boxes are invisible to them (and
vice versa for the {aud['other_user']}).</p>"""

    out.write_text(f"""<!doctype html><meta charset=utf-8>
<title>{aud['title']}</title><style>
body{{font-family:system-ui;background:{BG};color:{INK};max-width:980px;
margin:24px auto;padding:0 16px;line-height:1.45}}
table{{border-collapse:collapse;font-size:13.5px;margin:10px 0}}
td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}
td:first-child{{text-align:left}} th{{background:#f0f0ee}}
.big{{background:#e3f2e7;border:1px solid #7fbf8f;padding:12px 16px;
border-radius:6px;font-size:15px}}
.note{{background:#fff7e0;border:1px solid #e6c200;padding:8px 12px;
border-radius:4px}} h2{{margin-top:30px}}
.gloss td{{text-align:left;vertical-align:top}}
.gloss td:first-child{{white-space:nowrap}}
tr.pick{{background:#e3f2e7;font-weight:600}}
tr.pick2{{background:#fff4dd}} tr.cur{{background:#ececea}}
.sw-g{{background:#e3f2e7;padding:0 4px}}
.sw-a{{background:#fff4dd;padding:0 4px}}
.sw-c{{background:#ececea;padding:0 4px}}</style>

<h1>{aud['title']}</h1>
<p>Question: <b>which model, at which single confidence threshold, best
blurs {ns} without blurring things that aren't {ns}?</b> This is the
{aud['user']}'s setting: {ns} are blurred, {aud['others']} are not. For
this audience an <b>escape</b> (an unblurred {n}) is the product failure,
and a <b>false blur</b> (a {oth}, a child, or an object blurred by a
wrong {N} box) is the visible annoyance — false
"{aud['other'].title()}" boxes are invisible here. Each
model is tuned to its own best threshold first (§1; a fixed shared
threshold measures calibration, not quality), then all are measured on
the same 11,493 holdout images and on 3,000 person-free images.</p>

<div class=big><b>Recommendation: {best[0]} at threshold
{best[1]:.2f}.</b> It leaves only
{100 * best[2]['person_uncov_mean']:.1f}% of the average {n} visible,
while wrongly blurring {_wa(best[0], best[1])} of the image on average —
inside the production FP budget on both datasets.</div>

<h2>1 · Selected operating point — one threshold per model</h2>
<p>Question this section answers: <i>which single confidence does each
model ship with, and where did it come from?</i> The model scores every
box in [0,1]; boxes above the threshold blur. Every box was logged down
to score 0.001, so all thresholds are evaluated by replaying the logs
(§5) — no model re-runs. <b>The selection rule uses the report's only
two metrics: minimize the visible share of a {n} (TP side) subject to
the wrongly-blurred area (FP side) being no worse than production</b> —
the baseline is {inc} at {budget['conf']:.2f} (today's setting): at most
{bud_hold} of the image wrongly blurred on the holdout and {bud_pass} on
the person-free PASS set. Visibility rises and wrong blur falls as the
threshold rises, so the winner is simply the lowest threshold whose
wrong-blur area still beats production — the standard fixed-FP operating
point. The shipped value is ratified by the owner. One value per model,
used everywhere — no per-website or per-image tuning.</p>
<table><tr><th>model</th><th>selected threshold</th><th>rule pick
(min visibility in budget)</th><th>% of a {n} left visible at
selected</th></tr>
{oprows}</table>

<h2>2 · Headline comparison — each model at its selected threshold</h2>
<table><tr><th>model @ threshold</th><th>% of a {n} left visible</th>
<th>wrongly blurred % of image (holdout)</th>
<th>wrongly blurred % of image (PASS person-free)</th></tr>{rows}</table>
<p>Two metrics, two questions — lower is better on both. <b>% of a {n}
left visible</b>: for each {n} in the answer key, the share of her GT
box NOT covered by the union of fired {N} boxes (a box covering 95% of
a {n} leaves 5% of her visible); every {n} counts separately and
equally, then we average over all of them. This is the TP metric:
<i>how much {n} still gets through?</i> <b>Wrongly blurred % of
image</b>: per image, the area of fired {N} boxes that match NO {n} at
IoU ≥ 0.5, as a share of the picture, averaged over images — a box that
does match a {n} at IoU ≥ 0.5 is a legitimate blur and contributes
nothing, and boxes on ignore-region people are excused. This is the FP
metric: <i>how much of the screen gets blurred for no reason?</i> It is
one metric on two datasets: the holdout (judged by the machine answer
key) and PASS (3,000 human-verified person-free images, where every
fired box is wrong by construction — no reliance on our labels).</p>

<h2>3 · FPPI verification on person-free images (the PASS dataset)</h2>
{pass_html}

{tier_html}
{sweep_html}
{s045_html}
<h2>7 · Glossary — every metric, the question it answers, and how it is
computed</h2>
<table class=gloss><tr><th>term</th><th>the question it answers</th>
<th>how it is computed</th></tr>
<tr><td><b>selected threshold</b></td>
<td>Which single confidence does this model ship with?</td>
<td>The model scores every box 0–1; boxes above the threshold blur. Every
box was logged down to 0.001, so thresholds 0.05–0.90 are evaluated by
replaying the logs (§5). The selection rule below proposes the pick; the
owner ratifies. One value per model, everywhere.</td></tr>
<tr><td><b>selection rule (FP budget)</b></td>
<td>How is each model's threshold chosen from the two metrics?</td>
<td>Minimize the % of a {n} left visible (TP side) subject to the
wrongly-blurred area (FP side) at or below the production baseline (the
incumbent at 0.45), on both datasets. Visibility rises and wrong blur
falls as the threshold rises, so the pick is the lowest threshold whose
wrong-blur area still beats production (the standard fixed-FP /
Neyman–Pearson operating point). No exchange rate between a TP point and
an FP point is ever priced in; the earlier J and Udist balance scores
are retired.</td></tr>
<tr><td><b>% of a {n} left visible</b> (the TP-side metric)</td>
<td>How much {n} still gets through the blur?</td>
<td>Per {n} in the answer key: 1 − area(GT box ∩ union of fired {N}
boxes) ÷ area(GT box). A box covering 95% of a {n} leaves 5% of her
visible. Every {n} is computed separately and counts equally regardless
of size; the reported number is the average over all of them. Union
means two half-boxes that jointly cover her count as covering, and an
oversized box costs nothing here — its overflow is charged by the FP
metric instead. All geometry is box-vs-box (GT polygons are collapsed
to their bounding boxes; the extension renders blur as
rectangles).</td></tr>
<tr><td><b>wrongly blurred % of image</b> (the FP metric)</td>
<td>How much of the screen gets blurred for no reason? (One metric,
reported on two datasets — the dataset is the label, not the
metric.)</td>
<td>Per image: every fired {N} box is matched against the {ns} in the
answer key; a box with IoU ≥ 0.5 against some {n} (or against an
ignore-region person) is a legitimate blur and contributes nothing. The
union area of the remaining — unmatched — boxes, divided by the image
area, is the wrongly-blurred share; we average it over all images. On
the <b>holdout</b> that judgment uses our machine answer key; on
<b>PASS</b> — 3,000 human-verified person-free images — every fired box
is wrong by construction, so the number needs no labels at
all.</td></tr>
<tr><td><b>AP50</b></td>
<td>Ignoring thresholds entirely, CAN this model find and correctly
gender {ns}? (The capability ceiling the sweep moves beneath.)</td>
<td>All detections ranked by confidence; a detection is a hit if its box
overlaps a real {n} at IoU ≥ 0.5; the precision-recall curve is averaged
by the usual COCO 101-point procedure.</td></tr>
<tr><td><b>AP50-95</b></td>
<td>Are the boxes tight as well as right?</td>
<td>The same procedure repeated at IoU 0.50, 0.55 … 0.95 and averaged —
COCO's headline metric. A big AP50 → AP50-95 drop means "finds the
person, draws a loose box".</td></tr>
<tr><td><b>coverage tiers (t0–t4)</b></td>
<td>Does accuracy depend on how covered the person is?</td>
<td>During labeling, a vision model listed the visible body parts for
every person (an observable fact, not a judgment). A fixed written rule
maps that list to a tier; anything unrecognized goes to t4, never to a
covered tier. Tier labels group people — they never score models.</td></tr>
<tr><td><b>answer key (ground truth)</b></td>
<td>What are the models scored against?</td>
<td>11,493 images with every person boxed and labeled woman / man /
child by our labeling pipeline (a detector plus a second AI verifying
every box). It is machine-made — which is exactly why the FPPI check on
PASS exists (§3): it does not depend on these labels.</td></tr>
<tr><td><b>skipped ("ignore") people</b></td>
<td>What happens with people the labeling pipeline could not gender?</td>
<td>1,619 people (mostly very small figures) are excluded from both
sides of the ledger — model boxes on them are neither credited as
correct nor punished as wrong.</td></tr>
</table>

<h2>8 · What to keep in mind</h2>
<ul>
<li>The holdout's answer key is machine-generated (SAM3 + Gemini). The
FPPI check on PASS in §3 is human-verified and independent.</li>
<li>There is deliberately no separate child metric: a child under a {N}
blur matches no {n}, so the box counts as wrongly-blurred area (and as
a false box in §3), and a {n} misread as a child is simply an unblurred
{n} — fully charged by the visibility metric. Both
child directions are covered by the two headline metrics.</li>
<li>All numbers are measured in box space, not silhouette
space: a bounding box overstates a person's true outline (arms out,
seated poses), so "5% left visible" means 5% of the person's <i>box</i>.
Both GT and predictions share the distortion, so comparisons are fair —
but the absolute percentages inherit it.</li>
{aud['extra_caveats']}
<li>These numbers are single-image; video adds flicker, which is a
separate policy layer (EXP-2026-11).</li>
</ul>""")
    return chat


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m1" / "raw").mkdir(parents=True)
        dets1 = [{"det_index": 0, "cls": 0, "box_xyxy": [1, 1, 5, 5],
                  "conf": 0.4, "kept": False, "excluded": None},
                 {"det_index": 1, "cls": 1, "box_xyxy": [1, 1, 5, 5],
                  "conf": 0.4, "kept": False, "excluded": None}]
        for i, dets in enumerate([[], dets1]):
            (td / "m1" / "raw" / f"p{i}.json").write_text(json.dumps(
                {"image": f"p{i}.jpg", "width": 10, "height": 10,
                 "detections": dets}))
        pf = pass_fp(td / "m1" / "raw", 0.3, target_cls=0)
        assert pf["fp_per_100"] == 100.0 and pf["cls_fp_per_100"] == 50.0
        pfm = pass_fp(td / "m1" / "raw", 0.3, target_cls=1)
        assert pfm["cls_fp_per_100"] == 50.0, "male-class FP counting"

        # wrong-area metric: p1 has a woman GT exactly under the cls-0 box
        (td / "gt").mkdir()
        (td / "gt" / "p0.txt").write_text("")
        (td / "gt" / "p1.txt").write_text("0 0.3 0.3 0.4 0.4\n")
        wa = wrong_area_curves(td / "m1" / "raw", td / "gt", None,
                               [0.3], 0)
        assert wa[0.3] == 0.0, "IoU-1.0 matched box is a legitimate blur"
        wam = wrong_area_curves(td / "m1" / "raw", td / "gt", None,
                                [0.3], 1)
        assert abs(wam[0.3] - 0.08) < 1e-9, \
            "unmatched cls-1 box charges 16/100 px on 1 of 2 imgs"
        wap = wrong_area_curves(td / "m1" / "raw", None, None, [0.3], 0)
        assert abs(wap[0.3] - 0.08) < 1e-9, "person-free arm: all wrong"
        assert wrong_area_curves(td / "m1" / "raw", td / "gt", None,
                                 [0.5], 1)[0.5] == 0.0, "conf-gated"

        curves = {"m1": {"0.30": {
            "person_covered90_rate": 0.9, "person_uncov_mean": 0.06,
            "mean_exposure": 0.03, "falseblur_img_rate": 0.04,
            "fb_area_mean": 0.008, "fp_boxes_per_100": 4.0}}}
        fp_curves = {"holdout": {"m1": {0.3: 0.0}},
                     "pass": {"m1": {0.3: 0.0008}}}
        for mode, audk, want in (("male", "male", "Woman boxes"),
                                 ("female", "female", "Man boxes")):
            summary = {"modes": {mode: {"curves": curves}}}
            out = td / f"r_{mode}.html"
            pref = AUD[audk]["pref"]
            tiers_data = ([pref + "t0_covered"],
                          {"m1": {pref + "t0_covered": {
                              "person_uncov_mean": 0.3,
                              "n_persons": 10}}})
            ap_data = {pref + "t0_covered": {"m1": (0.75, 0.6)}}
            chat = build(summary, {"m1": 0.30}, td, out, AUD[audk],
                         tiers_data, ap_data, fp_curves=fp_curves)
            t = out.read_text()
            assert "Recommendation" in t and want in t, (mode, want)
            assert "left visible" in t and "wrongly blurred" in t, \
                "the two headline metrics"
            assert "0.080%" in t, "PASS wrong-area rendered"
            assert "no separate child metric" in t, "child-folding caveat"
            assert "Selected operating point" in t, "threshold section"
            assert "wrongly-blurred area (FP side)" in t, \
                "two-metric FP-budget selection described"
            assert "AP50 0.75" in t and "AP50-95 0.60" in t
            assert chat[0][3]["fp_per_100"] == 100.0
        t = (td / "r_female.html").read_text()
        assert "thobe" in t and "Men blur" in t, "female-audience wording"
        assert "shirtless-level) contains only ~31" in t, "t4 caveat"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary")
    ap.add_argument("--selected", help="model=conf,model=conf,...")
    ap.add_argument("--audience", choices=["male", "female"],
                    default="male")
    ap.add_argument("--pass-dir")
    ap.add_argument("--gt-labels")
    ap.add_argument("--ignore")
    ap.add_argument("--raw-root")
    ap.add_argument("--slices")
    ap.add_argument("--map-dir")
    ap.add_argument("--tier-cache-045")
    ap.add_argument("--tier-cache")
    ap.add_argument("--fp-cache", help="wrong-area curve cache (json)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    aud = AUD[args.audience]
    out = Path(args.out or ("WOMEN_REPORT_FINAL.html"
                            if args.audience == "male"
                            else "MEN_REPORT_FINAL.html"))
    selected = {}
    for part in args.selected.split(","):
        m, _, c = part.partition("=")
        selected[m.strip()] = float(c)
    summary = json.loads(Path(args.summary).read_text())
    tags = tier_tags(aud)
    tiers_data = tiers045 = ap_data = None
    if args.tier_cache and Path(args.tier_cache).exists():
        c = json.loads(Path(args.tier_cache).read_text())
        tiers_data = (c["tiers"], c["data"])
    if args.tier_cache_045 and Path(args.tier_cache_045).exists():
        c = json.loads(Path(args.tier_cache_045).read_text())
        tiers045 = (c["tiers"], c["data"])
    need = [(tiers_data is None, selected),
            (tiers045 is None, {m: 0.45 for m in selected})]
    missing = [sel for miss, sel in need if miss]
    if missing and args.gt_labels and args.raw_root and args.slices:
        tiers, outs = tier_metrics_multi(
            missing, Path(args.gt_labels), args.ignore, args.raw_root,
            args.slices, target=aud["target"], tags=tags)
        it = iter(outs)
        if tiers_data is None:
            tiers_data = (tiers, next(it))
            if args.tier_cache:
                Path(args.tier_cache).write_text(json.dumps(
                    {"tiers": tiers, "data": tiers_data[1],
                     "selected": selected}))
        if tiers045 is None:
            tiers045 = (tiers, next(it))
            if args.tier_cache_045:
                Path(args.tier_cache_045).write_text(json.dumps(
                    {"tiers": tiers, "data": tiers045[1],
                     "selected": {m: 0.45 for m in selected}}))
    if tiers_data:
        ap_data = tier_ap50(args.map_dir, list(selected), aud)

    # wrong-area (FP metric) curves: cached, else replayed from sidecars
    fp_curves = None
    if args.fp_cache and Path(args.fp_cache).exists():
        c = json.loads(Path(args.fp_cache).read_text())
        fp_curves = {ds: {m: {float(k): v for k, v in mm.items()}
                          for m, mm in c.get(ds, {}).items()}
                     for ds in ("holdout", "pass")}
    elif args.raw_root and args.gt_labels:
        grid = sorted(float(k) for k in
                      summary["modes"][aud["mode"]]["curves"]
                      [next(iter(selected))])
        fp_curves = {"holdout": {}, "pass": {}}
        for m in selected:
            fp_curves["holdout"][m] = wrong_area_curves(
                Path(args.raw_root) / m / "raw", args.gt_labels,
                args.ignore, grid, aud["target"])
            print(f"[fp] {m}: holdout wrong-area curve done")
            if args.pass_dir:
                fp_curves["pass"][m] = wrong_area_curves(
                    Path(args.pass_dir) / m / "raw", None, None, grid,
                    aud["target"])
                print(f"[fp] {m}: PASS wrong-area curve done")
        if args.fp_cache:
            Path(args.fp_cache).write_text(json.dumps(
                {ds: {m: {str(k): v for k, v in mm.items()}
                      for m, mm in fp_curves[ds].items()}
                 for ds in ("holdout", "pass")}))

    chat = build(summary, selected, args.pass_dir, out, aud,
                 tiers_data, ap_data, tiers045,
                 all_ap(args.map_dir, list(selected), aud["cls"]),
                 fp_curves=fp_curves)
    for m, cf, a, pf in chat:
        line = (f"{m:16s} @{cf:.2f}  visible "
                f"{100 * a['person_uncov_mean']:.1f}%")
        if fp_curves:
            v = fp_curves["holdout"].get(m, {}).get(cf)
            p = fp_curves["pass"].get(m, {}).get(cf)
            line += (f"  wrong-blur {100 * v:.3f}%" if v is not None else "")
            line += (f"  PASS {100 * p:.3f}%" if p is not None else "")
        print(line)
    print(f"[final] wrote {out}")


if __name__ == "__main__":
    main()
