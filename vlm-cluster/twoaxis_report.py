#!/usr/bin/env python3
"""EXP-2026-17 — accuracy report for the two-axis (gender x age) head.

DELIBERATELY SEPARATE from `build_error_gallery.py` and the shared scorers.
Every other model in this project emits ONE class per box, so every other tool
is built around a single predicted label and can only ever show a single
GT-vs-pred pair. This head emits TWO independent readings per box, and the
interesting failures are the ones where the axes disagree with the answer key
in different ways -- a woman read as a man while her age is read correctly is a
different defect from a child read as an adult woman. Folding that into the
shared 4-class gallery would hide exactly what this experiment exists to
measure, so this file scores and renders the axes SEPARATELY, and shows the
collapse as a third, derived row.

WHAT IT MEASURES, in three layers:

  1. detection  -- did a box land on the labeled person at all (IoU >= 0.5)
  2. per axis   -- gender argmax vs the human gender, age argmax vs the human
                   age, scored INDEPENDENTLY on the people that were detected
  3. collapsed  -- the single product label the blur policy actually acts on,
                   via two_axis.collapse(); this is the only layer comparable
                   to the other models in docs/MODEL_COMPARISON.md

HOW THE TWO AXES BECOME ONE LABEL (the question this report exists to answer
visually) -- `two_axis.collapse` is the blur policy written once:

    age == Child          -> Child            (never blurred, gender ignored)
    else gender == Woman  -> Woman
    else gender == Man    -> Man
    else                  -> UnknownGender

Two consequences worth seeing in the gallery rather than reading in prose:
  * a child's gender is DISCARDED by the collapse even though the head
    predicts it -- that information exists only in the two-axis view.
  * AgeUnknown falls through to the gender branch, i.e. an unreadable age is
    treated as an ADULT and therefore blurred. That is the safe direction for
    this product (an adult escaping the blur is the consequential error), and
    it means the AgeUnknown channel can never cause an adult to escape.

GROUND TRUTH mapping uses the SAME constants as training (two_axis.py), so the
answer key and the model are held to one definition:
    age <= 12          -> Child
    13 <= age <= 17    -> AgeUnknown   (the band; checked BEFORE the child cut)
    age >= 18          -> Adult
A model that answers AgeUnknown for a 15-year-old is therefore CORRECT here,
which is the whole point of the abstention channel.

Usage:
    python3 twoaxis_report.py --model /workspace/exp17/eval_snap/epochNN.pt \\
        --images /workspace/lagenda_eval/lagenda_yolo/images/val \\
        --gt-manifest /workspace/datasets/lagenda_full/eval_v2/gt.jsonl \\
        --gt-labels  /workspace/datasets/lagenda_full/eval_v2/labels \\
        --out /workspace/exp17/twoaxis_report.html --max-images 600

    python3 twoaxis_report.py --selftest    # no GPU, no data, no network
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import two_axis as TA

# reuse, never reimplement: the same greedy matcher every other scorer uses
from run_model_children import match_boxes

THUMB_PX = 150          # crop thumbnails, sized small enough to keep the HTML sane
CROP_PAD = 0.25         # context around the GT box, same padding the Spotlight crops used


# ----------------------------------------------------------------- ground truth
def gt_axes(age, gender):
    """LAGENDA's human answer key -> the two-axis taxonomy the model predicts.

    Deliberately built from two_axis's own constants rather than literals, so
    moving the band or the child cutoff moves the answer key with it.
    """
    g = {"F": TA.WOMAN, "M": TA.MAN}.get(str(gender).strip().upper()[:1], TA.GENDER_UNKNOWN)
    lo, hi = TA.AGE_UNKNOWN_BAND
    if age is None:
        a = TA.AGE_UNKNOWN
    elif lo <= age <= hi:               # band FIRST, exactly as two_axis_ids orders it
        a = TA.AGE_UNKNOWN
    elif age <= TA.CHILD_AGE_MAX:
        a = TA.CHILD
    else:
        a = TA.ADULT
    return g, a


def load_gt(manifest: Path, labels_dir: Path):
    """-> {image_name: [ {box_idx, age, gender, gid, aid} ]}.

    A gt.jsonl id is `<stem>_<line index into the label file>`; the box itself
    lives in labels/<stem>.txt at that line. Keeping the two joined by line
    number is the same convention Spotlight's provenance chain uses.
    """
    per_image = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        stem, _, idx = r["id"].rpartition("_")
        gid, aid = gt_axes(r.get("gt_age"), r.get("gt_gender"))
        per_image.setdefault(r["image"], []).append(
            {"stem": stem, "box_idx": int(idx), "age": r.get("gt_age"),
             "gender": r.get("gt_gender"), "gid": gid, "aid": aid})
    # attach pixel boxes
    out = {}
    for image, rows in per_image.items():
        lf = labels_dir / (Path(image).stem + ".txt")
        if not lf.exists():
            continue
        lines = [l.split() for l in lf.read_text().splitlines() if l.strip()]
        keep = []
        for r in rows:
            if r["box_idx"] >= len(lines):
                continue
            v = [float(x) for x in lines[r["box_idx"]][1:5]]
            r["nbox"] = v                       # normalized cx cy w h
            keep.append(r)
        if keep:
            out[image] = keep
    return out


def to_pixels(nbox, w, h):
    cx, cy, bw, bh = nbox
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


# ----------------------------------------------------------------------- scoring
class Tally:
    """Per-axis and collapsed counts, kept apart on purpose."""

    def __init__(self):
        self.n_gt = 0
        self.n_matched = 0
        self.gender = Counter()      # (gt_name, pred_name)
        self.age = Counter()
        self.collapsed = Counter()
        # the JOINT reading: (gt_gender, gt_age) -> (pred_gender, pred_age).
        # The per-axis matrices cannot answer 'which axis turned this child
        # into a woman' -- only the joint can, because the collapse is a
        # function of BOTH axes.
        self.joint = Counter()
        self.cases = []              # gallery rows

    def add_case(self, rec):
        self.cases.append(rec)

    def matrix(self, counter, labels):
        rows = []
        for t in labels:
            row = [counter[(t, p)] for p in labels]
            tot = sum(row)
            rows.append((t, row, tot, (counter[(t, t)] / tot) if tot else None))
        return rows

    def accuracy(self, counter):
        hit = sum(v for (t, p), v in counter.items() if t == p)
        tot = sum(counter.values())
        return hit, tot, (hit / tot if tot else 0.0)


def evaluate(detect, images_dir: Path, gt: dict, max_images: int, log=print):
    """detect(PIL.Image) -> (dets, axes); dets are (cls, x1, y1, x2, y2, conf)."""
    from PIL import Image

    t = Tally()
    names = sorted(gt)[:max_images] if max_images else sorted(gt)
    for i, image in enumerate(names, 1):
        p = images_dir / image
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        W, H = img.size
        rows = gt[image]
        boxes = [to_pixels(r["nbox"], W, H) for r in rows]
        t.n_gt += len(rows)

        dets, axes = detect(img)
        matched = match_boxes(boxes, [(0,) + tuple(d[1:5]) for d in dets], 0.5)

        for gi, (det, iou) in matched.items():
            di = next(k for k, d in enumerate(dets) if tuple(d[1:5]) == tuple(det[1:5]))
            ax, r = axes[di], rows[gi]
            t.n_matched += 1

            g_gt, g_pr = TA.CLASS_NAMES[r["gid"]], ax["gender_name"]
            a_gt, a_pr = TA.CLASS_NAMES[r["aid"]], ax["age_name"]
            c_gt = TA.COLLAPSED_NAMES[TA.collapse(r["gid"], r["aid"])]
            c_pr = TA.COLLAPSED_NAMES[TA.collapse(ax["gender_cls"], ax["age_cls"])]

            t.gender[(g_gt, g_pr)] += 1
            t.age[(a_gt, a_pr)] += 1
            t.collapsed[(c_gt, c_pr)] += 1
            t.joint[((g_gt, a_gt), (g_pr, a_pr))] += 1
            t.add_case({"image": image, "box": boxes[gi], "iou": round(iou, 3),
                        "age_years": r["age"], "gt_gender_raw": r["gender"],
                        "g_gt": g_gt, "g_pr": g_pr, "g_conf": ax["gender_conf"],
                        "a_gt": a_gt, "a_pr": a_pr, "a_conf": ax["age_conf"],
                        "c_gt": c_gt, "c_pr": c_pr, "conf": round(dets[di][5], 3)})
        if i % 100 == 0:
            log(f"  {i}/{len(names)} images · {t.n_matched} matched people")
    return t


# ------------------------------------------------------------------------- HTML
def thumb(images_dir: Path, image: str, box, px=THUMB_PX):
    from PIL import Image

    try:
        im = Image.open(images_dir / image).convert("RGB")
    except Exception:
        return ""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    crop = im.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                    min(im.width, int(x2 + pw)), min(im.height, int(y2 + ph))))
    crop.thumbnail((px, px))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _matrix_html(title, rows, labels, note=""):
    h = [f"<h3>{title}</h3>"]
    if note:
        h.append(f"<p class=note>{note}</p>")
    h.append("<table class=cm><tr><th>GT \\ predicted</th>"
             + "".join(f"<th>{l}</th>" for l in labels) + "<th>recall</th></tr>")
    for name, row, tot, rec in rows:
        cells = "".join(
            f"<td class='{'hit' if labels[j] == name else ('miss' if v else '')}'>{v or ''}</td>"
            for j, v in enumerate(row))
        r = f"{100 * rec:.1f}%" if rec is not None else "—"
        h.append(f"<tr><th>{name}</th>{cells}<td class=rec>{r}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def _cards(cases, images_dir, limit, embed=True):
    out = []
    for c in cases[:limit]:
        src = thumb(images_dir, c["image"], c["box"]) if embed else ""
        gbad = "bad" if c["g_gt"] != c["g_pr"] else "ok"
        abad = "bad" if c["a_gt"] != c["a_pr"] else "ok"
        cbad = "bad" if c["c_gt"] != c["c_pr"] else "ok"
        yrs = "?" if c["age_years"] is None else f"{c['age_years']:.0f}"
        out.append(f"""<div class=card>
  <img src="{src}" alt="">
  <div class=meta>
    <div class=row><span class=k>age (human)</span><span class=v>{yrs} yr · {c['gt_gender_raw']}</span></div>
    <div class=row><span class=k>gender</span><span class="v {gbad}">{c['g_gt']} &rarr; {c['g_pr']} <em>{c['g_conf']:.2f}</em></span></div>
    <div class=row><span class=k>age</span><span class="v {abad}">{c['a_gt']} &rarr; {c['a_pr']} <em>{c['a_conf']:.2f}</em></span></div>
    <div class="row collapsed"><span class=k>collapsed</span><span class="v {cbad}">{c['c_gt']} &rarr; {c['c_pr']}</span></div>
    <div class=row><span class=k>IoU / conf</span><span class=v>{c['iou']} / {c['conf']}</span></div>
  </div>
</div>""")
    return "\n".join(out)


CSS = """
:root{--bg:#fff;--fg:#16181d;--mut:#666;--line:#e3e6ea;--ok:#127a3d;--bad:#c0271a;
      --hit:#e6f5ec;--miss:#fdecea;--panel:#f7f8fa}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 10px;
   border-bottom:2px solid var(--line);padding-bottom:6px} h3{font-size:14px;margin:18px 0 8px}
.sub{color:var(--mut);margin:0 0 18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:16px 0}
.panel pre{margin:8px 0 0;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;
           white-space:pre-wrap;color:#22252b}
.note{color:var(--mut);margin:4px 0 8px;font-size:13px}
table{border-collapse:collapse;margin:6px 0 14px} td,th{border:1px solid var(--line);
      padding:5px 10px;text-align:right;font-variant-numeric:tabular-nums} th{background:var(--panel);text-align:left}
td.hit{background:var(--hit);font-weight:600} td.miss{background:var(--miss)} td.rec{font-weight:600}
.scorecard td,.scorecard th{text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px;display:flex;gap:10px;background:#fff}
.card img{width:110px;height:110px;object-fit:cover;border-radius:5px;background:var(--panel);flex:0 0 auto}
.meta{flex:1;min-width:0;font-size:12px}
.meta .row{display:flex;justify-content:space-between;gap:6px;padding:1px 0}
.meta .k{color:var(--mut)} .meta .v{text-align:right} .meta em{color:var(--mut);font-style:normal}
.collapsed{border-top:1px dashed var(--line);margin-top:3px;padding-top:3px;font-weight:600}
.ok{color:var(--ok)} .bad{color:var(--bad)}
.overflow{overflow-x:auto}
"""



def _blame_html(t, limit=14):
    """Decompose each COLLAPSED error into the two axis readings that caused it.

    `collapse` is a function of both axes, so a `Child -> Woman` error is not a
    gender failure -- it is an AGE failure (child read as adult) whose gender
    reading then became the visible label. This table is the only place that
    distinction is recoverable.
    """
    rows = []
    for (gt_pair, pr_pair), n in t.joint.items():
        c_gt = TA.COLLAPSED_NAMES[TA.collapse(TA.CLASS_NAMES.index(gt_pair[0]),
                                              TA.CLASS_NAMES.index(gt_pair[1]))]
        c_pr = TA.COLLAPSED_NAMES[TA.collapse(TA.CLASS_NAMES.index(pr_pair[0]),
                                              TA.CLASS_NAMES.index(pr_pair[1]))]
        if c_gt == c_pr:
            continue
        gender_wrong = gt_pair[0] != pr_pair[0]
        age_wrong = gt_pair[1] != pr_pair[1]
        blame = ("age only" if age_wrong and not gender_wrong else
                 "gender only" if gender_wrong and not age_wrong else
                 "both axes" if gender_wrong and age_wrong else "neither (?)")
        rows.append((n, c_gt, c_pr, gt_pair, pr_pair, blame))
    rows.sort(reverse=True)
    h = ["<h3>What actually caused each collapsed error</h3>",
         "<p class=note>The collapsed label is a function of BOTH axes, so a "
         "<code>Child &rarr; Woman</code> error is an <b>age</b> failure whose gender "
         "reading became the visible label &mdash; not a gender failure. This table is the "
         "only view where that is recoverable.</p>",
         "<table class=cm><tr><th>n</th><th>collapsed</th><th>gender axis</th>"
         "<th>age axis</th><th>blame</th></tr>"]
    for n, c_gt, c_pr, gp, pp, blame in rows[:limit]:
        gcell = f"{gp[0]} &rarr; {pp[0]}" + ("" if gp[0] != pp[0] else " <em>(ok)</em>")
        acell = f"{gp[1]} &rarr; {pp[1]}" + ("" if gp[1] != pp[1] else " <em>(ok)</em>")
        cls = "miss" if blame == "gender only" else ("hit" if blame == "age only" else "")
        h.append(f"<tr><td>{n}</td><th>{c_gt} &rarr; {c_pr}</th>"
                 f"<td style='text-align:left'>{gcell}</td>"
                 f"<td style='text-align:left'>{acell}</td>"
                 f"<td class='{cls}' style='text-align:left'>{blame}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def render(t: Tally, meta: dict, images_dir: Path, out: Path, max_cards=36, embed=True):
    G = list(TA.CLASS_NAMES[0:3])
    A = list(TA.CLASS_NAMES[3:6])
    C = list(TA.COLLAPSED_NAMES)

    gh, gt_, ga = t.accuracy(t.gender)
    ah, at_, aa = t.accuracy(t.age)
    ch, ct_, ca = t.accuracy(t.collapsed)
    det = t.n_matched / t.n_gt if t.n_gt else 0

    def sect(title, pred, note):
        rows = [c for c in t.cases if pred(c)]
        if not rows:
            return ""
        return (f"<h3>{title} <span class=note>({len(rows)} people"
                f"{', showing ' + str(max_cards) if len(rows) > max_cards else ''})</span></h3>"
                f"<p class=note>{note}</p><div class=grid>{_cards(rows, images_dir, max_cards, embed)}</div>")

    html = f"""<title>Two-axis accuracy — {meta['run']}</title>
<style>{CSS}</style>
<h1>Two-axis head: predicted vs label</h1>
<p class=sub>{meta['run']} · checkpoint <code>{meta['ckpt']}</code> · epoch {meta['epoch']} of {meta['total_epochs']}
 · {meta['dataset']} · generated {meta['when']}</p>

<div class=panel>
<b>How the head is read.</b> The model has {TA.NC} sigmoid channels, read as TWO argmax groups —
gender over channels 0–2 {G}, age over channels 3–5 {A}. Box confidence is the max over ALL six
channels, and NMS is class-agnostic, so box selection is identical to a normal single-label model.
<b>How the two axes become one label</b> (<code>two_axis.collapse</code> — this is the blur policy):
<pre>age == Child          -&gt; Child            # never blurred; the predicted gender is DISCARDED
else gender == Woman  -&gt; Woman
else gender == Man    -&gt; Man
else                  -&gt; UnknownGender</pre>
AgeUnknown falls through to the gender branch, i.e. an unreadable age is treated as an ADULT and
therefore blurred — the safe direction, and the reason the abstention channel can never let an
adult escape.
<b>Ground truth</b> uses the same constants as training: age &le; {TA.CHILD_AGE_MAX} = Child,
{TA.AGE_UNKNOWN_BAND[0]}–{TA.AGE_UNKNOWN_BAND[1]} = AgeUnknown (checked first), &ge; {TA.AGE_UNKNOWN_BAND[1] + 1} = Adult.
So answering <i>AgeUnknown</i> for a 15-year-old counts as CORRECT.
</div>

<h2>Scorecard</h2>
<table class=scorecard>
<tr><th>layer</th><th>metric</th><th>value</th></tr>
<tr><td>detection</td><td>labeled people found (IoU&ge;0.5)</td><td>{t.n_matched} / {t.n_gt} = <b>{100*det:.1f}%</b></td></tr>
<tr><td>axis 1</td><td>gender accuracy, on detected people</td><td>{gh} / {gt_} = <b>{100*ga:.1f}%</b></td></tr>
<tr><td>axis 2</td><td>age accuracy, on detected people</td><td>{ah} / {at_} = <b>{100*aa:.1f}%</b></td></tr>
<tr><td>collapsed</td><td>product label accuracy (blur policy applied)</td><td>{ch} / {ct_} = <b>{100*ca:.1f}%</b></td></tr>
</table>
<p class=note>The two axes are scored INDEPENDENTLY — a person counted wrong on gender may still be
right on age. Only the collapsed row is comparable with the single-label models in
docs/MODEL_COMPARISON.md.</p>

<h2>Confusion, per axis</h2>
<div class=overflow>
{_matrix_html("Gender axis (channels 0–2)", t.matrix(t.gender, G), G,
              "LAGENDA's answer key has no GenderUnknown, so that GT row is empty by construction — "
              "the column still shows how often the model abstains on a person whose gender IS known.")}
{_matrix_html("Age axis (channels 3–5)", t.matrix(t.age, A), A,
              "The AgeUnknown GT row is the 13–17 band. Adult&rarr;Child is the consequential direction: "
              "those adults escape the blur.")}
{_matrix_html("Collapsed product label", t.matrix(t.collapsed, C), C,
              "What the extension actually acts on, after the policy above.")}
</div>
{_blame_html(t)}

<h2>Cases</h2>
{sect("Both axes correct", lambda c: c['g_gt'] == c['g_pr'] and c['a_gt'] == c['a_pr'],
      "A sample of clean reads, so the error sections below can be judged against a baseline.")}
{sect("Gender wrong, age right", lambda c: c['g_gt'] != c['g_pr'] and c['a_gt'] == c['a_pr'],
      "The axes are independent: these people had their age read correctly.")}
{sect("Age wrong, gender right", lambda c: c['g_gt'] == c['g_pr'] and c['a_gt'] != c['a_pr'],
      "Includes the abstention failures — a teenager the model called Adult or Child instead of AgeUnknown.")}
{sect("Both axes wrong", lambda c: c['g_gt'] != c['g_pr'] and c['a_gt'] != c['a_pr'], "")}
{sect("Adult read as Child (the consequential error)",
      lambda c: c['a_gt'] == 'Adult' and c['a_pr'] == 'Child',
      "These adults are left unblurred, which is the error direction this product cares about most.")}
"""
    out.write_text(html)
    return out


# --------------------------------------------------------------------- selftest
def selftest():
    """No GPU, no network, no dataset: synthesize images + an answer key, run a
    stub detector, and assert the axes are scored independently and the collapse
    matches two_axis."""
    import tempfile

    from PIL import Image

    # the collapse contract this report explains, asserted rather than described
    assert TA.collapse(TA.WOMAN, TA.CHILD) == TA.C_CHILD, "child must ignore gender"
    assert TA.collapse(TA.WOMAN, TA.AGE_UNKNOWN) == TA.C_WOMAN, "unknown age -> adult -> blurred"
    assert gt_axes(15, "F") == (TA.WOMAN, TA.AGE_UNKNOWN), "13-17 must map to the band"
    assert gt_axes(10, "F") == (TA.WOMAN, TA.CHILD)
    assert gt_axes(40, "M") == (TA.MAN, TA.ADULT)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "images").mkdir()
        (td / "labels").mkdir()
        rows, people = [], [(40.0, "F"), (8.0, "M"), (15.0, "F")]
        for i, (age, g) in enumerate(people):
            name = f"img{i}.jpg"
            Image.new("RGB", (200, 200), (120, 130, 140)).save(td / "images" / name)
            (td / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.6\n")
            rows.append(json.dumps({"id": f"img{i}_0", "image": name,
                                    "gt_age": age, "gt_gender": g}))
        (td / "gt.jsonl").write_text("\n".join(rows) + "\n")

        gt = load_gt(td / "gt.jsonl", td / "labels")
        assert len(gt) == 3, gt

        # stub: always finds the box; gender always Man, age always correct-ish
        def stub(img):
            W, H = img.size
            box = to_pixels([0.5, 0.5, 0.4, 0.6], W, H)
            dets = [(0, box[0], box[1], box[2], box[3], 0.9)]
            axes = [{"gender_cls": TA.MAN, "gender_name": "Man", "gender_conf": 0.8,
                     "age_cls": TA.ADULT, "age_name": "Adult", "age_conf": 0.7}]
            return dets, axes

        t = evaluate(stub, td / "images", gt, 0, log=lambda *a: None)
        assert t.n_gt == 3 and t.n_matched == 3, (t.n_gt, t.n_matched)
        # gender: woman,man,woman vs always Man -> 1 hit
        assert t.accuracy(t.gender)[0] == 1, t.gender
        # age: adult,child,ageunknown vs always Adult -> 1 hit
        assert t.accuracy(t.age)[0] == 1, t.age
        # the axes really are independent: the 8-year-old boy is RIGHT on gender
        # and WRONG on age, which a single-label scorer could not express
        boy = [c for c in t.cases if c["age_years"] == 8.0][0]
        assert boy["g_gt"] == boy["g_pr"] == "Man" and boy["a_gt"] != boy["a_pr"], boy
        # collapsed: GT Woman/Child/Woman vs pred Man/Man/Man -> 0 hits
        assert t.accuracy(t.collapsed)[0] == 0, t.collapsed

        # the joint must attribute the boy's collapsed error to AGE, not gender:
        # GT (Man, Child) -> Child, pred (Man, Adult) -> Man. Same gender, so the
        # visible "Child -> Man" label is an age failure. This is the whole point
        # of the blame table.
        assert t.joint[(("Man", "Child"), ("Man", "Adult"))] == 1, t.joint
        blame = _blame_html(t)
        assert "age only" in blame, blame
        assert "Child &rarr; Man" in blame, blame

        out = render(t, {"run": "selftest", "ckpt": "none", "epoch": 0,
                         "total_epochs": 0, "dataset": "synthetic", "when": "—"},
                     td / "images", td / "report.html", embed=True)
        h = out.read_text()
        for must in ("Two-axis head", "collapse", "Gender axis", "Age axis",
                     "Collapsed product label", "data:image/jpeg;base64,"):
            assert must in h, f"report is missing {must!r}"
        assert len(h) > 4000, len(h)
    print("twoaxis_report selftest OK (axes scored independently; collapse matches two_axis)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--images")
    ap.add_argument("--gt-manifest")
    ap.add_argument("--gt-labels")
    ap.add_argument("--out", default="twoaxis_report.html")
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--max-cards", type=int, default=36, help="cards per gallery section")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json-out", help="also write the raw tallies as JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    for req in ("model", "images", "gt_manifest", "gt_labels"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_', '-')} is required")

    # imported here so --selftest needs no torch
    import torch

    from run_ultralytics_labels import TwoLabelModel

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    epoch = ck.get("epoch", -1) + 1
    total = (ck.get("train_args") or {}).get("epochs", "?")
    del ck

    m = TwoLabelModel(a.model, imgsz=a.imgsz, floor=a.floor, iou=a.iou, device=a.device)

    def detect(img):
        return m.detect(img), m.last_axes

    gt = load_gt(Path(a.gt_manifest), Path(a.gt_labels))
    print(f"answer key: {sum(len(v) for v in gt.values())} labeled people "
          f"in {len(gt)} images")
    t = evaluate(detect, Path(a.images), gt, a.max_images)

    meta = {"run": Path(a.model).parent.parent.name or "y26n_twoaxis",
            "ckpt": a.model, "epoch": epoch, "total_epochs": total,
            "dataset": f"LAGENDA v2 · {a.max_images or 'all'} images · conf {a.conf}",
            "when": os.popen("date -u +'%Y-%m-%d %H:%M UTC'").read().strip()}
    out = render(t, meta, Path(a.images), Path(a.out), max_cards=a.max_cards)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps({
            "meta": meta, "n_gt": t.n_gt, "n_matched": t.n_matched,
            "gender": {f"{k[0]}->{k[1]}": v for k, v in t.gender.items()},
            "age": {f"{k[0]}->{k[1]}": v for k, v in t.age.items()},
            "collapsed": {f"{k[0]}->{k[1]}": v for k, v in t.collapsed.items()},
            "joint": {f"{k[0][0]}+{k[0][1]}->{k[1][0]}+{k[1][1]}": v
                      for k, v in t.joint.items()},
        }, indent=2))
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    sys.exit(main())
