#!/usr/bin/env python3
"""
Visual side-by-side of the small-person candidates: for every sampled image,
one panel per model showing what that model actually predicted, with the
ground truth drawn alongside — so accuracy can be JUDGED, not just read off a
table.

Groups are the benchmark's own arms, kept separate and never pooled:
  crowd_small   real CrowdHuman photos, GT is person-only (no gender)
  synth h96..   the same 480 LAGENDA people shrunk to a target height
  paste_grey    SAM-cut people pasted on flat grey at 640

Each group gets its measured numbers at the top and a batch of images below.
Green = ground truth. Predictions are coloured by class (Woman magenta, Man
blue, Child orange) and drawn only at the production threshold (conf 0.45),
because that is what ships — the sidecars hold everything down to 0.001.

    python3 build_smallperson_gallery.py --out /workspace/exp19/gallery.html
    python3 build_smallperson_gallery.py --selftest
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_autolabel_on_manifest import seg_boxes          # noqa: E402
from run_model_children import match_boxes               # noqa: E402

SP = "/workspace/datasets/smallperson_v1"
B = "/workspace/exp19"
CONF = 0.45
MATCH_IOU = 0.5

# the four candidates the owner cares about; y26n_gradsupp deliberately dropped
MODELS = ["yolo11N-640", "y26n_noe2e_warm50-2", "y26n_sop50",
          "gelannfav14r4fw_gemlb_v2"]
# on disk the warm50 arm is named without the "noe2e"
DIR_ALIAS = {"y26n_noe2e_warm50-2": "y26n_warm50-2"}

CLS_COLOR = {0: (233, 30, 99), 1: (33, 150, 243), 2: (255, 152, 0)}
CLS_NAME = {0: "Woman", 1: "Man", 2: "Child"}
GT_COLOR = (0, 230, 60)


def arm_dir(model: str) -> str:
    return DIR_ALIAS.get(model, model)


def raw_dir_for(raw_tpl: str, model: str) -> Path:
    """Per-model dump dir. Templates carry a `%s` for the model's on-disk name;
    a plain path (selftest, or a single-model dir) is used as-is."""
    return Path(raw_tpl % arm_dir(model) if "%s" in raw_tpl else raw_tpl)


AVATAR_SIZES = ("a128", "a96", "a64", "a48")

AVATAR_METHOD = """
<section class="method"><h2>How this arm is built and scored</h2>
<h3>Building an avatar</h3>
<ol>
<li><b>Source people.</b> LAGENDA val images, restricted to people who carry a
<i>human</i> age and gender label and whose box is at least 200&nbsp;px tall. Class comes
from the project's own <code>gt_class</code> (Woman / Man / Child), never from a model.</li>
<li><b>The crop.</b> A square anchored on the top of the person's box and centred on it —
side = <code>min(box_width, box_height &times; 0.34)</code>, lifted 4% above the box top
because boxes clip hair. For a standing person that is head plus shoulders. It is taken
from the ORIGINAL photograph, so it is a photographic crop, exactly like a real profile
picture.</li>
<li><b>Never upscaled.</b> If the source square is smaller than the requested avatar the
person is skipped (896 were). Every avatar is therefore genuinely sharp — this arm holds
image quality constant and moves only size, which no other arm does.</li>
<li><b>Masked and placed.</b> Circle or rounded square, alternating, composited onto a
640&nbsp;px light canvas at a non-overlapping spot.</li>
<li><b>Paired across sizes.</b> The cast and their anchors are chosen ONCE at 160&nbsp;px
and re-rendered at every smaller size. <code>a160_0007</code> and <code>a56_0007</code>
hold the same people in the same places, so a score difference across sizes cannot be
blamed on different people or a different layout. Verified: 334 people at every size with
an identical 126&nbsp;Woman / 128&nbsp;Man / 80&nbsp;Child mix.</li>
</ol>
<h3>How a model is scored on it</h3>
<ul>
<li><b>Ground truth</b> is the placed avatar's square, class from the human label.</li>
<li><b>A match</b> is IoU&nbsp;&ge;&nbsp;0.5, greedy, one prediction per GT — the same
matcher every other scorer in this repo uses.</li>
<li><b>Detection recall</b> = matched GT / all GT, class ignored.</li>
<li><b>Class accuracy</b> = of those matched, how many carry the right class.</li>
<li><b>End-to-end</b> = found AND labelled correctly, per class. This is the product
metric: a woman detected but called Man is a woman who does not get blurred, so she counts
as a failure here and as a success under plain recall.</li>
<li><b>mAP</b> is separate and threshold-free (COCO 101-point, confidence-ordered), so it
rewards ranking quality rather than behaviour at one cut.</li>
</ul>
<p class="blurb"><b>Read both metric families.</b> They disagree here, and the reason is
calibration: <code>gelannfav14r4fw_gemlb_v2</code> has the best mAP at every size while
finishing last at conf&nbsp;0.45, because its median matched-detection confidence is 0.420
— below the shipped cut. Move its threshold to 0.25 and its 56&nbsp;px recall goes
47.6%&nbsp;&rarr;&nbsp;74.6%.</p>
<p class="dim"><b>Limits.</b> The crop is a geometric approximation from the person's box,
not a face detection. The source is LAGENDA general photography, not real scraped profile
pictures, so pose and lighting are photographic rather than selfie-typical. The canvas is
flat, not a rendered feed UI.</p></section>
"""


def groups(report):
    """`report` picks which benchmark the page covers. The two are never mixed:
    the crowd arm is real photographs with person-only GT (detection only), the
    synthetic arms carry human gender/age (detection AND classification)."""
    g = []
    if report == "avatars":
        for a in AVATAR_SIZES:
            px = a[1:]
            g.append(dict(
                name=f"avatars {px}px", images=f"{SP}/avatars/images",
                labels=f"{SP}/avatars/labels", raw=f"{B}/avatars/%s/raw",
                gendered=True, height=a, stat="avatars",
                map_json=f"{B}/avatars/%s/{a}_map.json",
                blurb=f"Facebook/LinkedIn-style profile pictures: head-and-shoulders "
                      f"crops taken from the original photo, masked to a circle or "
                      f"rounded square, and laid out on a 640 canvas at {px} px. "
                      f"Every avatar is at least a 1.5x downscale of a source that passed a "
                      f"calibrated sharpness gate, so they are CRISP like real profile "
                      f"pictures — this arm separates "
                      f"\u201csmall\u201d from \u201clow quality\u201d, which the "
                      f"other arms confound."))
        return g
    if report == "crowd":
        g.append(dict(
            name="crowd_small", images=f"{SP}/crowd_small/images",
            labels=f"{SP}/crowd_small/labels", raw=f"{B}/crowd_small/%s/raw",
            gendered=False, height=None, stat="crowd", map_json=None,
            blurb="800 real CrowdHuman photos, 16,314 people whose full body is "
                  "\u226496 px. Ground truth is person-only, so this arm measures "
                  "DETECTION only — the class colours are the model's claim, "
                  "nothing scores them."))
        return g
    for h in ("h96", "h64", "h48", "h32", "h24"):
        g.append(dict(
            name=f"synth_shrunk {h}", images=f"{SP}/synth_shrunk/{h}/images",
            labels=f"{SP}/synth_shrunk/{h}/labels", raw=f"{B}/synth/%s/{h}/raw",
            gendered=True, height=None, stat=f"synth:{h}",
            map_json=f"{B}/synth/%s/{h}_map.json",
            blurb=f"The same 480 LAGENDA people, whole scene shrunk so the tallest "
                  f"person is {h[1:]} px NATIVE. Canvas keeps the original size "
                  f"(median 1,280 px), so at 640 model input that person is only "
                  f"~{int(h[1:]) * 640 // 1280} px."))
    for h in ("h96", "h64", "h48", "h32", "h24"):
        g.append(dict(
            name=f"paste_grey {h}", images=f"{SP}/paste_grey/images",
            labels=f"{SP}/paste_grey/labels", raw=f"{B}/paste/%s/raw",
            gendered=True, height=h, stat="paste",
            map_json=f"{B}/paste/%s/{h}_map.json",
            blurb=f"SAM-cut people pasted on flat grey at 640, six per image, each "
                  f"{h[1:]} px tall — and because the canvas IS 640, {h[1:]} px "
                  f"native is {h[1:]} px at model input."))
    return g


def load_preds(raw_dir: Path, stem: str, conf: float):
    f = raw_dir / f"{stem}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return [(x["cls"], *x["box_xyxy"], x["conf"])
            for x in d.get("detections", []) if x.get("conf", 0) >= conf]


def full_detcls(labels_dir: Path, raw_dir: Path, conf: float, height=None):
    """Detection AND classification over the WHOLE arm, at the production
    threshold — computed from the log-raw sidecars and the GT labels, with no
    image decoding (the sidecars carry width/height).

    Three different questions, kept apart because they answer different things:
      det_recall   was the person FOUND at all (class ignored)
      cls_acc      given they were found, was the class right
      e2e[c]       was a person of class c found AND called c — the product
                   metric: a woman found but labelled Man is not blurred
    `height` filters paste_grey by its `h<NN>_` filename prefix.
    """
    tot = [0, 0, 0]                       # n_gt, found, class-right
    per = {c: [0, 0, 0] for c in (0, 1, 2)}
    if not raw_dir.exists():
        return None
    for f in sorted(raw_dir.glob("*.json")):
        if height and not f.stem.startswith("%s_" % height):
            continue
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        gt = seg_boxes(labels_dir / ("%s.txt" % f.stem), d["width"], d["height"]) or []
        if not gt:
            continue
        preds = [(x["cls"], *x["box_xyxy"], x["conf"])
                 for x in d.get("detections", []) if x.get("conf", 0) >= conf]
        mt = match_boxes([(g[1], g[2], g[3], g[4]) for g in gt], preds, MATCH_IOU)
        for gi, g in enumerate(gt):
            c = g[0]
            tot[0] += 1
            if c in per:
                per[c][0] += 1
            if gi in mt:
                tot[1] += 1
                if c in per:
                    per[c][1] += 1
                if mt[gi][0][0] == c:
                    tot[2] += 1
                    if c in per:
                        per[c][2] += 1
    if not tot[0]:
        return None
    rate = lambda k, n: (k / n) if n else None          # noqa: E731
    return {"n_gt": tot[0],
            "det_recall": rate(tot[1], tot[0]),
            "cls_acc": rate(tot[2], tot[1]),
            "e2e": {CLS_NAME[c]: rate(per[c][2], per[c][0]) for c in per},
            "det_by_cls": {CLS_NAME[c]: rate(per[c][1], per[c][0]) for c in per},
            "n_by_cls": {CLS_NAME[c]: per[c][0] for c in per}}


def map_stats(map_json_tpl, model: str):
    """The standardised threshold-free block: mAP50 / mAP75 / mAP50-95 and
    per-class AP50, straight from `map_eval.py` (COCO 101-point, conf-ordered
    matching, ignore regions honoured). Returns None when the arm has no class
    ground truth to compute it from."""
    if not map_json_tpl:
        return None
    f = Path(map_json_tpl % arm_dir(model) if "%s" in map_json_tpl else map_json_tpl)
    if not f.exists():
        return None
    try:
        j = json.loads(f.read_text())
    except ValueError:
        return None
    per = j.get("per_class") or {}
    return {"map50": j.get("map50"), "map75": j.get("map75"),
            "map50_95": j.get("map50_95"),
            "ap50": {c: (per.get(c) or {}).get("ap50") for c in
                     ("Woman", "Man", "Child")}}


def full_stats(kind: str, model: str):
    """The authoritative numbers for this group — read from the eval outputs over
    the WHOLE arm, not from the handful of images shown below. An 8-image sample
    is for looking at, never for quoting."""
    d = arm_dir(model)
    try:
        if kind == "crowd":
            j = json.loads(Path(f"{B}/crowd_small/{d}/eval/summary.json").read_text())
            return {"recall": j["detection_recall"]["rate"],
                    "n": j["detection_recall"]["n"],
                    "precision": j["detection_precision"]["rate"], "extra": None}
        if kind.startswith("synth:"):
            h = kind.split(":", 1)[1]
            j = json.loads(Path(f"{B}/synth/{d}/{h}_map.json").read_text())
            per = j.get("per_class", {})
            return {"recall": None, "n": None, "precision": None,
                    "map50": j.get("map50"),
                    "extra": {c: per.get(c, {}).get("ap50") for c in
                              ("Woman", "Man", "Child")}}
        if kind == "avatars":
            return None            # no per-size AP file; full_detcls carries this arm
        if kind == "paste":
            ngt = nf = nr = 0
            for line in open(f"{B}/paste/{d}/eval/sam_matches.jsonl"):
                r = json.loads(line)
                ngt += 1
                if r["status"] != "missed":
                    nf += 1
                    if r.get("pred_class") == r.get("gt_class"):
                        nr += 1
            return {"recall": nf / ngt if ngt else None, "n": ngt,
                    "precision": None,
                    "extra": {"class acc": nr / nf if nf else None}}
    except (OSError, KeyError, ValueError):
        return None
    return None


def score_group(images, labels_dir: Path, raw_tpl: str, models, conf):
    """Recall and class accuracy per model over the sampled images."""
    from PIL import Image
    out = {}
    for m in models:
        raw = raw_dir_for(raw_tpl, m)
        n_gt = n_found = n_right = 0
        for img in images:
            with Image.open(img) as im:
                w, h = im.size
            gt = seg_boxes(labels_dir / f"{img.stem}.txt", w, h) or []
            preds = load_preds(raw, img.stem, conf)
            if preds is None:
                continue
            gtb = [(g[1], g[2], g[3], g[4]) for g in gt]
            mt = match_boxes(gtb, preds, MATCH_IOU)
            n_gt += len(gtb)
            n_found += len(mt)
            for gi, (det, _j) in mt.items():
                if det[0] == gt[gi][0]:
                    n_right += 1
        out[m] = {"n_gt": n_gt, "found": n_found, "cls_right": n_right,
                  "recall": n_found / n_gt if n_gt else None,
                  "cls_acc": n_right / n_found if n_found else None}
    return out


CLS_INITIAL = {0: "W", 1: "M", 2: "C"}


def panel(img_path: Path, gt, preds, max_dim, redact=(), mode="model"):
    """One panel showing ONE thing.

    mode="gt"     : ground truth only, green.
    mode="model"  : that model's predictions only, coloured by predicted class,
                    plus any GT person it MISSED as a thin red box.

    They are separate panels on purpose. On the avatar arm the GT box *is* the
    avatar square and a model predicts almost exactly on it, so drawing both in
    one panel puts two rectangles a few pixels apart — which reads as a single
    dashed line and tells the viewer nothing.

    `redact` regions become flat hatched blocks labelled "hidden", so a
    deliberately concealed person can never be mistaken for a soft avatar.
    """
    from PIL import Image, ImageDraw
    from build_error_gallery import _b64_jpeg, _resize_max
    with Image.open(img_path) as im:
        img = im.convert("RGB")
    scale = min(1.0, max_dim / max(img.size))
    d = ImageDraw.Draw(img)

    for x1, y1, x2, y2 in redact:
        bx = (max(0, int(x1)), max(0, int(y1)),
              min(img.width, int(x2)), min(img.height, int(y2)))
        if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2:
            continue
        d.rectangle(bx, fill=(232, 232, 236))
        step = max(6, (bx[2] - bx[0]) // 6)
        for off in range(0, (bx[2] - bx[0]) + (bx[3] - bx[1]), step):
            d.line([(bx[0] + off, bx[1]), (bx[0], bx[1] + off)],
                   fill=(203, 203, 211), width=2)
        d.rectangle(bx, outline=(165, 165, 175), width=2)
        if (bx[2] - bx[0]) * scale > 42:
            d.text((bx[0] + 4, bx[1] + 3), "hidden", fill=(120, 120, 130))

    lw = max(2, int(max(img.size) / 300))

    def tag(box, text, colour):
        x1, y1, _x2, _y2 = box
        if (box[2] - box[0]) * scale < 26:
            return
        d.text((x1 + lw + 1, y1 + lw + 1), text, fill=colour)

    if mode == "gt":
        for c, x1, y1, x2, y2 in gt:
            d.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=lw)
            tag((x1, y1, x2, y2), CLS_INITIAL.get(c, "?"), GT_COLOR)
    else:
        gtb = [(g[1], g[2], g[3], g[4]) for g in gt]
        matched = match_boxes(gtb, preds or [], MATCH_IOU)
        for gi, g in enumerate(gt):
            if gi not in matched:          # a person this model did not find
                d.rectangle([g[1], g[2], g[3], g[4]], outline=(220, 40, 40), width=lw)
                tag((g[1], g[2], g[3], g[4]), "miss", (220, 40, 40))
        for c, x1, y1, x2, y2, cf in (preds or []):
            col = CLS_COLOR.get(c, (150, 150, 150))
            d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
            tag((x1, y1, x2, y2), CLS_INITIAL.get(c, "?"), col)
    return _b64_jpeg(_resize_max(img, max_dim), quality=72)


def _gate_text(root: Path):
    try:
        m = json.loads((root / "manifest.json").read_text())
        return ("VoL &ge; %g at &ge; %g&times; downscale" % (
            m.get("min_sharpness_vol128", 0), m.get("min_scale", 1)))
    except (OSError, ValueError):
        return "unrecorded"


def sharpness_audit(n_worst, max_dim=150, exclude_women=True):
    """The way to CONFIRM the set is crisp: measure every emitted avatar and
    show the WORST ones, magnified, with their scores. A sampled page can hide
    a bad tail; a worst-first page cannot."""
    from PIL import Image
    from build_error_gallery import _b64_jpeg
    from build_small_person_set import crop_sharpness

    root = Path(f"{SP}/avatars")
    if not (root / "labels").exists():
        return ""
    rows = []
    for f in sorted((root / "images").glob("*.jpg")):
        with Image.open(f) as im:
            img = im.convert("RGB")
        for cls, x1, y1, x2, y2 in (seg_boxes(root / "labels" / f"{f.stem}.txt",
                                              img.width, img.height) or []):
            if exclude_women and cls == 0:
                continue
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.width < 4:
                continue
            rows.append((crop_sharpness(crop), CLS_NAME.get(cls, "?"),
                         crop.width, crop))
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0])
    scores = [r[0] for r in rows]

    def pctl(q):
        return scores[min(len(scores) - 1, int(len(scores) * q / 100))]

    cards = "".join(
        "<figure><img src='data:image/jpeg;base64,%s'>"
        "<figcaption>%.0f<br><span class='dim'>%s &middot; %d px</span></figcaption>"
        "</figure>" % (
            _b64_jpeg(c.resize((max_dim, max_dim), Image.NEAREST), 88),
            sc, html.escape(cl), w)
        for sc, cl, w, c in rows[:n_worst])
    return ("<section class='method'><h2>Sharpness audit &mdash; how to confirm this "
            "yourself</h2><p class='blurb'>Every avatar was measured with "
            "variance-of-Laplacian normalised at 128&nbsp;px (the metric the Spotlight "
            "pipeline uses for <code>blurriness</code>). Below are the <b>%d least-sharp "
            "avatars in the entire set</b>, worst first, magnified, each captioned with "
            "its score. Worst-first, not sampled: if the bottom of the distribution is "
            "acceptable, everything above it is. Men and children only.</p>"
            "<p class='dim'>n = %d avatars measured (women excluded from this view) "
            "&middot; min %.0f &middot; p05 %.0f &middot; p25 %.0f &middot; median %.0f "
            "&middot; p75 %.0f &middot; max %.0f &middot; source gate was %s</p>"
            "<div class='zooms'>%s</div></section>" % (
                min(n_worst, len(rows)), len(scores), scores[0], pctl(5), pctl(25),
                pctl(50), pctl(75), scores[-1], _gate_text(root), cards))


def map_matrix(which, models, conf):
    """The whole size response in one table, straight from map_eval.py."""
    grps = [g for g in groups(which) if g.get("map_json")]
    if not grps:
        return ""
    cols = [g["name"].split()[-1] for g in grps]
    counts = []
    for g in grps:
        n = 0
        for f in sorted(Path(g["labels"]).glob("*.txt")):
            if g.get("height") and not f.stem.startswith("%s_" % g["height"]):
                continue
            n += len(seg_boxes(f, 1, 1) or [])
        counts.append(n)

    out = []
    for metric, key, blurb in (
            ("MAP50", "map50", "boxes at IoU 0.50 — the headline"),
            ("MAP50-95", "map50_95", "averaged over IoU 0.50:0.95 — box tightness counts"),
            ("WOMAN AP50", "woman", "the class the blur depends on")):
        rows = ""
        for m in models:
            cells = ""
            for g in grps:
                k = map_stats(g["map_json"], m) or {}
                v = (k.get("ap50") or {}).get("Woman") if key == "woman" else k.get(key)
                cells += "<td class='n s'>%s</td>" % ("&mdash;" if v is None
                                                      else "%.3f" % v)
            rows += "<tr><td>%s</td>%s</tr>" % (html.escape(m), cells)
        out.append("<h3>%s <span class='dim'>%s</span></h3>"
                   "<table><thead><tr><th>model</th>%s</tr></thead>"
                   "<tbody>%s</tbody></table>" % (
                       metric, html.escape(blurb),
                       "".join("<th>%s<br><span class='dim'>n=%d</span></th>"
                               % (html.escape(c), n) for c, n in zip(cols, counts)),
                       rows))
    return ("<section class='matrix'><h2>mAP across resolutions</h2>"
            "<p class='blurb'>Every cell is <code>map_eval.py</code> over that "
            "resolution's images only (COCO 101-point, confidence-ordered matching, "
            "ignore regions honoured), so it is threshold-free and independent of the "
            "conf-%s cut used elsewhere. Comparable down a column and across a row; NOT "
            "comparable to an arm whose canvas size differs. <b>n</b> is the number of "
            "labelled people scored.</p>%s</section>" % (conf, "".join(out)))


def build(out_path: Path, which, n_images, seed, conf, max_dim, models,
          hide_women=True, n_worst=32):
    from PIL import Image
    parts = []
    for grp in groups(which):
        name, blurb = grp["name"], grp["blurb"]
        gendered, height, raw_tpl = grp["gendered"], grp["height"], grp["raw"]
        img_dir, lab_dir = Path(grp["images"]), Path(grp["labels"])
        if not img_dir.exists():
            continue
        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                      and (not height or p.stem.startswith("%s_" % height)))
        with_gt = [p for p in imgs
                   if (seg_boxes(lab_dir / f"{p.stem}.txt", 1, 1) or [])]
        pool = with_gt or imgs
        if hide_women and gendered:
            # fewest women first: a page of entirely-hidden avatars shows nothing
            def n_women(q):
                return sum(1 for b in (seg_boxes(lab_dir / f"{q.stem}.txt", 1, 1) or [])
                           if b[0] == 0)
            pool = sorted(pool, key=n_women)[:max(n_images * 4, 24)]
        if not pool:
            continue
        sample = random.Random(seed).sample(pool, min(n_images, len(pool)))
        stats = score_group(sample, lab_dir, raw_tpl, models, conf)

        def pct(v):
            return "&mdash;" if v is None else "%.1f%%" % (v * 100)

        full = {m: full_detcls(lab_dir, raw_dir_for(raw_tpl, m), conf, height)
                for m in models}
        ap = {m: full_stats(grp["stat"], m) for m in models}
        mp = {m: map_stats(grp.get("map_json"), m) for m in models}

        if gendered:
            head = ("<tr><th rowspan='2'>model</th><th rowspan='2'>people</th>"
                    "<th colspan='5'>at conf %s &mdash; what ships</th>"
                    "<th colspan='6'>threshold-free (COCO mAP)</th></tr>"
                    "<tr><th>detection recall</th>"
                    "<th>class acc<br><span class='dim'>(of those found)</span></th>"
                    "<th>Woman e2e</th><th>Man e2e</th><th>Child e2e</th>"
                    "<th>MAP50</th><th>MAP75</th><th>MAP50-95</th>"
                    "<th>WOMAN AP50</th><th>MAN AP50</th><th>CHILD AP50</th></tr>"
                    % conf)
        else:
            head = ("<tr><th>model</th><th>people</th><th>detection recall</th>"
                    "<th>precision</th><th class='dim'>sample recall</th></tr>")

        rows = ""
        for m in models:
            f, a = full[m], ap[m] or {}
            if f is None:
                rows += "<tr><td>%s</td><td colspan='12'>no dump</td></tr>" % html.escape(m)
                continue
            if gendered:
                k = mp[m] or {}
                num = lambda v: "&mdash;" if v is None else "%.3f" % v   # noqa: E731
                rows += ("<tr><td>%s</td><td class='n'>%d</td><td class='n'>%s</td>"
                         "<td class='n'>%s</td><td class='n b'>%s</td><td class='n'>%s</td>"
                         "<td class='n'>%s</td>"
                         "<td class='n s'>%s</td><td class='n s'>%s</td><td class='n s'>%s</td>"
                         "<td class='n s'>%s</td><td class='n s'>%s</td>"
                         "<td class='n s'>%s</td></tr>") % (
                    html.escape(m), f["n_gt"], pct(f["det_recall"]), pct(f["cls_acc"]),
                    pct(f["e2e"]["Woman"]), pct(f["e2e"]["Man"]), pct(f["e2e"]["Child"]),
                    num(k.get("map50")), num(k.get("map75")), num(k.get("map50_95")),
                    num((k.get("ap50") or {}).get("Woman")),
                    num((k.get("ap50") or {}).get("Man")),
                    num((k.get("ap50") or {}).get("Child")))
            else:
                # crowd numbers come from eval_negatives_crowd.py, which drops
                # detections overlapping an ignore region BEFORE matching; this
                # page's own matcher does not, and reads ~1 pt higher.
                rows += ("<tr><td>%s</td><td class='n'>%d</td><td class='n'>%s</td>"
                         "<td class='n'>%s</td><td class='n dim'>%s</td></tr>") % (
                    html.escape(m), a.get("n") or f["n_gt"], pct(a.get("recall")),
                    pct(a.get("precision")), pct(stats[m]["recall"]))

        def redactions(img_path, gt):
            if not hide_women:
                return []
            if gendered:
                return [(b[1], b[2], b[3], b[4]) for b in gt if b[0] == 0]
            out, gtb = [], [(g[1], g[2], g[3], g[4]) for g in gt]
            for m in models:
                pr = load_preds(raw_dir_for(raw_tpl, m), img_path.stem, conf)
                if not pr:
                    continue
                for gi, (det, _j) in match_boxes(gtb, pr, MATCH_IOU).items():
                    if det[0] == 0:
                        out.append(gtb[gi])
            return out

        cards = []
        for p in sample:
            with Image.open(p) as im:
                w, h = im.size
            gt = seg_boxes(lab_dir / f"{p.stem}.txt", w, h) or []
            red = redactions(p, gt)
            panels = [
                f"<figure><figcaption class='mt gtp'>ground truth</figcaption>"
                f"<img src='data:image/jpeg;base64,"
                f"{panel(p, gt, None, max_dim, red, mode='gt')}'>"
                f"<figcaption>{len(gt)} people</figcaption></figure>"]
            for m in models:
                preds = load_preds(raw_dir_for(raw_tpl, m), p.stem, conf)
                if preds is None:
                    cap = "no dump"
                else:
                    gtb = [(g[1], g[2], g[3], g[4]) for g in gt]
                    mt = match_boxes(gtb, preds, MATCH_IOU)
                    right = sum(1 for gi, (det, _j) in mt.items()
                                if det[0] == gt[gi][0])
                    cap = (f"found {len(mt)}/{len(gt)} &middot; right class {right}"
                           f" &middot; {len(preds)} boxes")
                panels.append(
                    f"<figure><figcaption class='mt'>{html.escape(m)}</figcaption>"
                    f"<img src='data:image/jpeg;base64,"
                    f"{panel(p, gt, preds, max_dim, red)}'>"
                    f"<figcaption>{cap}</figcaption></figure>")
            cards.append(f"<div class='card'><h4>{html.escape(p.stem)} "
                         f"<span class='dim'>{w}&times;{h}</span></h4>"
                         f"<div class='row'>{''.join(panels)}</div></div>")

        note = ("<p class='dim'>All figures span the WHOLE arm at conf %s, not the %d "
                "images below. <b>End-to-end</b> = found AND labelled correctly; a woman "
                "found but called Man is not blurred, so that column is the product "
                "metric. Detection recall ignores class. The blue mAP block is "
                "threshold-free and not comparable across arms with different canvas "
                "sizes.</p>" % (conf, len(sample))
                if gendered else
                "<p class='dim'>Figures span the whole arm at conf %s and come from "
                "<code>eval_negatives_crowd.py</code>, the standard crowd scorer, which "
                "excludes detections landing on an ignore region before matching. Class "
                "accuracy is omitted: the ground truth is person-only, so there is no "
                "gender to be right or wrong about.</p>" % conf)

        parts.append(f"<section><h2>{html.escape(name)}</h2>"
                     f"<p class='blurb'>{html.escape(blurb)}</p>"
                     f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>"
                     f"{note}{''.join(cards)}</section>")

    is_synth = which in ("synthetic", "avatars")
    ttl = {"synthetic": "Small people \u2014 SYNTHETIC arms",
           "avatars": "Profile pictures \u2014 AVATAR arm",
           "crowd": "Small people \u2014 REAL CROWD photos"}[which]
    if which == "avatars":
        intro = ("Simulated social-media profile pictures \u2014 the head-and-shoulders "
                 "framing a feed is full of, which is a different distribution from the "
                 "full-body people these detectors were trained on. Crops come from the "
                 "original photograph, are never upscaled, and must clear a calibrated "
                 "sharpness gate, so the people are CRISP at every size: this arm asks "
                 "whether a profile picture gets blurred with image quality held "
                 "constant and only avatar size changing. Both <b>detection</b> and "
                 "<b>classification</b> are scored.")
    elif which == "synthetic":
        intro = ("Two synthetic arms, both carrying human gender/age labels, so both "
                 "<b>detection</b> and <b>classification</b> are scored. They differ in "
                 "one decisive way: <code>synth_shrunk</code> shrinks a whole scene onto "
                 "a canvas of the original size, so its \u201ch96\u201d person is only "
                 "~48&nbsp;px at model input; <code>paste_grey</code> pastes cut-out "
                 "people on a 640 canvas, where h96 really is 96&nbsp;px. Never pool "
                 "them.")
    else:
        intro = ("Real CrowdHuman photographs \u2014 the most realistic arm, and the one "
                 "to weight most heavily. Ground truth is person-only, so this page "
                 "scores <b>detection</b> alone.")

    redact_note = ("<p class='blurb' style='background:#fff3f3'><b>Women are not "
                   "rendered on this page.</b> Each is replaced by a flat hatched block "
                   "labelled \u201chidden\u201d \u2014 a solid fill, deliberately NOT a "
                   "blur, so a concealed person can never be mistaken for a soft avatar. "
                   "Boxes and every model's prediction are still drawn on top, so the "
                   "comparison is unaffected. All TABLE numbers are computed on the "
                   "unredacted images and include women in full."
                   + ("" if is_synth else " This arm's ground truth is person-only, so "
                      "gender is unknown and the substitute rule is: hide any person that "
                      "any of the four models called Woman.") + "</p>"
                   if hide_women else "")
    method = AVATAR_METHOD if which == "avatars" else ""
    audit = sharpness_audit(n_worst) if which == "avatars" else ""

    out_path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>{ttl}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:22px;background:#fff;color:#111;max-width:1600px}}
h1{{margin-bottom:4px}} h2{{margin-top:34px;border-bottom:2px solid #333;padding-bottom:4px}}
h4{{margin:0 0 6px;font-size:13px;font-weight:600}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0}}
.row{{display:flex;gap:10px;flex-wrap:wrap}}
figure{{margin:0;text-align:center}} figure img{{border:1px solid #bbb;display:block}}
figcaption{{font-size:11px;color:#555;margin-top:3px}}
.mt{{margin:0 0 3px;font-weight:600;color:#111}} .gtp{{color:#0a8a32}}
.dim{{color:#777;font-weight:400;font-size:12px}}
.blurb{{background:#f6f6f6;padding:8px 12px;border-radius:6px;font-size:13px}}
table{{border-collapse:collapse;margin:10px 0;font-size:13px}}
th,td{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
th{{background:#f0f0f0}} td.n{{text-align:right}}
td.b{{font-weight:700;background:#fff6fa}} td.s{{background:#f4f8ff}}
.method ol,.method ul{{font-size:13px;line-height:1.55}}
.method h3{{margin:16px 0 4px;font-size:14px}} .matrix h3{{margin:14px 0 4px;font-size:13px}}
.zooms{{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}}
.zooms figure img{{image-rendering:pixelated;border:2px solid #888}}
.key span{{display:inline-block;margin-right:14px;font-size:13px}}
.sw{{display:inline-block;width:12px;height:12px;border:2px solid;vertical-align:-2px;margin-right:4px}}
</style>
<h1>{ttl}</h1>
<p class="key">
<b>First panel = ground truth. Every other panel = that model's predictions only.</b><br>
<span><i class="sw" style="border-color:#00e63c"></i>ground truth (W/M/C)</span>
<span><i class="sw" style="border-color:#e91e63"></i>predicted Woman</span>
<span><i class="sw" style="border-color:#2196f3"></i>predicted Man</span>
<span><i class="sw" style="border-color:#ff9800"></i>predicted Child</span>
<span><i class="sw" style="border-color:#dc2828"></i>missed &mdash; GT this model did not find</span>
<span><i class="sw" style="border-color:#a5a5af;background:#e8e8ec"></i>hidden (woman)</span>
</p>
<p class="dim">GT and predictions are never drawn in the same panel: on these arms the GT
box IS the pasted square and models predict almost exactly on it, so overlaying them puts
two rectangles a few pixels apart and reads as one dashed line.</p>
<p class="blurb">{intro}<br><br>Four candidates, boxes drawn at the production threshold
<b>conf {conf}</b>. The arms disagree with each other, which is itself a finding — see
<code>MODEL_COMPARISON.md</code>.</p>
{redact_note}
{method}
{map_matrix(which, models, conf)}
{audit}
{''.join(parts)}""")
    return out_path


def _selftest():
    from PIL import Image
    tmp = Path("/tmp/_sp_gallery")
    for s in ("images", "labels", "raw"):
        (tmp / s).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 120), (40, 40, 40)).save(tmp / "images" / "a.jpg")
    (tmp / "labels" / "a.txt").write_text("0 0.25 0.5 0.2 0.5\n")   # 40x60 @ (30,30)
    (tmp / "raw" / "a.json").write_text(json.dumps({
        "image": "a.jpg", "width": 200, "height": 120, "detections": [
            {"cls": 0, "box_xyxy": [30, 30, 70, 90], "conf": 0.9},
            {"cls": 1, "box_xyxy": [10, 10, 20, 20], "conf": 0.10}]}))

    assert load_preds(tmp / "raw", "a", 0.45) == [(0, 30, 30, 70, 90, 0.9)], "conf filter"
    assert load_preds(tmp / "raw", "missing", 0.45) is None, "absent dump != empty dump"
    assert len(load_preds(tmp / "raw", "a", 0.05)) == 2

    st = score_group([tmp / "images" / "a.jpg"], tmp / "labels", str(tmp / "raw"),
                     ["m"], 0.45)["m"]
    assert st == {"n_gt": 1, "found": 1, "cls_right": 1, "recall": 1.0,
                  "cls_acc": 1.0}, st

    # a wrong class must count as found-but-misclassified, never as a miss
    (tmp / "raw" / "a.json").write_text(json.dumps({
        "image": "a.jpg", "width": 200, "height": 120,
        "detections": [{"cls": 1, "box_xyxy": [30, 30, 70, 90], "conf": 0.9}]}))
    st = score_group([tmp / "images" / "a.jpg"], tmp / "labels", str(tmp / "raw"),
                     ["m"], 0.45)["m"]
    assert st["found"] == 1 and st["cls_right"] == 0 and st["cls_acc"] == 0.0, st

    b64 = panel(tmp / "images" / "a.jpg",
                [(0, 30, 30, 70, 90)], [(0, 30, 30, 70, 90, 0.9)], 160)
    assert isinstance(b64, str) and len(b64) > 100
    assert arm_dir("y26n_noe2e_warm50-2") == "y26n_warm50-2", "on-disk alias"
    assert raw_dir_for("/x/%s/raw", "y26n_noe2e_warm50-2") == Path("/x/y26n_warm50-2/raw")
    assert raw_dir_for("/x/raw", "anything") == Path("/x/raw")
    assert arm_dir("y26n_sop50") == "y26n_sop50"
    cg = [g["name"] for g in groups("crowd")]
    sg = [g["name"] for g in groups("synthetic")]
    assert cg == ["crowd_small"], cg
    assert len(sg) == 10 and not set(cg) & set(sg), "reports must not share arms"
    assert all(g["gendered"] for g in groups("synthetic")), "synthetic GT is gendered"
    assert not groups("crowd")[0]["gendered"], "crowd GT is person-only"
    assert [g["height"] for g in groups("synthetic")][5:] == \
        ["h96", "h64", "h48", "h32", "h24"], "paste arm must split by height"

    print("build_smallperson_gallery selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--report", choices=["synthetic", "crowd", "avatars"],
                    help="synthetic = shrunk scenes + grey-canvas pastes, with "
                         "gender/age GT (detection AND classification); "
                         "crowd = real CrowdHuman photos, person-only GT "
                         "(detection only)")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--max-dim", type=int, default=380)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--show-women", action="store_true",
                    help="render women in the sample images too (default: their "
                         "pixels are blurred out; boxes are still drawn)")
    ap.add_argument("--n-worst", type=int, default=32,
                    help="avatars: how many least-sharp avatars to show in the audit")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.out or not a.report:
        ap.error("--out and --report are required (or use --selftest)")
    p = build(Path(a.out), a.report, a.n_images, a.seed, a.conf, a.max_dim,
              a.models, hide_women=not a.show_women, n_worst=a.n_worst)
    print(f"wrote {p}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
