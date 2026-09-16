#!/usr/bin/env python3
"""Self-contained HTML proof report for the two small-person benchmarks.

Shows the actual images used (sampled, GT drawn) and both metric tables
(mAP50 AND mAP50-95), clearly labeled. Women are ALWAYS redacted
(pixelate+blur of every GT-Woman region and every class-3/unknown ignore
region) — same rule as deploy_report / build_smallperson_gallery.
"""
import base64
import glob
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SP = Path("/workspace/datasets/smallperson_v1/avatars")
AV = Path("/workspace/exp19/avatars")
HS = Path("/workspace/deploycmp/holdout_small")
HIMG = Path("/workspace/datasets/haramblur_holdout/labeling/full/images")
DIMS = Path("/workspace/holdout_eval/v11n_shipped/raw")
OUT = Path("/workspace/deploycmp/SMALL_PERSON_PROOF.html")

MODELS = ["yolo11N-640", "y26n_sop50", "y26n_warm50-2",
          "gelannfav14r4fw_gemlb_v2", "gelannfav14r6_gemlb_v2"]
HOLD_ALIAS = {"yolo11N-640": "v11n_shipped", "y26n_sop50": "y26n_sop50",
              "y26n_warm50-2": "y26n_warm50",
              "gelannfav14r4fw_gemlb_v2": "gelan_r4fw_v2",
              "gelannfav14r6_gemlb_v2": "gelan_r6_v2"}
SIZES = ["a128", "a96", "a64", "a48"]
COLOR = {0: (211, 54, 155), 1: (43, 111, 212), 2: (232, 132, 44),
         3: (130, 130, 130)}
CNAME = {0: "Woman", 1: "Man", 2: "Child", 3: "Unknown"}
rng = random.Random(20260828)


def rows_of(path):
    """[(cls, x1, y1, x2, y2 normalized)] from YOLO box or polygon rows."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cls = int(float(p[0]))
            vals = [float(x) for x in p[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
            out.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        else:
            xs, ys = vals[0::2], vals[1::2]
            if len(xs) >= 3 and len(xs) == len(ys):
                out.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return out


def pixelate(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    reg = img.crop((x1, y1, x2, y2))
    small = reg.resize((max(1, (x2 - x1) // 14), max(1, (y2 - y1) // 14)))
    reg = small.resize((x2 - x1, y2 - y1), Image.NEAREST)
    reg = reg.filter(ImageFilter.GaussianBlur(3))
    img.paste(reg, (x1, y1))


def b64(img, q=80):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def draw_box(dr, box, color, width=3, label=None):
    dr.rectangle(box, outline=color, width=width)
    if label:
        dr.text((box[0] + 2, max(0, box[1] - 12)), label, fill=color)


def avatar_card(stem):
    img = Image.open(SP / "images" / (stem + ".jpg")).convert("RGB")
    rows = [(c, x1 * img.width, y1 * img.height, x2 * img.width, y2 * img.height)
            for c, x1, y1, x2, y2 in rows_of(SP / "labels" / (stem + ".txt"))]
    for c, *box in rows:
        if c == 0:
            pixelate(img, box)
    dr = ImageDraw.Draw(img)
    for c, *box in rows:
        draw_box(dr, box, COLOR[c], 3, CNAME[c][0])
    img = img.resize((480, 480))
    n = len(rows)
    return (f'<div class="card"><img src="data:image/jpeg;base64,{b64(img)}">'
            f'<div class="cap">{stem} · {n} people (GT drawn; women pixelated)'
            f'</div></div>')


def holdout_card(stem):
    matches = glob.glob(str(HIMG / (stem + ".*")))
    if not matches:
        return ""
    img = Image.open(matches[0]).convert("RGB")
    W, H = img.width, img.height
    scale_in = 640.0 / max(W, H)
    small = [(c, x1 * W, y1 * H, x2 * W, y2 * H)
             for c, x1, y1, x2, y2 in rows_of(HS / "labels" / (stem + ".txt"))]
    ign = [(c, x1 * W, y1 * H, x2 * W, y2 * H)
           for c, x1, y1, x2, y2 in rows_of(HS / "ignore" / (stem + ".txt"))]
    for c, *box in small + ign:
        if c in (0, 3):
            pixelate(img, box)
    dr = ImageDraw.Draw(img)
    for c, *box in ign:
        draw_box(dr, box, (160, 160, 160), 2)
    for c, *box in small:
        h_in = (box[3] - box[1]) * scale_in
        draw_box(dr, box, (30, 190, 60), 4, f"{CNAME[c][0]} {h_in:.0f}px")
    zooms = []
    for c, *box in small[:5]:
        pad = 0.3 * max(box[2] - box[0], box[3] - box[1])
        crop = img.crop((max(0, int(box[0] - pad)), max(0, int(box[1] - pad)),
                         min(W, int(box[2] + pad)), min(H, int(box[3] + pad))))
        if crop.width < 2 or crop.height < 2:
            continue
        zh = 150
        crop = crop.resize((max(1, int(crop.width * zh / crop.height)), zh),
                           Image.NEAREST)
        h_in = (box[3] - box[1]) * scale_in
        zooms.append(
            f'<div class="z"><img src="data:image/jpeg;base64,{b64(crop, 85)}">'
            f'<div class="zcap">{CNAME[c]} · {h_in:.0f}px @640</div></div>')
    if max(W, H) > 760:
        r = 760.0 / max(W, H)
        img = img.resize((int(W * r), int(H * r)))
    n64 = sum(1 for c, *b in small if (b[3] - b[1]) * scale_in <= 64)
    return (f'<div class="wide"><img src="data:image/jpeg;base64,{b64(img)}">'
            f'<div class="cap">{stem} · {len(small)} small (green, height at '
            f'model input labeled; {n64} of them &le;64px) · gray = ignore '
            f'(larger people / unknowns) · women &amp; unknowns pixelated</div>'
            f'<div class="zrow">{"".join(zooms)}</div></div>')


def table(headers, rows, hi=None):
    """rows: (label, [floatcells]); bold best per column at 3dp."""
    best = []
    for j in range(len(headers) - 1):
        vals = [f"{r[1][j]:.3f}" for r in rows]
        best.append(max(vals))
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = ""
    for lbl, cells in rows:
        tds = ""
        for j, v in enumerate(cells):
            s = f"{v:.3f}"
            tds += f"<td class='b'>{s}</td>" if s == best[j] else f"<td>{s}</td>"
        body += f"<tr><td class='m'>{lbl}</td>{tds}</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def pct_table(headers, rows):
    best = []
    for j in range(len(headers) - 1):
        best.append(max(f"{100*r[1][j]:.1f}" for r in rows))
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = ""
    for lbl, cells in rows:
        tds = ""
        for j, v in enumerate(cells):
            s = f"{100*v:.1f}%"
            cls = " class='b'" if f"{100*v:.1f}" == best[j] else ""
            tds += f"<td{cls}>{s}</td>"
        body += f"<tr><td class='m'>{lbl}</td>{tds}</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


# ---- data for tables ----
av_map = {}
for m in MODELS:
    av_map[m] = {}
    for s in SIZES:
        d = json.loads((AV / m / f"{s}_map.json").read_text())
        av_map[m][s] = (d["map50"], d["map50_95"],
                        (d.get("per_class", {}).get("Woman") or {}).get("ap50", 0))
E = json.loads((AV / "e2e_5model.json").read_text())
S = json.loads((HS / "small_slice_results.json").read_text())
CEN = json.loads((HS / "census.json").read_text())

sz_hdr = ["model", "128 px", "96 px", "64 px", "48 px"]
t_av50 = table(sz_hdr, [(m, [av_map[m][s][0] for s in SIZES]) for m in MODELS])
t_av5095 = table(sz_hdr, [(m, [av_map[m][s][1] for s in SIZES]) for m in MODELS])
t_avw50 = table(sz_hdr, [(m, [av_map[m][s][2] for s in SIZES]) for m in MODELS])
t_we2e = pct_table(sz_hdr, [(m, [E[m][s]["Woman"]["e2e"] / E[m][s]["Woman"]["n"]
                                 for s in SIZES]) for m in MODELS])
t_me2e = pct_table(sz_hdr, [(m, [E[m][s]["Man"]["e2e"] / E[m][s]["Man"]["n"]
                                 for s in SIZES]) for m in MODELS])

sm_rows50, sm_rows5095, sm_045 = [], [], []
for m in MODELS:
    d = S[HOLD_ALIAS[m]]
    pc = d["map"]["per_class"]
    sm_rows50.append((m, [d["map"]["map50"], pc["Woman"]["ap50"],
                          pc["Man"]["ap50"], pc["Child"]["ap50"]]))
    sm_rows5095.append((m, [d["map"]["map50_95"], pc["Woman"]["ap50_95"],
                            pc["Man"]["ap50_95"], pc["Child"]["ap50_95"]]))
    p = d["conf045"]["per_class"]
    sm_045.append((m, [d["conf045"]["det_recall"],
                       p["Woman"]["found"] / p["Woman"]["n"],
                       p["Woman"]["e2e"] / p["Woman"]["n"],
                       p["Man"]["found"] / p["Man"]["n"]]))
t_sm50 = table(["model", "mAP50", "Woman AP50", "Man AP50", "Child AP50"],
               sm_rows50)
t_sm5095 = table(["model", "mAP50-95", "Woman AP50-95", "Man AP50-95",
                  "Child AP50-95"], sm_rows5095)
t_sm045 = pct_table(["model", "det recall (all small)", "Woman found",
                     "Woman e2e", "Man found"], sm_045)

# ---- image proofs ----
av_cards = []
for s in SIZES:
    stems = sorted(p.stem for p in (SP / "labels").glob(f"{s}_*.txt"))
    for stem in rng.sample(stems, 3):
        av_cards.append(avatar_card(stem))

members = sorted(p.stem for p in (HS / "labels").glob("*.txt"))
by_coll = {}
for st in members:
    by_coll.setdefault(st.split("__")[0], []).append(st)
want = [("shiekhs", 5), ("women", 4), ("men", 3), ("child", 2), ("randoms", 1)]
h_cards = []
for coll, k in want:
    pool = by_coll.get(coll, [])
    for stem in rng.sample(pool, min(k, len(pool))):
        h_cards.append(holdout_card(stem))

css = """
body{font-family:-apple-system,Segoe UI,sans-serif;background:#fff;color:#1a1a1a;
     max-width:1180px;margin:24px auto;padding:0 16px}
h1{font-size:26px} h2{margin-top:38px;border-bottom:2px solid #ddd;padding-bottom:6px}
h3{margin-top:26px} .note{background:#f4f6f8;border-left:4px solid #8aa;
     padding:10px 14px;margin:12px 0;font-size:14px}
table{border-collapse:collapse;margin:14px 0;font-size:14px}
th,td{border:1px solid #ccc;padding:5px 10px;text-align:right}
th{background:#eef1f4} td.m{text-align:left;font-family:monospace;font-size:13px}
td.b{font-weight:700;background:#e8f5e9}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.card{width:300px} .card img{width:100%;border:1px solid #ccc}
.wide{margin:18px 0} .wide>img{max-width:100%;border:1px solid #ccc}
.cap{font-size:12.5px;color:#444;margin-top:4px}
.zrow{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.z img{height:150px;border:1px solid #bbb} .zcap{font-size:11.5px;color:#555}
.legend span{display:inline-block;margin-right:14px;font-size:13px}
"""
sbc = CEN["small_by_collection"]
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Small-person proof report</title><style>{css}</style></head><body>
<h1>Small-person benchmarks &mdash; proof report (five models)</h1>
<p>Companion to <b>SMALL_PERSON_FIVE_MODELS.md</b> (2026-08-28). Two separate
benchmarks, never pooled. Every table appears in <b>both mAP50 and
mAP50-95</b>. Tables marked "at conf 0.45" are outcome rates at the shipped
threshold &mdash; they are NOT AP numbers. All women in the images below are
pixelated (ground-truth Woman regions and unknown-gender regions); the table
numbers are computed on the unredacted images.</p>
<div class="legend"><b>Box colors:</b>
<span style="color:#d3369b">&#9632; Woman</span>
<span style="color:#2b6fd4">&#9632; Man</span>
<span style="color:#e8842c">&#9632; Child</span>
<span style="color:#1ebe3c">&#9632; small-slice GT person</span>
<span style="color:#a0a0a0">&#9632; ignore region</span></div>

<h2>A &middot; Avatars (simulated profile pictures)</h2>
<p>240 images, 1,364 people: sharp head-and-shoulders crops (LAGENDA source,
human gender/age, never upscaled) pasted on a 640&nbsp;px canvas as circles /
rounded squares at four person heights &mdash; 128 / 96 / 64 / 48 px. The same
341 people (139 Woman / 138 Man / 64 Child) appear at every size, so size is
the only variable.</p>
<h3>mAP50 by avatar size (threshold-free)</h3>{t_av50}
<h3>mAP50-95 by avatar size (strict boxes)</h3>{t_av5095}
<h3>Woman AP50 by avatar size</h3>{t_avw50}
<h3>Woman end-to-end at conf 0.45 &mdash; found AND called Woman (n=139/size; not an AP)</h3>{t_we2e}
<h3>Man end-to-end at conf 0.45 (n=138/size; not an AP)</h3>{t_me2e}
<div class="note">The two families disagree on purpose: the gelans rank boxes
best (mAP) but are calibrated low, so at the shipped 0.45 cut
<b>y26n_sop50</b> is the best Woman end-to-end at every size.</div>
<h3>Image proof &mdash; 3 random images per size (seed 20260828)</h3>
<div class="grid">{"".join(av_cards)}</div>

<h2>B &middot; QA-holdout small slice (real scrape data)</h2>
<p>Every person in the QA holdout whose ground-truth box height is
<b>&le;96&nbsp;px at model input</b> (native height &times; 640 / max(W,H)):
<b>1,587 of 16,139 people (9.8%) in 392 images</b>; 750 are &le;64 px, 379
&le;48 px. Class mix: Man 1,318 / Woman 209 / Child 60. By collection:
shiekhs {sbc.get("shiekhs", 0)}, women {sbc.get("women", 0)},
men {sbc.get("men", 0)}, child {sbc.get("child", 0)},
randoms {sbc.get("randoms", 0)}, women_hd {sbc.get("women_hd", 0)}.
People larger than the band become <i>ignore regions</i> (gray) &mdash; never
deletions &mdash; and the original unknown-gender ignores carry over, so the
score is a pure small-person number.</p>
<h3>mAP50 on the small slice (threshold-free)</h3>{t_sm50}
<h3>mAP50-95 on the small slice (strict boxes)</h3>{t_sm5095}
<h3>At conf 0.45 (outcome rates from the labels each model emitted; not AP)</h3>{t_sm045}
<div class="note">The production model finds 3 in 10 small people; the y26n
arms find ~6 in 10. On small women end-to-end, y26n_warm50-2 / y26n_sop50
lead (58.9% / 58.4%); gelan_r6 is last (35.4%).</div>
<h3>Image proof &mdash; 15 member images sampled across collections (seed 20260828)</h3>
<p>Green = a small-slice GT person, labeled with its height in model-input
pixels. Zoom strips under each image show those people at 150&nbsp;px
(nearest-neighbour, no smoothing) &mdash; this is the honest way to see how
little signal a small person carries.</p>
{"".join(h_cards)}

<h2>Method notes</h2>
<ul>
<li>GT drawn from the benchmark label files themselves (avatars:
<code>smallperson_v1/avatars/labels</code>; slice:
<code>deploycmp/holdout_small/labels+ignore</code>) &mdash; the exact files
the scorers read.</li>
<li>mAP: pycocotools-convention <code>map_eval.py</code> on floor-0.001
sidecars; slice scoring passes the ignore dir (IoA&ge;0.5 exclusion).</li>
<li>conf-0.45 tables: greedy IoU&ge;0.5 matching (project
<code>match_boxes</code>) against each model's emitted 0.45 labels; one code
path for all five models.</li>
<li>Redaction rule: every GT-Woman region and every unknown-gender region is
pixelated before rendering; no opt-out. Numbers are unaffected (computed on
originals).</li>
</ul>
</body></html>"""
OUT.write_text(html)
print("HTML_WRITTEN", OUT, len(html) // 1024, "KB",
      "avatar_cards", len(av_cards), "holdout_cards", len(h_cards))
