"""EXP-2026-10 — visual gallery of Lite's box corrections vs SAM3 vs GT.
Run on the pod:  python3 /tmp/box_gallery.py
Corrections are parsed as yxyx_1000 (Gemini's native convention, confirmed 2026-07-26)."""
import html as HT
from pathlib import Path
from PIL import Image, ImageDraw
from pipeline_v1_eval import load_verdicts, merge, build_crop, load_raw_dets
from crowd_headtohead import _b64_jpeg
from eval_negatives_crowd import load_odgt, _ioa, IGNORE_IOA
from run_model_children import iou, match_boxes

IMGS = Path('/workspace/datasets/crowdhuman/Images_sample500')
RAW  = Path('/workspace/exp10/raw_full/crowd')
GT_C, SAM_C, LITE_C = (0, 200, 0), (40, 90, 220), (235, 140, 20)

gt = load_odgt(Path('/workspace/datasets/crowdhuman/annotation_val.odgt'))
recs = load_verdicts(Path('/workspace/exp10/full_crowd'))
by = {}
for r in recs:
    by.setdefault(r['image'].rsplit('.', 1)[0], []).append(r)

cases = []
for stem, rr in by.items():
    e = gt.get(stem)
    if not e:
        continue
    vb = [p['vbox'] for p in e['persons']]
    kept = [('d', *r['box_img'], r) for r in rr
            if not any(_ioa(list(r['box_img']), ig) > IGNORE_IOA for ig in e['ignores'])]
    for gi, (d, j) in match_boxes(vb, kept, 0.5).items():
        r = d[5]
        bc = (r['v'] or {}).get('box_correction')
        if bc is None or not merge(r)['keep']:
            continue
        cw, ch = r['crop_wh']; ox, oy = r['crop_origin']; s = r.get('crop_scale', 1.0)
        a, b, c, dd = bc                                   # yxyx_1000 -> crop px
        cb = [b * cw / 1000, a * ch / 1000, dd * cw / 1000, c * ch / 1000]
        img_box = [cb[0] / s + ox, cb[1] / s + oy, cb[2] / s + ox, cb[3] / s + oy]
        cases.append(dict(stem=stem, r=r, gtb=vb[gi], sam_iou=j,
                          new_iou=iou(img_box, vb[gi]), crop_box=cb,
                          agree=iou(cb, r['sam_box_crop'])))

for c in cases:
    c['delta'] = c['new_iou'] - c['sam_iou']
cases.sort(key=lambda c: -c['delta'])

def render(c):
    li = int(c['r']['id'].rpartition('_')[2])
    p = IMGS / f"{c['stem']}.jpg"
    if not p.exists():
        return None
    dets, parts = load_raw_dets(RAW, c['stem'])
    with Image.open(p) as im:
        im = im.convert('RGB')
        cls, poly, box = dets[li]
        crop, origin, sam_box_crop, scale = build_crop(
            im, box, poly, True, 'outline', 320, parts=parts[li])
    ox, oy = origin
    d = ImageDraw.Draw(crop)
    g = c['gtb']
    d.rectangle([(g[0]-ox)*scale, (g[1]-oy)*scale, (g[2]-ox)*scale, (g[3]-oy)*scale],
                outline=GT_C, width=4)
    d.rectangle(sam_box_crop, outline=SAM_C, width=2)
    d.rectangle(c['crop_box'], outline=LITE_C, width=2)
    if max(crop.size) > 330:
        s2 = 330 / max(crop.size)
        crop = crop.resize((max(1, int(crop.width*s2)), max(1, int(crop.height*s2))))
    return _b64_jpeg(crop)

css = """body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;margin:24px;max-width:1240px}
h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:36px}
.c{display:inline-block;vertical-align:top;background:#fff;border:1px solid #ddd;
   border-radius:6px;padding:8px;margin:6px;max-width:350px}
.c img{max-width:330px}.c p{font-size:12px;margin:6px 0 0}
.win{color:#0a0;font-weight:700}.loss{color:#d00;font-weight:700}
.hd{background:#eef4ff;border-left:4px solid #2858dc;padding:10px 14px;font-size:13.5px}"""
doc = [f"<title>Box corrections: Lite vs SAM3 vs GT</title><style>{css}</style>",
       "<h1>Box corrections — where Lite helps, ties, and hurts</h1>",
       "<div class='hd'><b style='color:#00a000'>green = CrowdHuman ground truth</b> · "
       "<b style='color:#2858dc'>blue = SAM3 (what the pipeline emits today)</b> · "
       "<b style='color:#e08000'>orange = Lite's proposed correction</b><br>"
       f"{len(cases)} GT-matched detections where Lite offered a correction, parsed as "
       "yxyx_1000. 'delta' = Lite's IoU-vs-GT minus SAM3's. Sorted best → worst.</div>"]

buckets = [("Best cases — Lite clearly better (delta > +0.10)", lambda d: d > 0.10),
           ("Moderate wins (+0.02 to +0.10)", lambda d: 0.02 < d <= 0.10),
           ("Ties / negligible (-0.02 to +0.02)", lambda d: -0.02 <= d <= 0.02),
           ("Moderate losses (-0.10 to -0.02)", lambda d: -0.10 <= d < -0.02),
           ("Worst cases — Lite clearly worse (delta < -0.10)", lambda d: d < -0.10)]
for title, sel in buckets:
    sub = [c for c in cases if sel(c['delta'])]
    doc.append(f"<h2>{title} — {len(sub)} of {len(cases)}</h2>")
    if not sub:
        doc.append("<p>none</p>")
    for c in sub[:16]:
        b64 = render(c)
        if not b64:
            continue
        v = c['r']['v'] or {}
        cls_ = 'win' if c['delta'] > 0 else 'loss'
        doc.append(
            f"<div class='c'><img src='data:image/jpeg;base64,{b64}'>"
            f"<p><b>{HT.escape(c['r']['id'])}</b><br>"
            f"SAM3 IoU {c['sam_iou']:.2f} → Lite <b>{c['new_iou']:.2f}</b> "
            f"(<span class='{cls_}'>{c['delta']:+.2f}</span>)<br>"
            f"agreement with SAM3 box: {c['agree']:.2f}<br>"
            f"occl {v.get('occlusion')} {v.get('occlusion_percent')}% · "
            f"hq {v.get('highlight_quality')} · px_h {c['r'].get('person_px_height')}</p></div>")

out = Path('/workspace/exp10/box_corrections_gallery.html')
out.write_text("\n".join(doc))
print(f"{len(cases)} cases -> {out}")
for title, sel in buckets:
    print(f"  {title:52s} {sum(1 for c in cases if sel(c['delta']))}")