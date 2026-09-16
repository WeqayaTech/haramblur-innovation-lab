import json, os, base64, random, io
from PIL import Image, ImageDraw

def classify(v):
    g = v.get("gender"); a = v.get("age_group")
    if g in ("man", "woman") and a == "child":
        return 2
    if g == "woman":
        return 0
    if g == "man":
        return 1
    if a == "child":
        return 2
    return None

IMGDIRS = {"oiv7_train": "/workspace/open-images-v7/images/train",
           "oiv7_val": "/workspace/open-images-v7/images/val"}
CLS = {0: ("Woman", (255, 0, 200)), 1: ("Man", (30, 100, 255)),
       2: ("Child", (255, 150, 0)), 3: ("Unknown", (150, 150, 150))}
NEWCOL = (0, 200, 0)

def tag(d, xy, text, color):
    x, y = xy
    w = 6 * len(text) + 8
    y = max(0, y - 13)
    d.rectangle([x, y, x + w, y + 13], fill=color)
    d.text((x + 4, y + 1), text, fill=(255, 255, 255))

by_stem = {"oiv7_train": {}, "oiv7_val": {}}
for split in ["oiv7_train", "oiv7_val"]:
    for line in open("/workspace/spotlight/run/%s/verdicts_batch.jsonl" % split):
        try:
            r = json.loads(line)
        except Exception:
            continue
        v = r.get("v") or {}
        if v.get("verdict") != "not_person":
            continue
        c = classify(v)
        if c is None:
            continue
        r["_new_class"] = c
        by_stem[split].setdefault(r["image_stem"], []).append(r)

rng = random.Random(11)
train_stems = sorted(by_stem["oiv7_train"].keys())
val_stems = sorted(by_stem["oiv7_val"].keys())
sample_train = rng.sample(train_stems, min(52, len(train_stems)))
sample_val = val_stems

def render(stem, split):
    dets = by_stem[split][stem]
    img_name = dets[0]["image"]
    imgdir = IMGDIRS[split]
    lbldir = "/workspace/spotlight/run/%s/labels_unk3" % split
    im = Image.open(os.path.join(imgdir, img_name)).convert("RGB")
    W, H = im.size
    scale = 640.0 / max(W, H)
    if scale < 1:
        im = im.resize((int(W * scale), int(H * scale)))
    else:
        scale = 1.0
    d = ImageDraw.Draw(im)
    n_existing = 0
    lp = os.path.join(lbldir, stem + ".txt")
    if os.path.exists(lp):
        for ln in open(lp):
            t = ln.split()
            if len(t) < 7:
                continue
            c = int(float(t[0]))
            xs = [float(v) for v in t[1::2]]
            ys = [float(v) for v in t[2::2]]
            x1, y1, x2, y2 = min(xs) * W * scale, min(ys) * H * scale, max(xs) * W * scale, max(ys) * H * scale
            name, col = CLS.get(c, ("c%d" % c, (0, 0, 0)))
            d.rectangle([x1, y1, x2, y2], outline=col, width=2)
            tag(d, (x1, y1), name, col)
            n_existing += 1
    notes = []
    for r in dets:
        x1, y1, x2, y2 = [c * scale for c in r["box"]]
        v = r["v"]
        newc = r["_new_class"]
        name, col = CLS[newc]
        d.rectangle([x1, y1, x2, y2], outline=NEWCOL, width=4)
        tag(d, (x1, max(13, y1)), "NEW: " + name, NEWCOL)
        ea = v.get("estimated_age")
        bits = [v.get("gender"), v.get("age_group"),
                ("age %s" % int(ea)) if isinstance(ea, (int, float)) and ea > 0 else None,
                v.get("head_covering") if v.get("head_covering") not in (None, "none") else None,
                v.get("apparent_race") if v.get("apparent_race") not in (None, "unknown") else None]
        bits = [b for b in bits if b]
        hexcol = "#%02x%02x%02x" % col
        notes.append('det %s -> <b style="color:%s">%s</b>: %s' % (r["det_index"], hexcol, name, ", ".join(bits)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode(), n_existing, notes

cards = []
def build_cards(stems, split, label):
    out = []
    for stem in stems:
        try:
            b64, n_existing, notes = render(stem, split)
        except Exception as e:
            out.append("<!-- fail %s %s %s -->" % (split, stem, e))
            continue
        nl = "<br>".join(notes)
        out.append(
            '<div class="c"><div class="badge">%s</div><img src="data:image/jpeg;base64,%s">'
            '<div class="cap"><span class="m">%s &middot; %d existing labels (solid)</span><br>%s</div></div>'
            % (label, b64, stem, n_existing, nl)
        )
    return out

cards += build_cards(sample_train, "oiv7_train", "TRAIN")
cards += build_cards(sample_val, "oiv7_val", "VAL (all 62)")

TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>relabeled not_person to Woman/Man/Child</title><style>
body{font-family:-apple-system,Segoe UI,sans-serif;margin:20px;background:#fafafa;color:#222}
h1{font-size:21px}
.legend{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 14px;font-size:13.5px;max-width:960px;margin-bottom:14px}
.sw{display:inline-block;width:13px;height:13px;border-radius:2px;vertical-align:-2px;margin-right:4px}
.grid{display:flex;flex-wrap:wrap;gap:14px}
.c{width:660px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;position:relative}
.c img{max-width:100%;display:block;border-radius:4px}
.cap{font-size:12.5px;margin-top:6px;line-height:1.6}
.m{color:#888;font-size:11px}
.badge{position:absolute;top:14px;right:14px;background:#333;color:#fff;font-size:10.5px;padding:2px 7px;border-radius:10px;z-index:2}
</style></head><body>
<h1>Updated definition: not_person detections promoted to real labels</h1>
<div class="legend">
<b>Existing training labels (thin outline, already in labels_unk3):</b>
<span class="sw" style="background:rgb(255,0,200)"></span>Woman
<span class="sw" style="background:rgb(30,100,255)"></span>Man
<span class="sw" style="background:rgb(255,150,0)"></span>Child
<span class="sw" style="background:rgb(150,150,150)"></span>Unknown (class 3)
<br><b><span class="sw" style="background:rgb(0,200,0)"></span>Thick green = NEWLY PROMOTED label</b>
(was deleted as not_person; now added at the class shown in the tag, using Gemini gender+age:
age=child overrides gender and becomes Child). Caption lists every promoted detection's full cue set.
<br>Train: random 52 of 9,273 affected images (seed 11), 12,836 total promotable detections (2,812 Woman /
9,810 Man / 214 Child). Val: ALL 62 affected images, 82 detections (18 Woman / 61 Man / 3 Child).
Source: verdicts_batch.jsonl (verdict fields) plus raw/&lt;split&gt;/&lt;stem&gt;.txt (polygon geometry, class swapped).
</div>
<div class="grid">__CARDS__</div></body></html>"""

html = TEMPLATE.replace("__CARDS__", "\n".join(cards))
out_path = "/workspace/tmp_notperson_crops/relabeled_verification_gallery.html"
with open(out_path, "w") as f:
    f.write(html)
print("written", out_path, os.path.getsize(out_path) // 1024, "KB, cards:",
      len([c for c in cards if not c.startswith("<!--")]))
