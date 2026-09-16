#!/usr/bin/env python3
"""3 random examples from each of the 4 mutually exclusive abstention buckets.

Bucket definitions are copied verbatim from pass2.py so the examples illustrate
exactly the populations the table counts -- seeded reservoir sampling, no
cherry-picking.
"""
import json, random, io, base64, os, sys
sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from PIL import Image, ImageDraw
from spotlight_run import build_crop

VERDICTS = "/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl"
RAW  = "/workspace/spotlight/raw/oiv7_train"
IMGS = "/workspace/open-images-v7/images/train"
OUT  = "/workspace/exp_unk/buckets.json"
SEED, PER = 20260818, 6      # draw 6, keep the first 3 that render

def bucket_of(h, face):
    if h is None: return None
    if h < 64:        return "tiny"          # under 64 px, whatever the face
    if not face:      return "noface"        # >=64 px, no face reported
    return "face_small" if h < 192 else "face_large"

rng = random.Random(SEED)
res, seen = {}, {}
with open(VERDICTS) as fh:
    for line in fh:
        try: r = json.loads(line)
        except ValueError: continue
        v = r.get("v") or {}
        if str(v.get("verdict")) not in ("real_person", "depiction"): continue
        if str(v.get("gender")) != "unknown": continue          # abstentions only
        ex = v.get("exposed_body_parts") or []
        if isinstance(ex, str): ex = [ex]
        face = "face" in [str(e).lower() for e in ex]
        b = bucket_of(r.get("person_px_height"), face)
        if b is None: continue
        seen[b] = seen.get(b, 0) + 1
        lst = res.setdefault(b, [])
        if len(lst) < PER: lst.append(r)
        else:
            j = rng.randrange(seen[b])
            if j < PER: lst[j] = r
print("populations:", seen, flush=True)

def b64(img, maxside, q):
    im = img.copy()
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.LANCZOS)
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

out = {}
for b, recs in res.items():
    keep = []
    for r in recs:
        if len(keep) >= 3: break
        stem, di = r["image_stem"], r["det_index"]
        try:
            side = json.load(open(f"{RAW}/{stem}.json"))
            det = side["detections"][di]
            img = Image.open(f"{IMGS}/{stem}.jpg"); img.load()
        except Exception:
            continue
        if max(abs(p-q_) for p, q_ in zip(det["box"], r["box"])) >= 1.5:
            continue
        crop, _ = build_crop(img, r["box"], det.get("parts"))
        ctx = img.convert("RGB").copy()
        d = ImageDraw.Draw(ctx)
        d.rectangle(r["box"], outline=(0,220,90), width=max(2,int(min(ctx.size)/180)))
        v = r["v"]
        keep.append({"stem": stem, "det": di, "crop": b64(crop,260,74), "ctx": b64(ctx,340,66),
                     "h": r.get("person_px_height"), "sam_conf": r.get("sam_conf"),
                     "img_wh": r.get("img_wh"), "n_parts": r.get("n_parts"),
                     "age_group": v.get("age_group"), "est_age": v.get("estimated_age"),
                     "conf": v.get("confidence"), "parts": v.get("exposed_body_parts"),
                     "head": v.get("head_covering")})
    out[b] = keep
    print(b, len(keep), flush=True)
json.dump({"seed": SEED, "populations": seen, "buckets": out}, open(OUT,"w"))
print("bytes", os.path.getsize(OUT))
