#!/usr/bin/env python3
"""Stratified blind sample of Spotlight verdicts -> cards.json (base64 crops).

Renders each sampled detection exactly as Gemini saw it (spotlight_run.build_crop:
25% pad, green two-tone mask outline, upscaled to min side 320) plus a context
view of the whole image with the box drawn, so a human can judge "could anyone
tell?" both from the crop and from the scene.

Strata cover the abstentions AND matched controls where the model committed on
equally hard crops -- without the controls the gallery can only find
over-abstention, never under-abstention.
"""
import json, random, sys, io, base64, os
sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from PIL import Image, ImageDraw
from spotlight_run import build_crop

VERDICTS = "/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl"
RAW      = "/workspace/spotlight/raw/oiv7_train"
IMGS     = "/workspace/open-images-v7/images/train"
OUT      = "/workspace/exp_unk/cards.json"
SEED     = 20260817

# stratum -> (target_n, predicate on (rec, v, h, face, g, a))
def face_of(v):
    ex = v.get("exposed_body_parts") or []
    if isinstance(ex, str): ex = [ex]
    return "face" in [str(e).lower() for e in ex]

STRATA = {
 "A_unk_noface_tiny":    (20, lambda h,f,g,a: g=="unknown" and not f and h is not None and h < 64),
 "B_unk_noface_mid":     (20, lambda h,f,g,a: g=="unknown" and not f and h is not None and 64 <= h < 192),
 "C_unk_noface_large":   (20, lambda h,f,g,a: g=="unknown" and not f and h is not None and h >= 192),
 "D_unk_FACEVISIBLE":    (25, lambda h,f,g,a: g=="unknown" and f),
 "E_unk_very_large":     (20, lambda h,f,g,a: g=="unknown" and h is not None and h >= 320),
 "F_unk_gender_known_age":(15,lambda h,f,g,a: g=="unknown" and a in ("adult","child")),
 "G_committed_tiny":     (15, lambda h,f,g,a: g in ("man","woman") and h is not None and h < 64),
 "H_committed_noface":   (15, lambda h,f,g,a: g in ("man","woman") and not f and h is not None and h >= 64),
}

rng = random.Random(SEED)
res = {k: [] for k in STRATA}
seen = {k: 0 for k in STRATA}

with open(VERDICTS) as fh:
    for line in fh:
        try: r = json.loads(line)
        except ValueError: continue
        v = r.get("v") or {}
        if str(v.get("verdict")) != "real_person": continue
        g, a = str(v.get("gender")), str(v.get("age_group"))
        h, f = r.get("person_px_height"), face_of(v)
        for k, (n, pred) in STRATA.items():
            try:
                if not pred(h, f, g, a): continue
            except TypeError:
                continue
            seen[k] += 1                      # reservoir sampling keeps it unbiased
            if len(res[k]) < n: res[k].append(r)
            else:
                j = rng.randrange(seen[k])
                if j < n: res[k][j] = r

print("population per stratum:", seen, flush=True)

def b64(img, maxside, q):
    im = img.copy()
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.LANCZOS)
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

cards, fails = [], 0
for k, recs in res.items():
    for r in recs:
        stem, di = r["image_stem"], r["det_index"]
        try:
            side = json.load(open(f"{RAW}/{stem}.json"))
            det = side["detections"][di]
            img = Image.open(f"{IMGS}/{stem}.jpg")
            img.load()
        except Exception as e:
            fails += 1; continue
        box = r["box"]
        box_match = max(abs(p-q_) for p, q_ in zip(det["box"], box)) < 1.5
        crop, _ = build_crop(img, box, det.get("parts"))
        ctx = img.convert("RGB").copy()
        d = ImageDraw.Draw(ctx)
        w = max(2, int(min(ctx.size)/180))
        d.rectangle(box, outline=(0, 220, 90), width=w)
        v = r["v"]
        cards.append({
            "stratum": k, "stem": stem, "det": di,
            "crop": b64(crop, 300, 74), "ctx": b64(ctx, 380, 68),
            "crop_wh": list(crop.size), "img_wh": r.get("img_wh"),
            "h": r.get("person_px_height"), "blur": r.get("blurriness"),
            "sam_conf": r.get("sam_conf"), "n_parts": r.get("n_parts"),
            "box_match": box_match,
            "gender": v.get("gender"), "age_group": v.get("age_group"),
            "est_age": v.get("estimated_age"), "conf": v.get("confidence"),
            "head": v.get("head_covering"), "parts": v.get("exposed_body_parts"),
            "race": v.get("apparent_race"), "hq": v.get("highlight_quality"),
        })
rng.shuffle(cards)
json.dump({"seed": SEED, "population": seen, "render_failures": fails, "cards": cards},
          open(OUT, "w"))
print(f"cards={len(cards)} fails={fails} bytes={os.path.getsize(OUT):,}")
