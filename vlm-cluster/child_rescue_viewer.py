#!/usr/bin/env python3
"""
Eyeball the children the --child-by-age rule brings back into training.

One self-contained HTML file, one person per screen, left/right arrows to move.
For each rescued detection it shows the EXACT crop Gemini judged -- same 25%
padding, same mask outline, same 320px upscale, rendered by the production
build_crop() rather than a lookalike -- next to the full frame with the same
person outlined, because a 25px-tall person is not judgeable on its own.

Why this needs looking at: every one of these detections is
`confidence: low`, Gemini abstained on both gender and age_group, and SAM3
independently agreed "Child" on only 40.9% of them. Marking an adult as Child
is the one error direction that lets an adult through the blur unlooked-at, so
the sample is ordered to put the cases SAM3 DISAGREED with first.

Press x on anything that is not a child; the page keeps the list and a Copy
button, so a verification pass comes back as a list of stems rather than a
memory of "some looked wrong".

    python3 child_rescue_viewer.py \
        --dets   /workspace/spotlight/child_rescue_train_dets.jsonl \
        --images /workspace/open-images-v7/images/train \
        --raw    /workspace/spotlight/raw/oiv7_train \
        --out    /workspace/spotlight/child_rescue_view.html --limit 60

    python3 child_rescue_viewer.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from spotlight_run import (CLASS_NAME, MIN_CROP_SIDE, PAD, build_crop,
                            child_age_only, load_raw)

THUMB_MAX = 420            # display cap; the crop Gemini saw may be smaller
FRAME_MAX = 420


def pick(rows, limit, per_age_min=8, seed=0):
    """Stratified sample: every distinct estimated_age gets up to `per_age_min`
    before the remainder goes to the biggest bucket. A flat random sample of
    these would be ~91% the single value 10 and would never show the ages that
    actually look like a young child's."""
    by_age = defaultdict(list)
    for r in rows:
        by_age[r["v"]["estimated_age"]].append(r)
    rng = random.Random(seed)
    for v in by_age.values():
        rng.shuffle(v)
    out, leftovers = [], []
    for age in sorted(by_age):
        bucket = by_age[age]
        out += bucket[:per_age_min]
        leftovers += bucket[per_age_min:]
    if limit and len(out) > limit:
        out = out[:limit]
    # fill from the biggest remaining bucket(s), keeping the age mix visible
    leftovers.sort(key=lambda r: -len(by_age[r["v"]["estimated_age"]]))
    while limit and len(out) < limit and leftovers:
        out.append(leftovers.pop(0))
    # SAM3-disagreed first: those are the ones that could be adults
    out.sort(key=lambda r: (r["sam_class_id"] == 2, r["v"]["estimated_age"]))
    return out


def plain_crop(img: Image.Image, box):
    """The same padding and upscale as build_crop, with NOTHING drawn on it.
    EXP-2026-10 measured the green mask tint obscuring small/distant people
    badly enough to fail the pilot, so a verification pass needs to see the
    pixels without our own annotation sitting on top of the face."""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    c = img.crop((int(max(0, x1 - pw)), int(max(0, y1 - ph)),
                  int(min(img.width, x2 + pw)),
                  int(min(img.height, y2 + ph)))).convert("RGB")
    if max(c.size) < MIN_CROP_SIDE:
        s = MIN_CROP_SIDE / max(c.size)
        c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))),
                     Image.LANCZOS)
    return c


def _b64(im, cap):
    im = im.copy()
    if max(im.size) > cap:
        s = cap / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def render(rows, images_dir: Path, raw_dir: Path):
    """-> list of card dicts. Skips (and reports) anything whose image or
    sidecar is missing rather than dying halfway through a long render."""
    cards, skipped = [], []
    for n, r in enumerate(rows, 1):
        stem, di = r["image_stem"], r["det_index"]
        ip = images_dir / r.get("image", stem + ".jpg")
        if not ip.exists():
            hits = list(images_dir.rglob(stem + ".*"))
            if not hits:
                skipped.append((stem, "image not found"))
                continue
            ip = hits[0]
        rec = load_raw(raw_dir, stem)
        if rec is None:
            skipped.append((stem, "no raw sidecar"))
            continue
        dets = rec.get("detections", [])
        if di >= len(dets):
            skipped.append((stem, f"det_index {di} >= {len(dets)}"))
            continue
        det = dets[di]
        try:
            img = Image.open(ip)
            img.load()
        except Exception as e:                        # noqa: BLE001
            skipped.append((stem, f"unreadable image: {e}"))
            continue
        parts = [[(x, y) for x, y in p] for p in det.get("parts", [])]
        crop, _ = build_crop(img, det["box"], parts)
        clean = plain_crop(img, det["box"])
        frame = img.convert("RGB").copy()
        d = ImageDraw.Draw(frame)
        w = max(2, int(max(frame.size) / 250))
        d.rectangle([int(v) for v in det["box"]], outline=(255, 40, 40), width=w)
        v = r["v"]
        cards.append({
            "i": n, "stem": stem, "det": di,
            "crop": _b64(crop, THUMB_MAX), "clean": _b64(clean, THUMB_MAX),
            "frame": _b64(frame, FRAME_MAX),
            "age": v.get("estimated_age"), "conf": v.get("confidence"),
            "verdict": v.get("verdict"), "hq": v.get("highlight_quality"),
            "sam": CLASS_NAME.get(r["sam_class_id"], "?"),
            "sam_agrees": r["sam_class_id"] == 2,
            "px": r.get("person_px_height"),
            "race": v.get("apparent_race"), "head": v.get("head_covering"),
            "parts": ", ".join(v.get("exposed_body_parts") or []),
            "wh": r.get("img_wh"),
        })
        if n % 20 == 0:
            print(f"  rendered {len(cards)}/{len(rows)}", flush=True)
    return cards, skipped


HTML = """<!doctype html>
<meta charset="utf-8"><title>child-by-age rescues — verify</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#15171a;color:#e8e8e8;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{padding:10px 16px;background:#0f1113;border-bottom:1px solid #2a2f35;
        display:flex;gap:16px;align-items:center;flex-wrap:wrap;
        position:sticky;top:0}
 header b{font-size:15px}
 .hint{color:#8b949e;font-size:12px}
 main{padding:18px;display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap}
 .pane{text-align:center}
 .pane img{max-width:420px;border:1px solid #2a2f35;border-radius:4px;
           background:#000;display:block}
 .pane .lbl{color:#8b949e;font-size:12px;margin:6px 0 0}
 dl{display:grid;grid-template-columns:auto auto;gap:2px 12px;margin:0;
    align-content:start}
 dt{color:#8b949e}
 dd{margin:0;font-variant-numeric:tabular-nums}
 .tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px}
 .ok{background:#173a24;color:#7ee2a8}
 .warn{background:#3d2a12;color:#f3c078}
 .bad{background:#3d1a1a;color:#f39a9a}
 #marked{padding:0 16px 22px;color:#f39a9a;font-size:12px;
         word-break:break-all;max-width:1000px}
 button{background:#22272e;color:#e8e8e8;border:1px solid #39404a;
        border-radius:5px;padding:4px 10px;cursor:pointer;font-size:12px}
 kbd{background:#22272e;border:1px solid #39404a;border-radius:3px;
     padding:0 4px;font-size:11px}
</style>
<header>
  <b id="pos"></b>
  <span id="flag"></span>
  <span class="hint"><kbd>&larr;</kbd><kbd>&rarr;</kbd> move &nbsp;
    <kbd>x</kbd> mark "not a child" &nbsp; <kbd>f</kbd> full frame only</span>
  <button onclick="cp()">Copy marked list</button>
  <span class="hint" id="tally"></span>
</header>
<main>
  <div class="pane"><img id="clean"><p class="lbl">same crop, nothing drawn
    on it</p></div>
  <div class="pane"><img id="crop"><p class="lbl">the crop Gemini judged
    (25% padding, mask outlined, upscaled to 320px)</p></div>
  <div class="pane"><img id="frame"><p class="lbl">full frame, SAM3 box in
    red</p></div>
  <dl id="meta"></dl>
</main>
<div id="marked"></div>
<script>
const C = __CARDS__;
let i = 0, marked = new Set(), big = false;
const el = id => document.getElementById(id);
function draw(){
  const c = C[i];
  el('crop').src  = 'data:image/jpeg;base64,' + c.crop;
  el('clean').src = 'data:image/jpeg;base64,' + c.clean;
  el('frame').src = 'data:image/jpeg;base64,' + c.frame;
  el('crop').parentElement.style.display = big ? 'none' : 'block';
  el('clean').parentElement.style.display = big ? 'none' : 'block';
  el('pos').textContent = (i+1) + ' / ' + C.length;
  el('flag').innerHTML = marked.has(c.stem+':'+c.det)
    ? '<span class="tag bad">marked NOT a child</span>' : '';
  const t = c.sam_agrees
    ? '<span class="tag ok">SAM3 also said Child</span>'
    : '<span class="tag warn">SAM3 said ' + c.sam + '</span>';
  el('meta').innerHTML =
    row('estimated_age', c.age + '  <span class="tag warn">' + c.conf
        + ' confidence</span>')
  + row('SAM3 called it', t)
  + row('verdict', c.verdict)
  + row('highlight', c.hq)
  + row('person height', c.px + ' px' + (c.wh ? '  (image '+c.wh[0]+'x'+c.wh[1]+')' : ''))
  + row('apparent_race', c.race)
  + row('head_covering', c.head)
  + row('exposed', c.parts || '—')
  + row('image', c.stem + '  det ' + c.det);
  el('tally').textContent = marked.size + ' marked';
  el('marked').textContent = marked.size
    ? 'not a child: ' + [...marked].join('  ') : '';
}
const row = (k,v) => '<dt>'+k+'</dt><dd>'+v+'</dd>';
function cp(){
  navigator.clipboard.writeText([...marked].join('\\n'));
}
addEventListener('keydown', e => {
  if(e.key === 'ArrowRight' || e.key === 'j'){ i = Math.min(C.length-1, i+1); }
  else if(e.key === 'ArrowLeft' || e.key === 'k'){ i = Math.max(0, i-1); }
  else if(e.key === 'x'){
    const k = C[i].stem + ':' + C[i].det;
    marked.has(k) ? marked.delete(k) : marked.add(k);
  }
  else if(e.key === 'f'){ big = !big; }
  else return;
  e.preventDefault(); draw();
});
addEventListener('click', e => { if(e.target.tagName === 'IMG'){
  i = Math.min(C.length-1, i+1); draw(); }});
draw();
</script>
"""


def write_html(cards, out: Path, note=""):
    body = HTML.replace("__CARDS__", json.dumps(cards))
    if note:
        body = body.replace('<span class="hint" id="tally"></span>',
                            '<span class="hint" id="tally"></span>'
                            f'<span class="hint">{html.escape(note)}</span>')
    out.write_text(body)
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({len(cards)} people, {mb:.1f} MB)")
    if mb > 25:
        print("  NOTE: big for a browser tab -- use --limit for a quicker pass")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        imgs, raw = td / "img", td / "raw"
        imgs.mkdir(); raw.mkdir()
        rows = []
        for k, age in enumerate([1.0, 5.0, 10.0, 10.0]):
            stem = f"s{k}"
            Image.new("RGB", (300, 240), (90, 110, 130)).save(imgs / f"{stem}.jpg")
            (raw / f"{stem}.json").write_text(json.dumps({
                "image": f"{stem}.jpg", "width": 300, "height": 240,
                "detections": [{"cls": 1, "conf": 0.9, "box": [30, 20, 90, 200],
                                "parts": [[[30, 20], [90, 20], [90, 200], [30, 200]]]}]}))
            rows.append({
                "image_stem": stem, "det_index": 0, "image": f"{stem}.jpg",
                "sam_class_id": 2 if k == 0 else 1, "img_wh": [300, 240],
                "person_px_height": 180.0,
                "v": {"verdict": "real_person", "gender": "unknown",
                      "age_group": "unknown", "estimated_age": age,
                      "highlight_quality": "good", "confidence": "low",
                      "apparent_race": "unknown", "head_covering": "none",
                      "exposed_body_parts": ["face"]}})
        assert all(child_age_only(r["v"]) for r in rows)
        # stratification: 3 distinct ages must all appear in a 3-wide sample
        s = pick(rows, limit=3, per_age_min=1)
        assert len({r["v"]["estimated_age"] for r in s}) == 3, s
        # SAM3-disagreed cases come first
        assert s[0]["sam_class_id"] != 2 and s[-1]["sam_class_id"] == 2, s
        cards, skipped = render(pick(rows, limit=4, per_age_min=2), imgs, raw)
        assert len(cards) == 4 and not skipped, (len(cards), skipped)
        assert cards[0]["crop"] and cards[0]["frame"] and cards[0]["clean"]
        # the clean pane must NOT carry the outline colour the annotated one does
        assert cards[0]["clean"] != cards[0]["crop"]
        assert sum(c["sam_agrees"] for c in cards) == 1
        # a missing sidecar is skipped, not fatal
        (raw / "s0.json").unlink()
        c2, sk2 = render(rows, imgs, raw)
        assert len(c2) == 3 and sk2 and sk2[0][1] == "no raw sidecar", (c2, sk2)
        out = td / "v.html"
        write_html(cards, out, note="selftest")
        t = out.read_text()
        assert "__CARDS__" not in t and '"crop"' in t
        assert "ArrowRight" in t and "ArrowLeft" in t
        assert "id=\"clean\"" in t
        assert t.count("data:image/jpeg;base64") >= 1
    print("child_rescue_viewer.py self-tests passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", help="child_rescue_report.py --dump output")
    ap.add_argument("--images"); ap.add_argument("--raw"); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=60, help="0 = every rescue")
    ap.add_argument("--per-age-min", type=int, default=8)
    ap.add_argument("--only-sam-disagree", action="store_true",
                    help="only the cases SAM3 did NOT call Child")
    ap.add_argument("--only-sam-child", action="store_true",
                    help="only the cases SAM3 ALSO called Child -- i.e. exactly "
                         "what --child-by-age --child-require-sam emits")
    ap.add_argument("--min-px", type=int, default=0,
                    help="skip people shorter than this (person_px_height)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.dets and a.images and a.raw and a.out):
        ap.error("--dets --images --raw --out are required")
    rows = [json.loads(l) for l in open(a.dets) if l.strip()]
    rows = [r for r in rows if child_age_only(r["v"])]
    print(f"{len(rows):,} rescued detections in {a.dets}")
    if a.only_sam_disagree and a.only_sam_child:
        ap.error("--only-sam-disagree and --only-sam-child are opposites")
    if a.only_sam_disagree:
        rows = [r for r in rows if r["sam_class_id"] != 2]
        print(f"  {len(rows):,} of them SAM3 did NOT call Child")
    if a.only_sam_child:
        rows = [r for r in rows if r["sam_class_id"] == 2]
        print(f"  {len(rows):,} of them SAM3 ALSO called Child")
    if a.min_px:
        rows = [r for r in rows
                if isinstance(r.get("person_px_height"), (int, float))
                and r["person_px_height"] >= a.min_px]
        print(f"  {len(rows):,} of those are >= {a.min_px}px tall")
    sel = pick(rows, a.limit, a.per_age_min, a.seed)
    print(f"showing {len(sel):,} "
          f"(SAM3 disagreed on {sum(1 for r in sel if r['sam_class_id'] != 2)})")
    cards, skipped = render(sel, Path(a.images), Path(a.raw))
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:4]}")
    write_html(cards, Path(a.out),
               note=f"{len(rows):,} rescues total, {len(cards)} shown")


if __name__ == "__main__":
    main()
