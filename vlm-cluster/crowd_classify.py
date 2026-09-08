#!/usr/bin/env python3
"""Classify CrowdHuman GT persons with Gemini Flash-Lite (Spotlight prompt).

Detection here comes from CrowdHuman's HUMAN-ANNOTATED vboxes, not SAM3:
in crowds SAM3 silently misses ~1-in-4 people (EXP-2026-03/09), and an
unlabeled person in a training image teaches the detector that region is
background -- the exact error direction this product cannot afford. The GT
boxes are exhaustive, so only classification is delegated to the VLM.

The target person is indicated by the vbox rectangle outline: build_crop()
is called with parts=[] and its box-only fallback draws the same two-tone
outline the validated mask pipeline uses -- the crop path is reused
byte-for-byte, no new rendering code.

    # eyeball crops first, no API calls, no spend:
    python3 crowd_classify.py --gt .../gt_boxes.jsonl --images .../Images \
        --out .../verdicts_train.jsonl --save-crops /tmp/crops --limit 5

    # pilot (50 images ~ 1,100 persons ~ $0.75 live):
    python3 crowd_classify.py --gt .../gt_boxes.jsonl --images .../Images \
        --out .../verdicts_train.jsonl --limit 50

    python3 crowd_classify.py --selftest    # no network, no data

Resumable: (stem, box_index) pairs already in --out are skipped, output is
appended and flushed per verdict.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image

from spotlight_run import PROMPT, build_crop, parse_verdict

MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 400


def done_keys(out: Path):
    keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((r["stem"], r["box_index"]))
            except Exception:
                pass
    return keys


def vbox_to_xyxy(vbox):
    x, y, w, h = vbox
    return [x, y, x + w, y + h]


def load_parts(masks_dir, stem, box_index):
    """Per-person mask polygons from crowd_sam_masks.py sidecars. Returns []
    (box-outline fallback) when no sidecar exists -- but a masks run that is
    merely incomplete should FAIL LOUDLY at the caller, not silently degrade
    to the box outline the pilot already proved mislabels neighbors."""
    if not masks_dir:
        return []
    f = Path(masks_dir) / f"{stem}.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    for p in rec.get("persons", []):
        if p["box_index"] == box_index:
            return [[(x, y) for x, y in part] for part in p.get("parts", [])]
    return None


def run(gt_path: Path, images: Path, out: Path, limit, save_crops,
        masks_dir=None):
    skip = done_keys(out)
    if skip:
        print(f"[resume] {len(skip):,} verdicts already in {out.name}")
    client = None
    if not save_crops:
        from google import genai
        client = genai.Client()

    n_img = n_call = n_fail = 0
    t0 = time.time()
    with open(out, "a") as fh:
        for line in gt_path.read_text().splitlines():
            rec = json.loads(line)
            stem, boxes = rec["stem"], rec["boxes"]
            if limit and n_img >= limit:
                break
            n_img += 1
            ip = images / f"{stem}.jpg"
            if not ip.exists():
                print(f"  !! image missing: {ip.name}")
                continue
            img = Image.open(ip).convert("RGB")
            for i, b in enumerate(boxes):
                if (stem, i) in skip:
                    continue
                box = vbox_to_xyxy(b["vbox"])
                parts = load_parts(masks_dir, stem, i)
                if parts is None:
                    print(f"  !! no mask for {stem}#{i} — run crowd_sam_masks "
                          f"on this image first; skipping (box outline would "
                          f"mislabel neighbors)")
                    n_fail += 1
                    continue
                crop, _ = build_crop(img, box, parts)
                if save_crops:
                    d = Path(save_crops); d.mkdir(parents=True, exist_ok=True)
                    crop.save(d / f"{stem}_{i}.jpg", quality=90)
                    n_call += 1
                    continue
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=90)
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=[{"role": "user", "parts": [
                            {"text": PROMPT},
                            {"inline_data": {"mime_type": "image/jpeg",
                                             "data": buf.getvalue()}}]}],
                        config={"max_output_tokens": MAX_TOKENS},
                    )
                    text = resp.text or ""
                except Exception as e:
                    print(f"  !! API error {stem}#{i}: {e}")
                    n_fail += 1
                    continue
                v = parse_verdict(text)
                fh.write(json.dumps({
                    "stem": stem, "box_index": i, "vbox": b["vbox"],
                    "fbox": b.get("fbox"), "parse_ok": v is not None,
                    "v": v, "raw_text": text,
                    "source": ("crowd_gtbox_mask_live" if masks_dir
                               else "crowd_gtbox_live"),
                }) + "\n")
                fh.flush()
                n_call += 1
                if n_call % 50 == 0:
                    rate = n_call / max(1e-9, time.time() - t0)
                    print(f"  {n_call:,} verdicts ({rate:.1f}/s) "
                          f"img {n_img}, fails {n_fail}")
    what = "crops saved" if save_crops else "verdicts"
    print(f"done: {n_img} images, {n_call:,} {what}, {n_fail} API failures")


def _selftest():
    import tempfile
    from PIL import ImageDraw
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "Images").mkdir()
        img = Image.new("RGB", (640, 480), (90, 120, 90))
        ImageDraw.Draw(img).rectangle([200, 100, 320, 400], fill=(200, 180, 160))
        img.save(td / "Images" / "a,b1.jpg")
        gt = td / "gt.jsonl"
        gt.write_text(json.dumps(
            {"stem": "a,b1", "boxes": [{"vbox": [200, 100, 120, 300],
                                        "fbox": [195, 90, 130, 320]}]}) + "\n")
        run(gt, td / "Images", td / "v.jsonl", limit=None,
            save_crops=str(td / "crops"))
        crops = list((td / "crops").glob("*.jpg"))
        assert len(crops) == 1, crops
        c = Image.open(crops[0])
        assert max(c.size) >= 320          # upscale floor applied
        # resume logic: pre-seed a verdict, ensure it is skipped
        (td / "v.jsonl").write_text(json.dumps(
            {"stem": "a,b1", "box_index": 0, "v": None}) + "\n")
        assert done_keys(td / "v.jsonl") == {("a,b1", 0)}
        # verdict parsing round-trip via spotlight_run
        v = parse_verdict('{"verdict": "real_person", "gender": "woman", '
                          '"age_group": "adult", "estimated_age": 30}')
        assert v and v["verdict"] == "real_person"
    print("crowd_classify.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt"); ap.add_argument("--images"); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N images (pilot)")
    ap.add_argument("--save-crops", default=None,
                    help="write crops to this dir and make NO API calls")
    ap.add_argument("--masks", default=None,
                    help="crowd_sam_masks.py output dir; outlines the exact "
                         "person mask instead of the vbox rectangle")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.gt and a.images and a.out):
        ap.error("--gt --images --out are required")
    run(Path(a.gt), Path(a.images), Path(a.out), a.limit, a.save_crops,
        a.masks)


if __name__ == "__main__":
    main()
