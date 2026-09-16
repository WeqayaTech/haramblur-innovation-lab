#!/usr/bin/env python3
"""
Dump Spotlight-style crops for every SAM3 detection in child_v2 so a
vision-capable agent can view each one directly and produce the verdict
JSON itself -- no Gemini API call. Reuses build_crop()/load_raw() from
spotlight_run.py so a crop here is byte-identical to what the API path
would have sent, and the resulting verdicts.jsonl can be merged with the
project's own validated merge()/emit() logic via
`spotlight_run.py --emit-only` (no reimplementation of that logic here).

Usage:
    python3 dump_crops.py                  # all images with a raw sidecar
    python3 dump_crops.py --limit 20        # first 20 only (smoke test)
    python3 dump_crops.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/autolabel_pipeline_v2")

IMAGES = Path("/workspace/datasets/child_v2/images_unique")
RAW = Path("/workspace/datasets/child_v2/labels_sam3_raw")
CROPS = Path("/workspace/datasets/child_v2/crops")
INDEX = Path("/workspace/datasets/child_v2/crops_index.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    from spotlight_run import build_crop, load_raw
    from PIL import Image

    if args.selftest:
        # tiny synthetic image + fake detection, no pod data needed
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        (tmp / "images").mkdir()
        (tmp / "raw").mkdir()
        im = Image.new("RGB", (200, 300), (120, 140, 160))
        im.save(tmp / "images" / "t.jpg")
        sidecar = {"image": "t.jpg", "width": 200, "height": 300,
                   "detections": [{"cls": 1, "conf": 0.9,
                                    "box": [20, 30, 150, 250], "parts": []}]}
        (tmp / "raw" / "t.json").write_text(json.dumps(sidecar))
        rec = load_raw(tmp / "raw", "t")
        assert rec is not None
        det = rec["detections"][0]
        crop, scale = build_crop(im, det["box"], det.get("parts", []))
        assert crop.size[0] > 0 and crop.size[1] > 0
        print(f"selftest OK — build_crop produced a {crop.size} crop")
        return

    CROPS.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in IMAGES.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        imgs = imgs[:args.limit]

    n_crops, n_imgs = 0, 0
    mode = "a" if INDEX.exists() else "w"
    already = set()
    if INDEX.exists():
        for line in INDEX.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                already.add((r["image_stem"], r["det_index"]))

    with INDEX.open(mode) as idx:
        for p in imgs:
            rec = load_raw(RAW, p.stem)
            if rec is None:
                continue
            n_imgs += 1
            img = None
            for i, det in enumerate(rec.get("detections", [])):
                if (p.stem, i) in already:
                    continue
                if img is None:
                    img = Image.open(p).convert("RGB")
                parts = [[(x, y) for x, y in part] for part in det.get("parts", [])]
                crop, scale = build_crop(img, det["box"], parts)
                crop_path = CROPS / f"{p.stem}__det{i}.jpg"
                crop.save(crop_path, quality=90)
                idx.write(json.dumps({
                    "image_stem": p.stem, "det_index": i, "image": p.name,
                    "img_wh": [rec.get("width", img.width),
                               rec.get("height", img.height)],
                    "sam_class_id": det["cls"], "sam_conf": det.get("conf"),
                    "box": det["box"], "n_parts": len(parts),
                    "crop_path": str(crop_path),
                }) + "\n")
                n_crops += 1
            if img is not None:
                img.close()
    print(f"wrote {n_crops} new crops from {n_imgs} images with sidecars "
          f"-> {CROPS} (index: {INDEX})")


if __name__ == "__main__":
    main()
