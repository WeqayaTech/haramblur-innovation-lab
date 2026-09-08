#!/usr/bin/env python3
"""Box-prompted person masks for CrowdHuman GT boxes (SAM-ViT-huge).

The crowd pilot showed a plain vbox outline fails in dense scenes: the
rectangle contains neighbors and Gemini labels the wrong person. This stage
turns each GT box into that person's MASK, so the validated mask-outline
crop applies unchanged. SAM's native prompt type is the box -- detection
stays with CrowdHuman's human annotators, SAM only traces the person the
box already pins down (its crowd recall problem never enters the picture).

Writes one sidecar per image to --out:
  {"width": W, "height": H, "persons": [
      {"box_index": i, "vbox": [x,y,w,h], "box": [x1,y1,x2,y2],
       "iou_score": s, "parts": [[[x,y], ...], ...]}]}

    python3 crowd_sam_masks.py --gt .../gt_boxes.jsonl \
        --images /workspace/datasets/crowdhuman/Images \
        --out /workspace/spotlight/crowd_1k/masks_raw --limit 50

    python3 crowd_sam_masks.py --selftest    # mask->polygon only, no GPU

GPU: ~0.7-1.5 s/image on a 4090; the 1,000-image set is ~30 min.
Resumable: images with an existing sidecar are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def mask_to_parts(mask: np.ndarray, min_area=60, eps=1.8):
    """Binary mask -> list of polygon parts (one per connected component),
    mirroring the raw-labeler convention (parts kept separate, no bridges)."""
    import cv2
    m = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            parts.append([[int(x), int(y)] for x, y in ap])
    return parts


def run(gt_path: Path, images: Path, out: Path, limit, batch_boxes=32):
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading facebook/sam-vit-huge on {device} (cached on volume)...")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in gt_path.read_text().splitlines()]
    if limit:
        recs = recs[:limit]
    t0, done = time.time(), 0
    for rec in recs:
        stem = rec["stem"]
        op = out / f"{stem}.json"
        if op.exists():
            continue
        ip = images / f"{stem}.jpg"
        if not ip.exists():
            print(f"  !! image missing: {stem}")
            continue
        img = Image.open(ip).convert("RGB")
        boxes_xyxy = [[b["vbox"][0], b["vbox"][1],
                       b["vbox"][0] + b["vbox"][2],
                       b["vbox"][1] + b["vbox"][3]] for b in rec["boxes"]]
        persons = []
        # chunk the boxes: dense frames can carry 100+ and VRAM is finite
        for c0 in range(0, len(boxes_xyxy), batch_boxes):
            chunk = boxes_xyxy[c0:c0 + batch_boxes]
            inputs = proc(img, input_boxes=[chunk],
                          return_tensors="pt").to(device)
            with torch.no_grad():
                o = model(**inputs, multimask_output=True)
            masks = proc.image_processor.post_process_masks(
                o.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())[0]   # (N, 3, H, W)
            scores = o.iou_scores.cpu()[0]                 # (N, 3)
            for j in range(masks.shape[0]):
                best = int(scores[j].argmax())
                m = masks[j, best].numpy()
                i = c0 + j
                persons.append({
                    "box_index": i, "vbox": rec["boxes"][i]["vbox"],
                    "box": boxes_xyxy[i],
                    "iou_score": round(float(scores[j, best]), 4),
                    "parts": mask_to_parts(m)})
        op.write_text(json.dumps(
            {"width": img.width, "height": img.height, "persons": persons}))
        done += 1
        if done % 10 == 0:
            r = done / (time.time() - t0)
            print(f"  {done}/{len(recs)} imgs  {r:.2f} img/s  "
                  f"ETA {(len(recs)-done)/max(r,1e-9)/60:.0f} min", flush=True)
    print(f"done: {done} images -> {out}")


def _selftest():
    m = np.zeros((200, 200), dtype=bool)
    m[40:160, 60:120] = True            # torso blob
    m[10:30, 70:100] = True             # separate head blob (occlusion split)
    parts = mask_to_parts(m)
    assert len(parts) == 2, f"expected 2 parts, got {len(parts)}"
    xs = [p[0] for part in parts for p in part]
    ys = [p[1] for part in parts for p in part]
    assert 55 <= min(xs) <= 65 and 5 <= min(ys) <= 15
    tiny = np.zeros((50, 50), dtype=bool)
    tiny[10:12, 10:12] = True           # below min_area -> dropped
    assert mask_to_parts(tiny) == []
    print("crowd_sam_masks.py self-test passed (mask->polygon only; "
          "model path needs the pod GPU)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt"); ap.add_argument("--images"); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.gt and a.images and a.out):
        ap.error("--gt --images --out are required")
    run(Path(a.gt), Path(a.images), Path(a.out), a.limit)


if __name__ == "__main__":
    main()
