"""
RAW-LOGGING copy of the production autolabel_sam.py (EXP-2026-10).

Same pattern as EXP-2026-07's conf patch: the production script at
/workspace/autolabel/autolabel_sam.py stays untouched; this copy behaves
IDENTICALLY for the YOLO output (same prompts, conf threshold, mask-IoU NMS,
same mask_to_polygon flattening) and ADDITIONALLY writes one raw sidecar per
image next to the .txt:

    <stem>.json = {
      "image": name, "width": W, "height": H,
      "detections": [            # NMS survivors, SAME ORDER as the .txt lines
        {"cls": int, "conf": float, "box": [x1,y1,x2,y2] px,
         "parts": [[[x,y],...], ...]}   # per CONNECTED COMPONENT of the mask,
      ],                                # captured BEFORE the YOLO single-
      "suppressed": [ ...same shape... ]  # polygon flattening merges them
    }

Why: the YOLO format stores one polygon per person, so multi-part masks of
occluded people get stitched with bridge lines — unrecoverable exactly once
flattened. The sidecar preserves the raw readings (per-part geometry +
confidence + what NMS removed) so ANY post-processing can be redone later
without re-running the GPU. pipeline_v1_eval.py consumes the sidecars via
--raw-labels (exact per-part highlight rendering, no heuristics).

Usage identical to production, e.g.:
    python3 autolabel_sam_raw.py --input /workspace/exp10/trace_raw_imgs \
        --output /workspace/exp10/raw_labels \
        --classes "woman" "man" "child" --batch_size 1

NOTE on resume: like production, images with an existing .txt are skipped —
run into a FRESH output dir to guarantee sidecars exist for every image.
"""
import os
import glob
import json
import hashlib
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model


def compute_iou(mask1, mask2):
    """Unchanged from production."""
    intersection = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()


def mask_bbox(mask_tensor):
    """Tight (x1,y1,x2,y2) of a boolean mask, or None if empty. Used ONLY to
    skip provably-zero-IoU pairs in NMS (see below) — never changes output."""
    rows = torch.any(mask_tensor, dim=1)
    cols = torch.any(mask_tensor, dim=0)
    ys = torch.nonzero(rows, as_tuple=False)
    xs = torch.nonzero(cols, as_tuple=False)
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    return (int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1]))


def boxes_disjoint(a, b):
    """True when the two bboxes cannot intersect. If bboxes are disjoint the
    masks inside them are disjoint too, so mask IoU is EXACTLY 0 — skipping
    such pairs is bit-identical to computing them, and turns the O(n^2)
    full-resolution mask NMS from minutes/hours into milliseconds on dense
    crowd images (a 200-detection image was observed to hang the run)."""
    if a is None or b is None:
        return True
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def mask_to_polygon(mask_tensor, img_w, img_h):
    """Unchanged from production: flattens ALL parts into one normalized
    YOLO polygon (this is where the bridge stitching happens)."""
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    if contours:
        all_points = []
        for c in contours:
            if cv2.contourArea(c) < 30:
                continue
            epsilon = 0.001 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) >= 3:
                all_points.append(approx.reshape(-1, 2))
        if all_points:
            merged_points = np.concatenate(all_points, axis=0)
            flat_points = merged_points.flatten().astype(float)
            flat_points[0::2] /= img_w
            flat_points[1::2] /= img_h
            flat_points = np.clip(flat_points, 0, 1)
            polygons = flat_points.tolist()
    return polygons


def mask_to_parts(mask_tensor):
    """RAW addition: the same contours mask_to_polygon uses (same <30px area
    filter, same approxPolyDP simplification) but kept SEPARATE per connected
    component, in pixel coords, plus the pixel bounding box over all parts.
    Returns (parts, box_xyxy or None)."""
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        if cv2.contourArea(c) < 30:
            continue
        epsilon = 0.001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) >= 3:
            pts = approx.reshape(-1, 2).astype(float)
            parts.append([[round(float(x), 1), round(float(y), 1)]
                          for x, y in pts])
    if not parts:
        return [], None
    xs = [x for part in parts for x, _ in part]
    ys = [y for part in parts for _, y in part]
    box = [min(xs), min(ys), max(xs), max(ys)]
    return parts, box


def det_record(det, class_id_field="class_id"):
    parts, box = mask_to_parts(det['mask'])
    if box is None:
        return None
    return {"cls": det[class_id_field], "conf": round(det['score'], 4),
            "box": [round(v, 1) for v in box], "parts": parts}


def main():
    parser = argparse.ArgumentParser(
        description="RAW-logging autolabel (production-identical YOLO output "
                    "+ per-image raw sidecars).")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--classes", nargs='+', required=True)
    parser.add_argument("--output", type=str, default="labels")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--nms_iou", type=float, default=0.7)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    # Sharding: run the SAME command on N pods with --shard-index 0..N-1.
    # Split is by stable hash of the file path, so shards are disjoint and
    # complete regardless of pod start order, and each shard stays resumable.
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.shards):
        parser.error("--shard-index must be in [0, --shards)")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output, exist_ok=True)
    input_root = os.path.abspath(args.input)

    print(f"Loading SAM3 model (transformers) on {args.device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(args.device)
    model.eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(input_root, '**', ext),
                                     recursive=True))
    image_files.sort()
    if args.shards > 1:
        total = len(image_files)
        image_files = [p for p in image_files
                       if int(hashlib.md5(p.encode()).hexdigest(), 16)
                       % args.shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.shards}: {len(image_files)} "
              f"of {total} images")
    # Pre-filter work already done so the progress bar and ETA reflect REAL
    # remaining work (previously the bar counted fast-skipped files, so it
    # sprinted then appeared to stall).
    todo, skipped = [], 0
    for img_path in image_files:
        if "negative" in img_path:
            continue
        rel_dir = os.path.relpath(os.path.dirname(img_path), input_root)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if not args.overwrite and os.path.exists(
                os.path.join(args.output, rel_dir, f"{base_name}.txt")):
            skipped += 1
            continue
        todo.append((img_path, rel_dir, base_name))
    print(f"Found {len(image_files)} images · {skipped} already labeled · "
          f"{len(todo)} to do. Classes: {args.classes}")
    if not todo:
        print("Nothing to do.")
        return

    count = 0
    n_det_total = 0
    bar = tqdm(todo, desc="Labeling (raw)", unit="img", smoothing=0.05)
    for img_path, rel_dir, base_name in bar:
        label_dir = os.path.join(args.output, rel_dir)
        os.makedirs(label_dir, exist_ok=True)
        txt_path = os.path.join(label_dir, f"{base_name}.txt")
        raw_path = os.path.join(label_dir, f"{base_name}.json")
        if args.max_images is not None and count >= args.max_images:
            print(f"Reached max images limit ({args.max_images}). Stopping.")
            break
        try:
            pil_image = Image.open(img_path).convert("RGB")
            img_w, img_h = pil_image.size
            raw_detections = []

            total_classes = len(args.classes)
            for i in range(0, total_classes, args.batch_size):
                batch_classes = args.classes[i: i + args.batch_size]
                batch_ids = list(range(i, i + len(batch_classes)))
                batch_images = [pil_image] * len(batch_classes)
                inputs = processor(images=batch_images, text=batch_classes,
                                   return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    if args.device == "cuda":
                        with torch.autocast("cuda"):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                target_sizes = [(img_h, img_w)] * len(batch_classes)
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=args.conf, mask_threshold=0.5,
                    target_sizes=target_sizes)
                for j, result in enumerate(results):
                    class_id = batch_ids[j]
                    masks = result["masks"]
                    scores = result["scores"]
                    if len(scores) == 0:
                        continue
                    masks = masks.cpu()
                    scores = scores.cpu()
                    for k in range(len(scores)):
                        score = scores[k].item()
                        if score >= args.conf:
                            mask_tensor = masks[k]
                            if mask_tensor.dim() > 2:
                                mask_tensor = mask_tensor.squeeze(0)
                            raw_detections.append({
                                'score': score,
                                'class_id': class_id,
                                'mask': mask_tensor > 0})

            # NMS — same semantics as production, with a bit-identical
            # bbox prefilter (disjoint bboxes => mask IoU is exactly 0)
            raw_detections.sort(key=lambda x: x['score'], reverse=True)
            for det in raw_detections:
                det['bbox'] = mask_bbox(det['mask'])
            final_detections, suppressed = [], []
            for det in raw_detections:
                keep = True
                for saved_det in final_detections:
                    if boxes_disjoint(det['bbox'], saved_det['bbox']):
                        continue          # provably IoU 0 — skip the mask op
                    if compute_iou(det['mask'], saved_det['mask']) > args.nms_iou:
                        keep = False
                        break
                (final_detections if keep else suppressed).append(det)

            # YOLO output — unchanged — plus the raw sidecar, index-aligned:
            # a sidecar "detections" entry is appended EXACTLY when a .txt
            # line is written, so detection ids match across both files.
            yolo_segments = []
            sidecar = {"image": os.path.basename(img_path),
                       "width": img_w, "height": img_h,
                       "conf_threshold": args.conf, "nms_iou": args.nms_iou,
                       "classes": args.classes,
                       "detections": [], "suppressed": []}
            for det in final_detections:
                polygon = mask_to_polygon(det['mask'], img_w, img_h)
                if polygon:
                    poly_str = " ".join([f"{p:.6f}" for p in polygon])
                    yolo_segments.append(f"{det['class_id']} {poly_str}")
                    rec = det_record(det)
                    if rec is None:   # can't happen if polygon non-empty
                        rec = {"cls": det['class_id'],
                               "conf": round(det['score'], 4),
                               "box": None, "parts": []}
                    sidecar["detections"].append(rec)
            for det in suppressed:
                rec = det_record(det)
                if rec is not None:
                    sidecar["suppressed"].append(rec)

            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_segments))
            with open(raw_path, "w") as f:
                json.dump(sidecar, f)
            count += 1
            n_det_total += len(sidecar["detections"])
            bar.set_postfix(dets=n_det_total,
                            per_img=f"{n_det_total/count:.2f}", refresh=False)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("\nProcessing complete (YOLO labels + raw sidecars).")


if __name__ == "__main__":
    main()
