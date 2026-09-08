#!/usr/bin/env python3
"""
HARAMBLUR — Step 1: run the YOLO-MIT model on child Val images and capture
results (Label vs Model).

Selects N random Val images that have a Child (class 2) GT box, runs the trained
YOLO-MIT checkpoint in eager PyTorch (same path as the repo's predict_roi.py),
matches the model's detections to each GT child box by IoU, and records what the
model predicted there. Output feeds Step 2 (VLM) for a 3-way Label vs Model vs
VLM comparison.

    python run_model_children.py \
        --yaml /workspace/open-images-v7/dataset.yaml \
        --run-dir /workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4 \
        --repo /workspace/YOLO-MIT --imgsz 640 --n 5000 --out ./child_eval

    # local plumbing test (no GPU / no yolo package):
    python run_model_children.py --yaml ... --run-dir ... --mock --n 50 --out ./t

Outputs (in --out):
  results.jsonl  one record per GT child box:
      {id, image, gt_box_xyxy, gt_class, pred_class, pred_conf, match_iou, status}
      status = correct | misclassified | missed
  crops/<id>.jpg padded crop of the GT child box (for the VLM step)
  summary.json   counts + the child confusion breakdown
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw

import dataset_utils as du


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def match_boxes(gt_boxes, dets, min_iou):
    """Greedy, mutually-exclusive matching: each detection can match at most one
    GT box, each GT box gets at most one detection. Highest-IoU pairs are
    claimed first, so two nearby children in the same image can't both be
    'matched' to the same detection.

    Returns {gt_index: (det, iou)} for matched pairs only.
    """
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for di, d in enumerate(dets):
            j = iou(gt, d[1:5])
            if j >= min_iou:
                pairs.append((j, gi, di))
    pairs.sort(key=lambda p: p[0], reverse=True)  # best matches first
    used_gt, used_det, out = set(), set(), {}
    for j, gi, di in pairs:
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi)
        used_det.add(di)
        out[gi] = (dets[di], j)
    return out


def draw_match(img, gt_box, det_box, iou_val, pred_label):
    """Debug overlay: GT box in green, matched detection in red, IoU labeled."""
    ann = img.copy()
    draw = ImageDraw.Draw(ann)
    gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 200, 0), width=3)
    draw.text((gx1, max(0, gy1 - 14)), "GT: Child", fill=(0, 200, 0))
    if det_box is not None:
        dx1, dy1, dx2, dy2 = [int(v) for v in det_box[1:5]]
        draw.rectangle([dx1, dy1, dx2, dy2], outline=(220, 0, 0), width=2)
        draw.text((dx1, dy2 + 2), f"pred: {pred_label} iou={iou_val:.2f}",
                  fill=(220, 0, 0))
    return ann


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# child-image selection (Val, GT class == child)
# ---------------------------------------------------------------------------

def select_targets(cfg, root, split, target_indices, n, seed, names):
    """Yield (image_path, w, h, [(cls_id, box_xyxy), ...]) for up to n images
    that contain at least one box whose class is in target_indices (a set).
    Generalizes the original child-only selection to any class or class set.
    n <= 0 means unlimited (scan every matching image in the split)."""
    img_dir, lbl_dir = du.discover_split(cfg, root, split)
    imgs = list(du.list_images(img_dir))
    random.Random(seed).shuffle(imgs)
    picked = 0
    for img_path in imgs:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        boxes = du.yolo_boxes(du.label_path_for(img_path, img_dir, lbl_dir), w, h)
        matched = [(c, (x1, y1, x2, y2)) for (c, x1, y1, x2, y2) in boxes
                   if c in target_indices]
        if not matched:
            continue
        yield img_path, w, h, matched
        picked += 1
        if n > 0 and picked >= n:
            break


# ---------------------------------------------------------------------------
# YOLO-MIT model (mirrors predict_roi.py)
# ---------------------------------------------------------------------------

def find_model_config(run_dir: Path, repo: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    ov = run_dir / ".hydra" / "overrides.yaml"
    name = None
    if ov.exists():
        for line in ov.read_text().splitlines():
            t = line.strip().lstrip("- ").strip()
            if t.startswith("model="):
                name = t.split("=", 1)[1].strip()
                break
    if not name:
        raise SystemExit("Could not find model=... in .hydra/overrides.yaml; "
                         "pass --model-config explicitly")
    return repo / "yolo" / "config" / "model" / f"{name}.yaml"


class MitModel:
    def __init__(self, run_dir, repo, model_config, imgsz, conf, iou_nms,
                 max_dets, class_num, device):
        import torch
        from omegaconf import OmegaConf
        from yolo.model.yolo import create_model
        from yolo.utils.bounding_box_utils import Vec2Box
        from yolo.utils.model_utils import PostProcess
        self.torch = torch
        self.imgsz = imgsz
        self.device = torch.device(device)
        cfg_model = OmegaConf.load(str(model_config))
        model = create_model(cfg_model, weight_path=False, class_num=class_num)
        self._load_weights(model, Path(run_dir) / "weights" / "best.ckpt")
        model.eval()
        vec2box = Vec2Box(model, cfg_model.anchor, [imgsz, imgsz], self.device)
        self.model = model.to(self.device)
        nms = OmegaConf.create({"min_confidence": conf, "min_iou": iou_nms,
                                "max_bbox": max_dets})
        self.post = PostProcess(vec2box, nms)

    @staticmethod
    def _load_weights(model, ckpt_path):
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        clean = {}
        for k, v in state.items():
            if k.startswith("model.model."):
                clean[k[len("model."):]] = v
            elif k.startswith("ema.model."):
                continue
            else:
                clean[k] = v
        model.load_state_dict(clean, strict=False)

    def _preprocess(self, img: Image.Image):
        import torchvision.transforms.functional as TF
        size = self.imgsz
        ow, oh = img.size
        scale = min(size / ow, size / oh)
        nw, nh = int(round(ow * scale)), int(round(oh * scale))
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        left, top = (size - nw) // 2, (size - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
        tensor = TF.to_tensor(canvas).unsqueeze(0)
        rev = self.torch.tensor([[scale, left, top, left, top]],
                                dtype=self.torch.float32)
        return tensor, rev

    def detect(self, img: Image.Image):
        """Return list of (cls_id, x1,y1,x2,y2, conf) in ORIGINAL pixel coords."""
        tensor, rev = self._preprocess(img)
        tensor, rev = tensor.to(self.device), rev.to(self.device)
        with self.torch.no_grad():
            raw = self.model(tensor)
            dets = self.post(raw, rev_tensor=rev)
        out = []
        for row in dets[0].cpu().tolist():
            cls_id, x1, y1, x2, y2, conf = row
            out.append((int(cls_id), x1, y1, x2, y2, float(conf)))
        return out


class MockModel:
    """Random detections so selection/matching/output can be tested w/o GPU."""
    def __init__(self, class_num, seed=0):
        self.r = random.Random(seed)
        self.class_num = class_num

    def detect(self, img):
        w, h = img.size
        out = []
        for _ in range(self.r.randint(0, 2)):
            x1 = self.r.uniform(0, w * 0.5); y1 = self.r.uniform(0, h * 0.5)
            x2 = x1 + self.r.uniform(w * 0.2, w * 0.5)
            y2 = y1 + self.r.uniform(h * 0.2, h * 0.5)
            out.append((self.r.randint(0, self.class_num - 1), x1, y1, x2, y2,
                        self.r.uniform(0.3, 0.99)))
        return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="run YOLO-MIT on child val images")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--run-dir", required=True, help="the gelansfav14 run dir")
    ap.add_argument("--repo", default="/workspace/YOLO-MIT")
    ap.add_argument("--model-config", default=None,
                    help="override; else read from run-dir/.hydra/overrides.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=5000,
                    help="max images to scan; <= 0 means unlimited (whole split)")
    ap.add_argument("--child-class", type=int, default=2)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="target class NAMES to evaluate, e.g. --classes Woman Man "
                         "(default: just the --child-class index, for backward "
                         "compatibility with the original child-only behavior)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="MUST match training image_size (check the run's config)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="min IoU for a detection to count as the child's prediction")
    ap.add_argument("--pad", type=float, default=0.3, help="crop context padding")
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--debug-overlay", action="store_true",
                    help="also save crops_debug/<id>.jpg with GT (green) + "
                         "matched detection (red) boxes drawn, to visually "
                         "confirm Label and Model refer to the same object")
    ap.add_argument("--out", default="./child_eval")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out / "crops_debug"
    if args.debug_overlay:
        debug_dir.mkdir(parents=True, exist_ok=True)

    cfg = du.load_yaml(Path(args.yaml))
    root = du.resolve_root(cfg, Path(args.yaml))
    names = du.class_names(cfg)
    cls_name = lambda i: names.get(i, str(i))
    name2idx = {v: k for k, v in names.items()}
    if args.classes:
        target_indices = {name2idx[n] for n in args.classes if n in name2idx}
        if not target_indices:
            raise SystemExit(f"none of --classes {args.classes} found in {names}")
    else:
        target_indices = {args.child_class}
    print(f"[model] classes={names} · target={[cls_name(i) for i in target_indices]}")

    if args.mock:
        model = MockModel(class_num=len(names), seed=args.seed)
        print("[model] MOCK model (no GPU)")
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        mc = find_model_config(Path(args.run_dir), Path(args.repo), args.model_config)
        print(f"[model] config={mc} · imgsz={args.imgsz} · device={device}")
        model = MitModel(args.run_dir, args.repo, mc, args.imgsz, args.conf,
                         args.iou_nms, args.max_dets, len(names), device)

    f = (out / "results.jsonl").open("w")
    counts = {"correct": 0, "misclassified": 0, "missed": 0}
    per_class_counts = {cls_name(i): {"correct": 0, "misclassified": 0, "missed": 0}
                        for i in target_indices}
    confusion = {}   # gt class name -> {predicted class name -> count} (mismatches only)
    confusion_matrix = {cls_name(i): {} for i in target_indices}  # full matrix:
    # gt class name -> {pred class name OR 'missed' -> count}, includes diagonal
    n_imgs = n_boxes = 0
    t0 = time.time()

    for img_path, w, h, target_boxes in select_targets(
            cfg, root, args.split, target_indices, args.n, args.seed, names):
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            dets = model.detect(img)
            gt_boxes_only = [b for (_, b) in target_boxes]
            # greedy, mutually-exclusive matching so two same-image targets
            # can't both claim the same detection (see match_boxes doc)
            matched = match_boxes(gt_boxes_only, dets, args.match_iou)
            for bi, (gt_cls, gt) in enumerate(target_boxes):
                gt_name = cls_name(gt_cls)
                oid = f"{img_path.stem}_{bi}"
                m = matched.get(bi)
                if m is None:
                    best, best_iou = None, 0.0
                    pred_name, pred_conf, status = None, 0.0, "missed"
                else:
                    best, best_iou = m
                    pred_name = cls_name(best[0])
                    pred_conf = best[5]
                    status = "correct" if best[0] == gt_cls else "misclassified"
                    if status == "misclassified":
                        confusion.setdefault(gt_name, {})
                        confusion[gt_name][pred_name] = confusion[gt_name].get(pred_name, 0) + 1
                counts[status] += 1
                per_class_counts[gt_name][status] += 1
                matrix_col = "missed" if pred_name is None else pred_name
                confusion_matrix.setdefault(gt_name, {})
                confusion_matrix[gt_name][matrix_col] = (
                    confusion_matrix[gt_name].get(matrix_col, 0) + 1)

                # save padded crop of the GT box for the VLM step
                px1, py1, px2, py2 = du.pad_box(*gt, w, h, args.pad)
                crop = img.crop((px1, py1, px2, py2))
                crop.thumbnail((args.thumb, args.thumb))
                crop.save(crops_dir / f"{oid}.jpg")

                if args.debug_overlay:
                    ann = draw_match(img, gt, best, best_iou,
                                     pred_name or "none")
                    ann.thumbnail((args.thumb * 2, args.thumb * 2))
                    ann.save(debug_dir / f"{oid}.jpg")

                f.write(json.dumps({
                    "id": oid, "image": str(img_path),
                    "gt_class": gt_name,
                    "gt_box_xyxy": [int(v) for v in gt],
                    "pred_class": pred_name, "pred_conf": round(pred_conf, 4),
                    "match_iou": round(best_iou, 4), "status": status,
                }) + "\n")
                n_boxes += 1
        n_imgs += 1
        if n_imgs % 200 == 0:
            el = time.time() - t0
            print(f"[model] {n_imgs} imgs / {n_boxes} target boxes · "
                  f"{n_imgs/el:.1f} img/s", flush=True)
    f.close()

    total = max(1, n_boxes)
    per_class_summary = {}
    for cname, c in per_class_counts.items():
        t = max(1, sum(c.values()))
        per_class_summary[cname] = {
            **c, "accuracy_detected_only": round(
                c["correct"] / max(1, c["correct"] + c["misclassified"]), 4),
            "accuracy_full": round(c["correct"] / t, 4),
        }
    summary = {
        "target_boxes": n_boxes, "images": n_imgs,
        "target_classes": [cls_name(i) for i in target_indices],
        "counts": counts,
        "model_accuracy_detected_only": round(
            counts["correct"] / max(1, counts["correct"] + counts["misclassified"]), 4),
        "model_accuracy_full_population": round(counts["correct"] / total, 4),
        "per_class": per_class_summary,
        "misclassified_as": confusion,
        "confusion_matrix": confusion_matrix,  # label -> {pred-or-'missed' -> count}, full incl. diagonal
        "run_dir": args.run_dir, "imgsz": args.imgsz,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[model] {n_boxes} target boxes across {[cls_name(i) for i in target_indices]}")
    print(f"[model] correct: {counts['correct']} ({100*counts['correct']/total:.1f}%) · "
          f"misclassified: {counts['misclassified']} · missed: {counts['missed']}")
    print(f"[model] per-class: {per_class_summary}")
    print(f"[model] misclassified as: {confusion}")
    print(f"[model] -> {out/'results.jsonl'} · {out/'summary.json'}")


if __name__ == "__main__":
    main()
