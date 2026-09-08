#!/usr/bin/env python3
"""
Scale-robustness probe: shrink real images progressively and measure how each
candidate model's person-detection recall degrades as objects get smaller —
the regime LAGENDA and crowd scenes actually contain (distant/small people),
as opposed to the mostly-prominent subjects the standard four-benchmark
protocol scores.

Method: for each source image and each scale factor s, the image CONTENT is
resized by s and pasted onto a canvas of the ORIGINAL size (padded, not
cropped) so the composition/frame stays fixed and only the apparent person
size shrinks — exactly what a person moving further from the camera looks
like. GT boxes are scaled + offset identically. Every model sees the SAME
shrunk canvas at every scale, so the comparison isolates object size as the
only variable.

    python3 scale_robustness.py \
        --images /workspace/lagenda_eval/lagenda_yolo/images/val \
        --labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
        --model y26n_warm50:/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt \
        --model y26n_noe2e:/workspace/exp16/train/y26n_noe2e/weights/best.pt \
        --model y26n_gradsupp_nms:/workspace/exp14/train/y26n_gradsupp/weights/best_plain.pt:noe2e \
        --model v11n_shipped:/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt \
        --scales 1.0,0.75,0.5,0.35,0.25,0.15 --n 150 --out /workspace/exp18/scale_probe

    python3 scale_robustness.py --selftest    # synthetic, no data/GPU
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_yolo_labels(label_path: Path, w: int, h: int):
    """[(cls, x1, y1, x2, y2), ...] in pixel xyxy."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def shrink_and_pad(img, scale: float, fill=(114, 114, 114)):
    """Resize content by `scale`, paste centered onto a canvas of the
    ORIGINAL size. Returns (canvas, offset_x, offset_y)."""
    w, h = img.size
    if scale >= 0.999:
        return img, 0, 0
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    from PIL import Image
    small = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), fill)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas.paste(small, (ox, oy))
    return canvas, ox, oy


def scale_boxes(boxes, scale: float, ox: int, oy: int):
    return [(cls, x1 * scale + ox, y1 * scale + oy, x2 * scale + ox, y2 * scale + oy)
            for cls, x1, y1, x2, y2 in boxes]


def build_model(spec: str):
    """spec = 'name:path[:noe2e]'"""
    parts = spec.split(":")
    name, path = parts[0], parts[1]
    no_e2e = len(parts) > 2 and parts[2] == "noe2e"
    from run_ultralytics_labels import UltralyticsModel
    model = UltralyticsModel(path, imgsz=640, floor=0.25, iou=0.7,
                              device="cuda:0", no_e2e=no_e2e)
    return name, model


def run(images_dir, labels_dir, model_specs, scales, n, seed, out_dir,
        match_iou=0.3, log=print):
    from run_model_children import match_boxes

    imgs = sorted(p for p in Path(images_dir).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(seed).shuffle(imgs)
    imgs = imgs[:n] if n else imgs

    models = dict(build_model(s) for s in model_specs)
    log(f"models: {list(models)}")

    results = {name: {s: {"n_gt": 0, "n_matched": 0, "class_correct": 0}
                       for s in scales} for name in models}
    native_heights = []
    used = 0

    from PIL import Image
    for img_path in imgs:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gt = load_yolo_labels(Path(labels_dir) / f"{img_path.stem}.txt", w, h)
        if not gt:
            continue
        used += 1
        native_heights.extend(g[4] - g[2] for g in gt)
        for s in scales:
            canvas, ox, oy = shrink_and_pad(img, s)
            scaled_gt = scale_boxes(gt, s, ox, oy)
            gt_boxes = [(g[1], g[2], g[3], g[4]) for g in scaled_gt]
            for name, model in models.items():
                dets = model.detect(canvas)
                m = match_boxes(gt_boxes, dets, match_iou)
                r = results[name][s]
                r["n_gt"] += len(gt_boxes)
                r["n_matched"] += len(m)
                for gi, (det, _j) in m.items():
                    if det[0] == scaled_gt[gi][0]:
                        r["class_correct"] += 1
        if used % 25 == 0:
            log(f"  ...{used} images with GT processed")

    native_heights.sort()
    median_h = native_heights[len(native_heights) // 2] if native_heights else None

    summary = {"n_images_used": used, "match_iou": match_iou,
               "native_median_person_height_px": median_h,
               "scales": scales, "models": {}}
    for name in models:
        rows = []
        for s in scales:
            r = results[name][s]
            recall = r["n_matched"] / r["n_gt"] if r["n_gt"] else None
            cls_acc = (r["class_correct"] / r["n_matched"]
                       if r["n_matched"] else None)
            rows.append({"scale": s, "n_gt": r["n_gt"],
                         "recall": round(recall, 4) if recall is not None else None,
                         "class_acc_on_matched": round(cls_acc, 4) if cls_acc is not None else None})
        summary["models"][name] = rows

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"native median GT person height: {median_h:.0f}px "
        f"(over {len(native_heights)} boxes, {used} images)")
    log(f"{'model':<22}" + "".join(f"scale={s:<6}" for s in scales))
    for name, rows in summary["models"].items():
        log(f"{name:<22}" + "".join(
            f"{(r['recall'] if r['recall'] is not None else float('nan')):<11.3f}"
            for r in rows))
    return summary


def _selftest():
    from PIL import Image

    img = Image.new("RGB", (200, 100), (0, 0, 0))
    canvas, ox, oy = shrink_and_pad(img, 1.0)
    assert canvas.size == (200, 100) and (ox, oy) == (0, 0)

    canvas, ox, oy = shrink_and_pad(img, 0.5)
    assert canvas.size == (200, 100), "canvas must stay original size"
    assert (ox, oy) == (50, 25), f"expected centered offset (50,25), got ({ox},{oy})"

    boxes = [(0, 10.0, 10.0, 30.0, 50.0)]   # cls, x1,y1,x2,y2 — height 40
    scaled = scale_boxes(boxes, 0.5, 50, 25)
    cls, x1, y1, x2, y2 = scaled[0]
    assert abs(x1 - 55.0) < 1e-6 and abs(y1 - 30.0) < 1e-6, scaled
    assert abs((y2 - y1) - 20.0) < 1e-6, "height must halve under scale 0.5"

    lbl_dir = Path("/tmp/_scaletest_labels")
    lbl_dir.mkdir(exist_ok=True)
    (lbl_dir / "x.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    loaded = load_yolo_labels(lbl_dir / "x.txt", 100, 100)
    assert loaded == [(1, 40.0, 30.0, 60.0, 70.0)], loaded
    assert load_yolo_labels(lbl_dir / "missing.txt", 100, 100) == []

    class FakeMatchBoxes:
        pass

    # match_boxes reuse smoke test (no GPU/model needed)
    sys.path.insert(0, str(Path(__file__).parent))
    from run_model_children import match_boxes
    gt = [(0.0, 0.0, 10.0, 10.0)]
    dets = [(0, 1.0, 1.0, 11.0, 11.0, 0.9)]
    m = match_boxes(gt, dets, 0.3)
    assert 0 in m, m

    print("scale_robustness selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--model", action="append", default=[],
                     help="name:weights_path[:noe2e], repeatable")
    ap.add_argument("--scales", default="1.0,0.75,0.5,0.35,0.25,0.15")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not (args.images and args.labels and args.model and args.out):
        ap.error("--images/--labels/--model/--out required (or --selftest)")

    scales = [float(x) for x in args.scales.split(",")]
    run(args.images, args.labels, args.model, scales, args.n, args.seed, args.out,
        match_iou=args.match_iou)


if __name__ == "__main__":
    main()
