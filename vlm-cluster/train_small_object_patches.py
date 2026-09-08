#!/usr/bin/env python3
"""
Small-object augmentation for YOLO26n, continuing the validated warm-restart fine-tune.

Ports YOLO-MIT's `SmallObjectPatches` augmentation
(/workspace/YOLO-MIT/yolo/tools/data_augmentation.py:968-1076, ON at p=0.05 in every
YOLO-MIT/gelan/gemlb run since 2026-01) to Ultralytics' `BaseTransform` contract, unmodified
in its geometry: with probability p, the WHOLE training image is replaced by a grey canvas
holding every labeled object in that image, rescaled to fit inside `target_size` px (aspect
ratio preserved), placed at non-overlapping random positions. This is the training-time
mirror of the extension's own "upscale a small region before inference" path.

Why: the current best checkpoint (y26n_noe2e_warm50-2, EXP-2026-18) is strong everywhere
except small/distant people -- the fresh haramblur_holdout benchmark measured mAP50-95 by GT
box size at small~=0.267 vs large~=0.874. Spotlight's training labels have almost no small
people to learn from: it drops ~196k detections it can't confidently gender, median size
~58px, so the model rarely sees a correctly-labeled small person. YOLO-MIT's sibling
architectures (gelan/gemlb) train with this exact augmentation and have never lost to y26n on
crowd/small-person recall -- this ports the same trick, geometry unchanged, to the
ultralytics-trained head.

Mechanism (against pinned ultralytics==8.4.115 source, file:line refs from the pod install at
/usr/local/lib/python3.11/dist-packages/ultralytics):
  * `SmallObjectPatches` is a plain `BaseTransform` (data/augment.py:28) overriding
    `__call__` directly, the same way `CopyPaste`'s "flip" mode does (augment.py:1938-1948) --
    it needs shared state between the image edit and the box edit that the default
    get_params/apply_image/apply_instances split doesn't fit.
  * `SOPYOLODataset.build_transforms` (dataset.py:303-332) inserts it as the LAST pre-Format
    step: `transforms.insert(-1, ...)` on the Compose that `v8_transforms` already returns.
    At that point `labels["instances"]` is absolute-pixel xyxy, unnormalized -- Format is what
    normalizes and converts to xywh (dataset.py:319-331) -- matching exactly what the source
    YOLO-MIT algorithm expects (it denormalizes by hand at its own entry point).
  * `SOPTrainer.build_dataset` mirrors `build_yolo_dataset` (data/build.py:236-281) but
    constructs `SOPYOLODataset` in place of the stock `YOLODataset` it would otherwise pick.
    This is the ONLY trainer change -- no custom loss, no custom model class -- so a
    checkpoint trained with this trainer is a completely plain `DetectionModel` and needs NO
    special import to load anywhere (unlike gradsupp's checkpoints -- see
    train_gradsuppress.py's module docstring for that trap. The canonical-`__main__`-reimport
    guard below is kept anyway, cheaply, in case DataLoader workers ever spawn instead of
    fork).

Usage (pod), continuing the validated warm-restart recipe from
/workspace/exp18/train/y26n_noe2e_warm50-2/args.yaml:
    python3 train_small_object_patches.py \
        --data /workspace/exp12/spotlight_oiv7_local.yaml \
        --weights /workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt \
        --epochs 50 --imgsz 640 --optimizer MuSGD --lr0 0.003 --lrf 0.01 \
        --warmup-epochs 0 --sop-p 0.05 \
        --project /workspace/exp19/train --name y26n_sop50

    python3 train_small_object_patches.py --selftest   # needs torch+ultralytics, no data/GPU
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - ultralytics itself requires cv2; see module docstring
    cv2 = None

from ultralytics.data.augment import BaseTransform
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.instance import Instances


class SmallObjectPatches(BaseTransform):
    """Replace the whole image with a grey canvas of every object shrunk to fit
    `target_size` px, placed at non-overlapping random positions.

    Geometry is an unmodified port of YOLO-MIT's version (data_augmentation.py:968-1076):
    same target_size/background_color/max_attempts defaults, same
    scale-to-fit-longest-side-of-target rescale, same greedy
    random-position-with-retry placement using vectorized overlap checks.
    """

    def __init__(self, p: float = 0.05, target_size: int = 96,
                 background_color: tuple[int, int, int] = (114, 114, 114),
                 max_attempts: int = 50):
        self.p = p
        self.target_size = target_size
        self.background_color = background_color
        self.max_attempts = max_attempts

    def __call__(self, labels: dict) -> dict:
        if self.p <= 0 or random.random() >= self.p:
            return labels
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return labels

        img = labels["img"]
        h_img, w_img = img.shape[:2]

        instances.convert_bbox(format="xyxy")
        if instances.normalized:
            instances.denormalize(w_img, h_img)
        boxes = np.asarray(instances.bboxes, dtype=np.float64)
        cls = np.asarray(labels["cls"]).reshape(-1)
        if len(cls) != len(boxes):
            return labels  # defensive: never touch a mismatched batch

        target_size = self.target_size
        crops = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w_img, int(x2)), min(h_img, int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop_h, crop_w = y2i - y1i, x2i - x1i
            scale = min(target_size / crop_w, target_size / crop_h)
            new_w = max(1, int(round(crop_w * scale)))
            new_h = max(1, int(round(crop_h * scale)))
            resized = cv2.resize(img[y1i:y2i, x1i:x2i], (new_w, new_h),
                                  interpolation=cv2.INTER_LINEAR)
            crops.append((cls[i], resized, new_w, new_h))
        if not crops:
            return labels

        bg = self.background_color[: img.shape[2]]
        canvas = np.full(img.shape, bg, dtype=img.dtype)
        placed = np.empty((len(crops), 4), dtype=np.int32)
        placed_count = 0
        new_boxes, new_cls = [], []

        for cls_id, resized_crop, new_w, new_h in crops:
            max_x, max_y = w_img - new_w, h_img - new_h
            if max_x < 0 or max_y < 0:
                continue
            for _ in range(self.max_attempts):
                px = random.randint(0, max_x)
                py = random.randint(0, max_y)
                if placed_count > 0:
                    pl = placed[:placed_count]
                    if ((px < pl[:, 2]) & (px + new_w > pl[:, 0])
                            & (py < pl[:, 3]) & (py + new_h > pl[:, 1])).any():
                        continue
                canvas[py:py + new_h, px:px + new_w] = resized_crop
                placed[placed_count] = (px, py, px + new_w, py + new_h)
                placed_count += 1
                new_boxes.append([px, py, px + new_w, py + new_h])
                new_cls.append(cls_id)
                break

        if not new_boxes:
            return labels

        labels["img"] = canvas
        labels["instances"] = Instances(
            bboxes=np.array(new_boxes, dtype=np.float32),
            segments=[], bbox_format="xyxy", normalized=False,
        )
        labels["cls"] = np.array(new_cls, dtype=np.float32).reshape(-1, 1)
        return labels


class SOPYOLODataset(YOLODataset):
    """YOLODataset with SmallObjectPatches appended as the last training transform."""

    def __init__(self, *args, sop_p: float = 0.0, sop_target_size: int = 96, **kwargs):
        self.sop_p = sop_p
        self.sop_target_size = sop_target_size
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment and self.sop_p > 0:
            transforms.insert(-1, SmallObjectPatches(p=self.sop_p, target_size=self.sop_target_size))
        return transforms


class SOPTrainer(DetectionTrainer):
    """DetectionTrainer whose train-mode dataset carries SmallObjectPatches.
    Val mode always gets sop_p=0 -- the augmentation must never touch validation images."""

    sop_p = 0.05
    sop_target_size = 96

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        from ultralytics.utils.torch_utils import unwrap_model
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        pad = 0.0 if mode == "train" else 0.5
        return SOPYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=pad,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            sop_p=self.sop_p if mode == "train" else 0.0,
            sop_target_size=self.sop_target_size,
        )


# ---------------------------------------------------------------- selftest
def selftest():
    """No GPU, no dataset, no network -- but needs torch+ultralytics+cv2 (pod-only,
    same as train_gradsuppress.py's selftest). Asserts the contract points:
      1. p=0 is a strict no-op (image and instances untouched).
      2. Every object in the canvas is placed non-overlapping, inside bounds, and its
         longest side is at or under target_size.
      3. Pixels outside every placed patch equal the background color exactly.
      4. cls stays paired 1:1 with the surviving boxes (no silent misalignment).
      5. An image with zero instances is a no-op even at p=1.
    """
    random.seed(0)
    np.random.seed(0)

    img = np.random.randint(40, 220, (200, 300, 3), dtype=np.uint8)
    # two objects: one already small-ish, one large -- both must end up <= target_size
    boxes = np.array([[10, 10, 40, 60], [100, 20, 280, 190]], dtype=np.float32)
    cls = np.array([[0.0], [1.0]], dtype=np.float32)

    def fresh_labels():
        return {
            "img": img.copy(),
            "cls": cls.copy(),
            "instances": Instances(bboxes=boxes.copy(), segments=[], bbox_format="xyxy",
                                    normalized=False),
        }

    # 1. p=0 no-op
    t0 = SmallObjectPatches(p=0.0, target_size=64)
    out0 = t0(fresh_labels())
    assert np.array_equal(out0["img"], img), "p=0 must not touch the image"
    assert np.array_equal(out0["instances"].bboxes, boxes), "p=0 must not touch boxes"

    # 5. zero instances, p=1 -> no-op
    t1 = SmallObjectPatches(p=1.0, target_size=64)
    empty = fresh_labels()
    empty["instances"] = Instances(bboxes=np.zeros((0, 4), dtype=np.float32), segments=[],
                                   bbox_format="xyxy", normalized=False)
    empty["cls"] = np.zeros((0, 1), dtype=np.float32)
    out_empty = t1(empty)
    assert np.array_equal(out_empty["img"], img), "zero-instance image must be untouched"

    # 2-4. p=1 real run
    bg = (114, 114, 114)
    t = SmallObjectPatches(p=1.0, target_size=64, background_color=bg, max_attempts=50)
    out = t(fresh_labels())
    new_boxes = out["instances"].bboxes
    new_cls = out["cls"].reshape(-1)
    assert len(new_boxes) == 2, f"both objects should place on a mostly-empty 200x300 canvas, got {len(new_boxes)}"
    assert len(new_cls) == len(new_boxes), "cls must stay paired 1:1 with boxes"
    assert set(new_cls.tolist()) == {0.0, 1.0}, "both original classes must survive"

    h_img, w_img = img.shape[:2]
    canvas = out["img"]
    mask = np.zeros((h_img, w_img), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(new_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        assert 0 <= x1 < x2 <= w_img and 0 <= y1 < y2 <= h_img, (x1, y1, x2, y2)
        assert max(x2 - x1, y2 - y1) <= 64 + 1, f"box {i} exceeds target_size: {(x2 - x1, y2 - y1)}"
        assert not mask[y1:y2, x1:x2].any(), f"box {i} overlaps a previously placed box"
        mask[y1:y2, x1:x2] = True

    # every pixel NOT inside a placed box must be exactly the background color
    outside = canvas[~mask]
    assert (outside == np.array(bg, dtype=canvas.dtype)).all(), \
        "pixels outside placed patches must equal background_color exactly"

    print(f"SmallObjectPatches selftest OK ({len(new_boxes)}/2 objects placed, "
          f"canvas background verified, p=0/empty no-ops verified)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data yaml (same as the checkpoint being continued)")
    ap.add_argument("--weights", default="yolo26n.pt", help="checkpoint to continue from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--project", default="runs/sop")
    ap.add_argument("--name", default="sop")
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--optimizer", default=None,
                    help="e.g. MuSGD / SGD / AdamW. REQUIRED for --lr0 to take effect: "
                         "with the default optimizer=auto, ultralytics overrides lr0 "
                         "and momentum and picks its own (logs 'ignoring lr0').")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--lrf", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=None,
                    help="shorten for warm restarts from an already-trained ckpt")
    ap.add_argument("--sop-p", type=float, default=0.05,
                    help="SmallObjectPatches probability per training image (default 0.05, "
                         "matching YOLO-MIT's own production value)")
    ap.add_argument("--sop-target-size", type=int, default=96)
    ap.add_argument("--fraction", type=float, default=None,
                    help="cap the training set fraction, for smoke tests")
    ap.add_argument("--resume", metavar="CKPT",
                    help="resume an interrupted run from this checkpoint (last.pt), "
                         "continuing with SOPTrainer so SmallObjectPatches stays active "
                         "-- plain `yolo resume` would silently drop back to the stock "
                         "DetectionTrainer and lose the augmentation for the remaining epochs")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    from ultralytics import YOLO

    SOPTrainer.sop_p = args.sop_p
    SOPTrainer.sop_target_size = args.sop_target_size

    if args.resume:
        # resume=True makes ultralytics reload data/epochs/hyperparameters from the
        # checkpoint's own saved train_args -- passing them again here would be ignored
        # (and could silently diverge from what actually trained epochs 1-48).
        YOLO(args.resume).train(trainer=SOPTrainer, resume=True)
        return

    if not args.data:
        ap.error("--data required (or --selftest / --resume)")

    kw = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, project=args.project, name=args.name, device=0,
    )
    if args.freeze is not None:
        kw["freeze"] = args.freeze
    for k, v in (("optimizer", args.optimizer), ("lr0", args.lr0),
                 ("lrf", args.lrf), ("warmup_epochs", args.warmup_epochs),
                 ("fraction", args.fraction)):
        if v is not None:
            kw[k] = v
    if args.lr0 is not None and args.optimizer is None:
        ap.error("--lr0 without --optimizer: optimizer=auto ignores lr0 (see --help)")
    YOLO(args.weights).train(trainer=SOPTrainer, **kw)


if __name__ == "__main__":
    # Re-import self under the canonical module name, matching train_gradsuppress.py's
    # documented __main__-pickling discipline (see module docstring).
    import train_small_object_patches as _canonical

    sys.exit(_canonical.main())
