"""
Port of YOLO-MIT's SmallObjectPatches (yolo/tools/data_augmentation.py:968-1076)
to ultralytics' BaseTransform interface, for the yolo26s run.

Same algorithm: extract every object in the current training image, resize each
(aspect-preserved) to fit within target_size px, place them at random
non-overlapping positions on a target_size-canvas painted in the letterbox
grey (114,114,114) -- i.e. this REPLACES the whole training image with a
small-object composite, stochastically, per-sample, every epoch.

Ultralytics-side differences from the YOLO-MIT original (interface only, not
behavior): operates on labels['img'] / labels['instances'] / labels['cls']
instead of a raw (image, boxes) tuple; instances give pixel xyxy after
denormalize(); segments are dropped for repainted instances since a resized
crop has no meaningful polygon anymore (fine -- this dataset trains boxes only).
"""
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
from ultralytics.data.augment import BaseTransform
from ultralytics.utils.instance import Instances


class SmallObjectPatches(BaseTransform):
    def __init__(self, p: float = 0.05, target_size: int = 96, background_color=(114, 114, 114), max_attempts: int = 50):
        super().__init__()
        self.p = p
        self.target_size = target_size
        self.background_color = background_color
        self.max_attempts = max_attempts

    def __call__(self, labels: dict) -> dict:
        if random.random() >= self.p:
            return labels

        instances = labels["instances"]
        if len(instances) == 0:
            return labels

        img = labels["img"]
        h_img, w_img = img.shape[:2]
        target_size = self.target_size

        instances = deepcopy(instances)
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w_img, h_img)
        boxes = instances.bboxes  # (N,4) pixel xyxy
        cls = labels["cls"].reshape(-1)

        crops = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_h, crop_w = y2 - y1, x2 - x1
            scale = min(target_size / crop_w, target_size / crop_h)
            new_w, new_h = int(crop_w * scale), int(crop_h * scale)
            if new_w <= 0 or new_h <= 0:
                continue
            resized = cv2.resize(img[y1:y2, x1:x2], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            crops.append((cls[i], resized, new_w, new_h))

        if not crops:
            return labels

        canvas = np.full((h_img, w_img, 3), self.background_color, dtype=np.uint8)
        placed = np.empty((len(crops), 4), dtype=np.int32)
        placed_count = 0
        new_boxes, new_cls = [], []

        for cls_id, resized_crop, new_w, new_h in crops:
            max_x, max_y = w_img - new_w, h_img - new_h
            if max_x < 0 or max_y < 0:
                continue
            for _ in range(self.max_attempts):
                px, py = random.randint(0, max_x), random.randint(0, max_y)
                if placed_count > 0:
                    p_arr = placed[:placed_count]
                    if ((px < p_arr[:, 2]) & (px + new_w > p_arr[:, 0]) &
                        (py < p_arr[:, 3]) & (py + new_h > p_arr[:, 1])).any():
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
        labels["instances"] = Instances(np.array(new_boxes, dtype=np.float32), bbox_format="xyxy", normalized=False)
        labels["cls"] = np.array(new_cls, dtype=np.float32).reshape(-1, 1)
        return labels


def patch_v8_transforms(p: float = 0.05, target_size: int = 96):
    """Monkeypatch v8_transforms to append SmallObjectPatches at the end of the
    pipeline (after mosaic/copy_paste/perspective/mixup/hsv/flip), so it acts on
    the already-composed training sample, matching where YOLO-MIT applies it
    (post other augs, pre-Format).

    IMPORTANT: ultralytics/data/dataset.py does `from .augment import v8_transforms`
    -- a direct name import. build_transforms() (dataset.py:317) calls that
    already-bound local name, so patching ultralytics.data.augment.v8_transforms
    alone is a silent no-op at the real call site. Must patch the name inside
    ultralytics.data.dataset's own namespace instead.
    """
    import ultralytics.data.augment as aug_mod
    import ultralytics.data.dataset as dataset_mod

    orig_v8_transforms = aug_mod.v8_transforms

    def patched(dataset, imgsz, hyp):
        transforms = orig_v8_transforms(dataset, imgsz, hyp)
        transforms.append(SmallObjectPatches(p=p, target_size=target_size))
        return transforms

    aug_mod.v8_transforms = patched       # cosmetic consistency
    dataset_mod.v8_transforms = patched   # the actual call site (dataset.py:317)
