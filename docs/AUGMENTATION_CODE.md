# Augmentation — the actual code, per method

Every augmentation used by either training framework, quoted from the source that runs on the
pod. One entry per method, no repetition: where both frameworks implement the same idea, the
second entry shows only what differs.

Sources, both read on the pod 2026-08-16:

- **YOLO-MIT (v9 / gelan):** `/workspace/YOLO-MIT/yolo/tools/data_augmentation.py` (1,141 lines)
- **Ultralytics (YOLOv8 / YOLO11 / YOLO26 / YOLOE) 8.4.120:**
  `/usr/local/lib/python3.11/dist-packages/ultralytics/data/augment.py` (3,240 lines)

Which settings each run used is in `docs/TRAINING_CONFIG_HISTORY.md`. **ON/OFF markers below
refer to the production recipes** — the `*_gemlb_*` gelan runs and the stock-default
Ultralytics runs.

---

## 0. How the pipelines are assembled

**YOLO-MIT** — transforms are built from the config dict by name and run in listed order.
A scalar config value goes straight to the constructor's first arg (this is why
`Mosaic: 0.5` is a probability, and why `Mosaic=4096` means "always"):

```python
# yolo/tools/data_loader.py:155-166
def _create_composer(self, augment_cfg: DictConfig, disable_augs: bool):
    """Helper to build an AugmentationComposer."""
    transforms_list = []
    for aug, args in augment_cfg.items():
        # If we are disabling augs, check against the disable list
        if disable_augs and aug in self.disabled_augs:
            continue  # Skip this augmentation
        transform_obj = (
            getattr(augment, aug)(**args) if isinstance(args, DictConfig) else getattr(augment, aug)(args)
        )
```

```python
# yolo/tools/data_augmentation.py:27-48
def __call__(self, image: np.ndarray, boxes=torch.zeros(0, 5)):
    image = self.pre_size(image)  # presize for faster augmentations
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for transform in self.transforms:
        image, boxes = transform(image, boxes)

    image, boxes, rev_tensor = self.pad_resize(image, boxes)
    return image.transpose(2, 0, 1).copy(), boxes, rev_tensor
```

Boxes are `[cls, x1, y1, x2, y2]` **normalized corner** format throughout.

**Ultralytics** — the detection pipeline is fixed in code; only the probabilities/magnitudes
are configurable:

```python
# ultralytics/data/augment.py:2805-2846
mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
affine = RandomPerspective(
    degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale,
    shear=hyp.shear, perspective=hyp.perspective, size=(imgsz, imgsz), ...)

pre_transform = Compose([mosaic, affine])
if hyp.copy_paste_mode == "flip":
    pre_transform.insert(1, CopyPaste(dataset, p=hyp.copy_paste, mode=hyp.copy_paste_mode))
...
return Compose([
    pre_transform,
    MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
    CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
    Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None), flip_idx=flip_idx),
    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
    RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
    RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
])  # transforms
```

> **Two corrections to what our configs imply.**
> 1. **`erasing` and `auto_augment` are NOT in this list.** They appear only in
>    `classify_augmentations()` (augment.py:2899), the classification path. Every `args.yaml`
>    in this project records `erasing: 0.4` / `auto_augment: randaugment`, but for **detection
>    training they do nothing.**
> 2. `Albumentations` is instantiated unconditionally at `p=1.0` and would add
>    `Blur/MedianBlur/ToGray/CLAHE` at p=0.01 each — but its constructor is wrapped in
>    `except ImportError: pass` (augment.py:2135) and **albumentations is not installed on the
>    pod** (verified). So it is a no-op here.

---

## 1. Colour — HSV *(ON in both)*

Same idea, same values (`0.015 / 0.7 / 0.4`), two implementations. YOLO-MIT builds the LUT
from an absolute hue shift; Ultralytics samples one `uniform(-1, 1)` vector for all three.

```python
# YOLO-MIT data_augmentation.py:863-880
# Generate random gains
h_adj = np.random.uniform(-self.h_gain, self.h_gain) * 180
s_adj = np.random.uniform(1 - self.s_gain, 1 + self.s_gain)
v_adj = np.random.uniform(1 - self.v_gain, 1 + self.v_gain)

# Build LUT tables (256 entries each, uint8 — no full-image float32 allocation)
x = np.arange(256, dtype=np.float32)
lut_h = np.mod(x + h_adj, 180).astype(np.uint8)
lut_s = np.clip(x * s_adj, 0, 255).astype(np.uint8)
lut_v = np.clip(x * v_adj, 0, 255).astype(np.uint8)

img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(img_hsv)
img_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB), boxes
```

Ultralytics differs only in gain sampling and one guard:

```python
# ultralytics/data/augment.py:1477-1487
r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
x = np.arange(0, 256, dtype=r.dtype)
lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79
```

---

## 2. Horizontal / vertical flip *(HFlip ON in both; VFlip OFF in both)*

```python
# YOLO-MIT data_augmentation.py:221-243
class HorizontalFlip:
    def __call__(self, image, boxes):
        if random.random() < self.prob:
            image = cv2.flip(image, 1)          # horizontal
            if boxes.shape[0] > 0:
                boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes

class VerticalFlip:
    def __call__(self, image, boxes):
        if random.random() < self.prob:
            image = cv2.flip(image, 0)          # vertical
            if boxes.shape[0] > 0:
                boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes
```

Ultralytics splits image and boxes into two calls; the box maths lives in `Instances`:

```python
# ultralytics/data/augment.py:1565-1571 + 1585-1591
img = np.flipud(img) if params["direction"] == "vertical" else np.fliplr(img)
...
instances.flipud(params["h"]) / instances.fliplr(params["w"])
```

---

## 3. Geometry

### 3a. `RandomZoom` — YOLO-MIT's scale aug *(ON, always: `RandomZoom: 1`)*

Crop when `scale < 1`, pad when `scale > 1`. Note the asymmetry: **zoom-in picks a random
offset (a real random crop), zoom-out is always centred.**

```python
# YOLO-MIT data_augmentation.py:759-829
scale = np.random.uniform(self.scale_range[0], self.scale_range[1])   # default (0.5, 1.5)
new_height = max(1, int(original_height * scale))
new_width = max(1, int(original_width * scale))

if scale < 1.0:
    # --- ZOOM IN (Crop) ---
    top = random.randint(0, original_height - new_height)
    left = random.randint(0, original_width - new_width)
    image_out = image[top : top + new_height, left : left + new_width]

    if boxes.shape[0] > 0:
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
        boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_width)
        boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, new_height)
        boxes[:, [1, 3]] /= new_width
        boxes[:, [2, 4]] /= new_height
else:
    # --- ZOOM OUT (Pad) ---
    image_out = np.full((new_height, new_width, 3), self.fill_color, dtype=np.uint8)
    top = (new_height - original_height) // 2
    left = (new_width - original_width) // 2
    image_out[top : top + original_height, left : left + original_width] = image
    ...
    boxes = filter_boxes(boxes, old_boxes)   # remove extreme boxes
```

### 3b. `RandomPerspective` — Ultralytics' geometry *(ON: only `scale 0.5` and `translate 0.1` are non-zero)*

One composed 3×3 matrix. With our settings `degrees`, `shear`, `perspective` are all `0`, so
`P`, `S` are identity and `R` is pure scale — i.e. **this is our only scale/translate aug**.

```python
# ultralytics/data/augment.py:1126-1158
C = np.eye(3, dtype=np.float32)          # Center
C[0, 2] = -img.shape[1] / 2
C[1, 2] = -img.shape[0] / 2

P = np.eye(3, dtype=np.float32)          # Perspective
P[2, 0] = random.uniform(-self.perspective, self.perspective)
P[2, 1] = random.uniform(-self.perspective, self.perspective)

R = np.eye(3, dtype=np.float32)          # Rotation and Scale
a = random.uniform(-self.degrees, self.degrees)
s = random.uniform(1 - self.scale, 1 + self.scale)
R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

S = np.eye(3, dtype=np.float32)          # Shear
S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

T = np.eye(3, dtype=np.float32)          # Translation
T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]
T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]

M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
```

**The two differ in kind:** `RandomZoom` samples scale symmetrically around 1.0 and crops/pads
the *whole* frame; `RandomPerspective` warps about the image centre and always outputs
`(imgsz, imgsz)`.

### 3c. `RandomRotate90` — YOLO-MIT only *(OFF: `0` in every production run)*

Exists because 90° steps need no interpolation and the box maths is exact:

```python
# YOLO-MIT data_augmentation.py:253-268 (k == 1 branch; 180°/270° are the same shape)
k = random.randint(1, 3)               # 1=90°, 2=180°, 3=270°
boxes_old = boxes.clone()
if k == 1:
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if boxes.shape[0] > 0:
        boxes[:, 1] = 1 - boxes_old[:, 4]   # new_x_min = 1 - old_y_max
        boxes[:, 2] = boxes_old[:, 1]       # new_y_min = old_x_min
        boxes[:, 3] = 1 - boxes_old[:, 2]   # new_x_max = 1 - old_y_min
        boxes[:, 4] = boxes_old[:, 3]       # new_y_max = old_x_max
```

*(Ultralytics has no equivalent; `degrees: 90` in `dataset_v2_random_rotate_nomosaic/yolo11N-416`
went through the continuous `RandomPerspective` rotation instead.)*

---

## 4. Mosaic *(OFF in every gelan run; ON at 1.0 in every Ultralytics run)*

YOLO-MIT builds a fixed 2×2 grid where each quadrant is **resized to cover, then randomly
panned** — the mosaic centre is not jittered:

```python
# YOLO-MIT data_augmentation.py:345-383
mosaic_image = np.full((2 * img_sz, 2 * img_sz, 3), self.background_color, dtype=np.uint8)
quadrant_coords = [(0, 0), (img_sz, 0), (0, img_sz), (img_sz, img_sz)]

for (img, bxs), (paste_x, paste_y) in zip(data, quadrant_coords):
    h_img, w_img = img.shape[:2]
    scale = max(img_sz / h_img, img_sz / w_img)      # resize to "cover" the quadrant
    img_resized = cv2.resize(img, (int(np.ceil(w_img*scale)), int(np.ceil(h_img*scale))), ...)

    h_diff = max(0, new_h - img_sz); w_diff = max(0, new_w - img_sz)
    y_offset = random.randint(0, h_diff) if h_diff > 0 else 0     # random "pan"
    x_offset = random.randint(0, w_diff) if w_diff > 0 else 0

    img_cropped = img_resized[y_offset : y_offset + img_sz, x_offset : x_offset + img_sz]
    mosaic_image[paste_y : paste_y + img_sz, paste_x : paste_x + img_sz] = img_cropped
```

Ultralytics instead jitters the **shared centre** `(xc, yc)` and lets each tile be clipped by
the canvas edge — which is what makes objects fall partly outside the frame:

```python
# ultralytics/data/augment.py:523-534
if i == 0:  # top left
    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
elif i == 1:  # top right
    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
elif i == 2:  # bottom left
    ...
padw = x1a - x1b
padh = y1a - y1b
```

**Turning mosaic off late in training** — the two frameworks do the same thing by different
routes. Ultralytics (`close_mosaic: 10`, the stock default we always used) rebuilds the
dataloader for the last N epochs:

```python
# ultralytics/engine/trainer.py:454-456, 1086-1092
if epoch == (self.epochs - self.args.close_mosaic):
    self._close_dataloader_mosaic()
    self.train_loader.reset()
...
def _close_dataloader_mosaic(self):
    """Update dataloaders to stop using mosaic augmentation."""
    if hasattr(self.train_loader.dataset, "mosaic"):
        self.train_loader.dataset.mosaic = False
```

YOLO-MIT (`close_aug`, set to **0 = never** in all current runs) pre-builds a second composer
and swaps the pointer:

```python
# yolo/tools/data_loader.py:57-59, 244-253
self.disabled_augs = ["Mosaic", "MixUp", "CutMix", "VerticalFlip", "CopyPaste"]
self.full_transform    = self._create_composer(augment_cfg, disable_augs=False)
self.reduced_transform = self._create_composer(augment_cfg, disable_augs=True)
...
def set_augmentations(self, enable: bool):
    self.transform = self.full_transform if enable else self.reduced_transform
```

---

## 5. MixUp *(OFF everywhere)*

```python
# YOLO-MIT data_augmentation.py:476-491
lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

# Visibility checks — drop the labels of whichever image is too faint to see
if (1.0 - lam) < self.min_visibility:
    boxes2 = torch.zeros((0, 5))
if lam < self.min_visibility:
    boxes = torch.zeros((0, 5))

mixed_image = image.astype(np.float32) * lam + image2_rgb.astype(np.float32) * (1.0 - lam)
mixed_image = np.clip(mixed_image, 0, 255).astype(np.uint8)
merged_boxes = torch.cat((boxes, boxes2), dim=0)
```

The `min_visibility` label-dropping is YOLO-MIT's own addition; Ultralytics keeps all labels
from both images at any `lam`.

## 6. CutMix *(OFF everywhere)*

Paste a rectangle from a second image, then reassign labels **by box centre**:

```python
# YOLO-MIT data_augmentation.py:518-547
lam = np.random.beta(self.beta, self.beta)
cut_rat = np.sqrt(1.0 - lam)
cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
cx, cy = np.random.randint(w), np.random.randint(h)
x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)

image[y1:y2, x1:x2] = image2_rgb[y1:y2, x1:x2]

# drop image-1 boxes whose centre is now covered
mask_covered = (box_centers_x > x1) & (box_centers_x < x2) & (box_centers_y > y1) & (box_centers_y < y2)
boxes = boxes[~mask_covered]
# keep image-2 boxes whose centre landed inside the patch, clamped to it
boxes2_keep = boxes2[mask_inside]
boxes2_keep[:, 1] = boxes2_keep[:, 1].clamp(min=x1 / w, max=x2 / w)
```

## 7. CopyPaste *(OFF in current runs; used at 0.05–0.5 in the Nov-2025 `yolos-*` Ultralytics runs)*

YOLO-MIT pastes **box crops** (no masks), each independently flipped and rescaled:

```python
# YOLO-MIT data_augmentation.py:663-700
indices = torch.randperm(src_boxes.shape[0])
for idx in indices:
    if random.random() > self.paste_prob:
        continue
    x1, y1, x2, y2 = src_boxes_pixel[idx, 1:].int().tolist()
    patch = src_img[y1:y2, x1:x2]

    if random.random() > 0.5:
        patch = cv2.flip(patch, 1)
    scale = random.uniform(*self.scale_range)
    if scale != 1.0:
        patch = cv2.resize(patch, (new_p_w, new_p_h))

    px = random.randint(0, w_target - p_w)
    py = random.randint(0, h_target - p_h)
```

Ultralytics' `CopyPaste` requires **segmentation masks** and its `flip` mode mirrors an
instance onto the opposite side of the same image — a different operation despite the shared
name.

---

## 8. The two custom augmentations that exist only in YOLO-MIT

### 8a. `MaskSmallBoxes` *(ON; `min_area_px` was 7000 → 2048 → 1024 → 30 → 0 over the project)*

Grey-fills every box under an **area** threshold **and deletes its label** — so the model is
never asked to find it, and never punished for the pixels either:

```python
# YOLO-MIT data_augmentation.py:918-965
w_box = (boxes[:, 3] - boxes[:, 1]) * w_img
h_box = (boxes[:, 4] - boxes[:, 2]) * h_img
is_small = (w_box * h_box) < self.min_area_px          # AREA in px², not a side length

small_coords = boxes[is_small, 1:5]
small_coords[:, 0] *= w_img;  small_coords[:, 1] *= h_img
small_coords[:, 2] *= w_img;  small_coords[:, 3] *= h_img
coords_int = small_coords.long()
coords_int[:, 0].clamp_(0, w_img); ...

mean_color = 114
for x1, y1, x2, y2 in coords_np:
    if x2 > x1 and y2 > y1:
        image[y1:y2, x1:x2] = mean_color

keep_boxes = boxes[~is_small]                          # the label is removed too
return image, keep_boxes
```

`min_area_px: 7000` ≈ an 84×84 box — the Dec-2025 runs were deliberately erasing a large
share of small people. Note the fill is the literal constant `114`, despite the local name
`mean_color`.

### 8b. `SmallObjectPatches` *(ON at `p=0.05` in every run since 2026-01)*

Replaces the whole image with a grey canvas holding **every object rescaled to fit 96 px**,
placed at non-overlapping random positions — the training-time mirror of the extension's
"upscale a small region before inference" path:

```python
# YOLO-MIT data_augmentation.py:1012-1076
for i in range(n_boxes):                       # 1. extract + rescale every object
    crop_h, crop_w = y2 - y1, x2 - x1
    scale = min(target_size / crop_w, target_size / crop_h)      # target_size = 96
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    resized = cv2.resize(image[y1:y2, x1:x2], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    crops.append((boxes_np[i, 0], resized, new_w, new_h))

canvas = np.full((h_img, w_img, 3), self.background_color, dtype=np.uint8)

for cls_id, resized_crop, new_w, new_h in crops:                 # 2. place without overlap
    for _ in range(max_attempts):                                # max_attempts = 50
        px, py = random.randint(0, max_x), random.randint(0, max_y)
        if placed_count > 0:
            p = placed[:placed_count]
            if ((px < p[:, 2]) & (px + new_w > p[:, 0]) & (py < p[:, 3]) & (py + new_h > p[:, 1])).any():
                continue
        canvas[py : py + new_h, px : px + new_w] = resized_crop
        new_boxes.append([cls_id, px / w_img, py / h_img, (px + new_w) / w_img, (py + new_h) / h_img])
        break

return canvas, torch.tensor(new_boxes, dtype=boxes.dtype, device=boxes.device)
```

Because `scale = min(96/w, 96/h)`, objects **larger** than 96 px are scaled *down*: at `p=0.05`,
one image in twenty becomes a collage of ~96 px objects on grey.

---

## 9. Class masking — how "unknown" people are erased *(gelan `mask_classes: [3]`)*

Not an augmentation in the config list; it runs in the dataset before augmentation
(`data_loader.py` `_setup_class_masking`), and it is the mechanism behind the grey-masked
unknowns that lead crowd recall:

```python
# YOLO-MIT data_augmentation.py:85-104
def apply_region_mask(image, regions, color=(114, 114, 114)):
    """Paints the given regions of an image with a solid color, in-place."""
    if not regions:
        return image
    if np.isscalar(color):
        # cv2.fillPoly reads a scalar as Scalar(c, 0, 0), so expand it per channel
        color = (color,) * (image.shape[2] if image.ndim == 3 else 1)

    h_img, w_img = image.shape[:2]
    scale = np.array([w_img, h_img], dtype=np.float32)
    for region in regions:
        points = np.asarray(region, dtype=np.float32).reshape(-1, 2) * scale
        if points.shape[0] > 2:
            polygons.append(np.round(points).astype(np.int32))   # M > 2 → polygon
            continue
        (x1, y1), (x2, y2) = np.round(points).astype(np.int32)   # M == 2 → rectangle
```

With `use_polygon: true` the Spotlight segmentation polygon is filled, not the box — so the
person is erased at their own silhouette and the surrounding background survives. Ultralytics
has no equivalent; that is exactly the gap `train_gradsuppress.py` fills for YOLO26n by
zeroing the class gradient instead of painting the pixels.

---

## Summary — what is actually applied in production

| method | gelan `*_gemlb_*` | Ultralytics `y26n_*` / `yolo11N-640` |
|---|---|---|
| HSV 0.015/0.7/0.4 | ✅ `RandomHSV` | ✅ `RandomHSV` |
| horizontal flip 0.5 | ✅ `HorizontalFlip` | ✅ `RandomFlip(horizontal)` |
| scale / zoom | ✅ `RandomZoom: 1`, range (0.5, 1.5) | ✅ `RandomPerspective(scale=0.5)` |
| translate | — (crop offset only) | ✅ `RandomPerspective(translate=0.1)` |
| mosaic | ❌ `Mosaic: 0` | ✅ `p=1.0`, closed for last 10 epochs |
| vertical flip / rotate / shear / perspective | ❌ | ❌ (all `0`) |
| mixup / cutmix / copy-paste | ❌ | ❌ (all `0`) |
| mask small boxes | ✅ `MaskSmallBoxes` | ❌ none |
| small-object collage | ✅ `SmallObjectPatches: 0.05` | ❌ none |
| class grey-masking | ✅ `mask_classes: [3]` | ❌ (gradient suppression instead) |
| random erasing / randaugment | ❌ | ❌ **— config says `0.4`/`randaugment`, but detection ignores both** |
