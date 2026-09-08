#!/usr/bin/env python3
"""
HARAMBLUR — Dataset Diversity Audit (one-time spike)
====================================================

Quantifies the variance of the Person training set so we have a confidence
signal about coverage and gaps. Designed to run *inside the POD* where the
Open Images V7 volume is mounted at /workspace/open-images-v7/.

It crops every `Person` box (full body, partial body, body part, or face — all
count as one class for HaramBlur) and measures, per crop:

  * skin tone        -> ITA degrees -> Fitzpatrick-style band   (numpy/cv2)
  * scale / distance -> bbox area / image area                  (geometry)
  * capture quality  -> brightness, sharpness, resolution       (numpy/cv2)
  * view angle       -> frontal / profile / back                (CLIP, optional)
  * dress code       -> modest / casual / revealing / ...       (CLIP, optional)

Per dimension it reports the distribution + an imbalance score (normalized
entropy and Gini), then rolls everything up into one "diversity confidence"
number and a ranked list of under-represented buckets.

Ethnicity/nationality is deliberately NOT inferred per face — that is unreliable
and sensitive. Skin-tone distribution + (optional) dataset source metadata are
the diversity proxies instead.

Usage
-----
    python diversity_audit.py --yaml /workspace/open-images-v7/dataset.yaml \
        --split val --max-images 4000 --out ./audit_report

    # skip the CLIP-based dimensions (no torch / want it fast):
    python diversity_audit.py --yaml .../dataset.yaml --no-clip

Outputs (in --out dir): report.md, metrics.json, and one PNG chart per dimension.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: opencv-python. Install with requirements-audit.txt")

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: PyYAML. Install with requirements-audit.txt")


# ---------------------------------------------------------------------------
# Bucket definitions & CLIP prompts  (edit these freely — this is a spike)
# ---------------------------------------------------------------------------

# Fitzpatrick-style bands from ITA degrees (higher ITA = lighter skin).
FITZPATRICK_BANDS = [
    ("very_light", 55.0, math.inf),
    ("light", 41.0, 55.0),
    ("intermediate", 28.0, 41.0),
    ("tan", 19.0, 28.0),
    ("brown", 10.0, 19.0),
    ("dark", -math.inf, 10.0),
]

# Scale buckets as fraction of image area (proxy for distance, with caveats).
SCALE_BANDS = [
    ("tiny", 0.0, 0.01),       # far / crowd
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.20),
    ("large", 0.20, 0.50),
    ("dominant", 0.50, 1.01),  # close-up / fills frame
]

BRIGHTNESS_BANDS = [
    ("very_dark", 0, 50),
    ("dark", 50, 100),
    ("normal", 100, 160),
    ("bright", 160, 210),
    ("very_bright", 210, 256),
]

SHARPNESS_BANDS = [  # variance of Laplacian
    ("very_blurry", 0, 50),
    ("blurry", 50, 150),
    ("acceptable", 150, 500),
    ("sharp", 500, 1e9),
]

RESOLUTION_BANDS = [  # min(side) of the crop in pixels
    ("xs", 0, 32),
    ("s", 32, 64),
    ("m", 64, 128),
    ("l", 128, 256),
    ("xl", 256, 1e9),
]

VIEW_ANGLE_PROMPTS = {
    "frontal": "a photo of a person facing the camera",
    "profile": "a photo of a person seen from the side",
    "back": "a photo of a person seen from behind",
}

# Descriptive, neutral clothing buckets relevant to HaramBlur. Tune as needed.
DRESS_CODE_PROMPTS = {
    "fully_covered": "a person fully covered in modest clothing",
    "casual": "a person in casual everyday clothing",
    "formal": "a person in formal or business clothing",
    "athletic": "a person in athletic or sportswear",
    "revealing": "a person in swimwear or revealing clothing",
    "face_only": "a close-up photograph of a human face",
}


def band_of(value: float, bands) -> str:
    for name, lo, hi in bands:
        if lo <= value < hi:
            return name
    return bands[-1][0]


# ---------------------------------------------------------------------------
# Dataset loading  (auto-detects YOLO-txt or COCO-json layouts)
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_path: Path
    # list of person boxes in pixel xyxy; filled lazily for COCO, eagerly for txt
    boxes_xyxy: list = field(default_factory=list)
    img_w: int = 0
    img_h: int = 0


def load_dataset_yaml(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"Could not parse dataset yaml: {yaml_path}")
    return cfg


def class_names(cfg: dict) -> dict:
    """Return {class_id: name}. Handles `names:` (dict or list) and `class_list:`.

    For HaramBlur every class is a person type (Woman/Man/Child), so all of them
    are detection targets — there is no single 'Person' class to filter to.
    """
    classes = cfg.get("names") or cfg.get("class_list")
    if isinstance(classes, dict):              # {0: 'Woman', 1: 'Man', ...}
        return {int(k): str(v) for k, v in classes.items()}
    if isinstance(classes, list):              # ['Woman', 'Man', ...]
        return {i: str(v) for i, v in enumerate(classes)}
    return {0: "person"}


def resolve_root(cfg: dict, yaml_path: Path) -> Path:
    p = cfg.get("path")
    if p:
        root = Path(p)
        return root if root.is_absolute() else (yaml_path.parent / root)
    return yaml_path.parent


def _first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def discover_split(cfg: dict, root: Path, split_key: str):
    """Return (kind, image_dir, label_source). kind in {'yolo','coco'}.

    The yaml's train/val value IS the image dir relative to `path`
    (e.g. `train: images/train`), per Ultralytics convention.
    """
    split_val = cfg.get(split_key, split_key)  # e.g. 'images/train'

    # YOLO-txt layouts -------------------------------------------------------
    img_candidates = [
        root / split_val,                 # path/images/train  (this dataset)
        root / "images" / split_val,
        root / split_val / "images",
    ]
    for img_dir in img_candidates:
        if img_dir.is_dir():
            # Ultralytics convention: swap 'images' -> 'labels' in the path.
            lbl_dir = _first_existing(
                Path(str(img_dir).replace(f"{os.sep}images{os.sep}",
                                          f"{os.sep}labels{os.sep}", 1)),
                Path(str(img_dir).replace("images", "labels")),
                img_dir.parent / "labels" / img_dir.name,
                root / "labels" / Path(split_val).name,
            )
            if lbl_dir:
                return "yolo", img_dir, lbl_dir

    # COCO-json layout (fallback) -------------------------------------------
    img_dir = _first_existing(root / split_val, root / "images" / split_val)
    ann = _first_existing(
        root / "annotations" / f"instances_{Path(split_val).name}.json",
        root / "annotations" / f"{Path(split_val).name}.json",
    )
    if img_dir and ann:
        return "coco", img_dir, ann

    sys.exit(
        f"Could not locate images+labels for split '{split_val}' under {root}.\n"
        f"Tried YOLO and COCO layouts. Adjust the script's discover_split() to match the volume."
    )


def list_images(img_dir: Path):
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def label_path_for(img_path: Path, img_dir: Path, lbl_dir: Path) -> Path:
    rel = img_path.relative_to(img_dir).with_suffix(".txt")
    return lbl_dir / rel


def yolo_boxes(label_file: Path, w: int, h: int):
    """Parse normalized YOLO txt -> list of (class_id, x1,y1,x2,y2) in pixels.

    Every class is a person type here, so we keep all of them.
    """
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def load_coco_index(ann_file: Path):
    """Return ({image_filename: [(class_id, xyxy),...]}, {id:name}) for all categories."""
    with open(ann_file) as f:
        coco = json.load(f)
    catid_to_name = {c["id"]: str(c["name"]) for c in coco["categories"]}
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    out = {}
    for a in coco["annotations"]:
        x, y, bw, bh = a["bbox"]
        out.setdefault(id_to_file[a["image_id"]], []).append(
            (a["category_id"], x, y, x + bw, y + bh)
        )
    return out, catid_to_name


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def estimate_skin_ita(crop_bgr: np.ndarray):
    """Return ITA degrees over detected skin pixels, or None if too few."""
    if crop_bgr.size == 0:
        return None
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    if skin.sum() < 50:  # not enough skin pixels to trust
        return None
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * 100.0 / 255.0           # L* in [0,100]
    b = lab[:, :, 2] - 128.0                    # b* in [-128,127]
    Ls, bs = L[skin], b[skin]
    # robust: use medians to resist specular highlights / shadows
    L_med, b_med = float(np.median(Ls)), float(np.median(bs))
    if b_med == 0:
        return None
    return math.degrees(math.atan((L_med - 50.0) / b_med))


def estimate_quality(crop_bgr: np.ndarray):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


# ---- optional CLIP dimensions --------------------------------------------

class ClipScorer:
    """Lazy CLIP zero-shot scorer for view angle + dress code."""

    def __init__(self, device: str | None = None):
        import torch
        import open_clip
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._text_cache = {}

    def _text_feats(self, key: str, prompts: dict):
        if key not in self._text_cache:
            toks = self.tokenizer(list(prompts.values())).to(self.device)
            with self.torch.no_grad():
                tf = self.model.encode_text(toks)
                tf /= tf.norm(dim=-1, keepdim=True)
            self._text_cache[key] = (list(prompts.keys()), tf)
        return self._text_cache[key]

    def classify_multi(self, crops_rgb: list, prompt_sets: dict):
        """Encode the image batch ONCE, score it against several prompt sets.

        prompt_sets: {dim_key: {bucket: prompt}}
        returns      {dim_key: [(bucket, confidence) per crop]}
        """
        from PIL import Image
        imgs = self.torch.stack(
            [self.preprocess(Image.fromarray(c)) for c in crops_rgb]
        ).to(self.device)
        with self.torch.no_grad():
            vf = self.model.encode_image(imgs)
            vf /= vf.norm(dim=-1, keepdim=True)
            out = {}
            for key, prompts in prompt_sets.items():
                labels, tf = self._text_feats(key, prompts)
                sims = (vf @ tf.T).softmax(dim=-1)
                conf, idx = sims.max(dim=-1)
                idx = idx.cpu().numpy()
                conf = conf.cpu().numpy()
                out[key] = [(labels[i], float(c)) for i, c in zip(idx, conf)]
        return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalized_entropy(counts: Counter, all_buckets: list) -> float:
    total = sum(counts.values())
    if total == 0 or len(all_buckets) <= 1:
        return 0.0
    h = 0.0
    for b in all_buckets:
        p = counts.get(b, 0) / total
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(len(all_buckets))  # 1.0 = perfectly uniform


def gini(counts: Counter, all_buckets: list) -> float:
    vals = np.array([counts.get(b, 0) for b in all_buckets], dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    vals = np.sort(vals)
    n = len(vals)
    cum = np.cumsum(vals)
    return float((n + 1 - 2 * (cum / total).sum()) / n)


# Populated at runtime from the dataset yaml (e.g. ['Woman','Man','Child']).
CLASS_BUCKETS: list = []


def buckets_for(dim: str):
    return {
        "class_name": CLASS_BUCKETS,
        "skin_tone": [b[0] for b in FITZPATRICK_BANDS],
        "scale": [b[0] for b in SCALE_BANDS],
        "brightness": [b[0] for b in BRIGHTNESS_BANDS],
        "sharpness": [b[0] for b in SHARPNESS_BANDS],
        "resolution": [b[0] for b in RESOLUTION_BANDS],
        "view_angle": list(VIEW_ANGLE_PROMPTS.keys()),
        "dress_code": list(DRESS_CODE_PROMPTS.keys()),
    }[dim]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_bar_chart(counts: Counter, all_buckets: list, title: str, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    vals = [counts.get(b, 0) for b in all_buckets]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(all_buckets, vals, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("crops")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


class DebugCollector:
    """Reservoir-samples a few example crops per (dimension, bucket) so a human
    can eyeball whether the estimated buckets are actually correct."""

    def __init__(self, max_per_bucket: int, thumb: int = 150):
        self.max = max_per_bucket
        self.thumb = thumb
        self.samples: dict = {}   # (dim, bucket) -> list[(thumb_bgr, caption)]
        self.seen: dict = {}      # (dim, bucket) -> count seen

    def add(self, dim: str, bucket: str, crop_bgr, caption: str = ""):
        if self.max <= 0 or crop_bgr.size == 0:
            return
        key = (dim, bucket)
        n = self.seen.get(key, 0)
        self.seen[key] = n + 1
        thumb = self._fit(crop_bgr)
        bucket_list = self.samples.setdefault(key, [])
        if len(bucket_list) < self.max:
            bucket_list.append((thumb, caption))
        else:  # reservoir replacement keeps the sample representative
            j = random.randint(0, n)
            if j < self.max:
                bucket_list[j] = (thumb, caption)

    def _fit(self, crop_bgr):
        h, w = crop_bgr.shape[:2]
        s = self.thumb / max(h, w)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(crop_bgr, (nw, nh))
        canvas = np.zeros((self.thumb, self.thumb, 3), dtype=np.uint8)
        y0, x0 = (self.thumb - nh) // 2, (self.thumb - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def montage(self, dim: str, bucket: str):
        key = (dim, bucket)
        items = self.samples.get(key, [])
        if not items:
            return None
        cols = min(4, len(items))
        rows = math.ceil(len(items) / cols)
        cap_h = 22
        cell = self.thumb + cap_h
        sheet = np.full((rows * cell, cols * self.thumb, 3), 30, dtype=np.uint8)
        for i, (thumb, caption) in enumerate(items):
            r, c = divmod(i, cols)
            y, x = r * cell, c * self.thumb
            sheet[y:y + self.thumb, x:x + self.thumb] = thumb
            if caption:
                cv2.putText(sheet, caption, (x + 4, y + self.thumb + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1,
                            cv2.LINE_AA)
        return sheet


def write_debug(out: Path, collector: DebugCollector, dist: dict):
    """Write one montage per (dim, bucket) + an index for quick review."""
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    lines = ["# Debug montages — verify the estimated buckets\n",
             "Each image shows example crops the audit placed in that bucket. "
             "Captions show the measured value (ITA°, % area, brightness, "
             "sharpness, px, or CLIP confidence).\n"]
    for dim, counts in dist.items():
        lines.append(f"## {dim}\n")
        total = sum(counts.values()) or 1
        for bucket in buckets_for(dim):
            sheet = collector.montage(dim, bucket)
            if sheet is None:
                continue
            fname = f"{dim}__{bucket}.jpg"
            cv2.imwrite(str(dbg / fname), sheet)
            c = counts.get(bucket, 0)
            lines.append(f"### `{bucket}` — {c} crops ({100*c/total:.1f}%)")
            lines.append(f"![{dim} {bucket}](./{fname})\n")
    (dbg / "index.md").write_text("\n".join(lines))
    return dbg


def write_report(out: Path, summary: dict, dist: dict, charts: dict, meta: dict):
    lines = []
    lines.append("# HARAMBLUR — Dataset Diversity Audit\n")
    lines.append(f"- Dataset yaml: `{meta['yaml']}`")
    lines.append(f"- Split: **{meta['split']}**")
    lines.append(f"- Images scanned: **{meta['images_scanned']}**")
    lines.append(f"- Person crops analyzed: **{meta['crops']}**")
    lines.append(f"- CLIP dimensions: **{'on' if meta['clip'] else 'off'}**\n")

    if meta.get("debug_dir"):
        lines.append("> 🔍 **Verify the estimated buckets:** see "
                     "[`debug/index.md`](debug/index.md) for example crops per bucket.\n")

    conf = summary["diversity_confidence"]
    lines.append(f"## Overall diversity confidence: **{conf:.2f} / 1.00**")
    lines.append(
        "_Mean normalized entropy across measured dimensions "
        "(1.00 = perfectly balanced, 0 = single bucket). Treat as a relative signal, not an absolute grade._\n"
    )

    lines.append("## Per-dimension balance\n")
    lines.append("| Dimension | Balance (norm. entropy) | Gini | Most under-represented bucket |")
    lines.append("|---|---|---|---|")
    for dim, s in summary["dimensions"].items():
        lines.append(
            f"| {dim} | {s['entropy']:.2f} | {s['gini']:.2f} | "
            f"`{s['weakest_bucket']}` ({s['weakest_pct']:.1f}%) |"
        )
    lines.append("")

    lines.append("## Ranked coverage gaps\n")
    for i, g in enumerate(summary["gaps"], 1):
        lines.append(f"{i}. **{g['dimension']} / `{g['bucket']}`** — {g['pct']:.1f}% of crops")
    lines.append("")

    lines.append("## Distributions\n")
    for dim, counts in dist.items():
        lines.append(f"### {dim}")
        if dim in charts:
            lines.append(f"![{dim}]({charts[dim]})")
        total = sum(counts.values()) or 1
        for b in buckets_for(dim):
            c = counts.get(b, 0)
            lines.append(f"- `{b}`: {c} ({100*c/total:.1f}%)")
        lines.append("")

    lines.append("## Caveats\n")
    lines.append("- **Scale ≠ literal distance**: a close-up face and a far full body both shift this; read as relative size mix.")
    lines.append("- **Skin tone** is an ITA estimate over heuristic skin pixels; crops without visible skin are excluded.")
    lines.append("- **Ethnicity/nationality not inferred** — out of scope by design; use skin-tone + source metadata.")
    lines.append("- **CLIP dress/view** are zero-shot estimates; tune the prompts in the script for your taxonomy.")

    (out / "report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def iter_samples(kind, img_dir, label_src, names, max_images, seed):
    """Yield (img, [(class_name, x1,y1,x2,y2), ...], w, h)."""
    imgs = list(list_images(img_dir))
    random.Random(seed).shuffle(imgs)
    if max_images:
        imgs = imgs[:max_images]

    coco_index = coco_names = None
    if kind == "coco":
        coco_index, coco_names = load_coco_index(label_src)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if kind == "yolo":
            raw = yolo_boxes(label_path_for(img_path, img_dir, label_src), w, h)
            boxes = [(names.get(c, str(c)), x1, y1, x2, y2)
                     for (c, x1, y1, x2, y2) in raw]
        else:
            raw = coco_index.get(img_path.name, [])
            boxes = [(coco_names.get(c, str(c)), x1, y1, x2, y2)
                     for (c, x1, y1, x2, y2) in raw]
        if boxes:
            yield img, boxes, w, h


def main():
    ap = argparse.ArgumentParser(description="HaramBlur dataset diversity audit")
    ap.add_argument("--yaml", required=True, help="path to dataset.yaml")
    ap.add_argument("--split", default="val", help="split key in yaml (train/val)")
    ap.add_argument("--max-images", type=int, default=4000, help="0 = all (slow)")
    ap.add_argument("--max-crops", type=int, default=20000, help="cap analyzed crops")
    ap.add_argument("--min-crop-px", type=int, default=16, help="skip tinier boxes")
    ap.add_argument("--out", default="./audit_report")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-clip", action="store_true", help="skip view-angle & dress-code")
    ap.add_argument("--clip-batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="force CLIP device, e.g. cuda / cpu")
    ap.add_argument("--debug-samples", type=int, default=12,
                    help="example crops saved per bucket for visual verification (0 = off)")
    args = ap.parse_args()

    yaml_path = Path(args.yaml).expanduser()
    cfg = load_dataset_yaml(yaml_path)
    root = resolve_root(cfg, yaml_path)
    names = class_names(cfg)
    CLASS_BUCKETS[:] = [names[k] for k in sorted(names)]
    kind, img_dir, label_src = discover_split(cfg, root, args.split)
    print(f"[audit] layout={kind} images={img_dir} labels={label_src}")
    print(f"[audit] classes={CLASS_BUCKETS}")

    clip = None
    if not args.no_clip:
        try:
            clip = ClipScorer(device=args.device)
            print(f"[audit] CLIP ready on {clip.device}")
            if clip.device == "cpu":
                print("[audit] WARNING: CLIP on CPU is slow (expect 1h+ for 8k images). "
                      "Use --no-clip for a fast core pass, or fix CUDA for the full run.")
        except Exception as e:
            print(f"[audit] CLIP unavailable ({e}); continuing without view/dress dims")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dist = {d: Counter() for d in
            ["class_name", "skin_tone", "scale", "brightness", "sharpness", "resolution"]}
    if clip:
        dist["view_angle"] = Counter()
        dist["dress_code"] = Counter()

    debug = DebugCollector(args.debug_samples)
    clip_buf = []  # crops (rgb) awaiting CLIP

    def flush_clip():
        if not clip or not clip_buf:
            return
        results = clip.classify_multi(
            clip_buf,
            {"view_angle": VIEW_ANGLE_PROMPTS, "dress_code": DRESS_CODE_PROMPTS},
        )
        for key, preds in results.items():
            dist[key].update(b for b, _ in preds)
            for crop_rgb, (bucket, conf) in zip(clip_buf, preds):
                debug.add(key, bucket, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                          f"p={conf:.2f}")
        clip_buf.clear()

    n_imgs = n_crops = 0
    for img, boxes, w, h in iter_samples(kind, img_dir, label_src, names,
                                         args.max_images, args.seed):
        n_imgs += 1
        img_area = float(w * h)
        for (cls_name, x1, y1, x2, y2) in boxes:
            if n_crops >= args.max_crops:
                break
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            if xi2 - xi1 < args.min_crop_px or yi2 - yi1 < args.min_crop_px:
                continue
            crop = img[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue
            n_crops += 1

            # labeled class axis (real label, not estimated)
            dist["class_name"][cls_name] += 1
            debug.add("class_name", cls_name, crop)

            # geometry / quality (cheap, always on)
            frac = ((x2 - x1) * (y2 - y1)) / img_area
            b_scale = band_of(frac, SCALE_BANDS)
            dist["scale"][b_scale] += 1
            debug.add("scale", b_scale, crop, f"{frac*100:.1f}%")

            minside = min(crop.shape[:2])
            b_res = band_of(minside, RESOLUTION_BANDS)
            dist["resolution"][b_res] += 1
            debug.add("resolution", b_res, crop, f"{minside}px")

            bright, sharp = estimate_quality(crop)
            b_bright = band_of(bright, BRIGHTNESS_BANDS)
            dist["brightness"][b_bright] += 1
            debug.add("brightness", b_bright, crop, f"{bright:.0f}")

            b_sharp = band_of(sharp, SHARPNESS_BANDS)
            dist["sharpness"][b_sharp] += 1
            debug.add("sharpness", b_sharp, crop, f"{sharp:.0f}")

            ita = estimate_skin_ita(crop)
            if ita is not None:
                b_skin = band_of(ita, FITZPATRICK_BANDS)
                dist["skin_tone"][b_skin] += 1
                debug.add("skin_tone", b_skin, crop, f"ITA {ita:.0f}")

            if clip:
                clip_buf.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if len(clip_buf) >= args.clip_batch:
                    flush_clip()

        if n_crops >= args.max_crops:
            break
        if n_imgs % 500 == 0:
            print(f"[audit] {n_imgs} images / {n_crops} crops ...")
    flush_clip()

    print(f"[audit] done: {n_imgs} images, {n_crops} crops")

    # ---- scoring ----
    summary = {"dimensions": {}, "gaps": []}
    entropies = []
    for dim, counts in dist.items():
        all_b = buckets_for(dim)
        ent = normalized_entropy(counts, all_b)
        g = gini(counts, all_b)
        entropies.append(ent)
        total = sum(counts.values()) or 1
        weakest = min(all_b, key=lambda b: counts.get(b, 0))
        wpct = 100 * counts.get(weakest, 0) / total
        summary["dimensions"][dim] = {
            "entropy": ent, "gini": g,
            "weakest_bucket": weakest, "weakest_pct": wpct,
        }
        for b in all_b:
            summary["gaps"].append(
                {"dimension": dim, "bucket": b, "pct": 100 * counts.get(b, 0) / total}
            )
    summary["diversity_confidence"] = float(np.mean(entropies)) if entropies else 0.0
    summary["gaps"].sort(key=lambda g: g["pct"])
    summary["gaps"] = summary["gaps"][:12]

    # ---- charts ----
    charts = {}
    for dim, counts in dist.items():
        p = out / f"chart_{dim}.png"
        if save_bar_chart(counts, buckets_for(dim), dim, p):
            charts[dim] = p.name

    debug_dir = None
    if args.debug_samples > 0:
        debug_dir = write_debug(out, debug, dist)
        print(f"[audit] debug montages written to {debug_dir/'index.md'}")

    meta = {
        "yaml": str(yaml_path), "split": args.split,
        "images_scanned": n_imgs, "crops": n_crops, "clip": bool(clip),
        "debug_dir": str(debug_dir) if debug_dir else None,
    }
    (out / "metrics.json").write_text(json.dumps(
        {"meta": meta, "summary": summary,
         "distributions": {d: dict(c) for d, c in dist.items()}},
        indent=2))
    write_report(out, summary, {d: c for d, c in dist.items()}, charts, meta)

    print(f"\n[audit] diversity confidence: {summary['diversity_confidence']:.2f}")
    print(f"[audit] report written to {out/'report.md'}")


if __name__ == "__main__":
    main()
