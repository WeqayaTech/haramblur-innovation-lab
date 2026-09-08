"""Shared dataset loading for the VLM-cluster pipeline.

Handles the HaramBlur Open Images V7 YOLO layout:
    names: {0: Woman, 1: Man, 2: Child}
    path: /workspace/open-images-v7
    train: images/train   (labels at labels/train, Ultralytics convention)

Every class is a person type, so all boxes are kept.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("Missing dependency: PyYAML")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Could not parse dataset yaml: {yaml_path}")
    return cfg


def class_names(cfg: dict) -> dict:
    classes = cfg.get("names") or cfg.get("class_list")
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    if isinstance(classes, list):
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
    """Return (image_dir, label_dir). The yaml value is the image dir."""
    split_val = cfg.get(split_key, split_key)  # e.g. 'images/train'
    img_candidates = [
        root / split_val,
        root / "images" / split_val,
        root / split_val / "images",
    ]
    for img_dir in img_candidates:
        if img_dir.is_dir():
            lbl_dir = _first_existing(
                Path(str(img_dir).replace(f"{os.sep}images{os.sep}",
                                          f"{os.sep}labels{os.sep}", 1)),
                Path(str(img_dir).replace("images", "labels")),
                img_dir.parent / "labels" / img_dir.name,
                root / "labels" / Path(split_val).name,
            )
            if lbl_dir:
                return img_dir, lbl_dir
    raise SystemExit(
        f"Could not locate images+labels for split '{split_val}' under {root}."
    )


def list_images(img_dir: Path):
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def label_path_for(img_path: Path, img_dir: Path, lbl_dir: Path) -> Path:
    rel = img_path.relative_to(img_dir).with_suffix(".txt")
    return lbl_dir / rel


def yolo_boxes(label_file: Path, w: int, h: int):
    """Parse normalized YOLO txt -> list of (class_id, x1,y1,x2,y2) in pixels."""
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


def iter_images_with_boxes(cfg, root, split_key, names, max_images, seed):
    """Yield (image_path, img_bgr, [(class_name, x1,y1,x2,y2), ...])."""
    img_dir, lbl_dir = discover_split(cfg, root, split_key)
    imgs = list(list_images(img_dir))
    random.Random(seed).shuffle(imgs)
    if max_images:
        imgs = imgs[:max_images]
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        raw = yolo_boxes(label_path_for(img_path, img_dir, lbl_dir), w, h)
        boxes = [(names.get(c, str(c)), x1, y1, x2, y2)
                 for (c, x1, y1, x2, y2) in raw]
        if boxes:
            yield img_path, img, boxes


def pad_box(x1, y1, x2, y2, w, h, pad_frac):
    """Expand a box by pad_frac of its size, clamped to image bounds."""
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    nx1 = max(0, int(x1 - px))
    ny1 = max(0, int(y1 - py))
    nx2 = min(w, int(x2 + px))
    ny2 = min(h, int(y2 + py))
    return nx1, ny1, nx2, ny2
