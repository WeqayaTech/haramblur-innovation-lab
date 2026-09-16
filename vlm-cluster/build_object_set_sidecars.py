import json, os
from PIL import Image

LABELS_DIR = "/workspace/exp03/sam_labels/objects"
IMAGES_DIR = "/workspace/datasets/object_set"
OUT_DIR = "/workspace/exp03/sam_labels/objects"  # write .json sidecars alongside the .txt files

n_written = 0
n_empty = 0
n_missing_img = 0

for fname in sorted(os.listdir(LABELS_DIR)):
    if not fname.endswith(".txt"):
        continue
    stem = fname[:-4]
    img_path = os.path.join(IMAGES_DIR, stem + ".jpg")
    if not os.path.exists(img_path):
        n_missing_img += 1
        continue
    with Image.open(img_path) as im:
        W, H = im.size

    txt_path = os.path.join(LABELS_DIR, fname)
    detections = []
    with open(txt_path) as f:
        for line in f:
            parts_raw = line.split()
            if len(parts_raw) < 7:
                continue
            cls = int(float(parts_raw[0]))
            coords = [float(v) for v in parts_raw[1:]]
            xs_norm = coords[0::2]
            ys_norm = coords[1::2]
            xs = [x * W for x in xs_norm]
            ys = [y * H for y in ys_norm]
            box = [min(xs), min(ys), max(xs), max(ys)]
            poly = [[x, y] for x, y in zip(xs, ys)]
            detections.append({
                "cls": cls,
                "conf": None,
                "box": box,
                "parts": [poly],
            })

    if not detections:
        n_empty += 1

    sidecar = {
        "image": stem + ".jpg",
        "width": W,
        "height": H,
        "detections": detections,
        "suppressed": [],
    }
    out_path = os.path.join(OUT_DIR, stem + ".json")
    with open(out_path, "w") as f:
        json.dump(sidecar, f)
    n_written += 1

print(f"sidecars written: {n_written}, empty-detection images: {n_empty}, missing images: {n_missing_img}")
