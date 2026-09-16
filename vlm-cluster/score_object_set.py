import sys, os
from pathlib import Path
from PIL import Image
from run_autolabel_on_manifest import seg_boxes
from run_model_children import match_boxes, iou

GT_DIR = Path("/workspace/exp20/object_set_relabel/labels")
PRED_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/exp20/eval_object_set_v1/labels")
IMG_DIR = Path("/workspace/datasets/object_set")
CLASSES = {0: "Woman", 1: "Man", 2: "Child"}
MIN_IOU = 0.5

stems = sorted(p.stem for p in GT_DIR.glob("*.txt"))

n_gt = 0
n_matched = 0
n_class_correct = 0
n_fp = 0  # predictions not matched to any GT
n_images_with_gt = 0
n_images_with_fp = 0
per_class_gt = {0: 0, 1: 0, 2: 0}
per_class_matched = {0: 0, 1: 0, 2: 0}
confusions = []

for stem in stems:
    img_path = IMG_DIR / f"{stem}.jpg"
    if not img_path.exists():
        continue
    with Image.open(img_path) as im:
        W, H = im.size

    gt = seg_boxes(GT_DIR / f"{stem}.txt", W, H) or []
    pred_file = PRED_DIR / f"{stem}.txt"
    preds = seg_boxes(pred_file, W, H) or []

    if gt:
        n_images_with_gt += 1
    n_gt += len(gt)
    for g in gt:
        per_class_gt[g[0]] = per_class_gt.get(g[0], 0) + 1

    gt_boxes_only = [g[1:5] for g in gt]
    dets = [(p[0], p[1], p[2], p[3], p[4]) for p in preds]  # (cls, x1,y1,x2,y2)
    matches = match_boxes(gt_boxes_only, dets, MIN_IOU)

    matched_det_idx = set()
    for gi, (det, j) in matches.items():
        n_matched += 1
        gt_cls = gt[gi][0]
        pred_cls = det[0]
        per_class_matched[gt_cls] = per_class_matched.get(gt_cls, 0) + 1
        if pred_cls == gt_cls:
            n_class_correct += 1
        else:
            confusions.append((stem, CLASSES.get(gt_cls, gt_cls), CLASSES.get(pred_cls, pred_cls)))
        for di, d in enumerate(dets):
            if d is det:
                matched_det_idx.add(di)

    fp_this_image = len(dets) - len(matched_det_idx)
    n_fp += fp_this_image
    if fp_this_image > 0:
        n_images_with_fp += 1

print(f"=== object_set generalization score: {PRED_DIR} ===")
print(f"images: {len(stems)}")
print(f"GT people (never seen in training): {n_gt}  (by class: {[(CLASSES[k], v) for k, v in per_class_gt.items()]})")
print(f"images with >=1 GT person: {n_images_with_gt}")
print()
print(f"detection recall (GT found @ IoU>={MIN_IOU}): {n_matched}/{n_gt} = {n_matched/max(1,n_gt)*100:.1f}%")
print(f"classification accuracy on found people: {n_class_correct}/{n_matched} = {n_class_correct/max(1,n_matched)*100:.1f}%")
print(f"per-class recall: {[(CLASSES[k], f'{per_class_matched.get(k,0)}/{v}') for k, v in per_class_gt.items() if v>0]}")
print()
print(f"false positives (predictions matching no GT person): {n_fp}")
print(f"images with >=1 false positive: {n_images_with_fp} / {len(stems)}")
print(f"false-positive rate: {n_fp/259*100:.1f} per 100 images")
print()
if confusions:
    print("misclassifications (GT -> predicted):")
    for c in confusions[:20]:
        print(" ", c)
