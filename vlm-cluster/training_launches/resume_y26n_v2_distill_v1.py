import sys
sys.path.insert(0, "/workspace/exp20")

import cv2
cv2.setNumThreads(1)

from small_object_patches import patch_v8_transforms
patch_v8_transforms(p=0.05, target_size=96)

from ultralytics import YOLO

LAST = "/workspace/exp20/train/y26n_humanshaped_v2_distill_v1/weights/last.pt"
TEACHER = "/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt"

model = YOLO(LAST)
model.train(resume=LAST, distill_model=TEACHER, dis=6.0)
