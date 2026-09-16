import sys
sys.path.insert(0, "/workspace/exp20")

# Each of the 10 DataLoader workers is a separate PROCESS -- parallelism already
# comes from having 10 of them. Without this, cv2's internal thread pool defaults
# to the (misleading, host-level) nproc count inside each worker, oversubscribing
# the pod's real ~10-core cgroup allocation by 5-10x. Must be set before the
# DataLoader workers fork.
import cv2
cv2.setNumThreads(1)

from small_object_patches import patch_v8_transforms
patch_v8_transforms(p=0.05, target_size=96)

from ultralytics import YOLO

model = YOLO("yolo26s.pt")
model.train(
    data="/root/oiv7_local/dataset.yaml",
    epochs=100,
    patience=15,
    save_period=5,
    batch=0.85,
    imgsz=640,
    device="0",
    workers=10,
    project="/workspace/exp20/train",
    name="y26s_humanshaped_smallpatch_v1",
    optimizer="MuSGD",
    seed=0,
    deterministic=True,
    cls_remap=True,
    warmup_epochs=3.0,
    lr0=0.003,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, bgr=0.0,
    mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, copy_paste_mode="flip",
    auto_augment="randaugment", erasing=0.4, close_mosaic=10,
    multi_scale=0.0,
    plots=True,
    verbose=True,
)
