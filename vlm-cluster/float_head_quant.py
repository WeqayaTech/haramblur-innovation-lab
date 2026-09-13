#!/usr/bin/env python3
"""EXP-2026-21 arm D: static W8A8 TFLite with the detect head (or only its decode ops) left in float.

Mirrors ultralytics 8.4.146 `utils/export/litert.py` exactly (recipe.static_wi8_ai8, fp32 graph
I/O via NO_QUANTIZE on INPUT/OUTPUT, calibration = val-style LetterBox(scaleup=False) BGR→RGB
CHW /255), plus ONE extra rule: NO_QUANTIZE for every op whose scope matches --head-regex.

  --head-regex '.*Detect_23;.*|.*_NormalizeCoords;.*'   # decode ops only (convs stay int8)
  --head-regex '.*Detect_23.*|.*_NormalizeCoords.*'     # whole head in float

Input must be the fp32 .tflite produced by `yolo export format=tflite` (its trailing metadata.json
zip entry is copied to the output so LiteRTBackend can read names/imgsz).

usage: float_head_quant.py --src <fp32.tflite> --out <out.tflite> --calib <list.txt> --imgsz 640 --head-regex <re> [--n 500]
"""
import argparse, json, os, sys, time, zipfile

import cv2
import numpy as np


def calib_samples(list_file, imgsz, n):
    from ultralytics.data.augment import LetterBox
    lb = LetterBox((imgsz, imgsz), auto=False, scaleup=False)   # == ultralytics val transform (dataset.py)
    paths = [p.strip() for p in open(list_file) if p.strip()][:n]
    out = []
    for p in paths:
        im = cv2.imread(p)
        if im is None: continue
        im = lb(image=im)
        im = im[..., ::-1].transpose(2, 0, 1)                    # BGR->RGB, HWC->CHW (Format transform)
        im = np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0   # == litert.py: img.float()/255
        out.append({"args_0": im})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--head-regex", required=True); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    from ai_edge_quantizer import qtyping, quantizer, recipe
    t0 = time.time()
    samples = calib_samples(a.calib, a.imgsz, a.n)
    print(f"calibration samples: {len(samples)} shape={samples[0]['args_0'].shape} ({time.time() - t0:.0f}s)")
    qt = quantizer.Quantizer(a.src)
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
        qt.update_quantization_recipe(regex=".*", operation_name=op, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
    qt.update_quantization_recipe(regex=a.head_regex, operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                                  algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
    print("recipe rules:", len(qt.get_quantization_recipe()))
    res = qt.calibrate({"serving_default": samples})
    print(f"calibrated ({time.time() - t0:.0f}s)")
    qt.quantize(calibration_result=res).export_model(a.out, overwrite=True)
    # carry ultralytics metadata (names, imgsz, stride) so LiteRTBackend loads it like the originals
    try:
        with zipfile.ZipFile(a.src) as z: meta = z.read("metadata.json")
        with zipfile.ZipFile(a.out, "a", zipfile.ZIP_DEFLATED) as z: z.writestr("metadata.json", meta)
        print("metadata.json copied")
    except Exception as e:
        print("metadata copy skipped:", e)
    print(f"wrote {a.out} bytes={os.path.getsize(a.out)} src_bytes={os.path.getsize(a.src)} ratio={os.path.getsize(a.out) / os.path.getsize(a.src):.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
