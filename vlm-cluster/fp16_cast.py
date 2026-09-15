#!/usr/bin/env python3
"""Weight-cast an fp32 .tflite to float16 (ai_edge_quantizer float_casting; no calibration) and
carry the ultralytics metadata.json so LiteRTBackend loads it.  usage: fp16_cast.py <src> <dst>"""
import hashlib, os, sys, zipfile
from ai_edge_quantizer import qtyping, quantizer

src, dst = sys.argv[1], sys.argv[2]
qt = quantizer.Quantizer(src)
cfg = qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT))
qt.update_quantization_recipe(regex=".*", operation_name="*", op_config=cfg, algorithm_key="float_casting")
qt.quantize().export_model(dst, overwrite=True)
try:
    with zipfile.ZipFile(src) as z: meta = z.read("metadata.json")
    with zipfile.ZipFile(dst, "a", zipfile.ZIP_DEFLATED) as z: z.writestr("metadata.json", meta)
except Exception as e:
    print("metadata copy skipped:", e)
print(f"FP16_OK src={os.path.getsize(src)} dst={os.path.getsize(dst)} sha={hashlib.sha256(open(dst,'rb').read()).hexdigest()[:16]}")
