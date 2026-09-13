#!/usr/bin/env python3
"""Print dtype/quantization of a .tflite file's tensors (ai_edge_litert). Prints the
head-output tensors (shape [1,7,N] etc.) with their exact scale/zero-point, and a
`SCALE=… ZP=…` line for the int8 [1,7,N] tensor so shell scripts can eval it.

usage: tflite_tensor_inspect.py <file.tflite> [<file.tflite> ...] [--n 8400]
"""
import sys
from collections import Counter

import numpy as np
from ai_edge_litert.interpreter import Interpreter

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n = 8400
for a in sys.argv[1:]:
    if a.startswith("--n="): n = int(a.split("=")[1])
for f in args:
    it = Interpreter(model_path=f); it.allocate_tensors(); dets = it.get_tensor_details()
    print(f"== {f}")
    print(f"   tensors={len(dets)} dtypes={dict(Counter(np.dtype(t['dtype']).name for t in dets))}")
    for t in dets:
        shp = [int(x) for x in t["shape"]]
        if shp in ([1, 7, n], [1, n, 7], [1, 4, n], [1, 3, n], [1, 2, n]):
            q = t["quantization_parameters"]; sc = q["scales"]; zp = q["zero_points"]; dt = np.dtype(t["dtype"]).name
            print(f"   {t['index']:4d} {t['name'][:64]:64s} {dt:8s} shape={shp} "
                  f"scale={float(sc[0]) if len(sc) else None!r} zp={int(zp[0]) if len(zp) else None}")
            if shp == [1, 7, n] and dt == "int8" and len(sc):
                print(f"SCALE={float(sc[0])!r} ZP={int(zp[0])}")
    print("   outputs:", [(int(o["index"]), np.dtype(o["dtype"]).name) for o in it.get_output_details()])
