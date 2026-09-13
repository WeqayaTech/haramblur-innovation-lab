#!/usr/bin/env python3
"""List tensor scopes of a .tflite (ai_edge_litert): scope-prefix histogram, and full names of
head tensors (last dim == N anchors). Writes <out>.txt with every tensor. Used to build the
NO_QUANTIZE regex for the float-head arm of EXP-2026-21.

usage: tflite_scopes.py <file.tflite> <out.txt> [N=8400] [strip_prefix]
"""
import collections, sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

f, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8400
strip = sys.argv[4] if len(sys.argv) > 4 else "ultralytics.utils.export.engine._NormalizeCoords/"
it = Interpreter(model_path=f); it.allocate_tensors(); dets = it.get_tensor_details()
with open(out, "w") as fh:
    for t in dets:
        fh.write(f"{t['index']}\t{np.dtype(t['dtype']).name}\t{[int(x) for x in t['shape']]}\t{t['name']}\n")
pre = collections.Counter()
for t in dets:
    n = t["name"].replace(strip, "")
    pre["/".join(n.split("/")[:2])] += 1
print("=== top scope prefixes (wrapper prefix stripped) ===")
for k, v in pre.most_common(45): print(f"{v:4d}  {k[:120]}")
print(f"=== tensors whose last dim is {N} (head) ===")
for t in dets:
    s = [int(x) for x in t["shape"]]
    if s and s[-1] == N: print(t["index"], np.dtype(t["dtype"]).name, s, t["name"].replace(strip, "")[:150])
