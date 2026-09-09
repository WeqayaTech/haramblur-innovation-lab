#!/usr/bin/env python3
"""
bench_tflite.py — CPU latency benchmark for .tflite/.litert model files.

Pod-only (needs the real exported files and `ai_edge_litert`, not runnable
with --selftest against synthetic data). Fixed thread count rather than
`nproc` — see memory `runpod-nproc-thread-pinning`: nproc reports the RunPod
HOST inside the container, not the pod's actual allocation.

    python3 bench_tflite.py model_a_fp32.tflite model_a_int8.tflite ...

Input is random data matching each file's own input shape/dtype (int8 models
declare an integer input dtype; this benchmarks the interpreter's per-call
invoke() cost, not model accuracy — pair with map_eval.py / conf_sweep.py for
that). 20 warmup + 100 timed calls, median + p90 reported, matching this
repo's ONNX latency convention (docs/MODEL_EVAL_OVERVIEW.md).
"""
import os
import statistics
import sys
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter

NUM_THREADS = 4
WARMUP = 20
ITERS = 100


def bench_one(path):
    interp = Interpreter(model_path=path, num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    shape, dtype = inp["shape"], inp["dtype"]
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        x = rng.random(shape, dtype=np.float32).astype(dtype)
    else:
        info = np.iinfo(dtype)
        x = rng.integers(info.min, info.max, size=shape, dtype=dtype)
    idx = inp["index"]
    for _ in range(WARMUP):
        interp.set_tensor(idx, x)
        interp.invoke()
    times = []
    for _ in range(ITERS):
        interp.set_tensor(idx, x)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "file": path,
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "input_shape": list(shape),
        "median_ms": round(statistics.median(times), 2),
        "p90_ms": round(times[int(0.9 * len(times)) - 1], 2),
    }


if __name__ == "__main__":
    for r in (bench_one(p) for p in sys.argv[1:]):
        print(f"{r['file']}\t{r['size_mb']}MB\tshape={r['input_shape']}\t"
              f"median={r['median_ms']}ms\tp90={r['p90_ms']}ms")
