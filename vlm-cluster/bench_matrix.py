#!/usr/bin/env python3
"""Uniform CPU latency benchmark for every .tflite in the EXP-2026-22 matrix (one pod, one session).

Method (same for every file): LiteRT interpreter + XNNPACK, batch 1, fixed random input, N threads
pinned (OMP/interpreter), 20 warm-up + 100 timed invokes, repeated R times with the file order
interleaved (all files once, then again, ...), so a noisy neighbour hits different files each pass.
Reports per file and thread setting: median-of-medians (ms), p90 (of the pooled runs), spread
(max−min of the R medians, %), model-load time, file size, sha256. Environment is recorded.

usage: bench_matrix.py --out bench_matrix.json [--threads 1,4] [--repeats 3] [--warmup 20] [--runs 100]
                       [--models ...] [--sizes 640,416,320] [--tags fp32,fp16,int8,fdec] [--root DIR]
File resolution mirrors matrix_lib.sh::resolve (exp22 exports, then *_calib500, then exp21).
With --root DIR (any machine, e.g. a Mac) files are read from DIR/<run>/sz<SZ>/<run>_<tag>.tflite instead —
build that mirror with symlinks to wherever the local copies live.
"""
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, time

MODELS = ["yolo11N-640", "y26n_humanshaped_v2", "y26n_noe2e_warm50-2", "y26s_humanshaped_smallpatch_v1", "y26n_humanshaped_v2_distill_v1"]


ROOT = None   # set by --root: local mirror <root>/<run>/sz<sz>/<run>_<tag>.tflite


def resolve(run, tag, sz):
    if ROOT:
        p = os.path.join(ROOT, run, f"sz{sz}", f"{run}_{tag}.tflite")
        return p if os.path.isfile(p) and os.path.getsize(p) > 0 else None
    c = [f"/workspace/exp22/exports/{run}/sz{sz}/{run}_{tag}.tflite", f"/workspace/exports/{run}_calib500/sz{sz}/{run}_{tag}.tflite",
         f"/workspace/exp21/exports/{run}/sz{sz}/{run}_{tag}.tflite"]
    if tag == "fdec" and sz == 640: c.append(f"/workspace/exp21/exports/{run}/{run}_fhead_decode_out.tflite")
    for p in c:
        if os.path.isfile(p) and os.path.getsize(p) > 0: return p
    return None


def env_info():
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): cpu = line.split(":", 1)[1].strip(); break
    except Exception: pass
    if not cpu and sys.platform == "darwin":   # Apple silicon: brand string + performance/efficiency core split
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            pe = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"], capture_output=True, text=True).stdout.split()
            if len(pe) == 2: cpu += f" ({pe[0]}P+{pe[1]}E cores)"
            cpu += f", macOS {platform.mac_ver()[0]}"
        except Exception: pass
    q = -1; per = 100000
    try: q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
    except Exception: pass
    try: import ai_edge_litert; lv = getattr(ai_edge_litert, "__version__", "?")
    except Exception: lv = "?"
    return {"cpu_model": cpu, "nproc": os.cpu_count(), "cgroup_cores": (q / per) if q > 0 else None, "kernel": platform.release(),
            "python": platform.python_version(), "ai_edge_litert": lv, "loadavg_at_start": os.getloadavg(), "hostname": platform.node(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def bench_file(path, threads, warmup, runs):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    t0 = time.perf_counter()
    it = Interpreter(model_path=path, num_threads=threads); it.allocate_tensors()
    load_ms = (time.perf_counter() - t0) * 1000
    inp = it.get_input_details()[0]
    rng = np.random.default_rng(0)
    x = rng.random(inp["shape"], dtype=np.float32) if np.dtype(inp["dtype"]) == np.float32 else rng.integers(-128, 127, inp["shape"], dtype=inp["dtype"])
    for _ in range(warmup): it.set_tensor(inp["index"], x); it.invoke()
    ts = []
    for _ in range(runs):
        it.set_tensor(inp["index"], x); t = time.perf_counter(); it.invoke(); ts.append((time.perf_counter() - t) * 1000)
    return load_ms, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--threads", default="1,4"); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--sizes", default="640,416,320"); ap.add_argument("--tags", default="fp32,fp16,int8,fdec")
    ap.add_argument("--root", help="local mirror dir: <root>/<run>/sz<SZ>/<run>_<tag>.tflite (bench on any machine)")
    a = ap.parse_args()
    global ROOT; ROOT = a.root
    files = []
    for run in a.models.split(","):
        for sz in map(int, a.sizes.split(",")):
            for tag in a.tags.split(","):
                p = resolve(run, tag, sz)
                if p: files.append((run, tag, sz, p))
                else: print(f"MISSING {run} {tag} {sz}", flush=True)
    print(f"{len(files)} files, threads={a.threads}, repeats={a.repeats}", flush=True)
    results = {}   # key -> {threads: {...}}
    raw = {}
    for th in map(int, a.threads.split(",")):
        os.environ["OMP_NUM_THREADS"] = str(th); os.environ["TF_NUM_INTRAOP_THREADS"] = str(th)
        for rep in range(a.repeats):
            for run, tag, sz, p in files:
                key = f"{run}|{tag}|{sz}"
                # each measurement in a fresh process so thread settings and allocator state cannot leak between files
                code = f"import json,sys; sys.path.insert(0,'{os.path.dirname(os.path.abspath(__file__))}'); import bench_matrix as b; b.ROOT={ROOT!r}; l,t=b.bench_file({p!r},{th},{a.warmup},{a.runs}); print(json.dumps([l,t]))"
                out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
                try: load_ms, ts = json.loads(out.stdout.strip().splitlines()[-1])
                except Exception:
                    print(f"FAIL {key} th={th} rep={rep}: {out.stderr[-300:]}", flush=True); continue
                raw.setdefault(key, {}).setdefault(th, []).append({"load_ms": load_ms, "runs": ts})
                print(f"{key:60s} th={th} rep={rep} median={statistics.median(ts):7.2f} ms load={load_ms:6.0f} ms", flush=True)
    for run, tag, sz, p in files:
        key = f"{run}|{tag}|{sz}"; r = results.setdefault(key, {"run": run, "tag": tag, "size": sz, "path": p, "bytes": os.path.getsize(p),
                                                              "sha256_16": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], "threads": {}})
        for th, reps in raw.get(key, {}).items():
            meds = [statistics.median(x["runs"]) for x in reps]; pooled = sorted(v for x in reps for v in x["runs"])
            r["threads"][str(th)] = {"median_ms": round(statistics.median(meds), 2), "p90_ms": round(pooled[int(0.9 * len(pooled)) - 1], 2),
                                     "spread_pct": round(100 * (max(meds) - min(meds)) / min(meds), 1) if len(meds) > 1 else 0.0,
                                     "load_ms": round(statistics.median(x["load_ms"] for x in reps), 0), "repeats": len(reps)}
    json.dump({"env": env_info(), "method": {"warmup": a.warmup, "runs": a.runs, "repeats": a.repeats, "threads": a.threads, "delegate": "XNNPACK (LiteRT default)", "batch": 1, "input": "fixed random, seed 0"},
               "results": results}, open(a.out, "w"), indent=1)
    print("BENCH_MATRIX_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
