#!/usr/bin/env python3
"""One-screen status + sanity check for the EXP-2026-22 multi-pod run (run on any pod with the volume).

Shows: cells done / claimed / running per pod, stalled claims, failures, and accuracy sanity flags:
  * n_images != expected (4232 spotval / 11494 holdout)
  * mAP outside a plausible band
  * fp32 TFLite vs .pt  > 0.010 mAP50-95   (toolchain parity broken)
  * FP16 vs fp32 TFLite > 0.002            (weight cast broken)
  * INT8 / fdec vs .pt  outside [-0.12, +0.06] (export corrupt or unquantized)
usage: matrix_status.py [--eval /workspace/exp22/eval] [--claims /workspace/exp22/claims] [--logs /workspace/exp22/logs]
"""
import argparse, glob, json, os, re, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="/workspace/exp22/eval"); ap.add_argument("--claims", default="/workspace/exp22/claims"); ap.add_argument("--logs", default="/workspace/exp22/logs"); a = ap.parse_args()
pat = re.compile(r"(.+)_(pt|fp32|fp16|int8|fdec)_sz(\d+)_(spotval|holdout)$")
done = {}
for p in glob.glob(os.path.join(a.eval, "*_map.json")):
    m = pat.match(os.path.basename(p)[:-9])
    if not m: continue
    try: j = json.load(open(p))
    except Exception: print("CORRUPT map json (delete it so a runner rescores):", p); continue
    done[(m.group(4), m.group(1), m.group(2), int(m.group(3)))] = j
claims = {}
for d in glob.glob(os.path.join(a.claims, "*/")):
    k = os.path.basename(d.rstrip("/")); m = pat.match(k)
    if not m: continue
    owner = open(os.path.join(d, "owner")).read().split() if os.path.exists(os.path.join(d, "owner")) else ["?", "?", "?"]
    claims[(m.group(4), m.group(1), m.group(2), int(m.group(3)))] = {"host": owner[0], "t": owner[2] if len(owner) > 2 else "?", "age_min": (time.time() - os.path.getmtime(d)) / 60}
running = {k: v for k, v in claims.items() if k not in done}
print(f"=== {time.strftime('%H:%M:%S')}  done={len(done)}/150  claimed-not-done={len(running)}  unclaimed={150 - len(done) - len(running)}")
by_host = Counter(v["host"] for v in running.values())
print("running per host:", dict(by_host))
for k, v in sorted(running.items(), key=lambda kv: -kv[1]["age_min"]):
    flag = "  <-- STALLED?" if (v["age_min"] > (40 if k[0] == "holdout" else 20)) else ""
    print(f"  {k[0]:7s} {k[1]:32s} {k[2]:5s} {k[3]}  host={v['host'][:12]} since {v['t']} ({v['age_min']:.0f} min){flag}")
fails = []
for lg in glob.glob(os.path.join(a.logs, "*.log")):
    for line in open(lg, errors="ignore"):
        if line.startswith(("CELL_FAILED", "MISSING")) or "Traceback" in line: fails.append(f"{os.path.basename(lg)}: {line.strip()[:120]}")
print(f"failures: {len(fails)}"); [print("  " + f) for f in fails[:12]]
print("=== sanity flags")
n_flags = 0
for (ds, run, tag, sz), j in sorted(done.items()):
    exp = 4232 if ds == "spotval" else 11494
    if j.get("n_images") != exp: print(f"  n_images {j.get('n_images')} != {exp}: {ds} {run} {tag} {sz}"); n_flags += 1
    if not (0.25 <= j["map50_95"] <= 0.95): print(f"  implausible mAP50-95 {j['map50_95']}: {ds} {run} {tag} {sz}"); n_flags += 1
    ref = done.get((ds, run, "pt", sz))
    if ref:
        d = j["map50_95"] - ref["map50_95"]
        if tag == "fp32" and abs(d) > 0.010: print(f"  fp32 TFLite vs .pt Δ={d:+.4f}: {ds} {run} {sz}"); n_flags += 1
        if tag in ("int8", "fdec") and not (-0.12 <= d <= 0.06): print(f"  {tag} vs .pt Δ={d:+.4f} out of band: {ds} {run} {sz}"); n_flags += 1
    r32 = done.get((ds, run, "fp32", sz))
    if tag == "fp16" and r32 and abs(j["map50_95"] - r32["map50_95"]) > 0.002: print(f"  FP16 vs fp32 Δ={j['map50_95'] - r32['map50_95']:+.4f}: {ds} {run} {sz}"); n_flags += 1
print(f"flags: {n_flags}")
print("=== done cells by dataset/tag:", dict(Counter((k[0], k[2]) for k in done)))
