#!/usr/bin/env python3
"""Score the five models on the holdout small slice.

Two passes per model:
  1. map_eval (threshold-free AP; floor-0.001 sidecar views; big people +
     original unknowns as ignore regions).
  2. conf-0.45 operating point: greedy IoU>=0.5 matching (the project's
     match_boxes) of small GT boxes vs the labels each model actually
     emitted at 0.45 — detection recall, and per-class end-to-end
     (found AND called the right class).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
from run_model_children import match_boxes  # noqa: E402
try:
    from run_autolabel_on_manifest import seg_boxes  # noqa: E402
except Exception:
    from deploy_compare import seg_boxes  # noqa: E402

OUT = Path("/workspace/deploycmp/holdout_small")
ALIASES = ["v11n_shipped", "y26n_sop50", "y26n_warm50",
           "gelan_r4fw_v2", "gelan_r6_v2"]
DIMS_RAW = Path("/workspace/holdout_eval/v11n_shipped/raw")
CLS_NAME = {0: "Woman", 1: "Man", 2: "Child"}

members = sorted(p.stem for p in (OUT / "labels").glob("*.txt"))
dim_map = {}
for stem in members:
    d = json.loads((DIMS_RAW / (stem + ".json")).read_text())
    dim_map[stem] = (d["width"], d["height"])

procs = {}
for a in ALIASES:
    procs[a] = subprocess.Popen(
        [sys.executable, "map_eval.py", "--raw", str(OUT / f"raw_{a}"),
         "--gt-labels", str(OUT / "labels"),
         "--ignore-labels", str(OUT / "ignore"),
         "--expect-floor", "0.001",
         "--out", str(OUT / f"map_{a}.json")],
        cwd="/workspace/data_inspection_tools/vlm-cluster",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

results = {}
for a in ALIASES:
    per = {c: {"n": 0, "found": 0, "e2e": 0} for c in CLS_NAME.values()}
    pred_root = Path(f"/workspace/holdout_eval/{a}/labels")
    for stem in members:
        w, h = dim_map[stem]
        gt = seg_boxes(OUT / "labels" / (stem + ".txt"), w, h) or []
        preds = seg_boxes(pred_root / (stem + ".txt"), w, h) or []
        gt_boxes = [g[1:5] for g in gt]
        dets = [(p[0], p[1], p[2], p[3], p[4]) for p in preds]
        matched = match_boxes(gt_boxes, dets, 0.5)
        for gi, g in enumerate(gt):
            name = CLS_NAME.get(g[0])
            if name is None:
                continue
            per[name]["n"] += 1
            if gi in matched:
                per[name]["found"] += 1
                if matched[gi][0][0] == g[0]:
                    per[name]["e2e"] += 1
    tot_n = sum(v["n"] for v in per.values())
    tot_f = sum(v["found"] for v in per.values())
    results[a] = {"conf045": {
        "per_class": per,
        "det_recall": tot_f / tot_n if tot_n else None,
        "n_small": tot_n}}

for a in ALIASES:
    procs[a].wait()
    try:
        m = json.loads((OUT / f"map_{a}.json").read_text())
        results[a]["map"] = {
            "map50": m.get("map50"), "map50_95": m.get("map50_95"),
            "per_class": {c: {"ap50": (m.get("per_class", {}).get(c) or {}).get("ap50"),
                              "ap50_95": (m.get("per_class", {}).get(c) or {}).get("ap50_95")}
                          for c in CLS_NAME.values()}}
    except Exception as e:  # noqa: BLE001
        results[a]["map"] = {"error": str(e)}

(OUT / "small_slice_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("SCORE_DONE")
