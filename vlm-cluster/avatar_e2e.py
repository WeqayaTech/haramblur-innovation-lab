#!/usr/bin/env python3
"""Per-size per-class detection + end-to-end for ALL five models on the
avatars arm, from the conf-0.45 labels each model emitted — one code path
for every number so the table is apples-to-apples.

end-to-end = the person was found (greedy IoU>=0.5, project match_boxes)
AND called their own class. Canvas is 640x640 (manifest: canvas 640).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
from run_model_children import match_boxes  # noqa: E402

SP = Path("/workspace/datasets/smallperson_v1/avatars")
B = Path("/workspace/exp19/avatars")
MODELS = ["yolo11N-640", "y26n_sop50", "y26n_warm50-2",
          "gelannfav14r4fw_gemlb_v2", "gelannfav14r6_gemlb_v2"]
SIZES = ["a128", "a96", "a64", "a48"]
CLS_NAME = {0: "Woman", 1: "Man", 2: "Child"}
W = H = 640


def boxes(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = [float(x) for x in p[1:]]
        out.append((cls, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


gt_files = sorted((SP / "labels").glob("*.txt"))
results = {}
for m in MODELS:
    pred_root = B / m / "labels"
    per = {s: {c: {"n": 0, "found": 0, "e2e": 0} for c in CLS_NAME.values()}
           for s in SIZES}
    for f in gt_files:
        size = f.stem.split("_")[0]
        gt = boxes(f)
        dets = boxes(pred_root / f.name)
        matched = match_boxes([g[1:] for g in gt], dets, 0.5)
        for gi, g in enumerate(gt):
            name = CLS_NAME.get(g[0])
            if name is None:
                continue
            cell = per[size][name]
            cell["n"] += 1
            if gi in matched:
                cell["found"] += 1
                if matched[gi][0][0] == g[0]:
                    cell["e2e"] += 1
    results[m] = per

out = B / "e2e_5model.json"
out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("E2E_DONE")
