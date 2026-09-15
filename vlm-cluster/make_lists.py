#!/usr/bin/env python3
"""Write the EXP-2026-22 cell lists in priority order: cells_all.txt (everything; pt cells last so CPU pods
mop them up) and cells_pt.txt (pt cells first, holdout 640 first — for GPU pods). usage: make_lists.py <outdir>"""
import sys, os
MODELS=["yolo11N-640","y26n_humanshaped_v2","y26n_noe2e_warm50-2","y26s_humanshaped_smallpatch_v1","y26n_humanshaped_v2_distill_v1"]
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
spot=[("spotval",r,t,s) for r in MODELS for s in (640,416,320) for t in ("pt","fp32","fp16","int8","fdec")]
hold_core=[("holdout",r,t,s) for s in (640,416,320) for r in MODELS for t in ("int8","fdec")]
hold_opt=[("holdout",r,t,s) for s in (640,416,320) for r in MODELS for t in ("fp32","fp16")]
pt=[("holdout",r,"pt",s) for s in (640,416,320) for r in MODELS]
allcells=[c for c in spot if c[2]!="pt"]+[c for c in spot if c[2]=="pt"]+hold_core+hold_opt+pt
open(f"{out}/cells_all.txt","w").write("\n".join(" ".join(map(str,c)) for c in allcells)+"\n")
open(f"{out}/cells_pt.txt","w").write("\n".join(" ".join(map(str,c)) for c in pt+[c for c in spot if c[2]=="pt"])+"\n")
print(len(allcells), "cells;", len(pt), "holdout pt cells")
