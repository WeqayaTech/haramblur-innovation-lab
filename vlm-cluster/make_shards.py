#!/usr/bin/env python3
"""Plan EXP-2026-22 cells across pods (LPT balancing) and write one shard file per pod.
usage: make_shards.py --pods "name:cores:gpu|cpu,..." --out shards/ [--skip-optional]
Cost model (minutes, 5 threads): spotval 640/416/320 = 8/6/5, holdout = 21/14/10; y26s ×1.8; GPU pt cells 4 / 12.
Cells already scored on Spotlight-val (reused JSONs) cost 0 and go to the first pod."""
import argparse, math, os
MODELS=["yolo11N-640","y26n_humanshaped_v2","y26n_noe2e_warm50-2","y26s_humanshaped_smallpatch_v1","y26n_humanshaped_v2_distill_v1"]
TAGS=["pt","fp32","fp16","int8","fdec"]; SIZES=[640,416,320]
REUSE={(r,t,s) for r in MODELS[1:4] for t in ["pt","fp32","fp16","int8"] for s in SIZES} | {(r,"fdec",640) for r in MODELS[1:4]} | {("y26n_humanshaped_v2_distill_v1",t,s) for t in ["pt","fp32","int8","fdec"] for s in SIZES}
def cost(ds,run,tag,sz):
    if ds=="spotval" and (run,tag,sz) in REUSE: return 0.0
    if tag=="pt": return {"spotval":4,"holdout":12}[ds]
    base={"spotval":{640:8,416:6,320:5},"holdout":{640:21,416:14,320:10}}[ds][sz]
    return base*(1.8 if run.startswith("y26s") else 1.0)
ap=argparse.ArgumentParser(); ap.add_argument("--pods",required=True); ap.add_argument("--out",default="shards"); ap.add_argument("--skip-optional",action="store_true"); ap.add_argument("--done-dir",help="eval dir: skip cells whose _map.json exists"); ap.add_argument("--exclude-shards",help="dir of earlier shard files: skip cells already assigned"); a=ap.parse_args()
pods=[]
for spec in a.pods.split(","):
    f=spec.split(":"); n,c,k=f[:3]; slots=int(f[3]) if len(f)>3 else max(1,min(8,int(c)//5)); pods.append({"name":n,"cores":int(c),"gpu":k=="gpu","slots":slots})
skip=set()
if a.done_dir:
    import glob,re
    for q in glob.glob(os.path.join(a.done_dir,"*_map.json")):
        m=re.match(r"(.+)_(pt|fp32|fp16|int8|fdec)_sz(\d+)_(spotval|holdout)$",os.path.basename(q)[:-9])
        if m: skip.add((m.group(4),m.group(1),m.group(2),int(m.group(3))))
if a.exclude_shards:
    import glob
    for q in glob.glob(os.path.join(a.exclude_shards,"*.txt")):
        for line in open(q):
            t=line.split()
            if len(t)==4 and not line.startswith("#"): skip.add((t[0],t[1],t[2],int(t[3])))
cells=[]
for ds in ("spotval","holdout"):
    for run in MODELS:
        for tag in TAGS:
            for sz in SIZES:
                if a.skip_optional and ds=="holdout" and tag in ("fp32","fp16"): continue
                if (ds,run,tag,sz) in skip: continue
                cells.append((ds,run,tag,sz,cost(ds,run,tag,sz)))
gpu_cells=[c for c in cells if c[2]=="pt" and c[4]>0]; cpu_cells=[c for c in cells if c[2]!="pt" and c[4]>0]; free=[c for c in cells if c[4]==0]
def lpt(items, bins):  # bins: list of (podname, capacity_slots)
    load={b[0]:0.0 for b in bins}; assign={b[0]:[] for b in bins}
    for it in sorted(items,key=lambda x:-x[4]):
        b=min(bins,key=lambda b: load[b[0]]/b[1]); assign[b[0]].append(it); load[b[0]]+=it[4]
    return assign,{b[0]:load[b[0]]/b[1] for b in bins}
gpu_pods=[(p["name"],1) for p in pods if p["gpu"]]; cpu_pods=[(p["name"],p["slots"]) for p in pods if p["slots"]>0]
if not gpu_pods:
    cpu_cells += [(c[0],c[1],c[2],c[3],{"spotval":9,"holdout":25}[c[0]]) for c in gpu_cells]; gpu_cells=[]
ga,gl=lpt(gpu_cells,gpu_pods) if gpu_pods else ({},{}); ca,cl=lpt(cpu_cells,cpu_pods)
os.makedirs(a.out,exist_ok=True)
for p in pods:
    hdr=f"# {p['name']} cores={p['cores']} slots={p['slots']} gpu={p['gpu']}"
    cpu_lines=[" ".join(map(str,c[:4])) for c in ca.get(p['name'],[])]
    if p["name"]==pods[0]["name"]: cpu_lines=[" ".join(map(str,c[:4])) for c in free]+cpu_lines
    open(os.path.join(a.out,f"{p['name']}_cpu.txt"),"w").write("\n".join([hdr]+cpu_lines)+"\n")
    if p["gpu"]:
        open(os.path.join(a.out,f"{p['name']}_gpu.txt"),"w").write("\n".join([hdr]+[" ".join(map(str,c[:4])) for c in ga.get(p["name"],[])])+"\n")
    print(f"{p['name']:12s} cores={p['cores']:3d} slots={p['slots']} gpu={'Y' if p['gpu'] else 'n'}  cpu-cells={len(ca.get(p['name'],[])):3d} est≈{cl.get(p['name'],0):5.0f} min | gpu-cells={len(ga.get(p['name'],[])):2d} est≈{gl.get(p['name'],0):4.0f} min")
print(f"total cells={len(cells)} (free={len(free)}, gpu={len(gpu_cells)}, cpu={len(cpu_cells)}); wall ≈ {max(list(cl.values())+list(gl.values()) or [0]):.0f} min")
