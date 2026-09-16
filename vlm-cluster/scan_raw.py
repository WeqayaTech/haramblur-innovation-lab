import json, os, glob, sys
E="/workspace/exp22/eval"; total_bad=0; affected=[]
for d in sorted(glob.glob(E+"/*/raw")):
    cell=os.path.basename(os.path.dirname(d)); bad=0
    for f in os.listdir(d):
        p=os.path.join(d,f)
        try:
            if os.path.getsize(p)==0: raise ValueError("empty")
            json.load(open(p))
        except Exception:
            os.remove(p); bad+=1
            lab=os.path.join(os.path.dirname(d),"labels",f[:-5]+".txt")
            if os.path.exists(lab): os.remove(lab)
    if bad:
        total_bad+=bad; affected.append(cell); print(f"{cell}: removed {bad} corrupt sidecars", flush=True)
        for q in glob.glob(f"{E}/{cell}_map*.json")+glob.glob(f"{E}/{cell}_randoms.json"): os.remove(q)
print("TOTAL corrupt sidecars:", total_bad, "affected cells:", len(affected))
open("/workspace/exp22/affected_cells.txt","w").write("\n".join(affected)+"\n")
print("SCAN_DONE")
