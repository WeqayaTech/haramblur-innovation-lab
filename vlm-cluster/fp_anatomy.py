import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
from map_eval import iou, load_gt, load_ignore, _ioa
CONF=0.45
def load(raw):
    out={}
    for p in sorted(Path(raw).glob("*.json")):
        d=json.loads(p.read_text()); w,h=d["width"],d["height"]
        out[p.stem]=[(int(r["cls"]),float(r["conf"]),(r["box_xyxy"][0]/w,r["box_xyxy"][1]/h,r["box_xyxy"][2]/w,r["box_xyxy"][3]/h)) for r in d["detections"] if not r.get("excluded") and float(r["conf"])>=CONF]
    return out
gtd=Path("/workspace/spotlight/run/oiv7_val/labels"); igd=Path("/workspace/spotlight/run/oiv7_val/ignore_unk")
for m in ["y26n_humanshaped_v2","y26n_noe2e_warm50-2"]:
    for arm,raw in [("fp32",f"/workspace/quant_matrix_calib500/{m}/fp32tflite_sz640_spotval/raw"),("int8",f"/workspace/quant_matrix_calib500/{m}/sz640_spotval/raw")]:
        D=load(raw); stems=sorted(D); gt=load_gt(gtd,stems); ign=load_ignore(igd,stems)
        fp_total=dup_class=phantom=gt_dups=same_class_dup=0
        for s in stems:
            gb=[(c,tuple(b)) for c in gt.get(s,{}) for b in gt[s][c]]
            for i in range(len(gb)):
                for j in range(i+1,len(gb)):
                    if iou(gb[i][1],gb[j][1])>=0.5: gt_dups+=1
            dets=sorted(D[s],key=lambda x:-x[1]); used=[False]*len(gb)
            for c,conf,b in dets:
                best,bi=0,-1
                for j,(gc,g) in enumerate(gb):
                    if used[j] or gc!=c: continue
                    v=iou(b,g)
                    if v>best: best,bi=v,j
                if best>=0.5: used[bi]=True; continue
                if any(_ioa(list(b),g)>=0.5 for g in ign.get(s,[])): continue
                fp_total+=1
                on=max((iou(b,g) for gc,g in gb),default=0)
                if on>=0.5:
                    if any(gc==c and iou(b,g)>=0.5 for gc,g in gb): same_class_dup+=1
                    else: dup_class+=1
                else: phantom+=1
        print(f"{m} {arm}: FP={fp_total} | other-class box on a labelled person={dup_class} | same-class duplicate={same_class_dup} | on nothing={phantom} | GT duplicate pairs={gt_dups}", flush=True)
print("FP_ANATOMY_DONE", flush=True)
