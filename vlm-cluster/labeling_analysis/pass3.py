import json
from collections import Counter, defaultdict
P="/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl"
comp=defaultdict(Counter); est=defaultdict(Counter)
def bk(h,f):
    if h is None: return None
    if h<64: return "tiny"
    if not f: return "noface"
    return "face_small" if h<192 else "face_large"
for line in open(P):
    try: r=json.loads(line)
    except ValueError: continue
    v=r.get("v") or {}
    if str(v.get("verdict")) not in ("real_person","depiction"): continue
    if str(v.get("gender"))!="unknown": continue
    ex=v.get("exposed_body_parts") or []
    if isinstance(ex,str): ex=[ex]
    f="face" in [str(e).lower() for e in ex]
    b=bk(r.get("person_px_height"), f)
    if not b: continue
    comp[b][str(v.get("age_group"))]+=1
    ea=v.get("estimated_age")
    est[b]["none" if ea is None else ("0-3" if ea<=3 else ("4-12" if ea<=12 else ("13-17" if ea<18 else "18+")))]+=1
for b in ("tiny","noface","face_small","face_large"):
    t=sum(comp[b].values())
    print(f"{b:11} n={t:7,}  " + "  ".join(f"{k}={v:,} ({100*v/t:.0f}%)" for k,v in comp[b].most_common()))
    print("            est_age: " + "  ".join(f"{k}={v:,}" for k,v in sorted(est[b].items())))
