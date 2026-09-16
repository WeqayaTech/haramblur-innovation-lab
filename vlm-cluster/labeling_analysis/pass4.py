import json
from collections import Counter
P="/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl"
c=Counter(); ages=Counter()
for line in open(P):
    try: r=json.loads(line)
    except ValueError: continue
    v=r.get("v") or {}
    if str(v.get("verdict")) not in ("real_person","depiction"): continue
    if str(v.get("gender"))!="unknown": continue
    ex=v.get("exposed_body_parts") or []
    if isinstance(ex,str): ex=[ex]
    if "face" not in [str(e).lower() for e in ex]: continue
    h=r.get("person_px_height")
    if h is None or h<64: continue
    a=str(v.get("age_group")); conf=str(v.get("confidence")); ea=v.get("estimated_age")
    c[("child" if a=="child" else "not_child", conf)]+=1
    if a!="child":
        ages["none" if ea is None else ("0-3" if ea<=3 else ("4-12" if ea<=12 else ("13-17" if ea<18 else "18+")))]+=1
tot=sum(c.values())
print("face visible AND >=64px, gender abstained: total", f"{tot:,}")
for k,v in sorted(c.items()): print("  ",k,f"{v:,}")
res=sum(v for k,v in c.items() if k[0]=="not_child")
print("residual (not committed-child) =", format(res, ","))
print("  its est_age:", dict(ages))
