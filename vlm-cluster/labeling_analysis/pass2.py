import json
from collections import Counter, defaultdict
P="/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl"
by_verdict_g=defaultdict(Counter)
cell=Counter()          # (face, sizeband, modelconf) for unknown gender
per_image=Counter()     # dets per image
unk_per_image=Counter()
crowd=Counter()
justif=Counter()
def band(h):
    if h is None: return "na"
    for lo,hi in [(0,64),(64,192),(192,320),(320,10**9)]:
        if lo<=h<hi: return f"{lo}-{hi if hi<10**9 else 'inf'}"
    return "na"
for line in open(P):
    try: r=json.loads(line)
    except ValueError: continue
    v=r.get("v") or {}
    vd=str(v.get("verdict"))
    if vd not in ("real_person","depiction"): continue
    g=str(v.get("gender")); h=r.get("person_px_height")
    ex=v.get("exposed_body_parts") or []
    if isinstance(ex,str): ex=[ex]
    ex=[str(e).lower() for e in ex]
    face="face" if "face" in ex else "noface"
    by_verdict_g[vd][g]+=1
    per_image[r["image_stem"]]+=1
    if g=="unknown":
        unk_per_image[r["image_stem"]]+=1
        cell[(face,band(h),str(v.get("confidence")))]+=1
        # how many have SOME stated justification vs none
        small = h is not None and h<64
        justif["tiny(<64px)" if small else ("no_face" if face=="noface" else
              ("face+large" if (h or 0)>=192 else "face+small"))]+=1
# crowding: are unknowns concentrated in many-detection images?
tot_u=sum(unk_per_image.values())
for stem,n in unk_per_image.items():
    d=per_image[stem]
    crowd["1-2 dets" if d<=2 else ("3-5" if d<=5 else ("6-10" if d<=10 else "11+"))]+=n
allc=Counter()
for stem,d in per_image.items():
    allc["1-2 dets" if d<=2 else ("3-5" if d<=5 else ("6-10" if d<=10 else "11+"))]+=d
print("gender by verdict:", {k:dict(v) for k,v in by_verdict_g.items()})
print("\nunknown-gender justification buckets:", dict(justif), "total", sum(justif.values()))
print("\nquestionable cells (face visible, model says HIGH confidence):")
for k,v in sorted(cell.items(), key=lambda kv:-kv[1]):
    if k[0]=="face" and k[2]=="high": print("  ",k,f"{v:,}")
print("\nall cells top15:")
for k,v in sorted(cell.items(), key=lambda kv:-kv[1])[:15]: print("  ",k,f"{v:,}")
print("\nunknowns by image crowding:", dict(crowd))
print("all dets by image crowding:", dict(allc))
print("rate per bucket:", {k: round(100*crowd[k]/allc[k],1) for k in allc})
