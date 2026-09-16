import json, sys, collections
sys.path.insert(0, "/workspace/spotlight/childfix")
from spotlight_run import child_age_only
rows = [json.loads(l) for l in open(sys.argv[1])]
keep = [r for r in rows if child_age_only(r["v"])]
sam  = [r for r in keep if r["sam_class_id"] == 2]
def q(xs, lab):
    s = sorted(x for x in xs if x is not None)
    if not s: return f"  {lab}: n=0"
    f = lambda p: s[min(len(s)-1, int(p*len(s)))]
    return (f"  {lab}: n={len(s)}  p10={f(.1):.0f}  p25={f(.25):.0f}  "
            f"med={f(.5):.0f}  p75={f(.75):.0f}  p90={f(.9):.0f}")
print(f"guarded rescues            {len(keep):,}")
print(f"  of which SAM3 said Child {len(sam):,}  ({100*len(sam)/len(keep):.1f}%)")
print()
print("ages")
print("  all guarded:", sorted(collections.Counter(r["v"]["estimated_age"] for r in keep).items()))
print("  SAM3 agrees:", sorted(collections.Counter(r["v"]["estimated_age"] for r in sam).items()))
print()
print("person_px_height")
print(q([r.get("person_px_height") for r in keep], "all guarded"))
print(q([r.get("person_px_height") for r in sam],  "SAM3 agrees"))
print()
print("blurriness (variance of Laplacian; higher = sharper)")
print(q([r.get("blurriness") for r in keep], "all guarded"))
print(q([r.get("blurriness") for r in sam],  "SAM3 agrees"))
print()
for px in (0, 40, 60, 80, 100, 120):
    n = sum(1 for r in sam if (r.get("person_px_height") or 0) >= px)
    print(f"  SAM3 agrees AND person_px_height >= {px:>3}:  {n:>4}")
print()
print("  images touched:", len({r["image_stem"] for r in sam}))
print("  verdict:", dict(collections.Counter(r["v"]["verdict"] for r in sam)))
