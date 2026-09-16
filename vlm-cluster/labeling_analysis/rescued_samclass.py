import json, sys, collections, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spotlight_run import CLASS_NAME, child_age_only
rows = [json.loads(l) for l in open(sys.argv[1])]
keep = [r for r in rows if child_age_only(r["v"])]
print(f"rescued: {len(keep):,}")
c = collections.Counter(CLASS_NAME.get(r["sam_class_id"]) for r in keep)
for k, n in c.most_common():
    print(f"  SAM3 had called it {k:<6} {n:>6,}  {100*n/len(keep):5.1f}%")
print("\n  -> SAM3 independently said Child for "
      f"{100*c['Child']/len(keep):.1f}% of them")
hc = collections.Counter(r["v"].get("head_covering") for r in keep)
print("  head_covering:", dict(hc.most_common(4)))
print("  exposed face:", sum(1 for r in keep
                             if "face" in (r["v"].get("exposed_body_parts") or [])))
