"""Of the children the emit ALREADY keeps, how many had no gender?

If that number is large, the policy "a child is Child regardless of gender" is
not a change to make -- it is already what the production emit did.
"""
import json, sys, collections
sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import merge

g = collections.Counter(); grp = collections.Counter(); n = 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    v = (json.loads(line) or {}).get("v")
    m = merge(v)
    if m["keep"] and m["final_class"] == "Child":
        n += 1
        g[v.get("gender")] += 1
        grp[v.get("confidence")] += 1
print(f"kept as Child: {n:,}")
for k, c in g.most_common():
    print(f"  gender={str(k):<8} {c:>8,}  {100*c/n:5.1f}%")
print("  confidence:", dict(grp))
