"""What words did Gemini put in `gender` for the detections we DROPPED?

merge() only recognises "man" and "woman". Anything else falls through to
"Unknown" and the person is deleted. If Gemini answered "boy" or "girl", it
named a CHILD in the gender field itself -- and we threw the child away.
"""
import json, sys, collections
sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import merge

g = collections.Counter()
childwords = collections.Counter()
ages = collections.Counter()
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    v = (json.loads(line) or {}).get("v")
    m = merge(v)
    if m["keep"] and m["final_class"] == "Unknown":
        w = str(v.get("gender"))
        g[w] += 1
        if w in ("boy", "girl", "child", "kid", "baby", "infant", "toddler"):
            childwords[w] += 1
            ages[v.get("estimated_age")] += 1
tot = sum(g.values())
print(f"dropped unknown-gender: {tot:,}")
print("\nevery distinct value of the `gender` field in that pile")
for k, n in g.most_common():
    print(f"  {k:<12} {n:>9,}  {100*n/tot:6.2f}%")
print(f"\nchild-naming words: {sum(childwords.values()):,}  {dict(childwords)}")
if ages:
    print("  their estimated_age:", sorted((str(k), n) for k, n in ages.items()))
