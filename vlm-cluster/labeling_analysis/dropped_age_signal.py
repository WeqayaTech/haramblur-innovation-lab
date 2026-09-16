"""Is `estimated_age` informative for the detections Spotlight DROPPED?

If the same abstention that killed the gender also killed the age, then
filtering the dropped pile by estimated_age cannot find the children in it --
it only finds the ones whose placeholder number happened to be small.
"""
import json, sys, collections
sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import merge

exact = collections.Counter()
allunk = collections.Counter()   # exact age -> how many abstained on everything else
tot = collections.Counter()
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    v = r.get("v")
    m = merge(v)
    if not m["keep"] or m["final_class"] != "Unknown":
        continue
    a = v.get("estimated_age")
    key = "none" if a is None else a
    exact[key] += 1
    tot["dropped"] += 1
    if v.get("apparent_race") == "unknown" and v.get("exposed_body_parts") == ["none"]:
        allunk[key] += 1
    if v.get("confidence") == "low":
        tot["low_conf"] += 1

print(f"dropped unknown-gender: {tot['dropped']:,}   "
      f"confidence=low: {tot['low_conf']:,} "
      f"({100*tot['low_conf']/tot['dropped']:.1f}%)\n")
print("the 15 most common EXACT estimated_age values in the dropped pile")
print(f"{'age':>6} {'count':>9} {'% of dropped':>13}   {'also abstained on race+body':>28}")
for k, n in exact.most_common(15):
    print(f"{str(k):>6} {n:>9,} {100*n/tot['dropped']:>12.2f}%   "
          f"{allunk[k]:>9,} ({100*allunk[k]/n:>5.1f}%)")
distinct = len([k for k in exact if k != "none"])
top5 = sum(n for _, n in exact.most_common(5))
print(f"\ndistinct age values used: {distinct}   "
      f"top-5 values cover {100*top5/tot['dropped']:.1f}% of the pile")
