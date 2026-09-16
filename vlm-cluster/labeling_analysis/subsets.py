import json, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
def q(xs):
    s = sorted(xs)
    if not s: return "n=0"
    f = lambda p: s[min(len(s)-1, int(p*len(s)))]
    return f"n={len(s)} p10={f(.1):.0f} med={f(.5):.0f} p90={f(.9):.0f}"
V = lambda r, k: r["v"][k]
print(f"{len(rows)} candidates (dropped unknown-gender, estimated_age <= 12)\n")
sent = [r for r in rows if V(r,"estimated_age") <= 0]
allunk = sum(1 for r in sent if V(r,"apparent_race") == "unknown"
             and V(r,"exposed_body_parts") == ["none"])
print(f"  estimated_age <= 0 (impossible/placeholder)      {len(sent):>5}")
print(f"    of those, race AND body parts also abstained   {allunk:>5}")
bad = [r for r in rows if V(r,"highlight_quality") != "good"]
print(f"  highlight_quality != good (crop showed wrong)    {len(bad):>5}")
keep = [r for r in rows if V(r,"estimated_age") > 0 and V(r,"highlight_quality") == "good"]
print(f"  -> CREDIBLE subset (age>0 AND good highlight)    {len(keep):>5}")
print("       ages:", sorted(collections.Counter(V(r,"estimated_age") for r in keep).items()))
print("       conf:", dict(collections.Counter(V(r,"confidence") for r in keep)))
print("       verdict:", dict(collections.Counter(V(r,"verdict") for r in keep)))
print("       px height:", q([r["person_px_height"] for r in keep if r.get("person_px_height")]))
strict = [r for r in keep if V(r,"estimated_age") < 10]
print(f"  -> STRICT subset (also drops the round 10)       {len(strict):>5}")
print("       ages:", sorted(collections.Counter(V(r,"estimated_age") for r in strict).items()))
print("       conf:", dict(collections.Counter(V(r,"confidence") for r in strict)))
print("       px height:", q([r["person_px_height"] for r in strict if r.get("person_px_height")]))
print(f"\n  images touched (credible): "
      f"{len({r['image_stem'] for r in keep}):,} of 475,207")
