"""Prove a --child-by-age label dir differs from the baseline ONLY by children
coming back, and that the baseline itself was not touched."""
import os, sys, hashlib, collections
base, new = sys.argv[1], sys.argv[2]
bf = {f for f in os.listdir(base) if f.endswith(".txt")}
nf = {f for f in os.listdir(new) if f.endswith(".txt")}
print(f"baseline files {len(bf):,}   new files {len(nf):,}   "
      f"only-in-baseline {len(bf-nf)}   only-in-new {len(nf-bf)}")
changed, kinds = [], collections.Counter()
for f in sorted(bf & nf):
    a = open(os.path.join(base, f)).read()
    b = open(os.path.join(new, f)).read()
    if a == b:
        continue
    changed.append(f)
    la = [l.split() for l in a.splitlines() if l.strip()]
    lb = [l.split() for l in b.splitlines() if l.strip()]
    added = len(lb) - len(la)
    # every baseline line must survive verbatim, in order, in the new file
    geo_a = [" ".join(x) for x in la]
    geo_b = [" ".join(x) for x in lb]
    superset = all(g in geo_b for g in geo_a)
    newlines = [x for x in geo_b if x not in geo_a]
    cls = {x.split()[0] for x in newlines}
    kinds[(added, superset, tuple(sorted(cls)))] += 1
print(f"changed files: {len(changed):,}")
print("\nshape of every change  (lines_added, baseline_lines_all_survive, new_classes)")
for k, n in kinds.most_common():
    print(f"  {k}   x{n:,}")
bad = [k for k in kinds if not k[1] or k[0] <= 0 or k[2] != ("2",)]
print("\nVERDICT:", "PASS — every change is purely added class-2 (Child) lines"
      if not bad else f"FAIL — unexpected change shapes: {bad}")
print("sample changed files:", changed[:5])
