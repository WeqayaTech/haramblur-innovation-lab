"""Stage the ONLY images the --child-by-age rule can change.

The rule is strictly additive (asserted in spotlight_run._selftest): it can
turn a dropped detection into a Child and can do nothing else. So the set of
images whose label file changes is exactly the set of images holding at least
one rescued detection -- everything else is byte-identical to the baseline and
does not need re-emitting at all.

Symlinks those images' raw label files into a staging dir, so the UNMODIFIED
parallel_emit can be pointed at it (it globs raw/*.txt) with no new code path.
"""
import json, os, sys
sys.path.insert(0, "/workspace/spotlight/childfix")
from spotlight_run import child_age_only, merge

dets_path, raw_dir, stage_dir = sys.argv[1], sys.argv[2], sys.argv[3]
stems, n_dets = set(), 0
for line in open(dets_path):
    if not line.strip():
        continue
    r = json.loads(line)
    v = r.get("v")
    # re-derive from merge() itself; never trust the dump's own selection
    assert merge(v)["final_class"] == "Unknown", r["image_stem"]
    if child_age_only(v):
        assert merge(v, True)["final_class"] == "Child"
        stems.add(r["image_stem"])
        n_dets += 1
print(f"rescued detections {n_dets:,} across {len(stems):,} images")

os.makedirs(stage_dir, exist_ok=True)
missing = 0
for s in sorted(stems):
    src = os.path.join(raw_dir, s + ".txt")
    if not os.path.exists(src):
        missing += 1
        continue
    dst = os.path.join(stage_dir, s + ".txt")
    if not os.path.lexists(dst):
        os.symlink(src, dst)
print(f"staged {len(os.listdir(stage_dir)):,} raw label symlinks in {stage_dir}"
      f"   missing raw files: {missing}")
assert missing == 0, "a rescued image has no raw label file"
