"""Stage only the images the SAM3-confirmed child rescue can change."""
import json, os, sys
sys.path.insert(0, "/workspace/spotlight/childfix")
from spotlight_run import child_age_only, child_rescue_allowed, merge

dets_path, raw_dir, stage_dir = sys.argv[1], sys.argv[2], sys.argv[3]
stems, n = set(), 0
for line in open(dets_path):
    if not line.strip():
        continue
    r = json.loads(line)
    v = r["v"]
    assert merge(v)["final_class"] == "Unknown"
    if child_age_only(v) and child_rescue_allowed(r, require_sam_child=True):
        assert r["sam_class_id"] == 2
        stems.add(r["image_stem"]); n += 1
print(f"SAM3-confirmed rescues {n:,} across {len(stems):,} images")
os.makedirs(stage_dir, exist_ok=True)
for s in sorted(stems):
    src = os.path.join(raw_dir, s + ".txt")
    assert os.path.exists(src), src
    dst = os.path.join(stage_dir, s + ".txt")
    if not os.path.lexists(dst):
        os.symlink(src, dst)
print(f"staged {len(os.listdir(stage_dir)):,} raw label symlinks")
