import json, os, random, sys
d = "/workspace/spotlight/run/oiv7_train/labels"
stems = [json.loads(l)["image_stem"] for l in
         open("/workspace/spotlight/child_rescue_train_dets.jsonl")]
random.seed(0)
sizes = []
for s in random.sample(stems, min(400, len(stems))):
    p = os.path.join(d, s + ".txt")
    try:
        sizes.append(os.path.getsize(p))
    except OSError:
        pass
n = len(sizes); avg = sum(sizes)/n
print(f"sampled {n} label files: avg {avg:,.0f} B  min {min(sizes)}  max {max(sizes)}")
print(f"-> a full 475,207-file emit is roughly {avg*475207/1e9:.1f} GB")
