import argparse, json, os

# runs against the pod volume by default, or a local mirror of the
# result JSONs (see _smallperson_review/) so results are re-readable
# with no pod at all
_ap = argparse.ArgumentParser(description="EXP-2026-19 arms A+B summary")
_ap.add_argument("--base", default="/workspace/exp19")
B = _ap.parse_args().base
MODELS = ["yolo11N-640", "y26n_gradsupp", "y26n_warm50-2", "y26n_sop50", "gelannfav14r4fw_gemlb_v2"]

print("=== ARM A: crowd_small — REAL small people (800 imgs, 16,314 people <=96px full-body) ===")
hdr = ("model", "recall", "found", "precision", "light", "partial", "heavy", "clearFP/100")
print("%-26s%9s%8s%11s%8s%9s%8s%13s" % hdr)
for m in MODELS:
    f = "%s/crowd_small/%s/eval/summary.json" % (B, m)
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    r, p, o = d["detection_recall"], d["detection_precision"], d["recall_by_occlusion"]
    gv = lambda k: o.get(k, {}).get("rate", 0) * 100
    print("%-26s%8.2f%%%8d%10.1f%%%7.1f%%%8.1f%%%7.1f%%%13.2f" % (
        m, r["rate"] * 100, r["k"], p["rate"] * 100,
        gv("light"), gv("partial"), gv("heavy"),
        d["unmatched_breakdown"]["clear_fp_per_100_images"]))

print()
print("=== ARM B: synth_shrunk — the SAME 480 people at 5 apparent sizes (AP50) ===")
print("%-16s%8s%9s%8s%8s%8s" % ("model", "height", "mAP50", "Woman", "Man", "Child"))
for m in MODELS:
    for h in ["h96", "h64", "h48", "h32", "h24"]:
        f = "%s/synth/%s/%s_map.json" % (B, m, h)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        pc = d.get("per_class", {})
        g = lambda k: pc.get(k, {}).get("ap50", float("nan"))
        print("%-16s%8s%9.3f%8.3f%8.3f%8.3f" % (
            m, h, d.get("map50", float("nan")), g("Woman"), g("Man"), g("Child")))
