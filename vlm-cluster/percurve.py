import argparse, json, os
from collections import defaultdict

_ap = argparse.ArgumentParser(description="EXP-2026-19 arm C curves")
_ap.add_argument("--base", default="/workspace/exp19/paste")
_B = _ap.parse_args().base
HS = [96, 64, 48, 32, 24]
hdr = "model".ljust(16) + "metric".ljust(24) + "".join(("h%d" % h).ljust(8) for h in HS)
print(hdr); print("-" * len(hdr))
for m in ["yolo11N-640", "y26n_gradsupp", "y26n_warm50-2", "y26n_sop50", "gelannfav14r4fw_gemlb_v2"]:
    det = defaultdict(lambda: [0, 0]); cls = defaultdict(lambda: [0, 0])
    wom = defaultdict(lambda: [0, 0]); chi = defaultdict(lambda: [0, 0])
    for line in open(os.path.join(_B, m, "eval", "sam_matches.jsonl")):
        r = json.loads(line)
        h = int(r["image"].split("_")[0][1:])
        found = r["status"] != "missed"
        det[h][1] += 1; det[h][0] += found
        if found:
            cls[h][1] += 1; cls[h][0] += (r["pred_class"] == r["gt_class"])
        if r["gt_class"] == "Woman":
            wom[h][1] += 1; wom[h][0] += (found and r["pred_class"] == "Woman")
        if r["gt_class"] == "Child":
            chi[h][1] += 1
            chi[h][0] += (found and r["pred_class"] in ("Woman", "Man"))
    def row(name, d):
        print(m.ljust(16) + name.ljust(24) + "".join(
            ("%.1f" % (d[h][0] / d[h][1] * 100) if d[h][1] else "-").ljust(8) for h in HS))
    row("detection recall %", det)
    row("3-class acc (found) %", cls)
    row("WOMAN end-to-end %", wom)
    row("child->adult leak %", chi)
    print()
