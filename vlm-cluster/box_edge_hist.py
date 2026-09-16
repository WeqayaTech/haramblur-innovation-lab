#!/usr/bin/env python3
"""EXP-2026-21 arm A: is the INT8-vs-fp32 box-edge error quantized on a grid?

Reads two raw-dump dirs from run_ultralytics_labels.py (one JSON per image with
`width`, `height`, `detections[{cls, box_xyxy (original-image px), conf, kept}]`),
matches kept detections per image (same class, IoU >= --iou, conf >= --conf, greedy by IoU),
converts each of the four edge differences (A - B) into *model-input* pixels
(multiply by the letterbox gain min(S/w, S/h)), and reports:

  * a histogram of edge errors (bin --bin px),
  * the fraction of edge errors within +-tol of an integer multiple of the predicted
    grid step (--scale * S px; default 0.0042 = the measured int8 output scale),
  * the same fraction for a *null* grid (step shifted by half a tooth) as a control —
    a real comb makes the first number high and the second low; a smooth
    distribution makes them equal.

No numpy needed. --selftest runs on synthetic data.
"""
import argparse, json, math, os, sys
from collections import Counter

from run_model_children import iou


def load(d, conf):
    out = {}
    for f in os.listdir(d):
        if not f.endswith(".json"): continue
        j = json.load(open(os.path.join(d, f)))
        dets = [x for x in j["detections"] if x.get("kept") and x["conf"] >= conf]
        out[f] = (j["width"], j["height"], dets)
    return out


def match(da, db, thr):
    """Greedy same-class matching by IoU. Returns list of (boxA, boxB)."""
    pairs = []
    cands = []
    for i, a in enumerate(da):
        for k, b in enumerate(db):
            if a["cls"] != b["cls"]: continue
            v = iou(a["box_xyxy"], b["box_xyxy"])
            if v >= thr: cands.append((v, i, k))
    cands.sort(reverse=True)
    ua, ub = set(), set()
    for v, i, k in cands:
        if i in ua or k in ub: continue
        ua.add(i); ub.add(k); pairs.append((da[i]["box_xyxy"], db[k]["box_xyxy"]))
    return pairs


def analyse(errs, step, tol, binw, label):
    n = len(errs)
    if n == 0:
        print(f"{label}: no matched edges"); return {}
    on_grid = sum(1 for e in errs if abs(e / step - round(e / step)) * step <= tol)
    null = sum(1 for e in errs if abs((e + step / 2) / step - round((e + step / 2) / step)) * step <= tol)
    absmean = sum(abs(e) for e in errs) / n
    srt = sorted(abs(e) for e in errs); med = srt[n // 2]; p90 = srt[int(0.9 * n)]
    hist = Counter(round(e / binw) * binw for e in errs)
    print(f"\n== {label} ==")
    print(f"edges={n}  mean|err|={absmean:.3f} px  median={med:.3f}  p90={p90:.3f}")
    print(f"grid step={step:.3f} px  within ±{tol} px of k·step: {on_grid / n:.1%}   null grid (shifted ½ step): {null / n:.1%}")
    print("histogram (px, count) — peaks at k·step mean a comb:")
    lo, hi = -3 * step, 3 * step
    for c in sorted(k for k in hist if lo <= k <= hi):
        bar = "#" * min(60, int(60 * hist[c] / max(hist.values())))
        print(f"{c:+7.2f} {hist[c]:7d} {bar}")
    return {"edges": n, "mean_abs": absmean, "median": med, "p90": p90, "on_grid": on_grid / n, "null_grid": null / n}


def run(a_dir, b_dir, imgsz, scale, iou_thr, conf, tol, binw, label):
    A, B = load(a_dir, conf), load(b_dir, conf)
    errs = []
    by_size = {}   # box long side (model px) bucket -> list of |edge err|
    nimg = npair = 0
    for f in A:
        if f not in B: continue
        w, h, da = A[f]; _, _, db = B[f]
        gain = min(imgsz / w, imgsz / h)
        pairs = match(da, db, iou_thr)
        nimg += 1; npair += len(pairs)
        for ba, bb in pairs:
            e = [(ba[i] - bb[i]) * gain for i in range(4)]
            errs.extend(e)
            side = max(bb[2] - bb[0], bb[3] - bb[1]) * gain
            # stride-level proxy: YOLO assigns objects to P3/P4/P5 (stride 8/16/32) roughly by size
            bucket = "<64px (≈stride 8)" if side < 64 else "64–256px (≈stride 16)" if side < 256 else ">256px (≈stride 32)"
            by_size.setdefault(bucket, []).extend(abs(x) for x in e)
    print(f"{label}: images={nimg} matched boxes={npair}")
    r = analyse(errs, scale * imgsz, tol, binw, label)
    print("mean|edge err| by reference box size (a per-tensor int8 scale on the stride-unit regression tensor predicts ∝ stride):")
    for k in ("<64px (≈stride 8)", "64–256px (≈stride 16)", ">256px (≈stride 32)"):
        v = by_size.get(k, [])
        if v:
            v.sort(); print(f"  {k:24s} edges={len(v):6d} mean={sum(v) / len(v):.2f} px  median={v[len(v) // 2]:.2f}  p90={v[int(0.9 * len(v))]:.2f}")
            r[f"bucket {k}"] = {"edges": len(v), "mean": sum(v) / len(v), "median": v[len(v) // 2]}
    return r


def selftest():
    import random, tempfile
    random.seed(0)
    for kind in ("comb", "smooth"):
        ta, tb = tempfile.mkdtemp(), tempfile.mkdtemp()
        for n in range(200):
            w, h = 800, 600; gain = 640 / 800
            dets_b, dets_a = [], []
            for k in range(5):
                x1, y1 = random.uniform(0, 500), random.uniform(0, 400)
                b = [x1, y1, x1 + random.uniform(40, 200), y1 + random.uniform(40, 150)]
                if kind == "comb":
                    a = [v + (random.choice([-1, 0, 0, 1]) * 0.0042 * 640 + random.gauss(0, 0.05)) / gain for v in b]
                else:
                    a = [v + random.gauss(0, 1.5) / gain for v in b]
                dets_b.append({"cls": 1, "box_xyxy": b, "conf": 0.8, "kept": True})
                dets_a.append({"cls": 1, "box_xyxy": a, "conf": 0.8, "kept": True})
            for d, dets in ((ta, dets_a), (tb, dets_b)):
                json.dump({"image": f"{n}.jpg", "width": w, "height": h, "detections": dets}, open(f"{d}/{n}.json", "w"))
        r = run(ta, tb, 640, 0.0042, 0.7, 0.25, 0.25, 0.1, f"selftest-{kind}")
        if kind == "comb": assert r["on_grid"] > 0.9 and r["null_grid"] < 0.2, r
        else: assert abs(r["on_grid"] - r["null_grid"]) < 0.15, r
    print("\nSELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="raw dir, arm under test (int8)")
    ap.add_argument("--b", help="raw dir, reference (fp32 tflite)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--scale", type=float, default=0.0042, help="int8 output scale (normalized units)")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=0.25, help="px tolerance around a grid tooth")
    ap.add_argument("--bin", type=float, default=0.1)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", help="write summary here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    r = run(a.a, a.b, a.imgsz, a.scale, a.iou, a.conf, a.tol, a.bin, a.label or f"{a.a} vs {a.b}")
    if a.json: json.dump(r, open(a.json, "w"), indent=1)
