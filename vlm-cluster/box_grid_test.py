#!/usr/bin/env python3
"""EXP-2026-21 arm A (corrected): are a model's OWN box coordinates on the int8 output grid?

The first version of arm A (box_edge_hist.py) histogrammed int8 - fp32 edge differences and
looked for a comb. That was the wrong test: the difference of a gridded value and a continuous
value is continuous. The right test is on the int8 coordinates themselves.

For each kept detection in a raw dump, undo the letterbox (ultralytics LetterBox, center=True,
scaleup=True, auto=False) to get model-input pixels, convert to normalized cx, cy, w, h, and
measure the distance to the nearest multiple of the output-tensor scale. Reports the fraction
within --tol (in units of the step) — for a gridded output this is ~100 % and for a continuous
one (fp32 control) it is ~2*tol.

The scale must be the EXACT value from the .tflite tensor details (not the rounded 0.0042):
over 640 px a 1 % scale error walks off the grid after ~100 steps.
"""
import argparse, json, os, sys


def letterbox_params(w, h, S):
    gain = min(S / w, S / h)
    new_w, new_h = round(w * gain), round(h * gain)
    dw, dh = (S - new_w) / 2, (S - new_h) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    return gain, left, top


def run(raw_dir, S, scale, zp, tol, conf, label, limit=None):
    n = on = 0
    resid = []
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
    if limit: files = files[:limit]
    for f in files:
        j = json.load(open(os.path.join(raw_dir, f)))
        gain, left, top = letterbox_params(j["width"], j["height"], S)
        for d in j["detections"]:
            if not d.get("kept") or d["conf"] < conf: continue
            x1, y1, x2, y2 = d["box_xyxy"]
            # model-input px -> normalized; the tensor holds cx, cy, w, h
            X1, X2 = (x1 * gain + left) / S, (x2 * gain + left) / S
            Y1, Y2 = (y1 * gain + top) / S, (y2 * gain + top) / S
            vals = [(X1 + X2) / 2, (Y1 + Y2) / 2, X2 - X1, Y2 - Y1]
            for v in vals:
                q = (v / scale) + zp        # int8 code if on-grid
                r = abs(q - round(q))       # distance to nearest code, in steps
                resid.append(r); n += 1
                if r <= tol: on += 1
    resid.sort()
    print(f"{label}: values={n} on-grid(±{tol} step)={on / max(n,1):.1%}  median residual={resid[n // 2]:.3f} step  "
          f"(continuous data would give ≈{2 * tol:.0%} on-grid, median 0.25)")
    return {"values": n, "on_grid": on / max(n, 1), "median_resid": resid[n // 2] if n else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--scale", type=float, required=True, help="exact output-tensor scale from tensor details")
    ap.add_argument("--zp", type=float, default=-128)
    ap.add_argument("--tol", type=float, default=0.05, help="tolerance in grid steps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = run(a.raw, a.imgsz, a.scale, a.zp, a.tol, a.conf, a.label or a.raw, a.limit)
    if a.json: json.dump(r, open(a.json, "w"), indent=1)
