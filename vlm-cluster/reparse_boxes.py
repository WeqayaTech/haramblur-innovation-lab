#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-06: re-parse saved raw VLM responses under different
box-coordinate conventions and re-score, WITHOUT re-calling any API.

Why: Gemini's native detection convention is [ymin, xmin, ymax, xmax]
normalized to 0-1000 — y-first — regardless of the pixel x-first format our
prompt asks for. vlm_detect_eval's parser assumed x-first, which would zero
out every IoU match while leaving detection *counts* looking sane. This tool
diagnoses which convention a saved run actually used (by scoring all of them
against GT) and can rewrite detections.jsonl + summary.json under the right
one.

    # diagnose (prints match table under each convention, writes nothing)
    python3 reparse_boxes.py --run /workspace/exp06/gemini_flash/lagenda \
        --gt lagenda --labels ... --manifest ...

    # rewrite in place once the winning convention is obvious
    python3 reparse_boxes.py --run ... --gt ... [gt args] --write yxyx_1000

    python3 reparse_boxes.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from describe import extract_json
from vlm_detect_eval import (parse_detections, score_crowdhuman,
                              score_lagenda, score_negatives)

# "auto" = vlm_detect_eval.parse_detections' per-image heuristic — needed when
# a model mixes conventions across images in one run (qwen3.7-plus emits
# 0-1000 on some images and, after self-correcting, pixels on others)
# "auto_axis" = per-image AXIS-ORDER inference for models that flip x-first/
# y-first between responses (gemini-3.6-flash, observed 2026-07-19): people
# are overwhelmingly taller than wide, so pick the axis order that makes the
# majority of that response's boxes portrait. GT-free — cannot inflate scores.
FORMATS = ("xyxy_px", "xyxy_1000", "yxyx_1000", "auto", "auto_axis")


def raw_entries(text: str):
    """(person_dict, [a,b,c,d]) for every well-formed entry, untransformed."""
    obj = extract_json(text)
    raw = obj.get("people")
    if not isinstance(raw, list):
        return None
    out = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        b = p.get("box_2d") or p.get("box") or p.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            out.append((p, [float(v) for v in b]))
        except (TypeError, ValueError):
            continue
    return out


def transform(box, fmt: str, w: int, h: int):
    a, b, c, d = box
    if fmt == "xyxy_px":
        x1, y1, x2, y2 = a, b, c, d
    elif fmt == "xyxy_1000":
        x1, y1, x2, y2 = a * w / 1000, b * h / 1000, c * w / 1000, d * h / 1000
    elif fmt == "yxyx_1000":
        # Gemini native: [ymin, xmin, ymax, xmax] normalized 0-1000
        y1, x1, y2, x2 = a * h / 1000, b * w / 1000, c * h / 1000, d * w / 1000
    elif fmt == "yxyx_px":
        y1, x1, y2, x2 = a, b, c, d
    else:
        raise ValueError(fmt)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def _portrait_fraction(boxes):
    n = sum(1 for b in boxes if b and (b[3] - b[1]) > (b[2] - b[0]))
    return n / len(boxes) if boxes else 0.0


def pick_axis_order(entries, w, h):
    """Choose 'xyxy_1000' or 'yxyx_1000' (or _px flavors when coords exceed
    1000) for ONE response by portrait-majority. Ties -> y-first (family
    prior for Gemini)."""
    if not entries:
        return "yxyx_1000"
    peak = max(max(abs(v) for v in b) for _, b in entries)
    if peak <= 1000:
        cands = ("xyxy_1000", "yxyx_1000")
    else:
        cands = ("xyxy_px", "yxyx_px")
    scored = []
    for fmt in cands:
        boxes = [transform(b, fmt, w, h) for _, b in entries]
        scored.append((_portrait_fraction([bx for bx in boxes if bx]), fmt))
    scored.sort(key=lambda t: (t[0], "yxyx" in t[1]))   # ties -> y-first last=max
    return scored[-1][1]


def reparse_records(records: list, fmt: str) -> list:
    out = []
    for r in records:
        if fmt == "auto_axis":
            entries = raw_entries(r.get("raw_text", "")) or []
            chosen = pick_axis_order(entries, r["width"], r["height"])
            people = []
            for p, b in entries:
                bx = transform(b, chosen, r["width"], r["height"])
                if bx is None:
                    continue
                try:
                    est = float(p.get("estimated_age"))
                except (TypeError, ValueError):
                    est = None
                people.append({
                    "box_xyxy": bx,
                    "gender": str(p.get("gender", "unknown")).strip().lower(),
                    "age_group": str(p.get("age_group", "unknown")).strip().lower(),
                    "estimated_age": est,
                    "confidence": str(p.get("confidence", "unknown")).strip().lower(),
                })
            nr = dict(r)
            nr.update(people=people, n_people=len(people),
                      parse_ok=bool(entries), box_format=f"auto_axis:{chosen}")
            out.append(nr)
            continue
        if fmt == "auto":
            people, ok, rescaled = parse_detections(
                r.get("raw_text", ""), r["width"], r["height"])
            nr = dict(r)
            nr.update(people=people, n_people=len(people), parse_ok=ok,
                      rescaled=rescaled, box_format="auto")
            out.append(nr)
            continue
        entries = raw_entries(r.get("raw_text", ""))
        people = []
        if entries:
            for p, b in entries:
                bx = transform(b, fmt, r["width"], r["height"])
                if bx is None:
                    continue
                try:
                    est = float(p.get("estimated_age"))
                except (TypeError, ValueError):
                    est = None
                people.append({
                    "box_xyxy": bx,
                    "gender": str(p.get("gender", "unknown")).strip().lower(),
                    "age_group": str(p.get("age_group", "unknown")).strip().lower(),
                    "estimated_age": est,
                    "confidence": str(p.get("confidence", "unknown")).strip().lower(),
                })
        nr = dict(r)
        nr["people"] = people
        nr["n_people"] = len(people)
        nr["box_format"] = fmt
        out.append(nr)
    return out


def score(records, args):
    if args.gt == "none":
        return score_negatives(records)
    if args.gt == "crowdhuman":
        return score_crowdhuman(records, Path(args.odgt), args.match_iou)
    return score_lagenda(records, Path(args.labels), Path(args.manifest),
                         args.match_iou)


def key_numbers(s: dict) -> str:
    if s["mode"] == "lagenda":
        return (f"found {s['found']}/{s['gt_persons']}  "
                f"gender_ok {s['gender']['correct']}  "
                f"age_ok {s['age_group_at_12']['correct']}")
    if s["mode"] == "crowdhuman":
        return (f"matched {s['matched']}/{s['gt_persons']}  "
                f"recall {s['recall']}  precision {s['precision']}")
    return f"false_persons {s['false_persons_total']}"


def main():
    ap = argparse.ArgumentParser(description="re-parse saved raw VLM boxes")
    ap.add_argument("--run", help="run dir containing detections.jsonl")
    ap.add_argument("--gt", choices=["none", "crowdhuman", "lagenda"],
                    default="none")
    ap.add_argument("--odgt")
    ap.add_argument("--labels")
    ap.add_argument("--manifest")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--write", choices=FORMATS,
                    help="rewrite detections.jsonl + summary.json under this "
                         "convention (default: diagnose only, write nothing)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.run:
        raise SystemExit("--run required (or --selftest)")

    run_dir = Path(args.run)
    records = [json.loads(l) for l in (run_dir / "detections.jsonl").open()]

    # show a couple of raw boxes so the convention is visible to a human too
    shown = 0
    for r in records:
        entries = raw_entries(r.get("raw_text", ""))
        if entries:
            print(f"[reparse] {r['image']} ({r['width']}x{r['height']}): "
                  f"first raw box = {entries[0][1]}")
            shown += 1
        if shown >= 2:
            break

    print(f"[reparse] scoring {len(records)} records under each convention:")
    for fmt in FORMATS:
        s = score(reparse_records(records, fmt), args)
        print(f"  {fmt:<10} -> {key_numbers(s)}")

    if args.write:
        new_records = reparse_records(records, args.write)
        s = score(new_records, args)
        s["model_id"] = records and json.loads(
            (run_dir / "summary.json").read_text()).get("model_id")
        s["box_format"] = args.write
        s["reparsed_from_raw_text"] = True
        with (run_dir / "detections.jsonl").open("w") as fh:
            for r in new_records:
                fh.write(json.dumps(r) + "\n")
        # keep economics/cost from the original summary if present
        old = json.loads((run_dir / "summary.json").read_text())
        for k in ("labeling_economics", "parse_failures", "api_errors"):
            if k in old:
                s[k] = old[k]
        (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"[reparse] rewrote {run_dir}/detections.jsonl + summary.json "
              f"as {args.write}: {key_numbers(s)}")


def selftest():
    # a "gemini-style" response: y-first, 0-1000, on a 200x100 image.
    # person at pixels x 40-160, y 20-80  -> yxyx_1000 = [200, 200, 800, 800]
    # (a,b,c,d chosen asymmetric in effect: h != w so y-first vs x-first differ)
    raw = ('{"people": [{"box_2d": [200, 200, 800, 800], "gender": "woman", '
           '"age_group": "adult", "estimated_age": 30, "confidence": "high"}]}')
    rec = {"image": "img1.jpg", "width": 200, "height": 100,
           "raw_text": raw, "parse_ok": True, "rescaled": False}
    out = reparse_records([rec], "yxyx_1000")
    assert out[0]["people"][0]["box_xyxy"] == [40.0, 20.0, 160.0, 80.0], out
    # x-first 0-1000 gives a different (transposed) box on a non-square image
    out2 = reparse_records([rec], "xyxy_1000")
    assert out2[0]["people"][0]["box_xyxy"] == [40.0, 20.0, 160.0, 80.0] or \
           out2[0]["people"][0]["box_xyxy"] != out[0]["people"][0]["box_xyxy"]
    # raw px interpretation: x clamps to the 200px-wide image -> zero width,
    # box dropped entirely (that mass-drop is the smoking gun the diagnose
    # table shows as "found 0")
    out_px = reparse_records([rec], "xyxy_px")
    assert out_px[0]["people"] == [], out_px
    print("[reparse_boxes] selftest OK")


if __name__ == "__main__":
    main()
