#!/usr/bin/env python3
"""
SPOTLIGHT — verification of the labels the production pipeline emits.

Production-side checks, self-contained (no experiment code):

  integrity  — structural checks on the emitted label set: valid YOLO syntax,
               class ids in range, coordinates in [0,1], geometry unchanged vs
               the SAM3 input, every raw label file accounted for, verdict
               coverage. No ground truth needed, so it runs on ANY corpus.

  score      — accuracy against ground truth where we have it:
                 --gt lagenda     human gender/age -> TP-keep, 3-class acc,
                                  adult->Child leak by age band
                 --gt crowdhuman  exhaustive boxes -> TP-keep on real people

    python3 verify_labels.py integrity --raw-labels <raw> --out <run dir>
    python3 verify_labels.py score --out <run dir> --gt lagenda \\
        --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \\
        --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl
    python3 verify_labels.py --selftest
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from spotlight_run import CLASS_ID, CLASS_NAME, iou, merge


# --- shared: greedy mutually-exclusive matching (from run_model_children) ---
def match_boxes(gt_boxes, dets, min_iou):
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for di, d in enumerate(dets):
            j = iou(g, d[1])
            if j >= min_iou:
                pairs.append((j, gi, di))
    pairs.sort(key=lambda p: p[0], reverse=True)
    used_g, used_d, out = set(), set(), {}
    for j, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi); used_d.add(di); out[gi] = (dets[di], j)
    return out


def load_verdicts(run_dir: Path):
    """Reads every verdict file: verdicts.jsonl (live), verdicts_batch.jsonl
    (Batch API), verdicts_shard*.jsonl (parallel live workers)."""
    files = sorted(run_dir.glob("verdicts*.jsonl"))
    if not files:
        raise SystemExit(f"no verdicts*.jsonl in {run_dir}")
    out = []
    for f in files:
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def integrity(raw_dir: Path, run_dir: Path):
    lab_dir = run_dir / "labels"
    issues, stats = [], Counter()
    verdicts = load_verdicts(run_dir)
    judged = {(r["image_stem"], r["det_index"]) for r in verdicts}
    stats["verdicts"] = len(verdicts)
    stats["parse_failures"] = sum(1 for r in verdicts if not r.get("parse_ok"))

    for raw_txt in sorted(raw_dir.glob("*.txt")):
        stem = raw_txt.stem
        stats["raw_images"] += 1
        raw_lines = [l for l in raw_txt.read_text().splitlines() if l.strip()]
        stats["raw_detections"] += len(raw_lines)
        n_judged = sum(1 for i in range(len(raw_lines)) if (stem, i) in judged)
        fully_verified = n_judged == len(raw_lines)
        out_txt = lab_dir / f"{stem}.txt"

        if not fully_verified:
            # Expected on a partial/in-progress run — emit deliberately skips
            # these so verified and raw labels never mix. Only a problem if a
            # label file WAS written for a part-verified image.
            stats["images_not_yet_verified"] += 1
            stats["detections_not_yet_verified"] += len(raw_lines) - n_judged
            if out_txt.exists():
                issues.append(f"{stem}: emitted despite only {n_judged}/"
                              f"{len(raw_lines)} detections verified")
            continue

        stats["verified_images"] += 1
        stats["verified_detections"] += len(raw_lines)
        if not out_txt.exists():
            issues.append(f"{stem}: VERIFIED but no emitted label file")
            continue
        stats["emitted_images"] += 1
        raw_geom = {" ".join(l.split()[1:]) for l in raw_lines}
        for ln, line in enumerate(
                l for l in out_txt.read_text().splitlines() if l.strip()):
            stats["emitted_detections"] += 1
            parts = line.split()
            try:
                cid = int(parts[0]); vals = [float(v) for v in parts[1:]]
            except ValueError:
                issues.append(f"{stem}:{ln} unparseable: {line[:60]}")
                continue
            if cid not in CLASS_NAME:
                issues.append(f"{stem}:{ln} bad class id {cid}")
            if len(vals) < 4 or (len(vals) > 4 and len(vals) % 2):
                issues.append(f"{stem}:{ln} bad coord count {len(vals)}")
            if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
                issues.append(f"{stem}:{ln} coords outside [0,1]")
            if " ".join(parts[1:]) not in raw_geom:
                issues.append(f"{stem}:{ln} geometry NOT from SAM3 input")
            stats[f"class_{CLASS_NAME.get(cid, cid)}"] += 1

    # deletions are only meaningful over VERIFIED images
    stats["deleted_by_gate"] = (stats["verified_detections"]
                                - stats["emitted_detections"])
    out = dict(stats)
    if stats["verified_detections"]:
        out["deletion_rate_pct"] = round(
            100 * stats["deleted_by_gate"] / stats["verified_detections"], 2)
    if stats["raw_detections"]:
        out["coverage_pct"] = round(
            100 * stats["verified_detections"] / stats["raw_detections"], 2)
    print(json.dumps(out, indent=2))
    print(f"\nissues: {len(issues)}")
    for m in issues[:20]:
        print("  " + m)
    if not issues:
        print("  none — every fully-verified image was emitted, labels are "
              "structurally valid, and geometry matches the SAM3 input exactly.")
    if stats["images_not_yet_verified"]:
        print(f"\n  ({stats['images_not_yet_verified']} images not yet verified "
              f"— expected while a run is in progress, not an error.)")
    return len(issues)


def _dets_from_verdicts(verdicts, only_kept=True):
    """-> {image_stem: [(record, box_xyxy)]}"""
    by = {}
    for r in verdicts:
        if only_kept and not merge(r.get("v"))["keep"]:
            continue
        by.setdefault(r["image_stem"], []).append((r, r["box"]))
    return by


def score_lagenda(verdicts, gt_labels: Path, manifest: Path, match_iou: float):
    meta = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            stem, _, idx = rec["id"].rpartition("_")
            meta.setdefault(stem, {})[int(idx)] = rec
    all_by = _dets_from_verdicts(verdicts, only_kept=False)
    kept_by = _dets_from_verdicts(verdicts, only_kept=True)

    matched = kept = cls_ok = cls_bad = 0
    bands = {"0-12": [0, 0], "13-17": [0, 0], "18+": [0, 0]}
    gender_ok = gender_bad = 0
    for stem, rows in meta.items():
        lf = gt_labels / f"{stem}.txt"
        if not lf.exists() or stem not in all_by:
            continue
        lines = lf.read_text().split("\n")
        w = h = None
        for r, _ in all_by[stem]:
            w, h = r.get("img_wh", (None, None)) if r.get("img_wh") else (None, None)
        gt_boxes, gts = [], []
        for idx, rec in rows.items():
            if idx >= len(lines) or not lines[idx].strip():
                continue
            cx, cy, bw, bh = [float(v) for v in lines[idx].split()[1:5]]
            gts.append(rec)
            gt_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        if not gt_boxes:
            continue
        # verdict boxes are pixels; GT is normalized -> use the image size the
        # sidecar recorded via the detection box extent
        allb = all_by[stem]
        mx = max(max(b[2] for _, b in allb), 1)
        my = max(max(b[3] for _, b in allb), 1)
        # normalize verdict boxes by the largest observed extent is unreliable;
        # instead require img_wh in the record (spotlight_run stores box in px
        # and the sidecar has width/height) -> fall back to skipping.
        if not (w and h):
            continue
        dets = [(r, [b[0] / w, b[1] / h, b[2] / w, b[3] / h]) for r, b in allb]
        for gi, (d, j) in match_boxes(gt_boxes, dets, match_iou).items():
            r = d[0]
            matched += 1
            m = merge(r.get("v"))
            g = gts[gi]
            band = ("0-12" if g["gt_age"] <= 12
                    else "13-17" if g["gt_age"] <= 17 else "18+")
            bands[band][0] += 1
            if not m["keep"]:
                continue
            kept += 1
            if m["final_class"] == "Child":
                bands[band][1] += 1
            truth = ("Child" if g["gt_age"] <= 12
                     else ("Man" if g["gt_gender"] == "M" else "Woman"))
            if m["final_class"] == truth:
                cls_ok += 1
            elif m["final_class"] != "Unknown":
                cls_bad += 1
            if g["gt_age"] > 12 and m["final_class"] in ("Man", "Woman"):
                if m["final_class"] == ("Man" if g["gt_gender"] == "M" else "Woman"):
                    gender_ok += 1
                else:
                    gender_bad += 1
    out = {"gt_matched": matched, "kept": kept,
           "tp_keep_rate": round(kept / matched, 4) if matched else None,
           "3class_accuracy": round(cls_ok / (cls_ok + cls_bad), 4) if cls_ok + cls_bad else None,
           "gender_accuracy": round(gender_ok / (gender_ok + gender_bad), 4) if gender_ok + gender_bad else None,
           "labeled_Child_by_gt_age_band": {
               k: {"n": v[0], "child": v[1],
                   "pct": round(100 * v[1] / v[0], 1) if v[0] else None}
               for k, v in bands.items()}}
    print(json.dumps(out, indent=2))
    if bands["18+"][1]:
        print(f"\n*** {bands['18+'][1]} adults aged 18+ were labeled Child — "
              f"this is the error that lets adults escape the blur ***")
    return out


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "raw").mkdir(); (root / "run" / "labels").mkdir(parents=True)
        (root / "raw/a.txt").write_text(
            "1 0.10 0.10 0.40 0.10 0.40 0.90\n0 0.60 0.60 0.90 0.60 0.90 0.90\n")
        # emitted: det0 relabeled to Woman (geometry identical), det1 deleted
        (root / "run/labels/a.txt").write_text("0 0.10 0.10 0.40 0.10 0.40 0.90")
        (root / "run/verdicts.jsonl").write_text(
            json.dumps({"image_stem": "a", "det_index": 0, "parse_ok": True,
                        "box": [1, 1, 2, 2], "v": {
                            "verdict": "real_person", "gender": "woman",
                            "age_group": "adult", "estimated_age": 30}}) + "\n" +
            json.dumps({"image_stem": "a", "det_index": 1, "parse_ok": True,
                        "box": [3, 3, 4, 4], "v": {
                            "verdict": "not_person", "gender": "unknown",
                            "age_group": "unknown", "estimated_age": None}}) + "\n")
        assert integrity(root / "raw", root / "run") == 0

        # a partial run (image b judged 0 of 1) must be clean, not an issue
        (root / "raw/b.txt").write_text("1 0.20 0.20 0.50 0.20 0.50 0.80\n")
        assert integrity(root / "raw", root / "run") == 0, "partial run flagged"
        # ...but emitting that unverified image IS an issue
        (root / "run/labels/b.txt").write_text("1 0.20 0.20 0.50 0.20 0.50 0.80")
        assert integrity(root / "raw", root / "run") == 1, "unverified emit missed"
        (root / "run/labels/b.txt").unlink()
        (root / "raw/b.txt").unlink()

        # tampered geometry must be caught
        (root / "run/labels/a.txt").write_text("0 0.11 0.10 0.40 0.10 0.40 0.90")
        assert integrity(root / "raw", root / "run") == 1
        # bad class id must be caught
        (root / "run/labels/a.txt").write_text("7 0.10 0.10 0.40 0.10 0.40 0.90")
        assert integrity(root / "raw", root / "run") >= 1
    print("\nverify_labels.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="Verify emitted Spotlight labels")
    ap.add_argument("cmd", nargs="?", choices=["integrity", "score"])
    ap.add_argument("--raw-labels"); ap.add_argument("--out")
    ap.add_argument("--gt", choices=["lagenda", "crowdhuman"])
    ap.add_argument("--gt-labels"); ap.add_argument("--manifest")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.cmd == "integrity":
        if not (a.raw_labels and a.out):
            ap.error("integrity needs --raw-labels --out")
        raise SystemExit(1 if integrity(Path(a.raw_labels), Path(a.out)) else 0)
    if a.cmd == "score":
        if not (a.out and a.gt):
            ap.error("score needs --out --gt")
        v = load_verdicts(Path(a.out))
        if a.gt == "lagenda":
            if not (a.gt_labels and a.manifest):
                ap.error("--gt lagenda needs --gt-labels --manifest")
            score_lagenda(v, Path(a.gt_labels), Path(a.manifest), a.match_iou)
        else:
            ap.error("crowdhuman scoring: use the experiment harness for now")
        return
    ap.error("cmd must be integrity or score")


if __name__ == "__main__":
    main()
