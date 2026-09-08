#!/usr/bin/env python3
"""
Periodic small-object progress tracker for a LIVE training run.

Ultralytics only writes `best.pt`/`last.pt` by default (no per-epoch snapshots
unless --save-period is set, and this run wasn't launched with it). `last.pt` is
overwritten at the end of every epoch, so this script polls results.csv for new
epoch rows and, every `--every` epochs, copies last.pt aside BEFORE the next
epoch can overwrite it, then scores that snapshot with the exact same
smallperson_v1 protocol already used for the y26n_warm50-2 / y26n_gradsupp /
yolo11N-640 baseline (crowd_small real-photo recall + synth_shrunk mAP at
96/64/48/32/24px) -- so every row in the trend is directly comparable to that
baseline table, not a new metric.

    python3 track_small_object_progress.py \
        --run-dir /workspace/exp19/train/y26n_sop50 \
        --every 2 \
        --out /workspace/exp19/sop50_progress
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SP = Path("/workspace/datasets/smallperson_v1")
SIZES = ("h96", "h64", "h48", "h32", "h24")


def read_epochs_done(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with open(results_csv) as f:
        rows = list(csv.reader(f))
    return max(0, len(rows) - 1)  # minus header


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


def eval_checkpoint(ckpt: Path, out_dir: Path, epoch: int, log: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # crowd_small real-photo arm
    crowd_out = out_dir / "crowd_small"
    run(["python3", "-u", str(HERE / "run_ultralytics_labels.py"),
         "--engine", "ultralytics", "--model", str(ckpt),
         "--images", str(SP / "crowd_small" / "images"),
         "--out", str(crowd_out),
         "--conf", "0.45", "--floor", "0.001", "--iou", "0.7",
         "--imgsz", "640", "--device", "cuda:0"], log)
    run(["python3", "-u", str(HERE / "eval_negatives_crowd.py"),
         "--mode", "crowd", "--images", str(SP / "crowd_small" / "images"),
         "--pred-labels", str(crowd_out / "labels"),
         "--gt-odgt", str(SP / "crowd_small" / "annotations" / "persons.odgt"),
         "--out", str(crowd_out / "eval")], log)

    result = {"epoch": epoch, "ckpt": str(ckpt)}
    summary_f = crowd_out / "eval" / "summary.json"
    if summary_f.exists():
        s = json.loads(summary_f.read_text())
        result["crowd_dets_kept"] = s.get("dets_total_kept")
        result["crowd_n_gt"] = s.get("n_gt_persons")
        result["crowd_recall"] = s.get("detection_recall", {}).get("rate")
        result["crowd_precision"] = s.get("detection_precision", {}).get("rate")
        for occ in ("light", "partial", "heavy"):
            result[f"crowd_recall_{occ}"] = s.get("recall_by_occlusion", {}).get(occ, {}).get("rate")
        result["crowd_fp_per_100img"] = s.get("unmatched_breakdown", {}).get("clear_fp_per_100_images")

    # synth controlled arm, all 5 sizes
    for h in SIZES:
        h_out = out_dir / "synth" / h
        run(["python3", "-u", str(HERE / "run_ultralytics_labels.py"),
             "--engine", "ultralytics", "--model", str(ckpt),
             "--images", str(SP / "synth_shrunk" / h / "images"),
             "--out", str(h_out),
             "--conf", "0.45", "--floor", "0.001", "--iou", "0.7",
             "--imgsz", "640", "--device", "cuda:0"], log)
        map_out = out_dir / "synth" / f"{h}_map.json"
        run(["python3", "-u", str(HERE / "map_eval.py"),
             "--raw", str(h_out / "raw"),
             "--gt-labels", str(SP / "synth_shrunk" / h / "labels"),
             "--ignore-labels", str(SP / "synth_shrunk" / h / "ignore"),
             "--expect-floor", "0.001", "--out", str(map_out)], log)
        if map_out.exists():
            m = json.loads(map_out.read_text())
            result[f"{h}_map50"] = m.get("map50")
            result[f"{h}_map75"] = m.get("map75")
            result[f"{h}_map50_95"] = m.get("map50_95")
            by_area = m.get("by_area", {})
            result[f"{h}_mAP50_95_small"] = by_area.get("mAP50_95_small")
            result[f"{h}_mAP50_95_medium"] = by_area.get("mAP50_95_medium")
            result[f"{h}_mAP50_95_large"] = by_area.get("mAP50_95_large")
            for cls, v in m.get("per_class", {}).items():
                result[f"{h}_{cls}_ap50"] = v.get("ap50")
                result[f"{h}_{cls}_ap75"] = v.get("ap75")
                result[f"{h}_{cls}_ap50_95"] = v.get("ap50_95")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="ultralytics training run dir")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_csv = run_dir / "results.csv"
    last_pt = run_dir / "weights" / "last.pt"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "eval_log.txt"
    trend_csv = out / "trend.csv"

    done_epochs = set()
    if trend_csv.exists():
        with open(trend_csv) as f:
            for row in csv.DictReader(f):
                done_epochs.add(int(row["epoch"]))

    print(f"[tracker] watching {results_csv}, every={args.every}, out={out}", flush=True)
    fieldnames = None
    while True:
        n = read_epochs_done(results_csv)
        target = (n // args.every) * args.every
        if target > 0 and target not in done_epochs and last_pt.exists():
            snap = out / f"epoch{target}.pt"
            shutil.copy2(last_pt, snap)
            t0 = time.time()
            result = eval_checkpoint(snap, out / f"epoch{target}", target, log)
            result["eval_seconds"] = round(time.time() - t0, 1)
            snap.unlink(missing_ok=True)  # keep only the eval outputs, not every checkpoint copy

            if fieldnames is None:
                if trend_csv.exists():
                    with open(trend_csv) as f:
                        fieldnames = next(csv.DictReader(f)).keys()
                else:
                    fieldnames = list(result.keys())
            write_header = not trend_csv.exists()
            with open(trend_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow({k: result.get(k, "") for k in fieldnames})

            done_epochs.add(target)
            print(f"[tracker] epoch {target}: crowd_recall={result.get('crowd_recall')} "
                  f"crowd_precision={result.get('crowd_precision')} "
                  f"fp/100img={result.get('crowd_fp_per_100img')} | "
                  f"h96 map50={result.get('h96_map50')} map50-95={result.get('h96_map50_95')} | "
                  f"h64_map50-95={result.get('h64_map50_95')} h48={result.get('h48_map50_95')} "
                  f"h32={result.get('h32_map50_95')} h24={result.get('h24_map50_95')} "
                  f"(eval took {result['eval_seconds']}s)", flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
