#!/usr/bin/env python3
"""
Mirror a LIVE ultralytics training run's metrics into W&B without touching the
training process at all.

Ultralytics only wires up its W&B callback at trainer __init__ (see
ultralytics/utils/callbacks/wb.py, registered via add_integration_callbacks in
engine/trainer.py) -- once a run is already executing, there is no supported way
to inject a new callback into its live process. Rather than risk restarting a
multi-hour run to get logging, this script is a separate, read-only process: it
polls results.csv (the standard per-epoch training/val metrics ultralytics
already writes) and, optionally, a small-object trend.csv (from
track_small_object_progress.py) and forwards each NEW row to a W&B run via the
plain wandb.log() API. It never opens, writes to, or signals the training
process -- pure log-tailing.

    WANDB_API_KEY=... python3 wandb_mirror.py \
        --results-csv /workspace/exp19/train/y26n_sop50/results.csv \
        --trend-csv /workspace/exp19/sop50_progress/trend.csv \
        --project haramblur-smallobj --name y26n_sop50-mirror
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def read_rows(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float_dict(row: dict, skip=("ckpt",)) -> dict:
    out = {}
    for k, v in row.items():
        if k in skip or v in ("", None):
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True, help="ultralytics results.csv")
    ap.add_argument("--trend-csv", help="track_small_object_progress.py trend.csv (optional)")
    ap.add_argument("--project", default="haramblur-smallobj")
    ap.add_argument("--name", required=True)
    ap.add_argument("--poll-seconds", type=int, default=30)
    args = ap.parse_args()

    import wandb

    run = wandb.init(project=args.project, name=args.name, job_type="mirror",
                      config={"mirrors": args.results_csv,
                              "trend_source": args.trend_csv})
    print(f"[mirror] wandb run: {run.url}", flush=True)

    # Two independent metric families land out of order relative to each other
    # (small-object evals lag several epochs behind the training metrics they
    # were snapshotted from), which violates wandb's single global step being
    # monotonically increasing. Give each family its own x-axis instead of
    # relying on the implicit global step.
    wandb.define_metric("train/epoch")
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("small_obj/epoch")
    wandb.define_metric("small_obj/*", step_metric="small_obj/epoch")

    results_csv = Path(args.results_csv)
    trend_csv = Path(args.trend_csv) if args.trend_csv else None

    logged_epochs = set()
    logged_trend_epochs = set()

    while True:
        for row in read_rows(results_csv):
            epoch = int(float(row["epoch"]))
            if epoch in logged_epochs:
                continue
            metrics = to_float_dict(row)
            metrics = {f"train/{k}": v for k, v in metrics.items()}
            wandb.log(metrics)
            logged_epochs.add(epoch)
            print(f"[mirror] logged train epoch {epoch}", flush=True)

        if trend_csv is not None:
            for row in read_rows(trend_csv):
                epoch = int(float(row["epoch"]))
                if epoch in logged_trend_epochs:
                    continue
                metrics = to_float_dict(row)
                metrics = {f"small_obj/{k}": v for k, v in metrics.items()}
                wandb.log(metrics)
                logged_trend_epochs.add(epoch)
                print(f"[mirror] logged small-object epoch {epoch}", flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
