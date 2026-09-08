#!/usr/bin/env python3
"""
Spotlight pipeline — accurate cost/time estimation for a target corpus.

Cost is driven by two things that vary per corpus, so both are MEASURED
rather than assumed:

  1. detections per image   -> counted from SAM3 label files (free, no API)
  2. tokens per detection   -> measured from a real calibration run, because
                               crop size (and therefore image tokens) depends
                               on how big people are in your images

    # step 1 (free): scan labels you already have for the target corpus
    python3 estimate_cost.py scan --labels /workspace/exp10/raw_full/crowd

    # step 2 (~$0.10): calibrate tokens/detection on a small real sample,
    #                  then project. Reuses any pipeline run dir.
    python3 estimate_cost.py project \\
        --labels /workspace/exp10/raw_full/crowd \\
        --calibration-run /workspace/exp10/full_crowd \\
        --corpus-images 500000

    python3 estimate_cost.py --selftest

Rates come from model_pricing.json when filled, else the --rate flags
(defaults are the press-reported Flash-Lite figures; token counts are always
measured, only the $/token multiplier is assumed).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BATCH_DISCOUNT = 0.5          # Google Batch API list price vs standard
SEC_PER_CALL = 1.3            # measured sequential wall-clock per API call


def scan_labels(labels_dir: Path):
    """Detections per image from SAM3 output (.json sidecars or .txt labels).
    Free — no API, no images loaded."""
    counts = []
    sidecars = sorted(labels_dir.glob("*.json"))
    if sidecars:
        for f in sidecars:
            try:
                counts.append(len(json.loads(f.read_text()).get("detections", [])))
            except (ValueError, OSError):
                continue
        kind = "raw sidecars"
    else:
        for f in sorted(labels_dir.glob("*.txt")):
            counts.append(sum(1 for ln in f.read_text().splitlines() if ln.strip()))
        kind = "YOLO .txt labels"
    return counts, kind


def calibration(run_dir: Path):
    """Measured tokens/detection from a completed pipeline run."""
    cost = json.loads((run_dir / "cost_report.json").read_text())
    n = sum(1 for _ in (run_dir / "verdicts.jsonl").open())
    if cost["input_tokens"] == 0:
        raise SystemExit(
            f"{run_dir}/cost_report.json has 0 tokens — it was overwritten by a "
            "resumed run that made no new calls. Use a run dir whose calls were "
            "actually made, or re-run a fresh small sample into a new --out dir.")
    return cost["input_tokens"] / n, cost["output_tokens"] / n, n


def report(counts, kind, in_tok, out_tok, calib_n, corpus_images,
           rate_in, rate_out, prompt_note=""):
    imgs = len(counts)
    dets = sum(counts)
    dpi = dets / imgs if imgs else 0
    per_det = (in_tok * rate_in + out_tok * rate_out) / 1e6
    print(f"\nCORPUS SCAN  ({kind})")
    print(f"  images scanned        {imgs:,}")
    print(f"  detections            {dets:,}")
    print(f"  detections per image  mean {dpi:.2f} · median "
          f"{statistics.median(counts) if counts else 0:.0f} · "
          f"p90 {sorted(counts)[int(0.9*len(counts))-1] if counts else 0} · "
          f"max {max(counts) if counts else 0}")
    print(f"  images with 0 people  {sum(1 for c in counts if c == 0):,} "
          f"({100*sum(1 for c in counts if c == 0)/imgs if imgs else 0:.1f}%)")
    print(f"\nCALIBRATION  (measured on {calib_n:,} real detections){prompt_note}")
    print(f"  input tokens/detection   {in_tok:,.0f}")
    print(f"  output tokens/detection  {out_tok:,.0f}")
    print(f"  cost per 1,000 dets      ${per_det*1000:.3f} standard  "
          f"${per_det*1000*BATCH_DISCOUNT:.3f} batch")

    print(f"\nPROJECTION for {corpus_images:,} images at {dpi:.2f} people/image")
    proj_dets = corpus_images * dpi
    std, bat = proj_dets * per_det, proj_dets * per_det * BATCH_DISCOUNT
    hours = proj_dets * SEC_PER_CALL / 3600
    print(f"  detections to label   {proj_dets:,.0f}")
    print(f"  API cost              ${std:,.0f} standard   ${bat:,.0f} batch")
    print(f"  wall-clock            {hours:,.0f} h sequential  ·  "
          f"{hours/10:,.0f} h at 10x parallel  ·  batch: async")
    print(f"\n  sensitivity (people/image drives everything):")
    for mult, lab in ((0.5, "half as crowded"), (1.0, "as scanned"), (2.0, "twice as crowded")):
        d = corpus_images * dpi * mult
        print(f"    {lab:18s} {d:12,.0f} dets   ${d*per_det:9,.0f} std   "
              f"${d*per_det*BATCH_DISCOUNT:9,.0f} batch")
    print("\nNote: token counts are exact (from the API usage metadata). Rates "
          "$0.30 in / $2.50 out per 1M and the 50% Batch discount are VERIFIED "
          "against ai.google.dev/gemini-api/docs/pricing (2026-07-27).")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "lbl").mkdir()
        for i, n in enumerate([0, 1, 3, 7]):
            json.dump({"detections": [{} for _ in range(n)]},
                      open(root / f"lbl/img{i}.json", "w"))
        counts, kind = scan_labels(root / "lbl")
        assert sorted(counts) == [0, 1, 3, 7] and kind == "raw sidecars", counts

        run = root / "run"; run.mkdir()
        (run / "verdicts.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
        (run / "cost_report.json").write_text(json.dumps(
            {"input_tokens": 2000, "output_tokens": 200}))
        i_tok, o_tok, n = calibration(run)
        assert (i_tok, o_tok, n) == (1000, 100, 2), (i_tok, o_tok, n)
        report(counts, kind, i_tok, o_tok, n, 1000, 0.30, 2.50)
    print("\nestimate_cost.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="Spotlight cost/time estimator")
    ap.add_argument("cmd", nargs="?", choices=["scan", "project"])
    ap.add_argument("--labels", help="SAM3 label dir for the target corpus")
    ap.add_argument("--calibration-run", help="completed pipeline run dir")
    ap.add_argument("--corpus-images", type=int, default=500_000)
    ap.add_argument("--rate-in", type=float, default=0.30, help="$/1M input tokens")
    ap.add_argument("--rate-out", type=float, default=2.50, help="$/1M output tokens")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.labels:
        ap.error("--labels is required")
    counts, kind = scan_labels(Path(args.labels))
    if not counts:
        raise SystemExit(f"no label files found in {args.labels}")
    if args.cmd == "scan":
        imgs, dets = len(counts), sum(counts)
        print(f"{kind}: {imgs:,} images, {dets:,} detections, "
              f"{dets/imgs:.2f} per image (median "
              f"{statistics.median(counts):.0f}, max {max(counts)})")
        return
    if not args.calibration_run:
        ap.error("project needs --calibration-run")
    i_tok, o_tok, n = calibration(Path(args.calibration_run))
    report(counts, kind, i_tok, o_tok, n, args.corpus_images,
           args.rate_in, args.rate_out)


if __name__ == "__main__":
    main()
