#!/usr/bin/env python3
"""Score available Gemini 3.7 verdicts on the exact Sol-matched LAGENDA rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gemini-results",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199_gemini37/verdicts_gemini37.jsonl"),
    )
    parser.add_argument(
        "--sol-comparison",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199/human_comparison/comparison.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199_gemini37/human_comparison"),
    )
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
    from spotlight_run import merge  # type: ignore

    gemini_rows = read_jsonl(args.gemini_results)
    by_key = {(row["image_stem"], int(row["det_index"])): row for row in gemini_rows}
    if len(by_key) != len(gemini_rows):
        raise RuntimeError("duplicate Gemini result keys")

    output_rows: list[dict[str, Any]] = []
    matched = correct_gemini = correct_sol = correct_sam = kept = 0
    agreement_sol = agreement_sam = 0
    confusion: Counter[tuple[str, str]] = Counter()
    per_gt = defaultdict(lambda: Counter(total=0, gemini_correct=0, sol_correct=0, sam_correct=0))

    for row in read_jsonl(args.sol_comparison):
        if row.get("det_index") is None:
            continue
        key = row["image_stem"], int(row["det_index"])
        gemini = by_key.get(key)
        if gemini is None:
            continue
        decision = merge(gemini.get("v"))
        gemini_class = (
            decision["final_class"]
            if decision["keep"] and decision["final_class"] != "Unknown"
            else "Dropped"
        )
        enriched = dict(row)
        enriched.update(
            {
                "gemini_class": gemini_class,
                "gemini_verdict": gemini.get("v"),
                "gemini_model": gemini.get("model"),
                "gemini_usage": gemini.get("usage"),
            }
        )
        if row.get("status") == "matched":
            gt = row["gt_class"]
            sol_class = row["sol_class"]
            sam_class = row["sam_class"]
            enriched["gemini_correct"] = gemini_class == gt
            matched += 1
            kept += gemini_class != "Dropped"
            correct_gemini += gemini_class == gt
            correct_sol += sol_class == gt
            correct_sam += sam_class == gt
            agreement_sol += gemini_class == sol_class
            agreement_sam += gemini_class == sam_class
            confusion[(gt, gemini_class)] += 1
            per_gt[gt]["total"] += 1
            per_gt[gt]["gemini_correct"] += gemini_class == gt
            per_gt[gt]["sol_correct"] += sol_class == gt
            per_gt[gt]["sam_correct"] += sam_class == gt
        output_rows.append(enriched)

    labels = ["Woman", "Man", "Child", "Dropped"]
    metrics = {
        "dataset": "lagenda_full/fl1199",
        "gemini_model": "gemini-3.7-flash-medium",
        "gemini_verdicts_available": len(by_key),
        "expected_verdicts": 1884,
        "coverage": round(len(by_key) / 1884, 6),
        "matched_human_pairs_in_available_subset": matched,
        "gemini_kept_on_matched": kept,
        "gemini_accuracy_on_matched_subset": round(correct_gemini / matched, 6) if matched else None,
        "sol_accuracy_on_same_subset": round(correct_sol / matched, 6) if matched else None,
        "sam3_accuracy_on_same_subset": round(correct_sam / matched, 6) if matched else None,
        "gemini_sol_agreement_on_same_subset": round(agreement_sol / matched, 6) if matched else None,
        "gemini_sam3_agreement_on_same_subset": round(agreement_sam / matched, 6) if matched else None,
        "per_human_class": {name: dict(per_gt[name]) for name in ["Woman", "Man", "Child"]},
        "gemini_confusion_gt_rows": {
            gt: {pred: confusion[(gt, pred)] for pred in labels}
            for gt in ["Woman", "Man", "Child"]
        },
        "partial": len(by_key) != 1884,
        "caveat": (
            "Metrics cover all matched human rows."
            if len(by_key) == 1884
            else "Partial metrics cover only matched human rows whose Gemini verdict has completed."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
