#!/usr/bin/env python3
"""Score LAGENDA fl1199 SAM3 and Sol labels against the human 3-class labels."""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_NAME = {0: "Woman", 1: "Man", 2: "Child"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def yolo_box_to_xyxy(line: str, width: int, height: int) -> tuple[int, list[float]]:
    values = line.split()
    cls = int(values[0])
    cx, cy, bw, bh = map(float, values[1:5])
    return cls, [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/datasets/lagenda_full/fl1199"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/workspace/datasets/lagenda_full/eval_v2"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199/verdicts_sol.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199/human_comparison"),
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
    # run_model_children imports dataset_utils for its CLI path; match_boxes
    # itself has no dependency on it. Avoid requiring OpenCV merely to import
    # and reuse the project's exact matcher.
    sys.modules.setdefault("dataset_utils", types.ModuleType("dataset_utils"))
    from run_model_children import match_boxes  # type: ignore

    sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
    from spotlight_run import merge  # type: ignore

    dataset_root = args.dataset_root.resolve()
    eval_root = args.eval_root.resolve()
    results = read_jsonl(args.results.resolve())
    by_key = {(row["image_stem"], int(row["det_index"])): row for row in results}
    if len(by_key) != 1884:
        raise RuntimeError(f"expected 1,884 unique Sol verdicts, found {len(by_key):,}")

    gt_meta = {row["id"]: row for row in read_jsonl(eval_root / "gt.jsonl")}
    comparison: list[dict[str, Any]] = []
    gt_total = 0
    det_total = 0
    matched_total = 0
    correct_sol = 0
    correct_sam = 0
    sol_kept = 0
    matched_sol_dropped = 0
    confusion_sol: Counter[tuple[str, str]] = Counter()
    confusion_sam: Counter[tuple[str, str]] = Counter()
    per_gt = defaultdict(lambda: Counter(total=0, sol_correct=0, sam_correct=0, sol_dropped=0))

    for sidecar_path in sorted((dataset_root / "raw").glob("*.json")):
        stem = sidecar_path.stem
        sidecar = json.loads(sidecar_path.read_text())
        width, height = int(sidecar["width"]), int(sidecar["height"])
        detections = sidecar.get("detections", [])
        det_total += len(detections)

        box_lines = [line for line in (eval_root / "labels" / f"{stem}.txt").read_text().splitlines() if line.strip()]
        class_lines = [line for line in (eval_root / "labels_3class" / f"{stem}.txt").read_text().splitlines() if line.strip()]
        if len(box_lines) != len(class_lines):
            raise RuntimeError(f"human box/class count mismatch for {stem}")
        gt_total += len(box_lines)
        gt_boxes: list[list[float]] = []
        gt_classes: list[int] = []
        for box_line, class_line in zip(box_lines, class_lines):
            _box_cls, box = yolo_box_to_xyxy(box_line, width, height)
            class_id = int(class_line.split()[0])
            gt_boxes.append(box)
            gt_classes.append(class_id)

        det_tuples = [
            (index, *map(float, detection["box"]))
            for index, detection in enumerate(detections)
        ]
        matched = match_boxes(gt_boxes, det_tuples, args.match_iou)
        matched_det_indices = set()
        for gt_index, (det_tuple, match_iou) in matched.items():
            det_index = int(det_tuple[0])
            matched_det_indices.add(det_index)
            detection = detections[det_index]
            sol_row = by_key[(stem, det_index)]
            decision = merge(sol_row.get("v"))
            sol_class = decision["final_class"] if decision["keep"] and decision["final_class"] != "Unknown" else "Dropped"
            sam_class = CLASS_NAME[int(detection["cls"])]
            gt_class = CLASS_NAME[gt_classes[gt_index]]
            meta = gt_meta.get(f"{stem}_{gt_index}", {})
            matched_total += 1
            per_gt[gt_class]["total"] += 1
            if sol_class == "Dropped":
                matched_sol_dropped += 1
                per_gt[gt_class]["sol_dropped"] += 1
            else:
                sol_kept += 1
            if sol_class == gt_class:
                correct_sol += 1
                per_gt[gt_class]["sol_correct"] += 1
            if sam_class == gt_class:
                correct_sam += 1
                per_gt[gt_class]["sam_correct"] += 1
            confusion_sol[(gt_class, sol_class)] += 1
            confusion_sam[(gt_class, sam_class)] += 1
            comparison.append(
                {
                    "status": "matched",
                    "image_stem": stem,
                    "image": sidecar.get("image"),
                    "gt_index": gt_index,
                    "det_index": det_index,
                    "iou": round(float(match_iou), 6),
                    "gt_box": gt_boxes[gt_index],
                    "sam_box": detection["box"],
                    "gt_class": gt_class,
                    "gt_age": meta.get("gt_age"),
                    "gt_gender": meta.get("gt_gender"),
                    "sam_class": sam_class,
                    "sam_conf": detection.get("conf"),
                    "sol_class": sol_class,
                    "sol_verdict": sol_row.get("v"),
                    "sol_correct": sol_class == gt_class,
                    "sam_correct": sam_class == gt_class,
                }
            )

        for det_index, detection in enumerate(detections):
            if det_index in matched_det_indices:
                continue
            sol_row = by_key[(stem, det_index)]
            decision = merge(sol_row.get("v"))
            sol_class = decision["final_class"] if decision["keep"] and decision["final_class"] != "Unknown" else "Dropped"
            comparison.append(
                {
                    "status": "unmatched_detection",
                    "image_stem": stem,
                    "image": sidecar.get("image"),
                    "det_index": det_index,
                    "sam_box": detection["box"],
                    "sam_class": CLASS_NAME[int(detection["cls"])],
                    "sam_conf": detection.get("conf"),
                    "sol_class": sol_class,
                    "sol_verdict": sol_row.get("v"),
                }
            )
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched:
                continue
            gt_class = CLASS_NAME[gt_classes[gt_index]]
            meta = gt_meta.get(f"{stem}_{gt_index}", {})
            comparison.append(
                {
                    "status": "unmatched_human",
                    "image_stem": stem,
                    "image": sidecar.get("image"),
                    "gt_index": gt_index,
                    "gt_box": gt_box,
                    "gt_class": gt_class,
                    "gt_age": meta.get("gt_age"),
                    "gt_gender": meta.get("gt_gender"),
                }
            )

    labels = ["Woman", "Man", "Child", "Dropped"]
    metrics = {
        "dataset": "lagenda_full/fl1199",
        "human_labels": gt_total,
        "sam3_detections": det_total,
        "match_iou_threshold": args.match_iou,
        "matched_pairs": matched_total,
        "human_unmatched": gt_total - matched_total,
        "sam3_unmatched": det_total - matched_total,
        "sam3_recall_at_match_iou": round(matched_total / gt_total, 6),
        "sol_kept_on_matched": sol_kept,
        "sol_dropped_on_matched": matched_sol_dropped,
        "sol_accuracy_on_all_matched": round(correct_sol / matched_total, 6),
        "sol_accuracy_on_kept_matched": round(correct_sol / sol_kept, 6) if sol_kept else None,
        "sam3_class_accuracy_on_matched": round(correct_sam / matched_total, 6),
        "end_to_end_sol_correct_per_human": round(correct_sol / gt_total, 6),
        "per_human_class": {name: dict(per_gt[name]) for name in ["Woman", "Man", "Child"]},
        "sol_confusion_gt_rows": {
            gt: {pred: confusion_sol[(gt, pred)] for pred in labels}
            for gt in ["Woman", "Man", "Child"]
        },
        "sam3_confusion_gt_rows": {
            gt: {pred: confusion_sam[(gt, pred)] for pred in ["Woman", "Man", "Child"]}
            for gt in ["Woman", "Man", "Child"]
        },
        "caveat": (
            "Unmatched SAM3 detections are unmatched to LAGENDA's detector-derived human boxes; "
            "they are not automatically confirmed false positives."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparison)
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
