#!/usr/bin/env python3
"""Build a final-only Gemini vs batched-Sol dashboard for RunPod/Jupyter."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


PREFIXES = {
    "shiekhs": "shiekhs__",
    "children": "child__",
    "women": "women__",
    "men": "men__",
    "women_hd": "women_hd__",
    "randoms": "randoms__",
}

DISPLAY_NAMES = {
    "shiekhs": "Shiekhs",
    "children": "Children",
    "women": "Women",
    "men": "Men",
    "women_hd": "Women HD",
    "randoms": "Randoms",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def item_id(row: dict[str, Any]) -> str:
    return f"{row['image_stem']}--{int(row['det_index']):03d}"


def primary(verdict: dict[str, Any] | None) -> str:
    verdict = verdict or {}
    if verdict.get("gender") == "man":
        return "Man"
    if verdict.get("gender") == "woman":
        return "Child" if verdict.get("age_group") == "child" else "Woman"
    return "Unknown"


def compact(row: dict[str, Any]) -> dict[str, Any]:
    verdict = row.get("v") or {}
    return {
        "primary": primary(verdict),
        "verdict": verdict.get("verdict"),
        "gender": verdict.get("gender"),
        "age_group": verdict.get("age_group"),
        "estimated_age": verdict.get("estimated_age"),
        "confidence": verdict.get("confidence"),
        "highlight_quality": verdict.get("highlight_quality"),
        "head_covering": verdict.get("head_covering"),
        "apparent_race": verdict.get("apparent_race"),
        "exposed_body_parts": verdict.get("exposed_body_parts") or [],
    }


def labeled_compact(primary_label: str, **values: Any) -> dict[str, Any]:
    return {
        "primary": primary_label,
        "verdict": values.get("verdict"),
        "gender": values.get("gender"),
        "age_group": values.get("age_group"),
        "estimated_age": values.get("estimated_age"),
        "confidence": values.get("confidence"),
        "highlight_quality": values.get("highlight_quality"),
        "head_covering": values.get("head_covering"),
        "apparent_race": values.get("apparent_race"),
        "exposed_body_parts": values.get("exposed_body_parts") or [],
    }


def dataset_payload(
    collection: str,
    baseline_rows: list[dict[str, Any]],
    final_root: Path,
    dashboard_root: Path,
) -> dict[str, Any]:
    prefix = PREFIXES[collection]
    baseline = {
        item_id(row): row
        for row in baseline_rows
        if str(row.get("image_stem", "")).startswith(prefix)
    }
    verdict_path = final_root / f"{collection}-batched-b10" / "verdicts_sol_batched.jsonl"
    partial_path = final_root / f"{collection}-batched-b10" / "verdicts_sol_batched.partial.jsonl"
    use_final = verdict_path.is_file() and verdict_path.stat().st_size > 0
    sol_rows = load_jsonl(verdict_path if use_final else partial_path)
    sol = {str(row["item_id"]): row for row in sol_rows}
    progress = load_json(final_root / f"{collection}-batched-b10" / "progress.json")
    run_meta = load_json(final_root / f"{collection}-batched-b10" / "run_meta.json")

    compared_ids = [key for key in baseline if key in sol]
    disagreements: list[dict[str, Any]] = []
    for key in compared_ids:
        gemini_row = baseline[key]
        sol_row = sol[key]
        gemini = compact(gemini_row)
        batch = compact(sol_row)
        if gemini["primary"] == batch["primary"]:
            continue
        crop_path = Path(str(sol_row.get("crop_path") or ""))
        image_url = None
        if crop_path.is_file():
            relative = os.path.relpath(crop_path, dashboard_root)
            image_url = quote(relative.replace(os.sep, "/"), safe="/._-")
        disagreements.append(
            {
                "item_id": key,
                "image_url": image_url,
                "groups": [
                    "gemini_batch_disagreement",
                    f"{gemini['primary']}_to_{batch['primary']}",
                ],
                "person_px_height": gemini_row.get("person_px_height"),
                "image_dimensions": gemini_row.get("img_wh"),
                "gemini": gemini,
                "batch": batch,
            }
        )

    matches = sum(
        primary(baseline[key].get("v")) == primary(sol[key].get("v"))
        for key in compared_ids
    )
    gemini_counts = Counter(primary(row.get("v")) for row in baseline.values())
    sol_counts = Counter(primary(row.get("v")) for row in sol.values())
    expected = len(baseline)
    complete = len(compared_ids) == expected and expected > 0
    metrics = (
        run_meta
        if run_meta.get("status") == "complete"
        and int(run_meta.get("completed_detections") or 0) == expected
        and expected > 0
        else progress
    )
    usage = metrics.get("usage") or {}
    status = metrics.get("status") or ("not_started" if not sol else "running")

    return {
        "key": collection,
        "name": DISPLAY_NAMES[collection],
        "mode": "model_agreement",
        "title": f"Sol vs Gemini — {DISPLAY_NAMES[collection]}",
        "overview_description": "Primary labels are Man, Woman, Child, and Unknown.",
        "label_mix_description": f"All {expected:,} {DISPLAY_NAMES[collection]} detections.",
        "review_description": "Final production labeling: Gemini versus batched Sol.",
        "source_labels": [
            {"key": "gemini", "label": "Gemini", "class_name": ""},
            {"key": "batch", "label": "Final Sol", "class_name": "sol"},
        ],
        "group_options": [
            {"value": "any", "label": "All final disagreements"},
        ],
        "summary": {
            "all_count": expected,
            "compared_count": len(compared_ids),
            "final_match_count": matches,
            "final_disagreement_count": len(disagreements),
            "all_batch_primary": matches / len(compared_ids) if compared_ids else None,
            "failed_requests": metrics.get("failed_requests", 0),
            "credits": usage.get("charged_credits", 0),
            "elapsed_seconds": metrics.get("elapsed_seconds", 0),
            "workers": metrics.get("workers", 15),
            "batch_size": metrics.get("batch_size", 10),
            "model": metrics.get("model", "gpt-5.6-sol"),
            "service_tier": metrics.get("service_tier", "standard"),
            "status": status,
            "complete": complete,
        },
        "label_counts": {"gemini": dict(gemini_counts), "batch": dict(sol_counts)},
        "items": disagreements,
        "review_unavailable": None if disagreements else "No final disagreement images are available yet.",
    }


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def lagenda_payload(lagenda_root: Path, gemini_root: Path) -> dict[str, Any]:
    comparison_path = gemini_root / "human_comparison" / "comparison.jsonl"
    metrics_path = lagenda_root / "human_comparison" / "metrics.json"
    gemini_metrics_path = gemini_root / "human_comparison" / "metrics.json"
    verdicts_path = lagenda_root / "verdicts_sol.jsonl"
    gemini_verdicts_path = gemini_root / "verdicts_gemini37.jsonl"
    metrics = load_json(metrics_path)
    gemini_metrics = load_json(gemini_metrics_path)
    comparisons = load_jsonl(comparison_path)
    verdict_rows = load_jsonl(verdicts_path)
    gemini_verdict_rows = load_jsonl(gemini_verdicts_path)
    verdicts = {
        (row["image_stem"], int(row["det_index"])): row for row in verdict_rows
    }
    gemini_verdicts = {
        (row["image_stem"], int(row["det_index"])): row
        for row in gemini_verdict_rows
    }
    matched = [row for row in comparisons if row.get("status") == "matched"]
    disagreements: list[dict[str, Any]] = []
    for row in matched:
        key = (str(row["image_stem"]), int(row["det_index"]))
        sol_row = verdicts.get(key, {})
        gemini_row = gemini_verdicts.get(key, {})
        crop_path = str(gemini_row.get("crop_path") or sol_row.get("crop_path") or "")
        image_url = None
        if crop_path and Path(crop_path).is_file():
            image_url = quote(f"/files{crop_path}", safe="/._-")
        gt_age = row.get("gt_age")
        gt_gender = row.get("gt_gender")
        human = labeled_compact(
            str(row.get("gt_class") or "Unknown"),
            verdict="human_ground_truth",
            gender={"M": "man", "F": "woman"}.get(str(gt_gender), "unknown"),
            age_group=(
                "child"
                if isinstance(gt_age, (int, float)) and gt_age <= 12
                else "adult"
                if isinstance(gt_age, (int, float))
                else "unknown"
            ),
            estimated_age=gt_age,
            confidence="10 human votes",
            highlight_quality=f"IoU {float(row.get('iou') or 0):.3f}",
        )
        sam = labeled_compact(
            str(row.get("sam_class") or "Unknown"),
            verdict="sam3_detection",
            confidence=row.get("sam_conf"),
            highlight_quality=f"IoU {float(row.get('iou') or 0):.3f}",
        )
        batch = compact(sol_row)
        batch["primary"] = str(row.get("sol_class") or "Dropped")
        gemini = compact(gemini_row)
        gemini["primary"] = str(row.get("gemini_class") or gemini["primary"])
        if len({human["primary"], batch["primary"], gemini["primary"]}) == 1:
            continue
        groups: list[str] = []
        if batch["primary"] != human["primary"]:
            groups.append("human_sol_disagreement")
        if gemini["primary"] != human["primary"]:
            groups.append("human_gemini_disagreement")
        if gemini["primary"] != batch["primary"]:
            groups.append("gemini_sol_disagreement")
        if row.get("gemini_correct") and not row.get("sol_correct"):
            groups.append("gemini_wins")
        if row.get("sol_correct") and not row.get("gemini_correct"):
            groups.append("sol_wins")
        if not row.get("sol_correct") and not row.get("gemini_correct"):
            groups.append("both_wrong")
        if batch["primary"] == "Dropped":
            groups.append("sol_dropped")
        if gemini["primary"] == "Dropped":
            groups.append("gemini_dropped")
        groups.extend(
            [
                f"human_{human['primary']}_to_sol_{batch['primary']}",
                f"human_{human['primary']}_to_gemini_{gemini['primary']}",
            ]
        )
        width, height = sol_row.get("img_wh") or [None, None]
        disagreements.append(
            {
                "item_id": f"{key[0]}--{key[1]:03d}",
                "image_url": image_url,
                "groups": groups,
                "person_px_height": sol_row.get("person_px_height"),
                "image_dimensions": [width, height] if width and height else None,
                "iou": row.get("iou"),
                "human": human,
                "sam": sam,
                "batch": batch,
                "gemini": gemini,
            }
        )

    human_counts = Counter(str(row.get("gt_class") or "Unknown") for row in matched)
    sam_counts = Counter(str(row.get("sam_class") or "Unknown") for row in matched)
    sol_counts = Counter(str(row.get("sol_class") or "Dropped") for row in matched)
    gemini_counts = Counter(str(row.get("gemini_class") or "Dropped") for row in matched)
    correct_sol = sum(int(values.get("sol_correct", 0)) for values in metrics.get("per_human_class", {}).values())
    correct_gemini = sum(
        int(values.get("gemini_correct", 0))
        for values in gemini_metrics.get("per_human_class", {}).values()
    )
    usage = [row.get("usage") or {} for row in verdict_rows]
    started = [value for row in verdict_rows if (value := parse_time(row.get("started_at")))]
    finished = [value for row in verdict_rows if (value := parse_time(row.get("finished_at")))]
    elapsed = (max(finished) - min(started)).total_seconds() if started and finished else 0
    sol_accuracy = metrics.get("sol_accuracy_on_all_matched")
    sam_accuracy = metrics.get("sam3_class_accuracy_on_matched")
    gemini_accuracy = gemini_metrics.get("gemini_accuracy_on_matched_subset")
    gemini_sol_disagreements = sum(
        row.get("gemini_class") != row.get("sol_class") for row in matched
    )
    gemini_wins = sum(bool(row.get("gemini_correct")) and not bool(row.get("sol_correct")) for row in matched)
    sol_wins = sum(bool(row.get("sol_correct")) and not bool(row.get("gemini_correct")) for row in matched)
    return {
        "key": "lagenda",
        "name": "LAGENDA fl1199",
        "mode": "human_ground_truth",
        "title": "Gemini vs Sol vs Human — LAGENDA fl1199",
        "overview_description": "Human age and gender labels are the reference; Gemini 3.7 Flash, Sol, and raw SAM3 use the same matched boxes.",
        "label_mix_description": f"{len(matched):,} matched human/SAM3 box pairs at IoU ≥ {metrics.get('match_iou_threshold', 0.5):.1f}.",
        "review_description": "Every matched person where Human, final Sol, and Gemini 3.7 do not all agree; raw SAM3 is shown for context.",
        "source_labels": [
            {"key": "human", "label": "Human", "class_name": "human"},
            {"key": "sam", "label": "SAM3", "class_name": "sam"},
            {"key": "batch", "label": "Final Sol", "class_name": "sol"},
            {"key": "gemini", "label": "Gemini 3.7", "class_name": "gemini"},
        ],
        "group_options": [
            {"value": "any", "label": "All final-label disagreements"},
            {"value": "human_gemini_disagreement", "label": "Gemini differs from Human"},
            {"value": "human_sol_disagreement", "label": "Sol differs from Human"},
            {"value": "gemini_sol_disagreement", "label": "Gemini differs from Sol"},
            {"value": "gemini_wins", "label": "Gemini right, Sol wrong"},
            {"value": "sol_wins", "label": "Sol right, Gemini wrong"},
            {"value": "both_wrong", "label": "Gemini and Sol both wrong"},
            {"value": "sol_dropped", "label": "Sol dropped person"},
            {"value": "gemini_dropped", "label": "Gemini dropped person"},
        ],
        "label_filter_key": "human",
        "summary": {
            "all_count": len(matched),
            "compared_count": len(matched),
            "final_match_count": correct_gemini,
            "final_disagreement_count": len(disagreements),
            "all_batch_primary": sol_accuracy,
            "sol_accuracy": sol_accuracy,
            "gemini_accuracy": gemini_accuracy,
            "sam_accuracy": sam_accuracy,
            "accuracy_delta": (
                gemini_accuracy - sol_accuracy
                if isinstance(gemini_accuracy, (int, float)) and isinstance(sol_accuracy, (int, float))
                else None
            ),
            "gemini_correct": correct_gemini,
            "sol_correct": correct_sol,
            "gemini_error_count": len(matched) - correct_gemini,
            "sol_error_count": len(matched) - correct_sol,
            "gemini_sol_disagreement_count": gemini_sol_disagreements,
            "gemini_wins": gemini_wins,
            "sol_wins": sol_wins,
            "human_labels": metrics.get("human_labels"),
            "matched_pairs": metrics.get("matched_pairs"),
            "human_unmatched": metrics.get("human_unmatched"),
            "sam3_unmatched": metrics.get("sam3_unmatched"),
            "sam3_recall": metrics.get("sam3_recall_at_match_iou"),
            "sol_completed": len(verdicts),
            "gemini_completed": len(gemini_verdicts),
            "failed_requests": 0,
            "credits": round(sum(float(value.get("charged_credits", 0) or 0) for value in usage), 3),
            "elapsed_seconds": elapsed,
            "workers": 20,
            "batch_size": 1,
            "model": "Gemini 3.7 Flash + GPT-5.6 Sol",
            "service_tier": "subscription + standard",
            "config_chips": [
                "Gemini 3.7 Flash Medium",
                "GPT-5.6 Sol",
                "human ground truth",
                "1 crop / request",
                "1,884 shared SAM3 crops",
            ],
            "status": "complete" if len(verdicts) == 1884 and len(gemini_verdicts) == 1884 else "partial",
            "complete": len(verdicts) == 1884 and len(gemini_verdicts) == 1884 and len(matched) == metrics.get("matched_pairs"),
        },
        "label_counts": {
            "human": dict(human_counts),
            "sam": dict(sam_counts),
            "batch": dict(sol_counts),
            "gemini": dict(gemini_counts),
        },
        "items": disagreements,
        "review_unavailable": None if disagreements else "No final-label disagreements are available for review.",
        "caveat": (
            "Human age/gender labels are based on 10 crowdsourced votes; box geometry is detector-derived. "
            f"The {metrics.get('sam3_unmatched', 0):,} unmatched SAM3 detections are not automatically false positives. "
            f"Sources: {metrics_path}, {gemini_metrics_path}, {comparison_path}, {verdicts_path}, and {gemini_verdicts_path}."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("/workspace/datasets/haramblur_holdout/labeling/full/run/verdicts_batch.jsonl"),
    )
    parser.add_argument(
        "--final-root", type=Path, default=Path("/root/codex_sol_compare/final")
    )
    parser.add_argument(
        "--dashboard-root",
        type=Path,
        default=Path("/root/codex_sol_compare/comparison-dashboard"),
    )
    parser.add_argument(
        "--lagenda-root",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199"),
    )
    parser.add_argument(
        "--lagenda-gemini-root",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199_gemini37"),
    )
    args = parser.parse_args()
    args.dashboard_root.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_jsonl(args.baseline)
    datasets = {
        collection: dataset_payload(
            collection, baseline_rows, args.final_root, args.dashboard_root
        )
        for collection in ("shiekhs", "children", "women", "men", "women_hd", "randoms")
    }
    datasets["lagenda"] = lagenda_payload(
        args.lagenda_root.resolve(), args.lagenda_gemini_root.resolve()
    )
    payload = {
        "title": "Final labeling comparison",
        "datasets": datasets,
        "caveat": (
            "Gemini is a comparison reference, not human ground truth. "
            "Only final batched Sol labels are included; intermediate experiments and single-item Sol runs are excluded."
        ),
    }
    output = args.dashboard_root / "dashboard-data.json"
    encoded = json.dumps(payload, ensure_ascii=False)
    output.write_text(encoded, encoding="utf-8")
    (args.dashboard_root / "dashboard-data.js").write_text(
        f"window.DASHBOARD_DATA = {encoded};\n", encoding="utf-8"
    )
    template_path = args.dashboard_root / "static" / "index.html"
    app_path = args.dashboard_root / "static" / "app.js"
    styles_path = args.dashboard_root / "static" / "styles.css"
    if template_path.is_file() and app_path.is_file() and styles_path.is_file():
        safe_encoded = encoded.replace("<", "\\u003c")
        html = template_path.read_text(encoding="utf-8")
        html = html.replace(
            '<link rel="stylesheet" href="styles.css" />',
            f"<style>{styles_path.read_text(encoding='utf-8')}</style>",
        )
        html = html.replace(
            '<script src="dashboard-data.js"></script>',
            f"<script>window.DASHBOARD_DATA = {safe_encoded};</script>",
        )
        html = html.replace(
            '<script src="app.js" defer></script>',
            f"<script>{app_path.read_text(encoding='utf-8')}</script>",
        )
        (args.dashboard_root / "index.html").write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "datasets": {
                    key: {
                        "compared": value["summary"]["compared_count"],
                        "disagreements": len(value["items"]),
                        "status": value["summary"]["status"],
                    }
                    for key, value in payload["datasets"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
