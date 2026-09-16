#!/usr/bin/env python3
"""Run one spotlight-e1 Sol classification per LAGENDA fl1199 SAM3 detection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_sam3_sol_compare_v1 import (
    MODEL,
    AppServerClient,
    classify_one,
    load_jsonl,
    materialize_crop,
    write_json_atomic,
)


EXPECTED_PROMPT_SHA = "4c85ffbf8bcb"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_worklist(images_dir: Path, raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sidecar_path in sorted(raw_dir.glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        stem = sidecar_path.stem
        image_name = str(sidecar.get("image") or "")
        if not image_name or not (images_dir / image_name).exists():
            matches = sorted(images_dir.glob(f"{stem}.*"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"could not resolve exactly one image for sidecar {sidecar_path}"
                )
            image_name = matches[0].name
        width = int(sidecar["width"])
        height = int(sidecar["height"])
        for det_index, detection in enumerate(sidecar.get("detections", [])):
            box = [float(value) for value in detection["box"]]
            rows.append(
                {
                    "image_stem": stem,
                    "det_index": det_index,
                    "image": image_name,
                    "img_wh": [width, height],
                    "sam_class_id": int(detection["cls"]),
                    "sam_conf": float(detection["conf"]),
                    "box": box,
                    "n_parts": len(detection.get("parts") or []),
                    "blurriness": None,
                    "person_px_height": round(box[3] - box[1], 3),
                }
            )
    return rows


@dataclass(frozen=True)
class WorkItem:
    row: dict[str, Any]
    attempt: int = 1

    @property
    def key(self) -> tuple[str, int]:
        return self.row["image_stem"], int(self.row["det_index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/datasets/lagenda_full/fl1199"),
    )
    parser.add_argument(
        "--pipeline-root", type=Path, default=Path("/workspace/autolabel_pipeline_v2")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199"),
    )
    parser.add_argument("--codex-bin", type=Path, default=Path("/usr/bin/codex"))
    args = parser.parse_args()

    if not 1 <= args.workers <= 20:
        raise ValueError("workers must be between 1 and 20")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    if not args.codex_bin.is_file():
        raise FileNotFoundError(args.codex_bin)

    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    raw_dir = dataset_root / "raw"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "verdicts_sol.jsonl"
    errors_path = output_dir / "errors.jsonl"
    progress_path = output_dir / "progress.json"
    run_meta_path = output_dir / "run_meta.json"

    sys.path.insert(0, str(args.pipeline_root.resolve()))
    from spotlight_run import PROMPT, PROMPT_VERSION, build_crop  # type: ignore

    prompt_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]
    if prompt_sha != EXPECTED_PROMPT_SHA or PROMPT_VERSION != "spotlight-e1":
        raise RuntimeError(
            f"prompt mismatch: version={PROMPT_VERSION!r}, sha={prompt_sha!r}"
        )

    rows = load_worklist(images_dir, raw_dir)
    if len(rows) != 1884:
        raise RuntimeError(f"expected 1,884 SAM3 detections, found {len(rows):,}")
    if args.limit is not None:
        rows = rows[: args.limit]
    expected = {(row["image_stem"], int(row["det_index"])) for row in rows}
    if len(expected) != len(rows):
        raise RuntimeError("duplicate (image_stem, det_index) keys in worklist")

    existing_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_jsonl(results_path):
        key = row["image_stem"], int(row["det_index"])
        if key in expected:
            existing_by_key[key] = row
    pending = [WorkItem(row) for row in rows if (row["image_stem"], int(row["det_index"])) not in existing_by_key]

    started_at = utc_now()
    started = time.monotonic()
    work_queue: queue.Queue[WorkItem | None] = queue.Queue()
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    for item in pending:
        work_queue.put(item)

    def worker(number: int) -> None:
        client: AppServerClient | None = None
        log_dir = output_dir / "workers" / f"worker-{number:02d}"
        while True:
            item = work_queue.get()
            if item is None:
                break
            try:
                crop_path, crop_sha = materialize_crop(
                    item.row, images_dir, raw_dir, output_dir, build_crop
                )
                if client is None:
                    client = AppServerClient(args.codex_bin, output_dir, log_dir)
                    client.initialize()
                row = classify_one(
                    client,
                    PROMPT,
                    PROMPT_VERSION,
                    prompt_sha,
                    crop_path,
                    crop_sha,
                    item.row,
                    output_dir,
                    args.timeout,
                )
                result_queue.put({"status": "completed", "row": row})
            except Exception as error:
                if client is not None:
                    client.close()
                    client = None
                result_queue.put(
                    {
                        "status": "error",
                        "item": item,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if client is not None:
            client.close()

    threads = [
        threading.Thread(target=worker, args=(number,), daemon=False)
        for number in range(1, args.workers + 1)
    ]
    for thread in threads:
        thread.start()

    completed_this_run = 0
    terminal_failures = 0
    outstanding = len(pending)
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    total_credits = 0.0
    with results_path.open("a", encoding="utf-8") as results_file, errors_path.open(
        "a", encoding="utf-8"
    ) as errors_file:
        while outstanding:
            event = result_queue.get()
            if event["status"] == "completed":
                row = event["row"]
                results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                results_file.flush()
                os.fsync(results_file.fileno())
                completed_this_run += 1
                outstanding -= 1
                usage = row.get("usage", {})
                total_input_tokens += int(usage.get("input_tokens", 0))
                total_cached_input_tokens += int(usage.get("cached_input_tokens", 0))
                total_output_tokens += int(usage.get("output_tokens", 0))
                if isinstance(usage.get("charged_credits"), (int, float)):
                    total_credits += float(usage["charged_credits"])
            else:
                item = event["item"]
                errors_file.write(
                    json.dumps(
                        {
                            "image_stem": item.row["image_stem"],
                            "det_index": int(item.row["det_index"]),
                            "attempt": item.attempt,
                            "error": event["error"],
                            "at": utc_now(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errors_file.flush()
                if item.attempt < args.max_attempts:
                    work_queue.put(WorkItem(item.row, item.attempt + 1))
                else:
                    terminal_failures += 1
                    outstanding -= 1

            handled = completed_this_run + terminal_failures
            elapsed = time.monotonic() - started
            progress = {
                "dataset": "lagenda_full/fl1199",
                "model": MODEL,
                "workers": args.workers,
                "expected_detections": len(rows),
                "already_completed": len(existing_by_key),
                "completed_this_run": completed_this_run,
                "completed_total": len(existing_by_key) + completed_this_run,
                "terminal_failures": terminal_failures,
                "pending_or_running": outstanding,
                "elapsed_seconds": round(elapsed, 3),
                "effective_seconds_per_detection": round(elapsed / handled, 3) if handled else None,
                "updated_at": utc_now(),
            }
            write_json_atomic(progress_path, progress)
            if handled == 1 or handled % 10 == 0 or outstanding == 0:
                print(
                    f"progress {progress['completed_total']}/{len(rows)} · "
                    f"failed {terminal_failures} · {progress['effective_seconds_per_detection']}s/det",
                    flush=True,
                )

    for _ in threads:
        work_queue.put(None)
    for thread in threads:
        thread.join()

    final_by_key = {
        (row["image_stem"], int(row["det_index"])): row
        for row in load_jsonl(results_path)
        if (row["image_stem"], int(row["det_index"])) in expected
    }
    missing = sorted(expected - set(final_by_key))
    elapsed = time.monotonic() - started
    meta = {
        "dataset": "lagenda_full/fl1199",
        "pipeline": "SAM3 polygons -> spotlight-e1 crop -> gpt-5.6-sol",
        "model": MODEL,
        "workers": args.workers,
        "batch_size": 1,
        "one_request_per_detection": True,
        "fast_mode": False,
        "effort": "medium",
        "image_detail": "original",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": prompt_sha,
        "expected_detections": len(rows),
        "completed_detections": len(final_by_key),
        "missing_detections": len(missing),
        "missing_keys": [list(key) for key in missing],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "this_run_usage": {
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_input_tokens,
            "output_tokens": total_output_tokens,
            "charged_credits": round(total_credits, 6),
        },
        "status": "complete" if not missing else "partial",
    }
    write_json_atomic(run_meta_path, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
