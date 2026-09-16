#!/usr/bin/env python3
"""Classify one SAM3-highlighted detection per Sol request on the holdout set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


MODEL = "gpt-5.6-sol"
SOL_CREDITS_PER_MILLION = {
    "input": 125.0,
    "cached_input": 12.5,
    "output": 750.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def usage_credits(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not usage:
        return {"measurement": "unavailable", "charged_credits": None}
    input_tokens = int(usage.get("inputTokens", 0))
    cached_input_tokens = int(usage.get("cachedInputTokens", 0))
    output_tokens = int(usage.get("outputTokens", 0))
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    standard_credits = (
        uncached_input_tokens * SOL_CREDITS_PER_MILLION["input"]
        + cached_input_tokens * SOL_CREDITS_PER_MILLION["cached_input"]
        + output_tokens * SOL_CREDITS_PER_MILLION["output"]
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": int(usage.get("reasoningOutputTokens", 0)),
        "total_tokens": int(usage.get("totalTokens", input_tokens + output_tokens)),
        "standard_credits": round(standard_credits, 6),
        "credit_multiplier": 1.0,
        "charged_credits": round(standard_credits, 6),
        "measurement": "app-server-token-usage",
    }


class AppServerClient:
    def __init__(self, codex_bin: Path, cwd: Path, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [str(codex_bin), "app-server", "--stdio"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if not self.process.stdin or not self.process.stdout or not self.process.stderr:
            raise RuntimeError("failed to open app-server pipes")
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.deferred: list[dict[str, Any]] = []
        self.next_request_id = 1
        self.log_lock = threading.Lock()
        self.event_log = (log_dir / "app-server-events.jsonl").open("a", encoding="utf-8")
        self.stderr_log = (log_dir / "app-server-stderr.log").open("a", encoding="utf-8")
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.process.stdout
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                message = {"invalid_json": line.rstrip("\n")}
            with self.log_lock:
                self.event_log.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.event_log.flush()
            self.messages.put(message)

    def _read_stderr(self) -> None:
        assert self.process.stderr
        for line in self.process.stderr:
            with self.log_lock:
                self.stderr_log.write(line)
                self.stderr_log.flush()

    def send(self, message: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"app-server exited with {self.process.returncode}")
        assert self.process.stdin
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        request_id = self.next_request_id
        self.next_request_id += 1
        self.send({"method": method, "id": request_id, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {method}")
            message = self.messages.get(timeout=remaining)
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(
                        f"{method} failed: {json.dumps(message['error'], ensure_ascii=False)}"
                    )
                return message.get("result", {})
            self.deferred.append(message)

    def initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "haramblur_sam3_sol_compare",
                    "title": "HaramBlur SAM3 Sol Comparison",
                    "version": "1.0.0",
                },
                "capabilities": {"experimentalApi": True, "requestAttestation": False},
            },
        )
        if not result.get("userAgent"):
            raise RuntimeError("app-server initialization returned no userAgent")
        self.send({"method": "initialized", "params": {}})

    def wait_for_turn(
        self, thread_id: str, turn_id: str, timeout: float
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        final_text = ""
        token_usage: dict[str, Any] | None = None
        deadline = time.monotonic() + timeout
        pending = self.deferred
        self.deferred = []
        while True:
            if pending:
                message = pending.pop(0)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"turn {turn_id} timed out")
                message = self.messages.get(timeout=remaining)
            method = message.get("method")
            params = message.get("params", {})
            if "id" in message and method:
                raise RuntimeError(f"unexpected server request: {method}")
            if (
                method == "item/completed"
                and params.get("threadId") == thread_id
                and params.get("turnId") == turn_id
            ):
                item = params.get("item", {})
                if item.get("type") == "agentMessage" and item.get("text"):
                    final_text = item["text"]
            elif (
                method == "thread/tokenUsage/updated"
                and params.get("threadId") == thread_id
                and params.get("turnId") == turn_id
            ):
                token_usage = params.get("tokenUsage", {}).get("last")
            elif method == "turn/completed" and params.get("threadId") == thread_id:
                turn = params.get("turn", {})
                if turn.get("id") != turn_id:
                    self.deferred.append(message)
                    continue
                for item in turn.get("items", []):
                    if item.get("type") == "agentMessage" and item.get("text"):
                        final_text = item["text"]
                if turn.get("status") != "completed":
                    raise RuntimeError(
                        f"turn ended as {turn.get('status')}: {turn.get('error')}"
                    )
                if not final_text:
                    raise RuntimeError("completed turn returned no agent message")
                return final_text, token_usage, turn

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        with self.log_lock:
            self.event_log.close()
            self.stderr_log.close()


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["real_person", "depiction", "not_person"],
        },
        "gender": {"type": "string", "enum": ["man", "woman", "unknown"]},
        "age_group": {"type": "string", "enum": ["child", "adult", "unknown"]},
        "estimated_age": {"type": "number"},
        "highlight_quality": {
            "type": "string",
            "enum": ["good", "covers_wrong_object", "covers_multiple_people"],
        },
        "confidence": {"type": "string", "enum": ["high", "low"]},
        "apparent_race": {"type": "string"},
        "head_covering": {"type": "string"},
        "exposed_body_parts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "verdict",
        "gender",
        "age_group",
        "estimated_age",
        "highlight_quality",
        "confidence",
        "apparent_race",
        "head_covering",
        "exposed_body_parts",
    ],
}


@dataclass(frozen=True)
class WorkItem:
    baseline: dict[str, Any]
    attempt: int = 1

    @property
    def key(self) -> tuple[str, int]:
        return self.baseline["image_stem"], int(self.baseline["det_index"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def crop_path_for(output_dir: Path, image_stem: str, det_index: int) -> Path:
    shard = hashlib.sha256(image_stem.encode("utf-8")).hexdigest()[:2]
    return output_dir / "crops" / shard / f"{image_stem}--{det_index:03d}.jpg"


def materialize_crop(
    baseline: dict[str, Any],
    images_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    build_crop: Any,
) -> tuple[Path, str]:
    image_stem = baseline["image_stem"]
    det_index = int(baseline["det_index"])
    crop_path = crop_path_for(output_dir, image_stem, det_index)
    if not crop_path.exists():
        sidecar = json.loads((raw_dir / f"{image_stem}.json").read_text(encoding="utf-8"))
        detection = sidecar["detections"][det_index]
        image_path = images_dir / baseline["image"]
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        crop, _scale = build_crop(image, detection["box"], detection["parts"])
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = crop_path.with_name(f".{crop_path.name}.{os.getpid()}.tmp")
        crop.save(temporary, format="JPEG", quality=90)
        os.replace(temporary, crop_path)
    digest = hashlib.sha256()
    with crop_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return crop_path, digest.hexdigest()


def classify_one(
    client: AppServerClient,
    prompt: str,
    prompt_version: str,
    prompt_sha: str,
    crop_path: Path,
    crop_sha256: str,
    baseline: dict[str, Any],
    cwd: Path,
    timeout: float,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    thread_result = client.request(
        "thread/start",
        {
            "model": MODEL,
            "allowProviderModelFallback": False,
            "serviceTier": None,
            "cwd": str(cwd),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": True,
            "serviceName": "haramblur_sam3_spotlight_comparison",
            "baseInstructions": (
                "Inspect only the directly attached highlighted crop. Do not call tools, "
                "browse, read files, edit files, or do unrelated work. Return only the "
                "schema-constrained JSON requested by the user prompt."
            ),
            "developerInstructions": "",
        },
        timeout=60,
    )
    if thread_result.get("model") != MODEL:
        raise RuntimeError(
            f"requested {MODEL}, app-server selected {thread_result.get('model')}"
        )
    thread_id = thread_result["thread"]["id"]
    turn_result = client.request(
        "turn/start",
        {
            "threadId": thread_id,
            "input": [
                {"type": "text", "text": prompt, "text_elements": []},
                {"type": "localImage", "path": str(crop_path), "detail": "original"},
            ],
            "model": MODEL,
            "serviceTier": None,
            "effort": "medium",
            "outputSchema": VERDICT_SCHEMA,
        },
        timeout=60,
    )
    turn_id = turn_result["turn"]["id"]
    final_text, raw_usage, turn = client.wait_for_turn(
        thread_id, turn_id, timeout=timeout
    )
    verdict = json.loads(final_text)
    duration = time.monotonic() - started
    metadata_keys = [
        "image_stem",
        "det_index",
        "image",
        "img_wh",
        "sam_class_id",
        "sam_conf",
        "box",
        "n_parts",
        "blurriness",
        "person_px_height",
    ]
    row = {key: baseline.get(key) for key in metadata_keys}
    row.update(
        {
            "prompt_version": prompt_version,
            "prompt_sha": prompt_sha,
            "parse_ok": True,
            "v": verdict,
            "raw_text": final_text,
            "source": "codex-app-server",
            "model": MODEL,
            "service_tier": "standard",
            "fast_mode": False,
            "image_detail": "original",
            "crop_path": str(crop_path),
            "crop_sha256": crop_sha256,
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(duration, 3),
            "turn_duration_ms": turn.get("durationMs"),
            "usage": usage_credits(raw_usage),
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="shiekhs")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--speed-guard-warmup", type=float, default=600)
    parser.add_argument("--min-images-per-30m", type=float, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/datasets/haramblur_holdout/labeling/full"),
    )
    parser.add_argument(
        "--pipeline-root", type=Path, default=Path("/workspace/autolabel_pipeline_v2")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/workspace/codex_sol_compare/shiekhs")
    )
    parser.add_argument("--codex-bin", type=Path, default=Path("/root/.local/bin/codex"))
    args = parser.parse_args()

    if args.collection != "shiekhs":
        raise ValueError("this staged run is intentionally restricted to shiekhs")
    if not 1 <= args.workers <= 20:
        raise ValueError("workers must be between 1 and 20")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    if not args.codex_bin.is_file():
        raise FileNotFoundError(args.codex_bin)

    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    raw_dir = dataset_root / "raw"
    baseline_path = dataset_root / "run" / "verdicts_batch.jsonl"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "verdicts_sol.jsonl"
    errors_path = output_dir / "errors.jsonl"
    progress_path = output_dir / "progress.json"
    run_meta_path = output_dir / "run_meta.json"

    sys.path.insert(0, str(args.pipeline_root.resolve()))
    from spotlight_run import PROMPT, PROMPT_VERSION, build_crop  # type: ignore

    prompt_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]
    prefix = f"{args.collection}__"
    baseline_rows = [
        row
        for row in load_jsonl(baseline_path)
        if str(row.get("image_stem", "")).startswith(prefix)
    ]
    baseline_rows.sort(key=lambda row: (row["image_stem"], int(row["det_index"])))
    if args.limit is not None:
        baseline_rows = baseline_rows[: args.limit]
    expected_keys = {
        (row["image_stem"], int(row["det_index"])) for row in baseline_rows
    }
    if len(expected_keys) != len(baseline_rows):
        raise ValueError("baseline contains duplicate shiekhs detection keys")

    existing_rows = load_jsonl(results_path)
    completed_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in existing_rows:
        key = (row["image_stem"], int(row["det_index"]))
        if key in expected_keys:
            completed_by_key[key] = row
    pending = [
        WorkItem(row)
        for row in baseline_rows
        if (row["image_stem"], int(row["det_index"])) not in completed_by_key
    ]

    started_at = utc_now()
    started = time.monotonic()
    unique_image_count = len({row["image_stem"] for row in baseline_rows})
    detections_per_image = len(baseline_rows) / unique_image_count
    work_queue: queue.Queue[WorkItem | None] = queue.Queue()
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    for item in pending:
        work_queue.put(item)

    def worker(worker_number: int) -> None:
        client: AppServerClient | None = None
        log_dir = output_dir / "workers" / f"worker-{worker_number:02d}"
        while True:
            if stop_event.is_set():
                break
            item = work_queue.get()
            if item is None:
                break
            try:
                crop_path, crop_sha256 = materialize_crop(
                    item.baseline, images_dir, raw_dir, output_dir, build_crop
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
                    crop_sha256,
                    item.baseline,
                    output_dir,
                    args.timeout,
                )
                result_queue.put({"status": "completed", "row": row, "attempt": item.attempt})
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
    halted_reason: str | None = None
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
                error_row = {
                    "image_stem": item.baseline["image_stem"],
                    "det_index": int(item.baseline["det_index"]),
                    "attempt": item.attempt,
                    "error": event["error"],
                    "at": utc_now(),
                }
                errors_file.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                errors_file.flush()
                if item.attempt < args.max_attempts:
                    work_queue.put(WorkItem(item.baseline, item.attempt + 1))
                else:
                    terminal_failures += 1
                    outstanding -= 1

            finished_total = len(completed_by_key) + completed_this_run
            handled_this_run = completed_this_run + terminal_failures
            elapsed = time.monotonic() - started
            images_equivalent = handled_this_run / detections_per_image
            projected_images_per_30m = (
                images_equivalent * 1800 / elapsed if elapsed > 0 else None
            )
            progress = {
                "collection": args.collection,
                "model": MODEL,
                "service_tier": "standard",
                "fast_mode": False,
                "workers": args.workers,
                "expected_detections": len(baseline_rows),
                "already_completed": len(completed_by_key),
                "completed_this_run": completed_this_run,
                "completed_total": finished_total,
                "terminal_failures": terminal_failures,
                "pending_or_running": outstanding,
                "elapsed_seconds": round(elapsed, 3),
                "detections_per_source_image": round(detections_per_image, 6),
                "source_images_equivalent_this_run": round(images_equivalent, 3),
                "projected_source_images_per_30m": (
                    round(projected_images_per_30m, 3)
                    if projected_images_per_30m is not None
                    else None
                ),
                "effective_seconds_per_detection": (
                    round(elapsed / handled_this_run, 3) if handled_this_run else None
                ),
                "started_at": started_at,
                "updated_at": utc_now(),
            }
            write_json_atomic(progress_path, progress)
            if handled_this_run == 1 or handled_this_run % 10 == 0 or outstanding == 0:
                print(
                    f"progress {finished_total}/{len(baseline_rows)} · "
                    f"failed {terminal_failures} · {progress['effective_seconds_per_detection']}s/det",
                    flush=True,
                )

            if (
                elapsed >= args.speed_guard_warmup
                and projected_images_per_30m is not None
                and projected_images_per_30m < args.min_images_per_30m
            ):
                halted_reason = (
                    f"speed guard: projected {projected_images_per_30m:.2f} source images/30m "
                    f"is below {args.min_images_per_30m:.2f}"
                )
                print(f"HALTING: {halted_reason}", flush=True)
                stop_event.set()
                break

    if not halted_reason:
        for _ in range(args.workers):
            work_queue.put(None)
    for thread in threads:
        thread.join()

    all_results = load_jsonl(results_path)
    final_by_key = {
        (row["image_stem"], int(row["det_index"])): row
        for row in all_results
        if (row["image_stem"], int(row["det_index"])) in expected_keys
    }
    missing_keys = sorted(expected_keys - set(final_by_key))
    elapsed = time.monotonic() - started
    labels: dict[str, int] = {}
    for row in final_by_key.values():
        verdict = row.get("v", {})
        label = f"{verdict.get('verdict')}:{verdict.get('gender')}:{verdict.get('age_group')}"
        labels[label] = labels.get(label, 0) + 1
    run_meta = {
        "collection": args.collection,
        "pipeline": "SAM3 polygons -> spotlight crop -> gpt-5.6-sol",
        "model": MODEL,
        "service_tier": "standard",
        "fast_mode": False,
        "workers": args.workers,
        "batch_size": 1,
        "one_request_per_detection": True,
        "image_detail": "original",
        "crop_jpeg_quality": 90,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": prompt_sha,
        "baseline_path": str(baseline_path),
        "expected_images": len({row["image_stem"] for row in baseline_rows}),
        "expected_detections": len(baseline_rows),
        "completed_detections": len(final_by_key),
        "missing_detections": len(missing_keys),
        "missing_keys": [list(key) for key in missing_keys],
        "halted_reason": halted_reason,
        "speed_guard": {
            "warmup_seconds": args.speed_guard_warmup,
            "minimum_source_images_per_30m": args.min_images_per_30m,
            "detections_per_source_image": round(detections_per_image, 6),
        },
        "labels": labels,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "effective_seconds_per_detection": (
            round(elapsed / completed_this_run, 3) if completed_this_run else None
        ),
        "this_run_usage": {
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_input_tokens,
            "output_tokens": total_output_tokens,
            "charged_credits": round(total_credits, 6),
            "measurement": "sum-of-app-server-turns",
        },
        "status": (
            "complete"
            if not missing_keys
            else "halted_slow"
            if halted_reason
            else "partial"
        ),
    }
    write_json_atomic(run_meta_path, run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)
    return 0 if not missing_keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
