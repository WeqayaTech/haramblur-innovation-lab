#!/usr/bin/env python3
"""Classify LAGENDA fl1199 SAM3 spotlight crops with Gemini 3.7 Flash.

The worklist and crops are reused from the completed Sol run so the visual
input, prompt, schema, and one-detection-per-request unit are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "gemini-3.7-flash-medium"
EFFORT = "medium"
EXPECTED_PROMPT_SHA = "4c85ffbf8bcb"
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["real_person", "depiction", "not_person"]},
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class WorkItem:
    baseline: dict[str, Any]
    attempt: int = 1

    @property
    def key(self) -> tuple[str, int]:
        return self.baseline["image_stem"], int(self.baseline["det_index"])


def parse_stream(stdout: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    terminal = next((event["result"] for event in reversed(events) if event.get("event") == "result"), None)
    if not isinstance(terminal, dict):
        raise RuntimeError("Gemini stream did not contain a terminal result event")
    init = next((event.get("init") for event in events if event.get("event") == "init"), None)
    if not isinstance(init, dict) or init.get("model") != MODEL:
        raise RuntimeError(f"Gemini initialized an unexpected model: {init!r}")
    return terminal, events


def classify_one(
    agy_bin: Path,
    schema_path: Path,
    prompt: str,
    prompt_version: str,
    prompt_sha: str,
    baseline: dict[str, Any],
    timeout: float,
    output_dir: Path,
) -> dict[str, Any]:
    crop_path = Path(str(baseline["crop_path"])).resolve()
    if not crop_path.is_file():
        raise FileNotFoundError(crop_path)
    crop_sha = hashlib.sha256(crop_path.read_bytes()).hexdigest()
    expected_sha = baseline.get("crop_sha256")
    if expected_sha and crop_sha != expected_sha:
        raise RuntimeError(f"crop digest mismatch for {crop_path}")

    # Keep the CLI project rooted at output_dir while giving the @ resolver a
    # short project-relative path. This avoids its unreliable workspace-wide
    # fallback search for a few absolute paths without copying any image data.
    attachment_dir = output_dir / "attachments"
    attachment_dir.mkdir(exist_ok=True)
    attachment_path = attachment_dir / crop_path.name
    if attachment_path.is_symlink():
        attachment_path.unlink()
    if not attachment_path.exists():
        os.link(crop_path, attachment_path)
    command = [
        str(agy_bin),
        "--model", MODEL,
        "--effort", EFFORT,
        "--output-format", "stream-json",
        "--json-schema", str(schema_path),
        "--dangerously-skip-permissions",
        "--print-timeout", f"{int(timeout)}s",
        "-p",
        (
            f"Use view_file on exactly this image path: {attachment_path}. "
            "Do not search or inspect any other file.\n\n"
            f"{prompt}"
        ),
    ]
    started_at = utc_now()
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=output_dir,
        text=True,
        capture_output=True,
        timeout=timeout + 30,
        check=False,
    )
    duration = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"agy exit {completed.returncode}: {completed.stderr[-1200:]}"
        )
    terminal, events = parse_stream(completed.stdout)
    if terminal.get("status") != "SUCCESS":
        raise RuntimeError(f"Gemini status {terminal.get('status')}: {terminal.get('error')}")
    verdict = terminal.get("structured_output")
    if not isinstance(verdict, dict):
        response = str(terminal.get("response") or "").strip()
        try:
            verdict = json.loads(response)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"missing structured verdict: {response[-800:]}") from error
    missing = set(VERDICT_SCHEMA["required"]) - set(verdict)
    if missing:
        raise RuntimeError(f"verdict is missing fields: {sorted(missing)}")

    tool_names: list[str] = []
    for event in events:
        update = event.get("step_update")
        if not isinstance(update, dict):
            continue
        for key in ("tool_name", "name"):
            if isinstance(update.get(key), str):
                tool_names.append(update[key])

    row = {key: value for key, value in baseline.items() if key not in {"v", "raw_text", "usage"}}
    row.update(
        {
            "prompt_version": prompt_version,
            "prompt_sha": prompt_sha,
            "parse_ok": True,
            "v": verdict,
            "raw_text": terminal.get("response"),
            "source": "antigravity-cli",
            "model": MODEL,
            "effort": EFFORT,
            "one_request_per_detection": True,
            "crop_path": str(crop_path),
            "crop_sha256": crop_sha,
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(duration, 3),
            "conversation_id": terminal.get("conversation_id"),
            "num_turns": terminal.get("num_turns"),
            "usage": terminal.get("usage") or {},
            "tool_names": tool_names,
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--agy-bin", type=Path, default=Path("/root/.local/bin/agy"))
    parser.add_argument(
        "--pipeline-root", type=Path, default=Path("/workspace/autolabel_pipeline_v2")
    )
    parser.add_argument(
        "--sol-results",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199/verdicts_sol.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/codex_sol_compare/lagenda_fl1199_gemini37"),
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        raise ValueError("workers must be between 1 and 20")
    if not args.agy_bin.is_file():
        raise FileNotFoundError(args.agy_bin)

    sys.path.insert(0, str(args.pipeline_root.resolve()))
    from spotlight_run import PROMPT, PROMPT_VERSION  # type: ignore

    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
    if PROMPT_VERSION != "spotlight-e1" or prompt_sha != EXPECTED_PROMPT_SHA:
        raise RuntimeError(f"prompt mismatch: {PROMPT_VERSION=} {prompt_sha=}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "verdict_schema.json"
    write_json_atomic(schema_path, VERDICT_SCHEMA)
    results_path = output_dir / "verdicts_gemini37.jsonl"
    errors_path = output_dir / "errors.jsonl"
    progress_path = output_dir / "progress.json"
    run_meta_path = output_dir / "run_meta.json"

    baselines = read_jsonl(args.sol_results.resolve())
    by_key = {(row["image_stem"], int(row["det_index"])): row for row in baselines}
    if len(by_key) != 1884:
        raise RuntimeError(f"expected 1,884 unique Sol baselines, found {len(by_key):,}")
    rows = sorted(by_key.values(), key=lambda row: (row["image_stem"], int(row["det_index"])))
    if args.limit is not None:
        rows = rows[: args.limit]
    expected = {(row["image_stem"], int(row["det_index"])) for row in rows}
    existing = {
        (row["image_stem"], int(row["det_index"])): row
        for row in read_jsonl(results_path)
        if (row["image_stem"], int(row["det_index"])) in expected
    }
    pending = [WorkItem(row) for row in rows if (row["image_stem"], int(row["det_index"])) not in existing]

    work_queue: queue.Queue[WorkItem | None] = queue.Queue()
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    for item in pending:
        work_queue.put(item)

    def worker() -> None:
        while True:
            item = work_queue.get()
            if item is None:
                return
            try:
                result_queue.put(
                    {
                        "status": "completed",
                        "row": classify_one(
                            args.agy_bin, schema_path, PROMPT, PROMPT_VERSION,
                            prompt_sha, item.baseline, args.timeout, output_dir,
                        ),
                    }
                )
            except Exception as error:
                result_queue.put(
                    {"status": "error", "item": item, "error": f"{type(error).__name__}: {error}"}
                )

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    started_at = utc_now()
    started = time.monotonic()
    for thread in threads:
        thread.start()

    completed_this_run = 0
    failures = 0
    outstanding = len(pending)
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
    }
    with results_path.open("a") as results_file, errors_path.open("a") as errors_file:
        while outstanding:
            event = result_queue.get()
            if event["status"] == "completed":
                row = event["row"]
                results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                results_file.flush()
                os.fsync(results_file.fileno())
                completed_this_run += 1
                outstanding -= 1
                for key in usage_totals:
                    usage_totals[key] += int((row.get("usage") or {}).get(key, 0) or 0)
            else:
                item = event["item"]
                errors_file.write(json.dumps({
                    "image_stem": item.baseline["image_stem"],
                    "det_index": int(item.baseline["det_index"]),
                    "attempt": item.attempt,
                    "error": event["error"],
                    "at": utc_now(),
                }) + "\n")
                errors_file.flush()
                if item.attempt < args.max_attempts:
                    work_queue.put(WorkItem(item.baseline, item.attempt + 1))
                else:
                    failures += 1
                    outstanding -= 1

            handled = completed_this_run + failures
            elapsed = time.monotonic() - started
            progress = {
                "dataset": "lagenda_full/fl1199",
                "model": MODEL,
                "workers": args.workers,
                "expected_detections": len(rows),
                "already_completed": len(existing),
                "completed_this_run": completed_this_run,
                "completed_total": len(existing) + completed_this_run,
                "terminal_failures": failures,
                "pending_or_running": outstanding,
                "elapsed_seconds": round(elapsed, 3),
                "effective_seconds_per_detection": round(elapsed / handled, 3) if handled else None,
                "updated_at": utc_now(),
            }
            write_json_atomic(progress_path, progress)
            if handled == 1 or handled % 10 == 0 or outstanding == 0:
                print(
                    f"progress {progress['completed_total']}/{len(rows)} · failed {failures} · "
                    f"{progress['effective_seconds_per_detection']}s/det",
                    flush=True,
                )

    for _ in threads:
        work_queue.put(None)
    for thread in threads:
        thread.join()

    final = {
        (row["image_stem"], int(row["det_index"])): row
        for row in read_jsonl(results_path)
        if (row["image_stem"], int(row["det_index"])) in expected
    }
    missing = sorted(expected - set(final))
    meta = {
        "dataset": "lagenda_full/fl1199",
        "pipeline": "SAM3 polygons -> spotlight-e1 crop -> Gemini 3.7 Flash",
        "model": MODEL,
        "effort": EFFORT,
        "workers": args.workers,
        "batch_size": 1,
        "one_request_per_detection": True,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": prompt_sha,
        "expected_detections": len(rows),
        "completed_detections": len(final),
        "missing_detections": len(missing),
        "missing_keys": [list(key) for key in missing],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "this_run_usage": usage_totals,
        "status": "complete" if not missing else "partial",
    }
    write_json_atomic(run_meta_path, meta)
    print(json.dumps(meta, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
