"""
RunpodVolume — S3-compatible client for a RunPod network volume.

Reads credentials from environment (or .env):
  RUNPOD_API_KEY, RUNPOD_SECRET_KEY, RUNPOD_BUCKET, RUNPOD_REGION
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_REGION = "eu-ro-1"


def _endpoint_for_region(region: str) -> str:
    return f"https://s3api-{region}.runpod.io"


def _boto3():
    try:
        import boto3
        import botocore
        return boto3, botocore
    except ImportError:
        sys.exit("boto3 is not installed. Run:  pip install -r requirements.txt")


class RunpodVolume:
    """S3-compatible client for a RunPod network volume."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        endpoint_url: str | None = None,
        bucket: str | None = None,
        region: str | None = None,
    ):
        boto3, botocore = _boto3()

        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("RUNPOD_SECRET_KEY", "")
        self.bucket = bucket or os.environ.get("RUNPOD_BUCKET", "")
        self.region = region or os.environ.get("RUNPOD_REGION", DEFAULT_REGION)
        self.endpoint = (
            endpoint_url
            or os.environ.get("RUNPOD_ENDPOINT_URL")
            or _endpoint_for_region(self.region)
        )

        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required — set it in .env or export it")
        if not self.secret_key:
            raise ValueError("RUNPOD_SECRET_KEY is required — set it in .env or export it")
        if not self.bucket:
            raise ValueError("RUNPOD_BUCKET (volume ID) is required — set it in .env or export it")

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.api_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=botocore.config.Config(signature_version="s3v4"),
        )

    # ── Listing ───────────────────────────────────────────────────────────────

    def list(self, prefix: str = "") -> Iterator[dict]:
        """Yield object metadata dicts recursively under prefix.

        NOTE: RunPod S3 ignores Prefix without Delimiter, so we use a
        delimiter-based recursive traversal instead of a flat paginator.
        """
        yield from self._list_recursive(prefix)

    def _list_recursive(self, prefix: str, workers: int = 16) -> list[dict]:
        """Parallel recursive listing using Prefix+Delimiter (RunPod-compatible).

        Uses a thread pool so sibling directories are listed concurrently,
        making deep trees (e.g. git repos) much faster.
        """
        results: list[dict] = []
        queue = [prefix]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while queue:
                futures = {pool.submit(self._list_one_level, p): p for p in queue}
                queue = []
                for fut in as_completed(futures):
                    files, subdirs = fut.result()
                    results.extend(files)
                    queue.extend(subdirs)

        return results

    def _list_one_level(self, prefix: str) -> tuple[list[dict], list[str]]:
        """List a single directory level. Returns (files, subdir_prefixes)."""
        files: list[dict] = []
        subdirs: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for obj in page.get("Contents", []):
                if obj["Key"] != prefix:
                    files.append(obj)
            for cp in page.get("CommonPrefixes", []):
                subdirs.append(cp["Prefix"])
        return files, subdirs

    # ── Single file operations ────────────────────────────────────────────────

    def upload_file(self, local: str | Path, key: str, progress: bool = True) -> None:
        local = Path(local)
        if not local.is_file():
            raise FileNotFoundError(f"Not a file: {local}")
        callback = _ProgressCallback(local.stat().st_size, str(local)) if progress else None
        self._client.upload_file(str(local), self.bucket, key, Callback=callback)
        if progress:
            print()

    def download_file(self, key: str, local: str | Path, progress: bool = True) -> None:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        meta = self._client.head_object(Bucket=self.bucket, Key=key)
        size = meta["ContentLength"]
        callback = _ProgressCallback(size, key) if progress else None
        self._client.download_file(self.bucket, key, str(local), Callback=callback)
        if progress:
            print()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def generate_url(self, key: str, expires_in: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    # ── Sync ──────────────────────────────────────────────────────────────────

    def sync_pull(self, remote_prefix: str, local_dir: str | Path, workers: int = 8) -> tuple[int, int]:
        """Download files from remote_prefix → local_dir, skipping unchanged files.

        Lists directories in parallel, then downloads files in parallel.
        A file is skipped when it already exists locally with the same size.
        Returns (downloaded, skipped) counts.
        """
        local_dir = Path(local_dir)
        prefix = remote_prefix.rstrip("/") + "/"

        print(f"Listing {remote_prefix}/ ...")
        all_objects = self._list_recursive(prefix, workers=workers)
        if not all_objects:
            print(f"No files found under '{remote_prefix}/' on the volume.")
            return 0, 0

        to_download = []
        skipped = 0
        for obj in all_objects:
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            local_path = local_dir / rel
            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                skipped += 1
            else:
                to_download.append((key, obj["Size"], local_path))

        print(f"Found {len(all_objects)} files — {len(to_download)} to download, {skipped} already up-to-date.")

        if not to_download:
            return 0, skipped

        downloaded = 0
        lock = __import__("threading").Lock()

        def _download(item):
            key, size, local_path = item
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, key, str(local_path))
            rel = str(local_path.relative_to(local_dir))
            with lock:
                nonlocal downloaded
                downloaded += 1
                pct = downloaded / len(to_download) * 100
                print(f"  [{downloaded}/{len(to_download)}] {pct:.0f}%  {rel}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download, item) for item in to_download]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    print(f"  Warning: {exc}")

        return downloaded, skipped

    def sync_push(self, local_dir: str | Path, remote_prefix: str, workers: int = 8) -> tuple[int, int]:
        """Upload new/changed files from local_dir → remote_prefix on the volume.

        Lists remote and uploads changed files in parallel.
        A file is skipped when S3 already has it at the same size.
        Returns (uploaded, skipped) counts.
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise NotADirectoryError(local_dir)

        prefix = remote_prefix.rstrip("/") + "/"

        print(f"Listing remote {remote_prefix}/ ...")
        remote_index = {obj["Key"]: obj["Size"] for obj in self._list_recursive(prefix, workers=workers)}

        local_files = [f for f in local_dir.rglob("*") if f.is_file()]
        if not local_files:
            print(f"No local files found in {local_dir}/")
            return 0, 0

        to_upload = []
        skipped = 0
        for f in local_files:
            key = prefix + str(f.relative_to(local_dir))
            if remote_index.get(key) == f.stat().st_size:
                skipped += 1
            else:
                to_upload.append((f, key))

        print(f"Found {len(local_files)} local files — {len(to_upload)} to upload, {skipped} already up-to-date.")

        if not to_upload:
            return 0, skipped

        uploaded = 0
        lock = __import__("threading").Lock()

        def _upload(item):
            f, key = item
            self._client.upload_file(str(f), self.bucket, key)
            with lock:
                nonlocal uploaded
                uploaded += 1
                pct = uploaded / len(to_upload) * 100
                print(f"  [{uploaded}/{len(to_upload)}] {pct:.0f}%  {key}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_upload, item) for item in to_upload]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    print(f"  Warning: {exc}")

        return uploaded, skipped

    # ── Dataset shortcuts ─────────────────────────────────────────────────────

    def list_datasets(self) -> list[str]:
        seen: set[str] = set()
        for obj in self.list("datasets/"):
            parts = obj["Key"].split("/")
            if len(parts) >= 2 and parts[1]:
                seen.add(parts[1])
        return sorted(seen)

    def list_checkpoints(self, experiment: str | None = None) -> list[str]:
        prefix = f"checkpoints/{experiment}/" if experiment else "checkpoints/"
        return [obj["Key"] for obj in self.list(prefix)]


# ── Progress bar ──────────────────────────────────────────────────────────────

class _ProgressCallback:
    def __init__(self, total: int, label: str):
        self._total = total
        self._seen = 0
        self._label = label
        self._width = 40

    def __call__(self, chunk: int):
        self._seen += chunk
        pct = self._seen / self._total if self._total else 1
        filled = int(self._width * pct)
        bar = "█" * filled + "░" * (self._width - filled)
        mb_done = self._seen / 1_048_576
        mb_total = self._total / 1_048_576
        print(f"\r  [{bar}] {mb_done:.1f}/{mb_total:.1f} MB  {self._label}", end="", flush=True)
