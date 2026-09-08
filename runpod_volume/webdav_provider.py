"""
Read-write WebDAV provider backed by RunPod S3 (boto3).

Mounted via macOS's built-in `mount_webdav` — no third-party kernel
extension required. A short-lived in-memory cache avoids re-listing the
same S3 prefix on every repeated PROPFIND (VS Code/Finder issue many).
"""

from __future__ import annotations

import io
import time
from datetime import timezone
from threading import Lock

from wsgidav.dav_error import DAVError, HTTP_FORBIDDEN, HTTP_NOT_FOUND
from wsgidav.dav_provider import DAVCollection, DAVNonCollection, DAVProvider

_CACHE_TTL = 20  # seconds — long enough to absorb repeated PROPFINDs, short enough to stay fresh


def _parent_prefix(key: str) -> str:
    return key.rsplit("/", 1)[0] + "/" if "/" in key else ""


class _ListingCache:
    """Caches (dir_names, file_meta) per S3 prefix for a short TTL."""

    def __init__(self, ttl: float = _CACHE_TTL):
        self._ttl = ttl
        self._lock = Lock()
        self._store: dict[str, tuple[float, set, dict]] = {}

    def get(self, prefix: str):
        with self._lock:
            entry = self._store.get(prefix)
            if entry and time.time() - entry[0] < self._ttl:
                return entry[1], entry[2]
        return None

    def set(self, prefix: str, dir_names: set, file_meta: dict):
        with self._lock:
            self._store[prefix] = (time.time(), dir_names, file_meta)

    def invalidate(self, prefix: str):
        """Drop the cache entry for this prefix and every ancestor directory."""
        with self._lock:
            self._store.pop(prefix, None)
            parts = prefix.rstrip("/").split("/")
            for i in range(len(parts)):
                ancestor = "/".join(parts[:i]) + ("/" if i else "")
                self._store.pop(ancestor, None)


class _UploadBuffer(io.BytesIO):
    """BytesIO that survives .close() so we can read it back in end_write()."""

    def close(self):
        pass

    def really_close(self):
        super().close()


# ── S3 "directory" node (simulated from key prefixes) ────────────────────────

class S3Collection(DAVCollection):
    def __init__(self, path, environ, s3_client, bucket, prefix, cache: _ListingCache):
        super().__init__(path, environ)
        self._s3 = s3_client
        self._bucket = bucket
        self._prefix = prefix  # e.g. "" for root, "models/" for a subdir
        self._cache = cache
        self._dir_names: set | None = None
        self._file_meta: dict | None = None

    def _load(self):
        if self._dir_names is not None:
            return

        cached = self._cache.get(self._prefix)
        if cached:
            self._dir_names, self._file_meta = cached
            return

        dir_names: set = set()
        file_meta: dict = {}

        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"][len(self._prefix):].rstrip("/")
                dir_names.add(name)
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key == self._prefix:
                    continue
                name = key[len(self._prefix):]
                file_meta[name] = obj

        self._dir_names = dir_names
        self._file_meta = file_meta
        self._cache.set(self._prefix, dir_names, file_meta)

    def get_member_names(self):
        self._load()
        return sorted(self._dir_names) + sorted(self._file_meta.keys())

    def get_member(self, name):
        self._load()
        child_path = self.path.rstrip("/") + "/" + name

        if name in self._dir_names:
            child_prefix = self._prefix + name + "/"
            return S3Collection(child_path, self.environ, self._s3, self._bucket, child_prefix, self._cache)

        if name in self._file_meta:
            key = self._prefix + name
            try:
                meta = self._s3.head_object(Bucket=self._bucket, Key=key)
            except Exception:
                meta = self._file_meta[name]
            return S3Object(child_path, self.environ, self._s3, self._bucket, key, meta, self._cache)

        return None

    # ── Write operations ────────────────────────────────────────────────────

    def create_collection(self, name):
        """mkdir — S3 has no real directories, so create a zero-byte marker object."""
        key = self._prefix + name + "/"
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=b"")
        self._cache.invalidate(self._prefix)
        child_path = self.path.rstrip("/") + "/" + name
        return S3Collection(child_path, self.environ, self._s3, self._bucket, key, self._cache)

    def create_empty_resource(self, name):
        key = self._prefix + name
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=b"")
        self._cache.invalidate(self._prefix)
        child_path = self.path.rstrip("/") + "/" + name
        meta = self._s3.head_object(Bucket=self._bucket, Key=key)
        return S3Object(child_path, self.environ, self._s3, self._bucket, key, meta, self._cache)

    def support_recursive_delete(self):
        return True

    def delete(self):
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            self._s3.delete_objects(Bucket=self._bucket, Delete={"Objects": [{"Key": k} for k in batch]})
        self._cache.invalidate(self._prefix)


# ── S3 file node ──────────────────────────────────────────────────────────────

class S3Object(DAVNonCollection):
    def __init__(self, path, environ, s3_client, bucket, key, meta, cache: _ListingCache):
        super().__init__(path, environ)
        self._s3 = s3_client
        self._bucket = bucket
        self._key = key
        self._meta = meta
        self._cache = cache
        self._write_buffer: _UploadBuffer | None = None

    def get_content_length(self):
        return self._meta.get("ContentLength") or self._meta.get("Size", 0)

    def get_content_type(self):
        return self._meta.get("ContentType", "application/octet-stream")

    def get_last_modified(self):
        dt = self._meta.get("LastModified")
        if dt is None:
            return None
        return dt.astimezone(timezone.utc).timestamp()

    def support_etag(self):
        return True

    def get_etag(self):
        return self._meta.get("ETag", "").strip('"')

    def get_content(self):
        resp = self._s3.get_object(Bucket=self._bucket, Key=self._key)
        return resp["Body"]

    # ── Write operations ────────────────────────────────────────────────────

    def support_ranges(self):
        return False

    def begin_write(self, *, content_type=None):
        self._write_buffer = _UploadBuffer()
        return self._write_buffer

    def end_write(self, *, with_errors):
        buf = self._write_buffer
        self._write_buffer = None
        if buf is None:
            return
        if with_errors:
            buf.really_close()
            return
        data = buf.getvalue()
        buf.really_close()
        self._s3.put_object(Bucket=self._bucket, Key=self._key, Body=data)
        self._cache.invalidate(_parent_prefix(self._key))

    def delete(self):
        self._s3.delete_object(Bucket=self._bucket, Key=self._key)
        self._cache.invalidate(_parent_prefix(self._key))


# ── Provider ──────────────────────────────────────────────────────────────────

class S3Provider(DAVProvider):
    def __init__(self, s3_client, bucket: str):
        super().__init__()
        self._s3 = s3_client
        self._bucket = bucket
        self._cache = _ListingCache()
        self.readonly = False

    def get_resource_inst(self, path: str, environ: dict):
        """Walk the path via S3Collection.get_member() — never probe a leaf
        name with Prefix=<name>/ directly: RunPod's gateway echoes back any
        existing file as a bogus trailing-slash "directory" entry when asked
        that way, which would misclassify every file as a directory.
        """
        clean = path.strip("/")
        current = S3Collection("/", environ, self._s3, self._bucket, "", self._cache)

        if not clean:
            return current

        parts = clean.split("/")
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            member = current.get_member(part)
            if member is None:
                return None
            if is_last:
                return member
            if not isinstance(member, S3Collection):
                return None  # tried to descend into a file
            current = member

        return None
