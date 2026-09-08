#!/usr/bin/env python3
"""Verify the RunPod volume's staged datasets against the registry's expected state.

The checkable inventory lives in docs/DATASET_REGISTRY.md (the repo is the source
of truth); this script hardcodes the same expectations so it runs standalone on a
bare pod. Run it:

  - on a fresh pod, before any experiment: confirms the volume attached correctly
    and nothing was deleted/renamed since the registry was last updated;
  - after staging a new dataset: add its block here and to the registry table;
  - locally through the WebDAV mount:  python3 verify_datasets.py --root ./mnt/runpod

CPU-only, no deps beyond the stdlib. Exit code 0 = all present datasets pass;
1 = at least one FAIL (a dataset that exists but doesn't match expectations).
Datasets not yet staged (e.g. wider_attribute) report MISSING, which is not a
failure — the registry marks what's expected vs in-progress.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def count_images(d: Path, recursive=False):
    if not d.is_dir():
        return None
    it = d.rglob("*") if recursive else d.iterdir()
    return sum(1 for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def count_lines(f: Path):
    if not f.is_file():
        return None
    with open(f) as fh:
        return sum(1 for line in fh if line.strip())


# each check: (description, callable(root) -> (ok, detail))
def _lagenda(root):
    d = root / "lagenda_eval"
    if not d.is_dir():
        return None, "directory absent"
    return True, "present (legacy path; contents vary by experiment — see EXP-2026-01/02)"


def _crowdhuman(root):
    d = root / "datasets/crowdhuman"
    if not d.is_dir():
        return None, "directory absent"
    full = count_images(d / "Images")
    samp = count_images(d / "Images_sample500")
    odgt = count_lines(d / "annotation_val.odgt")
    ok = full == 4370 and samp == 500 and odgt == 4370
    return ok, f"Images={full} (want 4370) · Images_sample500={samp} (want 500) · odgt lines={odgt} (want 4370)"


def _pass3k(root):
    d = root / "datasets/pass_3k"
    if not d.is_dir():
        return None, "directory absent"
    n = count_images(d)
    return n == 3000, f"images={n} (want 3000, seed-51 sample of PASS.0.tar)"


def _object_set(root):
    d = root / "datasets/object_set"
    if not d.is_dir():
        return None, "directory absent"
    n = count_images(d)  # top level only — scorers read top level; junk lives in subdirs
    return n == 259, f"top-level images={n} (want 259 hand-verified — EXPENSIVE to rebuild, never delete)"


def _wider(root):
    d = root / "datasets/wider_attribute"
    if not d.is_dir():
        return None, "directory absent (staging planned — see DATASET_REGISTRY.md)"
    n = count_images(d / "images", recursive=True)
    anns = list((d / "annotations").rglob("*.json")) if (d / "annotations").is_dir() else []
    ok = n == 13789 and len(anns) >= 2
    return ok, f"images={n} (want 13789) · annotation JSONs={len(anns)} (want >=2: trainval+test)"


CHECKS = [
    ("lagenda_eval",     _lagenda),
    ("crowdhuman",       _crowdhuman),
    ("pass_3k",          _pass3k),
    ("object_set",       _object_set),
    ("wider_attribute",  _wider),
]


def main():
    ap = argparse.ArgumentParser(description="verify staged datasets against the registry")
    ap.add_argument("--root", default="/workspace",
                    help="volume root (default /workspace; use ./mnt/runpod locally)")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"volume root {root} not found — is the volume attached/mounted?")
        sys.exit(2)

    failed = False
    print(f"{'dataset':<18} {'status':<8} detail")
    print("-" * 100)
    for name, fn in CHECKS:
        ok, detail = fn(root)
        status = "MISSING" if ok is None else ("PASS" if ok else "FAIL")
        failed |= ok is False
        print(f"{name:<18} {status:<8} {detail}")
    print("-" * 100)
    print("registry: docs/DATASET_REGISTRY.md — update BOTH the table there and the"
          " expectations here when the volume changes.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
