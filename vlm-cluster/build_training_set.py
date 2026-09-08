#!/usr/bin/env python3
"""
Assemble the production training image set with FULL provenance, so images
used for training can never silently reappear in validation or test.

Two commands:

  reserve  — pick N images from a source pool (PASS, CrowdHuman, ...), copy or
             symlink them into a staging dir, and write:
               manifest.jsonl   one row per image: source dataset, original
                                path, staged path, sha1 of the file bytes
               used_<name>.txt  the basenames taken (for exclusion lists)
               held_out_<name>.txt  everything in the pool NOT taken
             Selection is seeded and idempotent: re-running never re-picks a
             different set, and never re-picks images an earlier reserve took.

  check    — verify a candidate eval/test set shares NO images with anything
             already reserved. Run this before every validation run.

    python3 build_training_set.py reserve --name pass10k \\
        --source /workspace/datasets/pass_3k --n 10000 \\
        --staging /workspace/train_set --seed 42

    python3 build_training_set.py check --staging /workspace/train_set \\
        --candidate /workspace/datasets/pass_3k_holdout

    python3 build_training_set.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def sha1(path: Path, chunk=1 << 20):
    h = hashlib.sha1()
    with path.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def load_manifest(staging: Path):
    mf = staging / "manifest.jsonl"
    if not mf.exists():
        return []
    return [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]


def reserve(name, source: Path, n, staging: Path, seed, link, hash_files):
    staging.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(staging)
    already = {r["source_path"] for r in rows}
    pool = list_images(source)
    if not pool:
        raise SystemExit(f"no images found under {source}")
    avail = [p for p in pool if str(p) not in already]
    rng = random.Random(seed)
    rng.shuffle(avail)
    picked = avail[: n if n > 0 else len(avail)]

    dest_dir = staging / "images"
    dest_dir.mkdir(exist_ok=True)
    new_rows = []
    for p in picked:
        dest = dest_dir / f"{name}__{p.name}"      # prefix prevents collisions
        if not dest.exists():
            if link:
                dest.symlink_to(p.resolve())
            else:
                shutil.copy2(p, dest)
        new_rows.append({"dataset": name, "source_path": str(p),
                         "staged_path": str(dest), "basename": p.name,
                         "sha1": sha1(p) if hash_files else None,
                         "seed": seed})
    with (staging / "manifest.jsonl").open("a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    taken = {r["source_path"] for r in rows + new_rows}
    (staging / f"used_{name}.txt").write_text(
        "\n".join(sorted(p.name for p in picked)) + "\n")
    held = [p for p in pool if str(p) not in taken]
    (staging / f"held_out_{name}.txt").write_text(
        "\n".join(sorted(p.name for p in held)) + "\n")

    print(f"[reserve] {name}: pool {len(pool)}, already reserved "
          f"{len(pool)-len(avail)}, newly taken {len(new_rows)}")
    print(f"  staged   -> {dest_dir}  ({'symlinks' if link else 'copies'})")
    print(f"  manifest -> {staging/'manifest.jsonl'} "
          f"({len(rows)+len(new_rows)} rows total)")
    print(f"  HELD OUT for eval/test: {len(held)} images "
          f"-> {staging/f'held_out_{name}.txt'}")
    return new_rows


def check(staging: Path, candidate: Path):
    rows = load_manifest(staging)
    used_paths = {r["source_path"] for r in rows}
    used_names = {r["basename"] for r in rows}
    used_sha = {r["sha1"] for r in rows if r.get("sha1")}
    cand = list_images(candidate)
    by_path = [p for p in cand if str(p) in used_paths]
    by_name = [p for p in cand if p.name in used_names and str(p) not in used_paths]
    by_sha = []
    if used_sha:
        for p in cand:
            if str(p) not in used_paths and p.name not in used_names:
                if sha1(p) in used_sha:
                    by_sha.append(p)
    print(f"[check] candidate {candidate}: {len(cand)} images vs "
          f"{len(rows)} reserved")
    print(f"  same source path : {len(by_path)}")
    print(f"  same filename    : {len(by_name)}")
    print(f"  same file bytes  : {len(by_sha)}  (content-level duplicate)")
    total = len(by_path) + len(by_name) + len(by_sha)
    if total:
        print(f"\n  *** {total} OVERLAPPING IMAGES — this candidate set is NOT "
              f"clean for eval/test ***")
        for p in (by_path + by_name + by_sha)[:10]:
            print(f"    {p}")
    else:
        print("\n  CLEAN: no overlap with reserved training images.")
    return total


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "pool"; src.mkdir()
        for i in range(20):
            (src / f"img{i:03d}.jpg").write_bytes(bytes([i]) * 64)
        stg = root / "stage"

        r1 = reserve("passX", src, 5, stg, 42, link=False, hash_files=True)
        assert len(r1) == 5
        held = (stg / "held_out_passX.txt").read_text().split()
        assert len(held) == 15, held

        # idempotent + non-overlapping on a second reserve
        r2 = reserve("passX", src, 5, stg, 42, link=False, hash_files=True)
        assert len(r2) == 5
        assert not ({x["source_path"] for x in r1} & {x["source_path"] for x in r2})
        assert len(load_manifest(stg)) == 10

        # a candidate built from held-out images is clean
        cand_ok = root / "cand_ok"; cand_ok.mkdir()
        for name in held[:4]:
            shutil.copy2(src / name, cand_ok / name)
        assert check(stg, cand_ok) == 0

        # a candidate containing a reserved image (renamed!) is caught by sha1
        cand_bad = root / "cand_bad"; cand_bad.mkdir()
        used = json.loads((stg / "manifest.jsonl").read_text().splitlines()[0])
        shutil.copy2(used["source_path"], cand_bad / "totally_different_name.jpg")
        assert check(stg, cand_bad) == 1
    print("\nbuild_training_set.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="Provenance-tracked training set")
    ap.add_argument("cmd", nargs="?", choices=["reserve", "check"])
    ap.add_argument("--name"); ap.add_argument("--source")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--staging"); ap.add_argument("--candidate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (uses disk)")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha1 (faster, but loses renamed-duplicate detection)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.cmd == "reserve":
        if not (a.name and a.source and a.staging):
            ap.error("reserve needs --name --source --staging")
        reserve(a.name, Path(a.source), a.n, Path(a.staging), a.seed,
                link=not a.copy, hash_files=not a.no_hash)
    elif a.cmd == "check":
        if not (a.staging and a.candidate):
            ap.error("check needs --staging --candidate")
        raise SystemExit(1 if check(Path(a.staging), Path(a.candidate)) else 0)
    else:
        ap.error("cmd must be reserve or check")


if __name__ == "__main__":
    main()
