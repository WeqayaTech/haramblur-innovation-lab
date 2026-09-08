#!/usr/bin/env python3
"""
EXP-2026-17 — where does two-axis pairing break? Checks all THREE levels.

The 1-epoch smoke raised on an unpaired row that the runbook's on-disk pair
check could not see, because the two live at different levels:

  L1  raw .txt on disk        -- what parallel_emit.py wrote
  L2  ultralytics label cache -- what training actually loads. `verify_image_label`
                                 reduces each polygon to a box and then drops
                                 DUPLICATE [cls, cx, cy, w, h] rows
                                 (`np.unique(lb, axis=0)`). Two near-duplicate
                                 SAM3 detections whose polygons differ on disk
                                 can collapse to the same box here; if they
                                 share one axis's class, that row is deleted and
                                 its partner is orphaned. Invisible at L1.
  L3  after augmentation      -- mosaic/perspective/clip, i.e. what the loss sees

L1 is a full-corpus scan. L2 builds the real dataset through ultralytics so the
cache path is exercised exactly as training does. Both report offending files.

    python3 exp17_check_pairs.py --labels $TREE/labels/train          # L1 only
    python3 exp17_check_pairs.py --data /workspace/exp17/twoaxis.yaml # L1 + L2
    python3 exp17_check_pairs.py --selftest
"""
from __future__ import annotations

import argparse
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import two_axis as TA

GEN_LO, GEN_HI = TA.GROUPS[0]
AGE_LO, AGE_HI = TA.GROUPS[1]


def classify_rows(clses):
    """-> (n_gender, n_age). The invariant is n_gender == n_age."""
    g = sum(1 for c in clses if GEN_LO <= c < GEN_HI)
    a = sum(1 for c in clses if AGE_LO <= c < AGE_HI)
    return g, a


def _l1_one(p: Path):
    """-> (verdict, stem, n_gender, n_age). Geometry must match within a pair."""
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        return ("empty", p.stem, 0, 0)
    clses = [int(r[0]) for r in rows]
    g, a = classify_rows(clses)
    if g != a:
        return ("axis_count_mismatch", p.stem, g, a)
    if len(rows) % 2:
        return ("odd_rows", p.stem, g, a)
    for gr, ar in zip(rows[0::2], rows[1::2]):
        if not (GEN_LO <= int(gr[0]) < GEN_HI and AGE_LO <= int(ar[0]) < AGE_HI):
            return ("row_order", p.stem, g, a)
        if gr[1:] != ar[1:]:
            return ("geometry_mismatch", p.stem, g, a)
    return ("ok", p.stem, g, a)


def level1(labels: Path, workers: int, log=print):
    fs = sorted(labels.glob("*.txt"))
    log(f"[L1] {len(fs):,} label files on disk")
    tally, examples = Counter(), {}
    with Pool(workers) as pool:
        for verdict, stem, g, a in pool.imap_unordered(_l1_one, fs, chunksize=512):
            tally[verdict] += 1
            if verdict not in ("ok", "empty") and verdict not in examples:
                examples[verdict] = (stem, g, a)
    for k, n in tally.most_common():
        log(f"  {k:<22} {n:>10,}")
    for k, (stem, g, a) in examples.items():
        log(f"  e.g. {k}: {stem}  gender_rows={g} age_rows={a}")
    return tally


def level2(data_yaml: str, split: str, log=print):
    """Build the dataset ultralytics builds, then check the CACHED labels."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(data_yaml)
    args = get_cfg(DEFAULT_CFG)
    args.data = data_yaml
    ds = build_yolo_dataset(args, data[split], 1, data, mode="val", rect=False)
    log(f"[L2] {len(ds.labels):,} images in the ultralytics cache")

    tally, examples = Counter(), {}
    n_rows = 0
    for lab in ds.labels:
        clses = [int(c) for c in lab["cls"].flatten().tolist()]
        n_rows += len(clses)
        g, a = classify_rows(clses)
        if not clses:
            tally["empty"] += 1
        elif g != a:
            tally["axis_count_mismatch"] += 1
            examples.setdefault("axis_count_mismatch", (lab["im_file"], g, a))
        else:
            tally["ok"] += 1
    log(f"[L2] {n_rows:,} label rows after caching")
    for k, n in tally.most_common():
        log(f"  {k:<22} {n:>10,}")
    for k, (f, g, a) in examples.items():
        log(f"  e.g. {k}: {f}  gender_rows={g} age_rows={a}")
    if tally["axis_count_mismatch"]:
        log("  ^ these are invisible at L1: verify_image_label reduced two "
            "polygons to the same box and dropped one as a duplicate row")
    return tally


def _selftest():
    import tempfile
    poly = "0.1 0.1 0.4 0.1 0.4 0.9"
    cases = {
        "a": (f"0 {poly}\n3 {poly}\n", "ok"),
        "b": ("", "empty"),
        "c": (f"0 {poly}\n", "axis_count_mismatch"),          # orphan gender row
        "d": (f"0 {poly}\n3 {poly}\n3 {poly}\n", "axis_count_mismatch"),
        "e": (f"3 {poly}\n0 {poly}\n", "row_order"),
        "f": (f"0 {poly}\n3 0.2 0.2 0.5 0.2 0.5 0.9\n", "geometry_mismatch"),
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for stem, (body, _) in cases.items():
            (d / f"{stem}.txt").write_text(body)
        for stem, (_, want) in cases.items():
            got = _l1_one(d / f"{stem}.txt")[0]
            assert got == want, f"{stem}: got {got}, want {want}"
        t = level1(d, 2, log=lambda *a: None)
        assert t["ok"] == 1 and t["empty"] == 1 and t["axis_count_mismatch"] == 2, t
    print("exp17_check_pairs.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="label dir for the L1 disk scan")
    ap.add_argument("--data", help="data yaml -- also runs the L2 cache check")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.labels:
        level1(Path(a.labels), a.workers)
    if a.data:
        level2(a.data, a.split)
    if not (a.labels or a.data):
        ap.error("--labels and/or --data required (or --selftest)")


if __name__ == "__main__":
    main()
