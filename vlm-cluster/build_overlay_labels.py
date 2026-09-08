#!/usr/bin/env python3
"""
Build a training label dir as an OVERLAY on an existing one, without copying it.

A label-policy change that touches a small fraction of the corpus does not
justify duplicating the whole thing: the Spotlight train labels are ~11 GB
across 475,207 files, the --child-by-age rescue rewrites 1,139 of them, and the
RunPod volume quota is invisible to `df` and has silently truncated writes on
this project three times. So the new dir is one symlink per image, pointing at
the overlay file where one exists and at the baseline otherwise.

Two properties this tool enforces rather than assumes:

  * the BASELINE IS NEVER WRITTEN TO. Every path created lives under --out, and
    --out is refused if it is inside the baseline or already holds files.
  * the overlay only ADDS. Every baseline line must survive verbatim in the
    overlay's version of that file, or the run aborts before creating anything
    -- a policy that was supposed to be additive silently dropping a person is
    the failure worth catching here, and it is cheap to check on the small side.

Any trainer that reads a label dir reads symlinks transparently, and nothing
writes to labels at train time. This is the same pattern as the project's
existing symlink image trees (crowd_sample_imgs, exp12/train_tree).

    python3 build_overlay_labels.py \
        --baseline /workspace/spotlight/run/oiv7_train/labels \
        --overlay  /workspace/spotlight/childfix/changed_labels \
        --out      /workspace/spotlight/run/oiv7_train/labels_child_by_age

    python3 build_overlay_labels.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _txt_names(d: Path):
    return {f for f in os.listdir(d) if f.endswith(".txt")}


def check_additive(baseline: Path, overlay: Path, names):
    """Every baseline line must reappear verbatim in the overlay file. Returns
    (n_changed, added_class_counts). Raises on a non-additive difference."""
    changed = 0
    added = {}
    for n in sorted(names):
        b = [l for l in (baseline / n).read_text().splitlines() if l.strip()]
        o = [l for l in (overlay / n).read_text().splitlines() if l.strip()]
        if b == o:
            continue
        changed += 1
        missing = [l for l in b if l not in o]
        if missing:
            raise SystemExit(
                f"NOT ADDITIVE: {n} lost {len(missing)} baseline line(s), "
                f"first: {missing[0][:60]!r}")
        for l in o:
            if l not in b:
                added[l.split()[0]] = added.get(l.split()[0], 0) + 1
    return changed, added


def build(baseline: Path, overlay: Path, out: Path, relative=True):
    baseline, overlay, out = (p.resolve() for p in (baseline, overlay, out))
    if out == baseline or baseline in out.parents:
        raise SystemExit(f"refusing to write inside the baseline dir: {out}")
    if out.exists() and _txt_names(out):
        raise SystemExit(f"{out} already holds .txt files -- refusing to mix "
                         f"two policies in one dir; delete it or pick another")

    t0 = time.time()
    base_names = _txt_names(baseline)
    over_names = _txt_names(overlay)
    print(f"baseline {len(base_names):,} files   overlay {len(over_names):,} "
          f"files   (listed in {time.time()-t0:.0f}s)", flush=True)
    extra = over_names - base_names
    if extra:
        raise SystemExit(f"overlay has {len(extra)} files not in the baseline, "
                         f"e.g. {sorted(extra)[:3]} -- the overlay must be a "
                         f"subset, otherwise the image set changed too")

    n_changed, added = check_additive(baseline, overlay, over_names)
    print(f"overlay is additive: {n_changed:,} of {len(over_names):,} files "
          f"differ, added lines by class: "
          f"{ {k: v for k, v in sorted(added.items())} }", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    t1, n_over, n_base = time.time(), 0, 0
    for i, n in enumerate(sorted(base_names), 1):
        src = (overlay if n in over_names else baseline) / n
        if n in over_names:
            n_over += 1
        else:
            n_base += 1
        dst = out / n
        target = os.path.relpath(src, out) if relative else src
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(target, dst)
        if i % 100000 == 0:
            r = i / (time.time() - t1)
            print(f"  {i:,}/{len(base_names):,}  {r:.0f} links/s  "
                  f"ETA {(len(base_names)-i)/r/60:.1f} min", flush=True)
    print(f"\n{out}\n  {n_over + n_base:,} symlinks "
          f"({n_over:,} -> overlay, {n_base:,} -> baseline) "
          f"in {(time.time()-t1)/60:.1f} min")
    return {"links": n_over + n_base, "to_overlay": n_over,
            "to_baseline": n_base, "changed_files": n_changed,
            "added_lines_by_class": added}


def verify(baseline: Path, overlay: Path, out: Path, sample=500):
    """Read back through the symlinks: content must equal the overlay where one
    exists and the baseline everywhere else. Counting files, not trusting the
    build's own tally -- the Spotlight run's lesson."""
    import random
    names = sorted(_txt_names(out))
    over = _txt_names(overlay)
    dangling = [n for n in names if not (out / n).exists()]
    print(f"symlinks in out: {len(names):,}   dangling: {len(dangling)}")
    assert not dangling, dangling[:5]
    assert len(names) == len(_txt_names(baseline)), "file count != baseline"
    random.seed(0)
    pick = random.sample(names, min(sample, len(names)))
    pick += sorted(over)[:min(50, len(over))]      # always check overlay files
    for n in pick:
        got = (out / n).read_text()
        want = ((overlay if n in over else baseline) / n).read_text()
        assert got == want, f"content mismatch through symlink: {n}"
    n_ov = sum(1 for n in pick if n in over)
    print(f"content verified on {len(pick)} files ({n_ov} of them overlay "
          f"files) -- all match")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        b, o = td / "base", td / "over"
        b.mkdir(); o.mkdir()
        (b / "a.txt").write_text("1 0.1 0.1 0.4 0.9\n")
        (b / "b.txt").write_text("0 0.2 0.2 0.5 0.8\n")
        (b / "c.txt").write_text("")                       # empty is valid
        # overlay adds a Child line to a.txt, leaves the rest alone
        (o / "a.txt").write_text("1 0.1 0.1 0.4 0.9\n2 0.7 0.7 0.9 0.9\n")
        st = build(b, o, td / "out")
        assert st == {"links": 3, "to_overlay": 1, "to_baseline": 2,
                      "changed_files": 1, "added_lines_by_class": {"2": 1}}, st
        verify(b, o, td / "out")
        assert (td / "out/a.txt").is_symlink()
        assert (td / "out/a.txt").read_text().splitlines()[1].startswith("2 ")
        assert (td / "out/b.txt").read_text() == (b / "b.txt").read_text()
        # the baseline was not touched
        assert (b / "a.txt").read_text() == "1 0.1 0.1 0.4 0.9\n"
        # relative links survive the dir being moved as a whole
        os.rename(td / "out", td / "moved")
        assert (td / "moved/a.txt").read_text().count("\n") == 2

        # a non-additive overlay must abort BEFORE creating anything
        (o / "b.txt").write_text("2 0.9 0.9 0.95 0.95\n")   # replaces, not adds
        try:
            build(b, o, td / "out2")
        except SystemExit as e:
            assert "NOT ADDITIVE" in str(e), e
        else:
            raise AssertionError("accepted a non-additive overlay")
        assert not (td / "out2").exists() or not _txt_names(td / "out2")
        (o / "b.txt").unlink()

        # refuses to write into the baseline, and refuses a populated out dir
        try:
            build(b, o, b)
        except SystemExit as e:
            assert "refusing to write inside" in str(e), e
        else:
            raise AssertionError("wrote into the baseline")
        try:
            build(b, o, td / "moved")
        except SystemExit as e:
            assert "already holds .txt" in str(e), e
        else:
            raise AssertionError("mixed two policies in one dir")

        # an overlay file with no baseline counterpart changes the image set
        (o / "zz.txt").write_text("2 0.1 0.1 0.2 0.2\n")
        try:
            build(b, o, td / "out3")
        except SystemExit as e:
            assert "not in the baseline" in str(e), e
        else:
            raise AssertionError("accepted an overlay outside the baseline")
    print("build_overlay_labels.py self-tests passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline"); ap.add_argument("--overlay"); ap.add_argument("--out")
    ap.add_argument("--absolute", action="store_true",
                    help="absolute symlink targets (default: relative, so the "
                         "dir keeps working if the tree is moved as a whole)")
    ap.add_argument("--verify-sample", type=int, default=500)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.baseline and a.overlay and a.out):
        ap.error("--baseline --overlay --out are required")
    b, o, out = Path(a.baseline), Path(a.overlay), Path(a.out)
    build(b, o, out, relative=not a.absolute)
    verify(b, o, out, a.verify_sample)


if __name__ == "__main__":
    main()
