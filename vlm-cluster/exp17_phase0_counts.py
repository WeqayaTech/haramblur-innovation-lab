#!/usr/bin/env python3
"""
EXP-2026-17 Phase 0 — the gate, run BEFORE spending any GPU.

Two counts over the existing Spotlight verdicts. Both are cheap, need no API
calls and no GPU, and both can kill or reshape the experiment:

  1. How many detections Gemini judged a CHILD still carry a usable gender.
     The two-axis design predicts a child's gender, which spotlight_run.merge()
     threw away -- but EXP-2026-01 found VLMs abstain more on young children.
     If most children come back gender-unknown they simply get class 2 and the
     design is unaffected; either way we want the number before training, not
     after.

  2. How many detections land in AgeUnknown under the chosen band. If it is a
     rounding error the class is dead weight; if it is a large slice, the age
     head is mostly being taught to abstain and the band needs narrowing. The
     band is a judgement call taken from EXP-2026-10's measured gradient, NOT a
     calibrated threshold -- this is what calibrates it.

Also prints the age histogram around the band edges, so the choice of 13-17 can
be argued from this project's own data rather than from the EXP-2026-10 write-up.

    python3 exp17_phase0_counts.py --verdicts /workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl
    python3 exp17_phase0_counts.py --selftest
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import two_axis as TA


def _why_age_unknown(v, band, lowconf):
    """Which of the three triggers fired. Mirrors the branch order in
    two_axis.two_axis_ids -- they have very different remedies, so reporting
    only the total hides which lever (if any) is worth pulling."""
    age, grp = v.get("estimated_age"), v.get("age_group")
    if lowconf and v.get("confidence") == "low":
        return "low_confidence"
    if isinstance(age, (int, float)) and band[0] <= age <= band[1]:
        return "teen_band"
    if grp not in ("child", "adult"):
        return "age_group_unknown"
    return "other"          # unreachable; a canary if the rule ever drifts


def count(paths, band=TA.AGE_UNKNOWN_BAND, lowconf=False):
    """-> (stats dict, age histogram). Streams; never loads the file (253 MB)."""
    st = Counter()
    hist = Counter()
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                v = (json.loads(line) or {}).get("v")
                st["verdicts"] += 1
                ids = TA.two_axis_ids(v, band, lowconf)
                if ids is None:
                    st["not_person"] += 1
                    continue
                gid, aid = ids
                st["people"] += 1
                if aid == TA.AGE_UNKNOWN:
                    st["why_" + _why_age_unknown(v, band, lowconf)] += 1
                st[f"gender_{TA.CLASS_NAMES[gid]}"] += 1
                st[f"age_{TA.CLASS_NAMES[aid]}"] += 1
                # count 1: does a child keep a gender?
                if aid == TA.CHILD:
                    st["child_gender_known" if gid != TA.GENDER_UNKNOWN
                       else "child_gender_unknown"] += 1
                age = (v or {}).get("estimated_age")
                if isinstance(age, (int, float)):
                    hist[int(age) // 5 * 5] += 1
    return st, hist


def report(st, hist, band, log=print):
    n = st["people"] or 1
    log(f"verdicts        {st['verdicts']:,}")
    log(f"  not a person  {st['not_person']:,}")
    log(f"  people        {st['people']:,}")
    log("")
    log("GATE 1 — do children keep a gender?")
    kids = st["child_gender_known"] + st["child_gender_unknown"]
    if kids:
        log(f"  children            {kids:,}")
        log(f"  gender readable     {st['child_gender_known']:,} "
            f"({100 * st['child_gender_known'] / kids:.1f}%)")
        log(f"  gender unreadable   {st['child_gender_unknown']:,} "
            f"({100 * st['child_gender_unknown'] / kids:.1f}%)  -> class 2")
    else:
        log("  no children found — check the verdicts path")
    log("")
    log(f"GATE 2 — how big is AgeUnknown at band {band[0]}-{band[1]}?")
    for k in ("Adult", "Child", "AgeUnknown"):
        log(f"  {k:<12} {st['age_' + k]:>12,}  ({100 * st['age_' + k] / n:5.2f}%)")
    au = st["age_AgeUnknown"] or 1
    log("  WHY each AgeUnknown fired -- the remedies differ completely:")
    for k, what in (("teen_band", "estimated age inside the band  -> narrowing the band moves THIS only"),
                    ("age_group_unknown", "Gemini said age_group=unknown  -> a labeler abstention, not a band choice"),
                    ("low_confidence", "verdict confidence low"),
                    ("other", "UNREACHABLE -- the rule has drifted, investigate")):
        cnt = st["why_" + k]        # NOT `n` -- that is the people count below
        if cnt or k in ("teen_band", "age_group_unknown"):
            log(f"    {k:<18} {cnt:>10,}  ({100 * cnt / au:5.1f}% of AgeUnknown)  {what}")
    log("  (a rounding error means the class is dead weight; a large slice "
        "means the age head mostly learns to abstain)")
    log("")
    log("gender axis")
    for k in ("Woman", "Man", "GenderUnknown"):
        log(f"  {k:<14} {st['gender_' + k]:>12,}  ({100 * st['gender_' + k] / n:5.2f}%)")
    log("")
    log("estimated_age histogram (5-year buckets) — argue the band from this")
    for lo in sorted(hist):
        mark = "  <- band" if band[0] <= lo + 4 and lo <= band[1] else ""
        log(f"  {lo:>3}-{lo + 4:<3} {hist[lo]:>12,}{mark}")


def _selftest():
    import tempfile
    rows = [
        {"v": {"verdict": "real_person", "gender": "man",
               "age_group": "unknown", "estimated_age": 44.0}},   # labeler abstained
        {"v": {"verdict": "real_person", "gender": "woman",
               "age_group": "child", "estimated_age": 7.0}},      # girl
        {"v": {"verdict": "real_person", "gender": "unknown",
               "age_group": "child", "estimated_age": 5.0}},      # child, no gender
        {"v": {"verdict": "real_person", "gender": "man",
               "age_group": "adult", "estimated_age": 15.0}},     # teen -> AgeUnknown
        {"v": {"verdict": "real_person", "gender": "man",
               "age_group": "adult", "estimated_age": 40.0}},
        {"v": {"verdict": "not_person"}},
    ]
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "v.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        st, hist = count([f])
        # narrowing the band CANNOT touch the labeler's own abstention
        st2, _ = count([f], band=(99, 99))
    assert st["verdicts"] == 6 and st["people"] == 5 and st["not_person"] == 1
    assert st["child_gender_known"] == 1 and st["child_gender_unknown"] == 1
    assert st["age_AgeUnknown"] == 2, st           # the 15-year-old + the abstention
    assert st["why_teen_band"] == 1 and st["why_age_group_unknown"] == 1, st
    assert st["why_other"] == 0, "the why-rule drifted from two_axis_ids"
    assert st["age_Child"] == 2 and st["age_Adult"] == 1
    assert st["gender_GenderUnknown"] == 1
    assert st2["age_AgeUnknown"] == 1 and st2["why_age_group_unknown"] == 1, st2
    assert st2["why_teen_band"] == 0 and st2["age_Adult"] == 2, st2
    report(st, hist, TA.AGE_UNKNOWN_BAND, log=lambda *a: None)   # smoke the printer
    print("exp17_phase0_counts.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", nargs="*", default=[],
                    help="verdicts*.jsonl file(s), or a run dir containing them")
    ap.add_argument("--age-unknown-band", default="13,17", metavar="LO,HI")
    ap.add_argument("--age-unknown-lowconf", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.verdicts:
        ap.error("--verdicts required (or --selftest)")
    paths = []
    for v in a.verdicts:
        p = Path(v)
        paths += sorted(p.glob("verdicts*.jsonl")) if p.is_dir() else [p]
    if not paths:
        ap.error("no verdicts*.jsonl found at the given path(s)")
    lo, hi = (int(x) for x in a.age_unknown_band.split(","))
    print(f"[phase0] {len(paths)} verdict file(s)")
    st, hist = count(paths, (lo, hi), a.age_unknown_lowconf)
    report(st, hist, (lo, hi))


if __name__ == "__main__":
    main()
