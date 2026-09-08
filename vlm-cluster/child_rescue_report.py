#!/usr/bin/env python3
"""
How many CHILDREN did the Spotlight emit throw away for having no gender?

spotlight_run.merge() decides a detection's class in this order:

    age_group == "child" and estimated_age <= 12   -> Child
    gender == "man" / "woman"                      -> Man / Woman
    otherwise                                      -> "Unknown"  (dropped)

So a person is only saved by the age branch when Gemini committed to
`age_group: "child"`. When Gemini abstained on age_group but still wrote
`estimated_age: 2`, the record falls through to the gender branch, finds
"unknown" there too, and the child is deleted from training -- even though the
verdict says, in the very same JSON object, that it is a toddler.

This tool streams the verdict ledger and counts exactly that population, plus
the mirror case (a young estimated_age that was kept as an ADULT because the
gender branch answered first). Both are re-emittable for $0: no API calls, no
GPU, the verdicts already exist.

Every number is computed from the SAME merge() the production emit uses --
imported, not re-implemented -- so "currently dropped" here means what the
label files on disk actually did.

    python3 child_rescue_report.py \
        --verdicts /workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl \
        --out /workspace/spotlight/child_rescue_train.json

    python3 child_rescue_report.py --selftest
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from spotlight_run import CHILD_AGE_MAX, merge


def _age_bucket(age):
    """Year-by-year up to 20 (that is where the whole question lives), then
    coarse. Returns a sort key and a label."""
    if age is None:
        return (999, "none")
    a = int(age)
    if a <= 20:
        return (a, str(a))
    lo = min(a // 10 * 10, 90)
    return (lo, f"{lo}-{lo + 9}")


def _pct(n, d):
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


def count(paths, dump=None):
    """-> stats Counter, age histograms, height samples. Streams the file.

    `dump` is an open file handle: every detection in the rescuable population
    is written to it verbatim (one JSON per line, Gemini's raw_text included),
    so the decision can be argued from what the model actually said rather
    than from these aggregates."""
    st = Counter()
    hist_dropped = Counter()          # every dropped-unknown detection
    hist_all = Counter()
    grp_of_rescuable = Counter()      # why did a <=12 y/o reach the gender branch
    conf_of_rescuable = Counter()
    heights_rescuable = []
    heights_dropped = []

    for p in paths:
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                v = rec.get("v")
                st["verdicts"] += 1
                m = merge(v)
                age = (v or {}).get("estimated_age")
                grp = (v or {}).get("age_group")
                h = rec.get("person_px_height")
                young = isinstance(age, (int, float)) and age <= CHILD_AGE_MAX

                if not m["keep"]:
                    st["not_person"] += 1
                    continue
                st["kept_or_dropped_people"] += 1
                hist_all[_age_bucket(age)] += 1
                cls = m["final_class"]

                if cls == "Unknown":
                    st["dropped_unknown"] += 1
                    hist_dropped[_age_bucket(age)] += 1
                    if h:
                        heights_dropped.append(h)
                    if young:
                        # THE POPULATION THIS TOOL EXISTS FOR
                        st["rescuable_child"] += 1
                        if dump is not None:
                            dump.write(json.dumps(rec) + "\n")
                        grp_of_rescuable[str(grp)] += 1
                        conf_of_rescuable[str((v or {}).get("confidence"))] += 1
                        if h:
                            heights_rescuable.append(h)
                    elif isinstance(age, (int, float)):
                        if age <= 17:
                            st["dropped_teen_13_17"] += 1
                            if grp == "child":
                                st["dropped_teen_13_17_grp_child"] += 1
                        else:
                            st["dropped_adult"] += 1
                    else:
                        st["dropped_age_none"] += 1
                        if grp == "child":
                            st["dropped_age_none_grp_child"] += 1
                else:
                    st[f"kept_{cls}"] += 1
                    if cls != "Child" and young:
                        # mirror case: gender answered first, so a verdict
                        # that reads <=12 was written into the label file as
                        # an adult. Reported, deliberately NOT auto-fixed --
                        # see the report text.
                        st["kept_adult_but_young"] += 1
                        st[f"kept_adult_but_young_{cls}"] += 1
                        st[f"kept_adult_but_young_grp_{grp}"] += 1
    return (st, hist_dropped, hist_all, grp_of_rescuable, conf_of_rescuable,
            heights_rescuable, heights_dropped)


def _quantiles(xs):
    if not xs:
        return {}
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return {"n": len(s), "p10": q(.10), "median": q(.50), "p90": q(.90)}


def report(st, hist_dropped, hist_all, grp, conf, h_res, h_drop, log=print):
    people = st["kept_or_dropped_people"] or 1
    log(f"verdicts                  {st['verdicts']:,}")
    log(f"  not a person (deleted)  {st['not_person']:,}")
    log(f"  people                  {st['kept_or_dropped_people']:,}")
    log("")
    log("WHAT THE PRODUCTION EMIT DID (spotlight_run.merge, unchanged)")
    for c in ("Woman", "Man", "Child"):
        log(f"  kept as {c:<6}          {st['kept_' + c]:>10,}  {_pct(st['kept_' + c], people)}")
    log(f"  dropped unknown-gender  {st['dropped_unknown']:>10,}  {_pct(st['dropped_unknown'], people)}")
    log("")
    log("THE DROPPED PILE, BY estimated_age")
    log(f"  age <= {CHILD_AGE_MAX} (CHILDREN)     {st['rescuable_child']:>10,}"
        f"  {_pct(st['rescuable_child'], st['dropped_unknown'])} of dropped")
    log(f"  age 13-17 (teen band)   {st['dropped_teen_13_17']:>10,}"
        f"  {_pct(st['dropped_teen_13_17'], st['dropped_unknown'])}"
        f"   (of which age_group=child: {st['dropped_teen_13_17_grp_child']:,})")
    log(f"  age 18+                 {st['dropped_adult']:>10,}"
        f"  {_pct(st['dropped_adult'], st['dropped_unknown'])}")
    log(f"  no estimated_age at all {st['dropped_age_none']:>10,}"
        f"  {_pct(st['dropped_age_none'], st['dropped_unknown'])}"
        f"   (of which age_group=child: {st['dropped_age_none_grp_child']:,})")
    log("")
    log(f"WHY THOSE <= {CHILD_AGE_MAX} y/o WERE NOT CAUGHT BY THE age_group BRANCH")
    for k, n in grp.most_common():
        log(f"  age_group = {k:<10}   {n:>10,}  {_pct(n, st['rescuable_child'])}")
    log("  verdict confidence: " + ", ".join(f"{k}={n:,}" for k, n in conf.most_common()))
    log("")
    log("SIZE (person_px_height)")
    log(f"  rescuable children  {_quantiles(h_res)}")
    log(f"  all dropped people  {_quantiles(h_drop)}")
    log("")
    log("MIRROR CASE — kept as an ADULT although estimated_age <= "
        f"{CHILD_AGE_MAX} (gender branch answered first)")
    log(f"  total                   {st['kept_adult_but_young']:>10,}"
        f"   (Woman {st['kept_adult_but_young_Woman']:,} / Man {st['kept_adult_but_young_Man']:,})")
    for k, n in sorted(st.items()):
        if k.startswith("kept_adult_but_young_grp_"):
            log(f"    age_group = {k.split('grp_')[1]:<10} {n:>10,}")
    log("")
    log("AGE HISTOGRAM — dropped-unknown detections only")
    for key in sorted(hist_dropped):
        lab = key[1]
        n = hist_dropped[key]
        bar = "#" * min(60, int(60 * n / max(hist_dropped.values())))
        log(f"  {lab:>6} {n:>9,}  {bar}")


def _selftest():
    import tempfile
    rows = [
        # a toddler Gemini would not gender AND would not age-group: the case
        {"person_px_height": 40, "v": {"verdict": "real_person", "gender": "unknown",
         "age_group": "unknown", "estimated_age": 2.0, "confidence": "low"}},
        {"person_px_height": 55, "v": {"verdict": "real_person", "gender": "unknown",
         "age_group": "unknown", "estimated_age": 3.0, "confidence": "high"}},
        # committed child -> already kept today, must NOT be counted as rescuable
        {"person_px_height": 300, "v": {"verdict": "real_person", "gender": "unknown",
         "age_group": "child", "estimated_age": 6.0, "confidence": "high"}},
        # genuinely unreadable adult -> stays dropped
        {"person_px_height": 30, "v": {"verdict": "real_person", "gender": "unknown",
         "age_group": "adult", "estimated_age": 40.0, "confidence": "low"}},
        # teen, dropped, NOT a child under the <=12 definition
        {"person_px_height": 90, "v": {"verdict": "real_person", "gender": "unknown",
         "age_group": "unknown", "estimated_age": 15.0, "confidence": "low"}},
        # mirror case: young age but gender known -> written as an adult
        {"person_px_height": 200, "v": {"verdict": "real_person", "gender": "woman",
         "age_group": "unknown", "estimated_age": 9.0, "confidence": "high"}},
        # not a person
        {"person_px_height": 10, "v": {"verdict": "not_person", "gender": "unknown",
         "age_group": "unknown", "estimated_age": None, "confidence": "low"}},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        d = Path(td) / "dump.jsonl"
        with open(d, "w") as fh:
            st, hd, ha, grp, conf, hr, hdp = count([p], dump=fh)
        dumped = [json.loads(l) for l in d.read_text().splitlines() if l.strip()]
        assert len(dumped) == 2, dumped
        assert {r["v"]["estimated_age"] for r in dumped} == {2.0, 3.0}, dumped
    assert st["verdicts"] == 7 and st["not_person"] == 1, st
    assert st["kept_or_dropped_people"] == 6, st
    assert st["kept_Child"] == 1 and st["kept_Woman"] == 1, st
    assert st["dropped_unknown"] == 4, st
    assert st["rescuable_child"] == 2, st            # the 2 y/o and the 3 y/o
    assert st["dropped_teen_13_17"] == 1 and st["dropped_adult"] == 1, st
    assert grp["unknown"] == 2, grp
    assert st["kept_adult_but_young"] == 1, st
    assert st["kept_adult_but_young_Woman"] == 1, st
    assert _quantiles(hr)["median"] == 55, _quantiles(hr)
    report(st, hd, ha, grp, conf, hr, hdp, log=lambda *a: None)
    print("child_rescue_report.py self-tests passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="write the stats as JSON too")
    ap.add_argument("--dump", default=None,
                    help="write every rescuable detection (with raw_text) here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.verdicts:
        ap.error("--verdicts is required")
    dump = open(a.dump, "w") if a.dump else None
    try:
        res = count([Path(p) for p in a.verdicts], dump=dump)
    finally:
        if dump:
            dump.close()
    report(*res)
    if a.out:
        st, hd, ha, grp, conf, hr, hdp = res
        Path(a.out).write_text(json.dumps({
            "child_age_max": CHILD_AGE_MAX,
            "stats": dict(st),
            "hist_dropped": {k[1]: n for k, n in sorted(hd.items())},
            "hist_all_people": {k[1]: n for k, n in sorted(ha.items())},
            "age_group_of_rescuable": dict(grp),
            "confidence_of_rescuable": dict(conf),
            "height_rescuable": _quantiles(hr),
            "height_all_dropped": _quantiles(hdp),
        }, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
