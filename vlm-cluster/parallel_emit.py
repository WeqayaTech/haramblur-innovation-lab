#!/usr/bin/env python3
"""Parallel drop-in for spotlight_run.py --emit-only.

The serial emit walks 475k label files one network round-trip at a time
(~35 img/s ~ 4 hours). This runs the SAME per-image transform -- merge(),
CLASS_ID and the keep/delete/relabel rules are imported from spotlight_run,
not reimplemented -- across N processes with progress lines, and lands in
minutes. The selftest executes BOTH implementations on the same synthetic
run and asserts identical label bytes, identical stats, and identical audit
content (order-independent: worker completion order is nondeterministic).

    python3 parallel_emit.py \
        --raw-labels /workspace/spotlight/raw/oiv7_train \
        --run /workspace/spotlight/run/oiv7_train \
        --out /workspace/spotlight/run/oiv7_train/labels --workers 24

    python3 parallel_emit.py --selftest
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

from spotlight_run import CLASS_ID, CLASS_NAME, child_rescue_allowed, merge
# EXP-2026-17: two rows per person, identical geometry -- one naming the gender
# class, one the age class. Taxonomy + collapse policy live in two_axis.py.
from two_axis import (ADULT, AGE_UNKNOWN, AGE_UNKNOWN_BAND, CHILD,
                      GENDER_UNKNOWN, collapse, two_axis_ids)
from two_axis import CLASS_NAMES as TWO_AXIS_NAMES

# Workers inherit these via fork (copy-on-write) -- the verdict map is built
# once in the parent, never pickled per task.
_VERDICTS: dict = {}
_RAW: Path = None
_OUT: Path = None
_KEEP_UNKNOWN = False
_ALLOW_UNVERIFIED = False
_UNKNOWN_CLASS = None      # int -> emit unknown-gender adults as this class id
_TWO_AXIS = False          # emit two rows per person (gender row + age row)
_AGE_BAND = AGE_UNKNOWN_BAND
_LOWCONF_AGE_UNKNOWN = False
_CHILD_BY_AGE = False      # gender-less child kept on its estimated_age alone
_REQUIRE_SAM_CHILD = False # ...and only if SAM3 called it Child too
_CHILD_MIN_PX = 0          # ...and only above this person_px_height


def _init(verdicts, raw, out, keep_unknown, allow_unverified,
          unknown_class=None, two_axis=False, age_band=AGE_UNKNOWN_BAND,
          lowconf_age_unknown=False, child_by_age=False,
          require_sam_child=False, child_min_px=0):
    global _VERDICTS, _RAW, _OUT, _KEEP_UNKNOWN, _ALLOW_UNVERIFIED, \
        _UNKNOWN_CLASS, _TWO_AXIS, _AGE_BAND, _LOWCONF_AGE_UNKNOWN, \
        _CHILD_BY_AGE, _REQUIRE_SAM_CHILD, _CHILD_MIN_PX
    _VERDICTS, _RAW, _OUT = verdicts, raw, out
    _KEEP_UNKNOWN, _ALLOW_UNVERIFIED = keep_unknown, allow_unverified
    _UNKNOWN_CLASS = unknown_class
    _TWO_AXIS, _AGE_BAND = two_axis, age_band
    _LOWCONF_AGE_UNKNOWN = lowconf_age_unknown
    _CHILD_BY_AGE = child_by_age
    _REQUIRE_SAM_CHILD, _CHILD_MIN_PX = require_sam_child, child_min_px

def _one_image(stem: str):
    """Mirror of spotlight_run.emit()'s per-image body. Returns
    (stats_delta, audit_rows, emitted:bool)."""
    st = {"kept": 0, "deleted": 0, "relabeled": 0,
          "dropped_unknown": 0, "no_verdict": 0, "child_by_age": 0,
          "gender_unknown": 0, "age_unknown": 0, "child": 0, "adult": 0}
    audit = []
    lines = [l for l in (_RAW / f"{stem}.txt").read_text().splitlines()
             if l.strip()]
    verdicts = _VERDICTS.get(stem, {})
    if not _ALLOW_UNVERIFIED and len(verdicts) < len(lines):
        st["no_verdict"] += len(lines) - len(verdicts)
        return st, audit, False

    if _TWO_AXIS:
        kept_lines = []
        for i, line in enumerate(lines):
            r = verdicts.get(i)
            if r is None:
                # no verdict: the raw SAM3 line has no two-axis reading, and
                # emitting it unpaired would crash the trainer's pair check
                st["no_verdict"] += 1
                continue
            ids = two_axis_ids(r.get("v"), _AGE_BAND, _LOWCONF_AGE_UNKNOWN)
            if ids is None:
                st["deleted"] += 1
                audit.append({"image": stem, "det": i, "action": "delete",
                              "was": CLASS_NAME.get(r["sam_class_id"]),
                              "verdict": (r.get("v") or {}).get("verdict")})
                continue
            gid, aid = ids
            st["kept"] += 1
            st["gender_unknown"] += gid == GENDER_UNKNOWN
            st["age_unknown"] += aid == AGE_UNKNOWN
            st["child"] += aid == CHILD
            st["adult"] += aid == ADULT
            # audit only the abstentions — the deviations from a definite
            # label are what a reviewer needs to see, one row per person
            # would be 1.19M rows of noise
            if gid == GENDER_UNKNOWN or aid == AGE_UNKNOWN:
                audit.append({"image": stem, "det": i, "action": "abstain",
                              "gender": TWO_AXIS_NAMES[gid],
                              "age": TWO_AXIS_NAMES[aid],
                              "estimated_age": (r.get("v") or {}).get("estimated_age")})
            geom = line.split()[1:]     # SAM3 polygon preserved byte-for-byte
            kept_lines.append(" ".join([str(gid)] + geom))
            kept_lines.append(" ".join([str(aid)] + geom))
        (_OUT / f"{stem}.txt").write_text("\n".join(kept_lines))
        return st, audit, True

    kept_lines = []
    for i, line in enumerate(lines):
        r = verdicts.get(i)
        if r is None:
            st["no_verdict"] += 1
            kept_lines.append(line)
            continue
        m = merge(r.get("v"), _CHILD_BY_AGE and child_rescue_allowed(
            r, _REQUIRE_SAM_CHILD, _CHILD_MIN_PX))
        if not m["keep"]:
            st["deleted"] += 1
            audit.append({"image": stem, "det": i, "action": "delete",
                          "was": CLASS_NAME.get(r["sam_class_id"]),
                          "verdict": (r.get("v") or {}).get("verdict")})
            continue
        if m.get("via") == "age_only":
            # a gender-less child kept on its age alone: always audited,
            # because this is the population the rule exists to recover
            st["child_by_age"] += 1
            audit.append({"image": stem, "det": i, "action": "child_by_age",
                          "was": CLASS_NAME.get(r["sam_class_id"]),
                          "estimated_age": (r.get("v") or {}).get("estimated_age"),
                          "person_px_height": r.get("person_px_height")})
        if m["final_class"] == "Unknown":
            if _UNKNOWN_CLASS is not None:
                # Trainer contract: unknown-gender people survive as their
                # own class so the training pipeline can mask those regions
                # at load time (neither positive nor negative signal),
                # instead of emit silently deleting a real person from
                # supervision. Counted under dropped_unknown so totals stay
                # comparable with the baseline emit; not counted as a
                # relabel.
                st["dropped_unknown"] += 1
                audit.append({"image": stem, "det": i,
                              "action": "kept_as_unknown_class"})
                kept_lines.append(" ".join([str(_UNKNOWN_CLASS)]
                                           + line.split()[1:]))
                continue
            elif not _KEEP_UNKNOWN:
                st["dropped_unknown"] += 1
                audit.append({"image": stem, "det": i,
                              "action": "drop_unknown_gender"})
                continue
            else:
                new_id = r["sam_class_id"]
        else:
            new_id = CLASS_ID[m["final_class"]]
        if new_id != r["sam_class_id"]:
            st["relabeled"] += 1
            audit.append({"image": stem, "det": i, "action": "relabel",
                          "was": CLASS_NAME.get(r["sam_class_id"]),
                          "now": m["final_class"]})
        st["kept"] += 1
        kept_lines.append(" ".join([str(new_id)] + line.split()[1:]))
    (_OUT / f"{stem}.txt").write_text("\n".join(kept_lines))
    return st, audit, True


def run(raw: Path, run_dir: Path, out: Path, workers, keep_unknown,
        allow_unverified, unknown_class=None, two_axis=False,
        age_band=AGE_UNKNOWN_BAND, lowconf_age_unknown=False,
        child_by_age=False, require_sam_child=False, child_min_px=0):
    t0 = time.time()
    by_img = {}
    n_v = 0
    for f in sorted(run_dir.glob("verdicts*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # slim record: emit only needs v + sam_class_id; dropping
            # raw_text keeps the fork image ~5x smaller. person_px_height is
            # carried ONLY in --child-by-age mode, where it is both a guard
            # (--child-min-px) and the field a reviewer wants in the audit --
            # 1.58M extra dict entries are not worth paying for otherwise.
            slim = {"v": r.get("v"), "sam_class_id": r["sam_class_id"]}
            if child_by_age:
                slim["person_px_height"] = r.get("person_px_height")
            by_img.setdefault(r["image_stem"], {})[r["det_index"]] = slim
            n_v += 1
    print(f"[emit] {n_v:,} verdicts for {len(by_img):,} images loaded "
          f"({time.time()-t0:.0f}s)", flush=True)

    stems = sorted(p.stem for p in raw.glob("*.txt"))
    out.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "kept": 0, "deleted": 0, "relabeled": 0,
             "dropped_unknown": 0, "no_verdict": 0, "child_by_age": 0,
             "images_skipped_unverified": 0,
             "gender_unknown": 0, "age_unknown": 0, "child": 0, "adult": 0}
    audit_f = (out / "_audit.jsonl").open("w")
    t1, done = time.time(), 0
    with Pool(workers, initializer=_init,
              initargs=(by_img, raw, out, keep_unknown, allow_unverified,
                        unknown_class, two_axis, age_band,
                        lowconf_age_unknown, child_by_age,
                        require_sam_child, child_min_px)) as pool:
        for st, audit, emitted in pool.imap_unordered(_one_image, stems,
                                                      chunksize=256):
            done += 1
            for k, v in st.items():
                stats[k] += v
            if emitted:
                stats["images"] += 1
            else:
                stats["images_skipped_unverified"] += 1
            for row in audit:
                audit_f.write(json.dumps(row) + "\n")
            if done % 50000 == 0:
                r = done / (time.time() - t1)
                print(f"  {done:,}/{len(stems):,}  {r:.0f} img/s  "
                      f"ETA {(len(stems)-done)/r/60:.1f} min", flush=True)
    audit_f.close()
    (out / "_emit_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def _selftest():
    import tempfile
    import spotlight_run
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = td / "raw"; raw.mkdir()
        poly = "0.1 0.1 0.4 0.1 0.4 0.9"
        # img a: delete + relabel(child->line stays) + keep man
        (raw / "a.txt").write_text(f"0 {poly}\n2 {poly}\n1 {poly}\n")
        # img b: unknown-gender drop
        (raw / "b.txt").write_text(f"0 {poly}\n")
        # img c: partially verified -> skipped
        (raw / "c.txt").write_text(f"1 {poly}\n1 {poly}\n")
        # img d: a 15-year-old -- the band every model measured as a coin flip
        (raw / "d.txt").write_text(f"1 {poly}\n")
        V = [
            {"image_stem": "a", "det_index": 0, "sam_class_id": 0,
             "v": {"verdict": "not_person", "gender": "unknown",
                   "age_group": "unknown", "estimated_age": None}},
            {"image_stem": "a", "det_index": 1, "sam_class_id": 2,
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult", "estimated_age": 34.0}},
            {"image_stem": "a", "det_index": 2, "sam_class_id": 1,
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult", "estimated_age": 40.0}},
            {"image_stem": "b", "det_index": 0, "sam_class_id": 0,
             "v": {"verdict": "real_person", "gender": "unknown",
                   "age_group": "adult", "estimated_age": None}},
            {"image_stem": "c", "det_index": 0, "sam_class_id": 1,
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult", "estimated_age": 30.0}},
            {"image_stem": "d", "det_index": 0, "sam_class_id": 1,
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult", "estimated_age": 15.0}},
        ]
        rd = td / "run"; rd.mkdir()
        (rd / "verdicts_batch.jsonl").write_text(
            "\n".join(json.dumps(v) for v in V) + "\n")
        # serial reference
        s_stats = spotlight_run.emit(raw, rd / "verdicts_batch.jsonl",
                                     td / "out_serial", keep_unknown=False)
        # parallel under test
        p_stats = run(raw, rd, td / "out_par", workers=3,
                      keep_unknown=False, allow_unverified=False)
        # the parallel emit reports extra two-axis counters the serial
        # reference has no concept of; compare on the serial key set so the
        # equivalence guarantee stays exact rather than being relaxed
        assert {k: p_stats[k] for k in s_stats} == s_stats, \
            f"stats differ:\n{s_stats}\n{p_stats}"
        assert set(s_stats) <= set(p_stats), "parallel emit lost a stats key"
        s_files = {p.name: p.read_text() for p in (td / "out_serial").glob("*.txt")}
        p_files = {p.name: p.read_text() for p in (td / "out_par").glob("*.txt")}
        assert s_files == p_files, "label files differ"
        sa = sorted((td / "out_serial" / "_audit.jsonl").read_text().splitlines())
        pa = sorted((td / "out_par" / "_audit.jsonl").read_text().splitlines())
        assert sa == pa, "audit content differs"
        # unknown-class mode: the dropped-unknown detection survives as class 3
        u_stats = run(raw, rd, td / "out_unk", workers=2,
                      keep_unknown=False, allow_unverified=False,
                      unknown_class=3)
        assert u_stats["dropped_unknown"] == 1
        b_lines = (td / "out_unk" / "b.txt").read_text().splitlines()
        assert b_lines and b_lines[0].startswith("3 "), b_lines
        # geometry identical to the source line, only the class id changed
        src = (raw / "b.txt").read_text().split()[1:]
        assert b_lines[0].split()[1:] == src
        # kept/relabeled counters unchanged vs baseline
        assert u_stats["kept"] == p_stats["kept"]
        assert u_stats["relabeled"] == p_stats["relabeled"]

        # --- --child-by-age: a gender-less child kept on its age alone ---
        # own raw/run dirs so the baseline equivalence fixtures above stay
        # exactly as they were
        kraw = td / "raw_kid"; kraw.mkdir()
        (kraw / "e.txt").write_text(f"1 {poly}\n0 {poly}\n")
        krd = td / "run_kid"; krd.mkdir()
        KV = lambda i, c, **kw: {
            "image_stem": "e", "det_index": i, "sam_class_id": c,
            "v": {"verdict": "real_person", "gender": "unknown",
                  "age_group": "unknown", "estimated_age": 3.0,
                  "highlight_quality": "good", "confidence": "low", **kw}}
        (krd / "verdicts_batch.jsonl").write_text("\n".join(json.dumps(v) for v in [
            KV(0, 1),                              # toddler -> rescued
            KV(1, 0, estimated_age=0.0)]) + "\n")  # placeholder age -> stays dropped
        k_off = run(kraw, krd, td / "kid_off", workers=2, keep_unknown=False,
                    allow_unverified=False)
        k_on = run(kraw, krd, td / "kid_on", workers=2, keep_unknown=False,
                   allow_unverified=False, child_by_age=True)
        assert k_off["kept"] == 0 and k_off["dropped_unknown"] == 2, k_off
        assert k_off["child_by_age"] == 0, k_off
        assert k_on["kept"] == 1 and k_on["dropped_unknown"] == 1, k_on
        assert k_on["child_by_age"] == 1, k_on
        e_lines = (td / "kid_on" / "e.txt").read_text().splitlines()
        assert len(e_lines) == 1 and e_lines[0].startswith("2 "), e_lines
        assert e_lines[0].split()[1:] == (kraw / "e.txt").read_text(
            ).splitlines()[0].split()[1:], "geometry must be untouched"
        # the equivalence guarantee has to hold in the new mode too, or the
        # parallel emit is no longer a drop-in for the serial reference
        k_ser = spotlight_run.emit(kraw, krd / "verdicts_batch.jsonl",
                                   td / "kid_serial", keep_unknown=False,
                                   child_by_age=True)
        assert {k: k_on[k] for k in k_ser} == k_ser, (k_ser, k_on)
        assert {p.name: p.read_text() for p in (td / "kid_serial").glob("*.txt")} \
            == {p.name: p.read_text() for p in (td / "kid_on").glob("*.txt")}, \
            "label files differ in --child-by-age mode"
        assert sorted((td / "kid_serial" / "_audit.jsonl").read_text().splitlines()) \
            == sorted((td / "kid_on" / "_audit.jsonl").read_text().splitlines())

        # --child-require-sam: det 0's SAM class is 1, det 1's is 0, so
        # demanding SAM3 agreement rescues nobody here
        k_sam = run(kraw, krd, td / "kid_sam", workers=2, keep_unknown=False,
                    allow_unverified=False, child_by_age=True,
                    require_sam_child=True)
        assert k_sam["child_by_age"] == 0 and k_sam["kept"] == 0, k_sam
        k_sam_ser = spotlight_run.emit(kraw, krd / "verdicts_batch.jsonl",
                                       td / "kid_sam_serial", keep_unknown=False,
                                       child_by_age=True, require_sam_child=True)
        assert {k: k_sam[k] for k in k_sam_ser} == k_sam_ser, (k_sam_ser, k_sam)
        # ...and with a SAM-Child detection it does rescue, px floor permitting
        krd2 = td / "run_kid2"; krd2.mkdir()
        (krd2 / "verdicts_batch.jsonl").write_text("\n".join(json.dumps(v) for v in [
            dict(KV(0, 2), person_px_height=150.0),
            dict(KV(1, 2), person_px_height=20.0)]) + "\n")
        k2 = run(kraw, krd2, td / "kid2", workers=2, keep_unknown=False,
                 allow_unverified=False, child_by_age=True,
                 require_sam_child=True)
        assert k2["child_by_age"] == 2, k2
        k3 = run(kraw, krd2, td / "kid3", workers=2, keep_unknown=False,
                 allow_unverified=False, child_by_age=True,
                 require_sam_child=True, child_min_px=100)
        assert k3["child_by_age"] == 1, k3
        assert k3["dropped_unknown"] == 1, k3

        # --- EXP-2026-17 two-axis emit ---------------------------------
        # Run with the band on AND off. The band changes what the model is
        # TAUGHT; it must not change the 3-class projection, because people
        # in it were already adults under merge() and AgeUnknown collapses
        # back to Adult. That makes the equivalence below unconditional.
        for tag, band in (("out_2ax", (13, 17)), ("out_2ax_noband", (99, 99))):
            t_stats = run(raw, rd, td / tag, workers=2, keep_unknown=False,
                          allow_unverified=False, two_axis=True, age_band=band)
            collapsed = {}
            for f in sorted((td / tag).glob("*.txt")):
                rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
                assert len(rows) % 2 == 0, f"{f.name}: odd row count, a pair is broken"
                out = []
                for g, a in zip(rows[0::2], rows[1::2]):
                    gid, aid = int(g[0]), int(a[0])
                    assert 0 <= gid <= 2, f"{f.name}: {gid} is not a gender class"
                    assert 3 <= aid <= 5, f"{f.name}: {aid} is not an age class"
                    assert g[1:] == a[1:], \
                        f"{f.name}: paired rows must share geometry byte-for-byte"
                    assert g[1:] == poly.split(), f"{f.name}: geometry not preserved"
                    cid = collapse(gid, aid, keep_unknown_gender=False)
                    if cid is not None:
                        out.append(" ".join([str(cid)] + g[1:]))
                collapsed[f.name] = "\n".join(out)
            assert collapsed == p_files, (
                f"[{tag}] collapsing two-axis labels must reproduce the 3-class "
                f"emit exactly:\n{collapsed}\n{p_files}")
            # same people kept, and every one of them carries both axes
            assert t_stats["kept"] == p_stats["kept"] + p_stats["dropped_unknown"], \
                "two-axis keeps the unknown-gender people the 3-class emit drops"
            assert t_stats["deleted"] == p_stats["deleted"]
            assert t_stats["gender_unknown"] == 1, t_stats     # image b
        # the band is what moved: the 15-year-old is AgeUnknown only with it on
        a_on = json.loads((td / "out_2ax" / "_emit_stats.json").read_text())
        a_off = json.loads((td / "out_2ax_noband" / "_emit_stats.json").read_text())
        assert a_on["age_unknown"] == 1 and a_off["age_unknown"] == 0, (a_on, a_off)
        assert a_on["adult"] + 1 == a_off["adult"], (a_on, a_off)
    print("parallel_emit.py self-test passed "
          "(byte-identical to spotlight_run.emit; --unknown-class, "
          "--child-by-age and --two-axis verified, incl. 3-class collapse "
          "equivalence)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-labels"); ap.add_argument("--run"); ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--keep-unknown-gender", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--unknown-class", type=int, default=None,
                    help="emit unknown-gender people as this class id "
                         "(e.g. 3) instead of dropping them, so the trainer "
                         "can mask those regions at load time")
    ap.add_argument("--two-axis", action="store_true",
                    help="EXP-2026-17: emit TWO rows per person with identical "
                         f"geometry, classes {list(TWO_AXIS_NAMES)} — one naming "
                         "the gender, one the age. Nothing is dropped: an "
                         "unreadable gender or age gets its own class")
    ap.add_argument("--age-unknown-band", default="13,17", metavar="LO,HI",
                    help="inclusive estimated_age range emitted as AgeUnknown "
                         "(default 13,17 — the band EXP-2026-10 measured as a "
                         "coin flip). Pass an empty range to disable")
    ap.add_argument("--child-by-age", action="store_true",
                    help="a child needs no gender: when Gemini gave no gender "
                         "but a usable estimated_age (1-12) and judged the "
                         "right crop, emit Child instead of dropping the "
                         "detection. Excludes estimated_age <= 0 (Gemini's "
                         "placeholder) and bad-highlight crops — see "
                         "spotlight_run.merge for the measured reason")
    ap.add_argument("--child-require-sam", action="store_true",
                    help="with --child-by-age: only rescue when SAM3 ALSO "
                         "classed the detection Child, so the label rests on "
                         "two independent opinions rather than one "
                         "low-confidence Gemini read")
    ap.add_argument("--child-min-px", type=int, default=0,
                    help="with --child-by-age: skip rescues whose "
                         "person_px_height is below this")
    ap.add_argument("--age-unknown-lowconf", action="store_true",
                    help="also emit AgeUnknown when the verdict's "
                         "confidence field is 'low'")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.raw_labels and a.run and a.out):
        ap.error("--raw-labels --run --out are required")
    if a.two_axis and (a.unknown_class is not None or a.keep_unknown_gender):
        ap.error("--two-axis has its own GenderUnknown class; it cannot be "
                 "combined with --unknown-class / --keep-unknown-gender")
    try:
        lo, hi = (int(x) for x in a.age_unknown_band.split(","))
    except ValueError:
        ap.error("--age-unknown-band wants LO,HI (two integers)")
    if a.two_axis and a.child_by_age:
        ap.error("--two-axis reads age and gender independently; nothing is "
                 "dropped for a missing gender, so --child-by-age is moot")
    run(Path(a.raw_labels), Path(a.run), Path(a.out), a.workers,
        a.keep_unknown_gender, a.allow_unverified, a.unknown_class,
        a.two_axis, (lo, hi), a.age_unknown_lowconf, a.child_by_age,
        a.child_require_sam, a.child_min_px)


if __name__ == "__main__":
    main()
