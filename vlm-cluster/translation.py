#!/usr/bin/env python3
"""
Translation layer — map raw (gender, age) into a customizable label taxonomy.

Both sides of an evaluation emit RAW two-axis data:
  - human GT (e.g. LAGENDA): numeric age + gender (M/F)
  - VLM describe.py output:  apparent_age_band + apparent_gender

A `Translation` is a pure, declarative mapping over (age_bucket x gender) that
BOTH sides pass through identically, so any taxonomy — the production
{Woman, Man, Child}, a two-axis Gender+Age view, gender-only, child-vs-adult —
is just a different combination of the same two axes. Add a new taxonomy by
adding one `Translation` to the registry; nothing else changes.

Canonical age buckets are the describe.py band vocabulary. Numeric ages map into
the same buckets using the ranges documented in describe.py's OBJECT_PROMPT, so
GT (numeric) and VLM (band) are always comparable.

Run `python translation.py` to execute the built-in self-tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# canonical age buckets (== describe.py apparent_age_band vocabulary)
# ---------------------------------------------------------------------------
ALL_BUCKETS = ["infant", "toddler", "child", "preteen", "teenager",
               "young_adult", "adult", "middle_aged", "elderly", "senior"]
UNKNOWN = "unknown"

# Project definition of "Child": preteen and younger (puberty boundary),
# matching label_errors.CHILD_STAGES. Teenager is post-puberty (adult side).
CHILD_BUCKETS = {"infant", "toddler", "child", "preteen"}
TEEN_BUCKETS = {"teenager"}
ADULT_BUCKETS = {"young_adult", "adult", "middle_aged", "elderly", "senior"}

# numeric upper edges -> band (from describe.py OBJECT_PROMPT ranges)
_YEAR_EDGES = [(0, "infant"), (3, "toddler"), (9, "child"), (12, "preteen"),
               (17, "teenager"), (29, "young_adult"), (49, "adult"),
               (64, "middle_aged"), (79, "elderly")]  # else -> senior


def age_bucket_from_years(age) -> str:
    """Numeric age -> canonical band bucket. None/negative -> 'unknown'."""
    if age is None:
        return UNKNOWN
    try:
        n = int(age)
    except (TypeError, ValueError):
        return UNKNOWN
    if n < 0:
        return UNKNOWN
    for edge, name in _YEAR_EDGES:
        if n <= edge:
            return name
    return "senior"


def age_bucket_from_band(band) -> str:
    """VLM apparent_age_band -> canonical bucket. Unknown/unmapped -> 'unknown'."""
    b = (band or "").strip().lower()
    return b if b in ALL_BUCKETS else UNKNOWN


def norm_gender(x) -> str:
    """Any gender encoding -> {'man', 'woman', 'unknown'}."""
    g = str(x or "").strip().lower()
    if g in ("f", "female", "woman", "w"):
        return "woman"
    if g in ("m", "male", "man"):
        return "man"
    return "unknown"


# ---------------------------------------------------------------------------
# reusable per-axis rules
# ---------------------------------------------------------------------------
def _gender_label(gender: str) -> Optional[str]:
    return {"woman": "Woman", "man": "Man"}.get(gender)  # None if unknown


def _age_group_coarse(bucket: str) -> Optional[str]:
    if bucket in CHILD_BUCKETS:
        return "child"
    if bucket in TEEN_BUCKETS:
        return "teen"
    if bucket in ("young_adult", "adult", "middle_aged"):
        return "adult"
    if bucket in ("elderly", "senior"):
        return "senior"
    return None  # unknown


# ---------------------------------------------------------------------------
# Head / Translation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Head:
    """One classification axis of a taxonomy.

    `fn(gender, bucket) -> class | None`. None means the record is
    not-applicable / undecidable for this head (e.g. gender of a child, or an
    unknown age) and is counted as coverage, never as a wrong answer.
    """
    name: str
    classes: tuple
    fn: Callable[[str, str], Optional[str]]

    def label(self, gender: str, bucket: str) -> Optional[str]:
        return self.fn(gender, bucket)


@dataclass(frozen=True)
class Translation:
    """A taxonomy = one or more independent heads. Single-head schemes collapse
    the two axes into flat classes; multi-head schemes (e.g. two_axis) keep the
    axes separate and are scored per head."""
    name: str
    heads: tuple

    def labels(self, gender: str, bucket: str) -> dict:
        """{head_name: class | None} for a raw (gender, bucket) pair."""
        return {h.name: h.label(gender, bucket) for h in self.heads}


# --- head rule functions ----------------------------------------------------
def _prod3(gender: str, bucket: str) -> Optional[str]:
    if bucket == UNKNOWN:
        return None
    if bucket in CHILD_BUCKETS:
        return "Child"
    return _gender_label(gender)            # Woman/Man, or None if gender unknown


def _gender_adults_only(gender: str, bucket: str) -> Optional[str]:
    if bucket == UNKNOWN or bucket in CHILD_BUCKETS:
        return None                          # gender N/A for children / unknown age
    return _gender_label(gender)


def _make_child_vs_adult(child_buckets: frozenset) -> Callable[[str, str], Optional[str]]:
    def fn(gender: str, bucket: str) -> Optional[str]:
        if bucket == UNKNOWN:
            return None
        return "Child" if bucket in child_buckets else "Adult"
    return fn


def _make_gender(exclude: frozenset):
    """Gender head that abstains for the given buckets (and unknown age)."""
    def fn(gender: str, bucket: str):
        if bucket == UNKNOWN or bucket in exclude:
            return None
        return _gender_label(gender)
    return fn


def _gender_any(gender: str, bucket: str) -> Optional[str]:
    """Gender head that ignores age entirely — abstains only when the gender
    itself is unreadable. Every other gender head here abstains on an unknown
    age bucket too, which re-couples the axes; a genuinely two-axis taxonomy
    must not do that (a clear face of unclear age still has a clear gender)."""
    return _gender_label(gender)


# --- registry ---------------------------------------------------------------
_UNDER10_CHILD = frozenset({"infant", "toddler", "child"})     # 9/10 edge (age <=9)
_PUBERTY_CHILD = frozenset(CHILD_BUCKETS)                       # 12/13 edge
_MINOR_CHILD = frozenset(CHILD_BUCKETS | TEEN_BUCKETS)          # 17/18 edge

TRANSLATIONS = {
    "production_3class": Translation(
        "production_3class",
        (Head("class", ("Woman", "Man", "Child"), _prod3),)),

    "two_axis": Translation(
        "two_axis",
        (Head("gender", ("Woman", "Man"), _gender_adults_only),
         Head("age", ("child", "teen", "adult", "senior"),
              lambda g, b: _age_group_coarse(b)))),

    # EXP-2026-17: the two-axis detector head. Unlike "two_axis" above, the
    # gender axis is scored on EVERYONE (children included — the head predicts
    # it) and does not abstain on an unknown age, and the age axis is the
    # product's binary Adult/Child at the puberty edge rather than 4 bands.
    # Teens are deliberately NOT special-cased here: GT stays definite so the
    # 13-17 band can be scored as "did the model correctly answer AgeUnknown".
    "two_axis_full": Translation(
        "two_axis_full",
        (Head("gender", ("Woman", "Man"), _gender_any),
         Head("age", ("Adult", "Child"), _make_child_vs_adult(_PUBERTY_CHILD)))),

    "gender_only": Translation(
        "gender_only",
        (Head("gender", ("Woman", "Man"), _gender_adults_only),)),

    "child_vs_adult_9": Translation(            # young-child boundary (age <=9)
        "child_vs_adult_9",
        (Head("stage", ("Child", "Adult"), _make_child_vs_adult(_UNDER10_CHILD)),)),

    "gender_9plus": Translation(                # gender scored on everyone age >=10
        "gender_9plus",
        (Head("gender", ("Woman", "Man"), _make_gender(_UNDER10_CHILD)),)),

    "child_vs_adult": Translation(              # puberty boundary (12/13)
        "child_vs_adult",
        (Head("stage", ("Child", "Adult"), _make_child_vs_adult(_PUBERTY_CHILD)),)),

    "child_vs_adult_18": Translation(           # legal-minor boundary (17/18)
        "child_vs_adult_18",
        (Head("stage", ("Child", "Adult"), _make_child_vs_adult(_MINOR_CHILD)),)),

    "age_band": Translation(
        "age_band",
        (Head("age", ("child", "teen", "adult", "senior"),
              lambda g, b: _age_group_coarse(b)),)),
}


def get(name: str) -> Translation:
    if name not in TRANSLATIONS:
        raise KeyError(f"unknown translation '{name}'. "
                       f"available: {sorted(TRANSLATIONS)}")
    return TRANSLATIONS[name]


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------
def _selftest():
    # numeric and band map into the same bucket
    assert age_bucket_from_years(8) == "child"
    assert age_bucket_from_band("child") == "child"
    assert age_bucket_from_years(8) == age_bucket_from_band("child")
    assert age_bucket_from_years(0) == "infant"
    assert age_bucket_from_years(15) == "teenager"
    assert age_bucket_from_years(45) == "adult"
    assert age_bucket_from_years(90) == "senior"
    assert age_bucket_from_years(-1) == UNKNOWN
    assert age_bucket_from_band("bogus") == UNKNOWN

    # gender normalization
    assert norm_gender("F") == "woman" and norm_gender("male") == "man"
    assert norm_gender("") == "unknown" and norm_gender("unclear") == "unknown"

    # production_3class == compare_classes.vlm_truth semantics:
    #   child age -> Child; else gender -> Woman/Man; unknown gender/age -> None
    p3 = get("production_3class").heads[0]
    cases = [
        ("woman", "child", "Child"), ("man", "preteen", "Child"),
        ("woman", "adult", "Woman"), ("man", "teenager", "Man"),
        ("unknown", "adult", None), ("woman", UNKNOWN, None),
    ]
    for g, b, want in cases:
        assert p3.label(g, b) == want, (g, b, p3.label(g, b), want)

    # gender_only returns None for children and unknown age
    go = get("gender_only").heads[0]
    assert go.label("woman", "child") is None
    assert go.label("woman", "adult") == "Woman"
    assert go.label("unknown", "adult") is None

    # two_axis has two independent heads
    ta = get("two_axis")
    assert [h.name for h in ta.heads] == ["gender", "age"]
    lab = ta.labels("man", "child")
    assert lab == {"gender": None, "age": "child"}      # gender N/A for child
    lab = ta.labels("woman", "adult")
    assert lab == {"gender": "Woman", "age": "adult"}

    # child_vs_adult thresholds differ on teenagers
    assert get("child_vs_adult").heads[0].label("man", "teenager") == "Adult"
    assert get("child_vs_adult_18").heads[0].label("man", "teenager") == "Child"

    # two_axis_full: the axes are genuinely independent — gender is answered
    # for children and for unknown-age people, age is answered for
    # unknown-gender people. This is what "two_axis" above does NOT do.
    taf = get("two_axis_full")
    assert [h.name for h in taf.heads] == ["gender", "age"]
    assert taf.labels("woman", "child") == {"gender": "Woman", "age": "Child"}
    assert taf.labels("man", "adult") == {"gender": "Man", "age": "Adult"}
    assert taf.labels("woman", UNKNOWN) == {"gender": "Woman", "age": None}
    assert taf.labels("unknown", "adult") == {"gender": None, "age": "Adult"}
    assert taf.labels("unknown", UNKNOWN) == {"gender": None, "age": None}
    # teenagers stay definite in GT (Adult at the 12/13 edge) so the model's
    # AgeUnknown answers on that band can be scored as abstention, not error
    assert taf.labels("man", "teenager")["age"] == "Adult"
    # contrast: the older two_axis scheme couples the axes
    assert get("two_axis").labels("woman", "child")["gender"] is None

    print("translation.py self-tests passed")


if __name__ == "__main__":
    _selftest()
