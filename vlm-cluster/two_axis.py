#!/usr/bin/env python3
"""
EXP-2026-17 — the two-axis (gender x age) taxonomy, defined once.

A person is described by two INDEPENDENT axes instead of one entangled class:

    channels 0-2   GENDER  {Woman, Man, GenderUnknown}
    channels 3-5   AGE     {Adult, Child, AgeUnknown}

In a label file that is two rows per person with identical geometry, one
naming the gender class and one naming the age class. In the model head it is
6 sigmoid channels read as two argmax groups.

Both explicit Unknowns are load-bearing: they are why nothing has to be dropped
for being unreadable (Spotlight discarded 196,119 people, median ~58 px, for
having no gender slot) and why no per-axis loss masking is needed.

This module is stdlib-only and imported by the trainer (train_twolabel.py), the
label emit (parallel_emit.py) and the eval decode (run_ultralytics_labels.py),
so the taxonomy and the collapse policy cannot drift between them.

    python3 two_axis.py     # self-tests
"""
from __future__ import annotations

CLASS_NAMES = ("Woman", "Man", "GenderUnknown", "Adult", "Child", "AgeUnknown")
NC = 6
GROUPS = ((0, 3), (3, 6))          # half-open channel ranges, one per axis
GROUP_NAMES = ("gender", "age")

WOMAN, MAN, GENDER_UNKNOWN, ADULT, CHILD, AGE_UNKNOWN = range(6)
GENDER_ID = {"woman": WOMAN, "man": MAN}     # anything else -> GenderUnknown

# The band where every model this project has measured is unreliable
# (EXP-2026-10: 13-15 -> 76.5% Child, 16-17 -> 14.3%, 18+ -> 0%). Labelling it
# AgeUnknown stops training the model on what is close to a coin flip.
AGE_UNKNOWN_BAND = (13, 17)
CHILD_AGE_MAX = 12                 # == spotlight_run.CHILD_AGE_MAX

# The collapsed label space the existing scorers consume. The ORDER matters:
# conf_sweep.py indexes positionally (classes[0] = Woman, classes[:2] = the
# adults, classes[2] = Child), so keeping UnknownGender last means every
# existing metric keeps its meaning with nothing but --classes passed.
COLLAPSED_NAMES = ("Woman", "Man", "Child", "UnknownGender")
C_WOMAN, C_MAN, C_CHILD, C_UNKNOWN_GENDER = range(4)


def two_axis_ids(v, band=AGE_UNKNOWN_BAND, lowconf_age_unknown=False):
    """One Spotlight/Gemini verdict -> (gender_id, age_id), or None if the
    detection is not a person.

    The axes are read INDEPENDENTLY from the same verdict — unlike
    spotlight_run.merge(), which collapses them and throws a child's gender
    away. `band` is a definition choice recorded here, not a field read out of
    the verdict: the Spotlight prompt forced a child/adult call, so there is no
    ready-made age-unknown bucket the way there is for gender.
    """
    if v is None or v.get("verdict") not in ("real_person", "depiction"):
        return None
    gid = GENDER_ID.get(v.get("gender"), GENDER_UNKNOWN)

    age, grp = v.get("estimated_age"), v.get("age_group")
    if lowconf_age_unknown and v.get("confidence") == "low":
        aid = AGE_UNKNOWN
    elif isinstance(age, (int, float)) and band[0] <= age <= band[1]:
        aid = AGE_UNKNOWN
    elif grp not in ("child", "adult"):
        aid = AGE_UNKNOWN
    elif grp == "child" and (age is None or age <= CHILD_AGE_MAX):
        aid = CHILD
    else:
        aid = ADULT
    return gid, aid


def collapse(gid, aid, keep_unknown_gender=True):
    """Two-axis ids -> a COLLAPSED_NAMES id, or None to drop the detection.

    This IS the product's blur policy, written once: Child is the only class
    never blurred, an unreadable age defaults to Adult (= blur = the safe
    direction), and only an unreadable gender has nothing to say.

    `keep_unknown_gender=True` keeps those people in the label file as their
    own class so class-agnostic Component 1 recall is unaffected and Component
    3 can score them as an abstention rather than as a wrong answer. Passing
    False drops them, which is what a strict 3-class projection does.
    """
    if aid == CHILD:
        return C_CHILD
    if gid == WOMAN:
        return C_WOMAN
    if gid == MAN:
        return C_MAN
    return C_UNKNOWN_GENDER if keep_unknown_gender else None


def split_scores(scores):
    """A length-6 score sequence -> (gender_id, gender_conf, age_id, age_conf).

    The decode the browser must also implement: one argmax per group. The
    detection confidence is deliberately NOT computed here — see
    person_conf — because which scalar gates a box is a policy question,
    while the axis readings are not.
    """
    out = []
    for lo, hi in GROUPS:
        grp = list(scores[lo:hi])
        j = max(range(len(grp)), key=grp.__getitem__)
        out += [lo + j, grp[j]]
    return tuple(out)


def person_conf(scores):
    """The scalar a detection is thresholded and NMS'd on: max over ALL
    channels. Every person has exactly two hot channels at training time, so
    the max is whichever axis the model is more sure of. Keeping it as the
    plain max (rather than, say, the gender confidence) is what makes the
    decode bit-comparable with stock single-label NMS — see the parity check
    in run_ultralytics_labels.selftest."""
    return max(scores)


def _selftest():
    assert len(CLASS_NAMES) == NC
    assert sum(hi - lo for lo, hi in GROUPS) == NC

    V = lambda **kw: {"verdict": "real_person", "gender": "man",
                      "age_group": "adult", "estimated_age": 30.0, **kw}
    assert two_axis_ids(V()) == (MAN, ADULT)
    assert two_axis_ids(V(gender="woman")) == (WOMAN, ADULT)
    assert two_axis_ids(V(gender="unclear")) == (GENDER_UNKNOWN, ADULT)
    # a child keeps its gender — the whole point vs merge()
    assert two_axis_ids(V(gender="woman", age_group="child",
                          estimated_age=7.0)) == (WOMAN, CHILD)
    # the teen band abstains, and it wins over the age_group call
    assert two_axis_ids(V(estimated_age=15.0)) == (MAN, AGE_UNKNOWN)
    assert two_axis_ids(V(age_group="child", estimated_age=15.0)) == (MAN, AGE_UNKNOWN)
    assert two_axis_ids(V(estimated_age=15.0), band=(99, 99)) == (MAN, ADULT)
    assert two_axis_ids(V(age_group="unknown", estimated_age=None)) == (MAN, AGE_UNKNOWN)
    # a 'child' older than the cutoff is an adult, matching merge()'s fallthrough
    assert two_axis_ids(V(age_group="child", estimated_age=30.0)) == (MAN, ADULT)
    assert two_axis_ids(V(confidence="low"), lowconf_age_unknown=True)[1] == AGE_UNKNOWN
    assert two_axis_ids(V(confidence="low"))[1] == ADULT      # off by default
    assert two_axis_ids({"verdict": "not_person"}) is None
    assert two_axis_ids(None) is None

    # collapse: children never blurred, unknown age defaults to adult
    assert collapse(WOMAN, CHILD) == C_CHILD
    assert collapse(GENDER_UNKNOWN, CHILD) == C_CHILD
    assert collapse(WOMAN, AGE_UNKNOWN) == C_WOMAN      # default-to-adult
    assert collapse(MAN, ADULT) == C_MAN
    assert collapse(GENDER_UNKNOWN, ADULT) == C_UNKNOWN_GENDER
    assert collapse(GENDER_UNKNOWN, ADULT, keep_unknown_gender=False) is None

    # decode
    s = [0.1, 0.9, 0.2, 0.7, 0.3, 0.1]
    assert split_scores(s) == (MAN, 0.9, ADULT, 0.7)
    assert person_conf(s) == 0.9
    s2 = [0.1, 0.2, 0.8, 0.1, 0.1, 0.6]
    assert split_scores(s2) == (GENDER_UNKNOWN, 0.8, AGE_UNKNOWN, 0.6)

    print("two_axis.py self-tests passed")


if __name__ == "__main__":
    _selftest()
