#!/usr/bin/env python3
"""
HARAMBLUR — confirmed LABEL ERRORS from the hard-negative descriptions.

A label error is a case where the model prediction AND the VLM's independent read
agree against the dataset label. We can confirm these for the AGE-arbitrable
confusions using the VLM's apparent_age_band:

  * label is Man/Woman, model predicted Child, VLM age is a child  -> a CHILD
    mislabeled as an adult class.
  * label is Child, model predicted Man/Woman, VLM age is an adult -> an ADULT
    mislabeled as Child.

Gender-only confusions (Man<->Woman) are NOT confirmable here (we don't store an
independent VLM gender verdict), so they go to a separate "needs_review" bucket.

    python label_errors.py --in ./hardneg --out ./hardneg/label_errors

Outputs: report.md (groups + montages), montages/<category>.jpg, and
label_errors.csv (image paths to fix), needs_review.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from cluster import load_records, montage

# Puberty-based boundary: Child = preteen and younger; Man/Woman = teenager and
# up (post-puberty). This is the project definition of "Child" — gender-agnostic,
# age-only.
CHILD_STAGES = {"infant", "toddler", "child", "preteen"}
ADULT_STAGES = {"teenager", "young_adult", "adult", "middle_aged", "elderly", "senior"}


def life_stage(age: str) -> str:
    if age in CHILD_STAGES:
        return "child"
    if age in ADULT_STAGES:
        return "adult"
    return "ambiguous"


def classify(rec: dict):
    """Return (category, confirmed) for a hard-neg record."""
    o = rec.get("object", {})
    label = o.get("true_label", "?")
    pred = o.get("model_pred", "?")
    stage = life_stage(o.get("apparent_age_band", "unknown"))

    if label == pred:
        return None, False  # not a disagreement

    # if the focus box was ambiguous (multiple same-class boxes), don't confirm
    if o.get("focus_ambiguous") == "yes":
        return f"needs_review__ambiguous_focus__{label}_as_{pred}", False

    # age-arbitrable label errors
    if pred == "Child" and label in ("Man", "Woman") and stage == "child":
        return f"labeled_{label}__really_Child", True
    if pred in ("Man", "Woman") and label == "Child" and stage == "adult":
        return f"labeled_Child__really_{pred}", True

    # everything else (gender swaps, or age too ambiguous to confirm)
    return f"needs_review__{label}_as_{pred}", False


def main():
    ap = argparse.ArgumentParser(description="confirmed label errors")
    ap.add_argument("--in", dest="indir", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--montage-samples", type=int, default=20)
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.indir]
    crops_dirs = [d / "crops" for d in in_dirs]
    out = Path(args.out) if args.out else in_dirs[0] / "label_errors"
    mont_dir = out / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(in_dirs)
    groups = defaultdict(list)
    confirmed_flag = {}
    for r in recs:
        cat, confirmed = classify(r)
        if cat is None:
            continue
        groups[cat].append(r)
        confirmed_flag[cat] = confirmed

    confirmed = {c: g for c, g in groups.items() if confirmed_flag[c]}
    review = {c: g for c, g in groups.items() if not confirmed_flag[c]}

    def montage_for(cat, members):
        ids = [m["id"] for m in members]
        random.Random(0).shuffle(ids)
        sheet = montage(ids[:args.montage_samples], crops_dirs)
        if sheet is not None:
            import cv2
            name = f"{cat}.jpg"
            cv2.imwrite(str(mont_dir / name), sheet)
            return f"montages/{name}"
        return None

    def write_csv(path, cats):
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["category", "id", "image", "label", "model_pred",
                        "vlm_age", "confidence"])
            for cat, members in cats.items():
                for m in members:
                    o = m["object"]
                    w.writerow([cat, m["id"], m["image"], o.get("true_label"),
                                o.get("model_pred"), o.get("apparent_age_band"),
                                round(m.get("confidence", 0), 3)])

    write_csv(out / "label_errors.csv", confirmed)
    write_csv(out / "needs_review.csv", review)

    total = len(recs)
    L = ["# HARAMBLUR — confirmed label errors\n",
         f"- Records analyzed: **{total}**",
         f"- Confirmed label errors: **{sum(len(g) for g in confirmed.values())}**",
         f"- Needs review (gender swaps / ambiguous age): "
         f"**{sum(len(g) for g in review.values())}**\n",
         "A *confirmed* label error = model prediction AND VLM age agree against "
         "the dataset label. Open each montage to verify.\n",
         "## Confirmed label errors (fix these labels)\n"]
    for cat, members in sorted(confirmed.items(), key=lambda kv: -len(kv[1])):
        link = montage_for(cat, members)
        ages = ", ".join(f"{a}:{n}" for a, n in
                         Counter(m["object"].get("apparent_age_band") for m in members).most_common(4))
        L.append(f"### {cat} — {len(members)} cases")
        if link:
            L.append(f"![{cat}]({link})")
        L.append(f"- VLM ages: {ages}")
        L.append(f"- list: see `label_errors.csv` (filter category={cat})\n")

    L.append("## Needs review — not age-confirmable (mostly gender swaps)\n")
    for cat, members in sorted(review.items(), key=lambda kv: -len(kv[1])):
        link = montage_for(cat, members)
        L.append(f"### {cat} — {len(members)} cases")
        if link:
            L.append(f"![{cat}]({link})")
        L.append("")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[label_errors] {total} records")
    print(f"[label_errors] confirmed: {sum(len(g) for g in confirmed.values())} "
          f"in {len(confirmed)} categories")
    for cat, members in sorted(confirmed.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(members):4d}  {cat}")
    print(f"[label_errors] report -> {out/'report.md'}  ·  csv -> {out/'label_errors.csv'}")


if __name__ == "__main__":
    main()
