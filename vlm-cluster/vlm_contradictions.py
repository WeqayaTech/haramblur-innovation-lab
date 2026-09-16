#!/usr/bin/env python3
"""
HARAMBLUR — find VLM self-contradictions on gender calls.

The VLM's apparent_gender field can carry the same bias the production model
has: men in head coverings / flowing robes (e.g. ghutra, thobe) sometimes get
read as feminine. We don't need a new VLM pass to check this — every record
already captured facial_hair, head_covering, and garment_type INDEPENDENTLY
of apparent_gender in the same response. If those contradict the gender call,
that's a red flag the "confirmed" label_error / model_error for that record
may itself be wrong (VLM fooled the same way as the model).

Pure post-processing of EXISTING compare*.jsonl — no GPU, no re-describing.

    python vlm_contradictions.py --in ./all_eval/compare --out ./all_eval/compare/contradictions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from compare_child import load_records_all_shards
from cluster import flat, montage

# Strong masculine-presenting cues that contradict an apparent_gender="woman" call.
WOMAN_CONTRADICTORS = {
    "facial_hair": {"moustache", "short_beard", "full_beard"},
    "head_covering": {"ghutra_keffiyeh", "turban"},
    "garment_type": {"thobe_robe"},
}
# Strong feminine-presenting cues that contradict an apparent_gender="man" call.
MAN_CONTRADICTORS = {
    "head_covering": {"hijab"},
    "garment_type": {"abaya", "dress"},
}


def find_contradiction(o: dict) -> str | None:
    gender = o.get("apparent_gender")
    if gender == "woman":
        for field, bad_values in WOMAN_CONTRADICTORS.items():
            v = o.get(field)
            if v in bad_values:
                return f"apparent_gender=woman but {field}={v}"
    elif gender == "man":
        for field, bad_values in MAN_CONTRADICTORS.items():
            v = o.get(field)
            if v in bad_values:
                return f"apparent_gender=man but {field}={v}"
    return None


def main():
    ap = argparse.ArgumentParser(description="find VLM gender self-contradictions")
    ap.add_argument("--in", dest="indir", nargs="+", required=True,
                    help="compare_classes.py / compare_gender.py output dir(s)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--montage-samples", type=int, default=24)
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.indir]
    crops_dirs = [d.parent / "crops" if (d.parent / "crops").is_dir() else d / "crops"
                  for d in in_dirs]
    out = Path(args.out) if args.out else in_dirs[0] / "contradictions"
    mont_dir = out / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records_all_shards(in_dirs[0]) if len(in_dirs) == 1 else \
        [r for d in in_dirs for r in load_records_all_shards(d)]
    print(f"[contradict] {len(recs)} records loaded")

    flagged = []
    for r in recs:
        o = r.get("object", {})
        reason = find_contradiction(o)
        if reason:
            flagged.append((r, reason))

    print(f"[contradict] {len(flagged)} self-contradictory gender calls found "
          f"({100*len(flagged)/max(1,len(recs)):.1f}% of scanned)")

    # group by (label, model_pred, category, reason-field) for review
    groups = {}
    for r, reason in flagged:
        field = reason.split("but ")[1].split("=")[0]
        key = (r.get("label"), r.get("model_pred"), r.get("category"), field)
        groups.setdefault(key, []).append(r["id"])

    L = ["# VLM gender self-contradictions\n",
         f"- Records scanned: **{len(recs)}**",
         f"- Self-contradictory: **{len(flagged)}** "
         f"({100*len(flagged)/max(1,len(recs)):.1f}%)\n",
         "A contradiction = the VLM said `apparent_gender=woman` but also reported "
         "a strongly masculine cue (beard, ghutra/turban, thobe) in the SAME "
         "response — or the reverse (man + hijab/abaya/dress). These are exactly "
         "the cases where the VLM may share the model's own bias (e.g. reading "
         "robed/head-covered men as feminine), so any `label_error`/`model_error` "
         "verdict resting on them should be treated as unverified, not confirmed.\n",
         "| label | model_pred | category | contradicting field | count | montage |",
         "|---|---|---|---|---|---|"]
    for (label, pred, cat, field), ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        name = f"contradict_{label}_{pred}_{cat}_{field}".replace("/", "_")
        sheet = montage(ids[:args.montage_samples], crops_dirs,
                        title=f"{label}->{pred} {cat}: gender contradicts {field} (n={len(ids)})")
        link = "—"
        if sheet is not None:
            cv2.imwrite(str(mont_dir / f"{name}.jpg"), sheet)
            link = f"[crops](montages/{name}.jpg)"
        L.append(f"| {label} | {pred} | {cat} | {field} | {len(ids)} | {link} |")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(
        {"scanned": len(recs), "contradictory": len(flagged),
         "groups": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in groups.items()}},
        (out / "contradictions.json").open("w"), indent=2)

    print(f"[contradict] -> {out/'report.md'}")


if __name__ == "__main__":
    main()
