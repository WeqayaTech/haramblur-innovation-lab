#!/usr/bin/env python3
"""
HARAMBLUR VLM-cluster — field report (balance + verification).

Skips embeddings/clustering entirely. Works straight off the VLM's structured
fields to answer the questions that actually matter:

  1. Is the data balanced?      -> distribution per field + imbalance score
  2. What do we need more of?   -> sparse cells in field cross-tabs
  3. Is the description right?   -> a montage of crops per bucket, to eyeball

Run AFTER describe.py, on one or more run dirs:

    python field_report.py --in ./run1 ./run_full --out ./field_report

Open field_report/report.md. Each bucket links a montage so you can confirm
e.g. that crops labeled orientation=back really are back-facing.
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from cluster import flat, load_records, montage  # reuse tested helpers

# Fields worth auditing for balance. Edit freely.
DEFAULT_FIELDS = [
    "class", "apparent_age_band", "gender_read_confidence", "age_read_confidence",
    "facial_hair", "hair_length", "head_covering", "garment_type",
    "apparent_attire_region", "skin_tone_mst", "skin_tone_confidence",
    "clothing_coverage", "orientation", "visible_part", "crowd",
    "setting", "scene_type", "occlusion", "pose",
]

# Pairs whose sparse combinations = candidate data to collect.
# These target the known failure modes (beardless men, robed figures, etc.).
DEFAULT_CROSS = [
    ("class", "apparent_age_band"),
    ("class", "gender_read_confidence"),
    ("class", "facial_hair"),
    ("class", "garment_type"),
    ("class", "head_covering"),
    ("class", "apparent_attire_region"),
    ("class", "skin_tone_mst"),
    ("class", "crowd"),
]


def norm_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c)
    return h / math.log(len(counts))  # 1.0 = balanced across observed buckets


def save_montage(ids, crops_dirs, path, n, seed=0):
    ids = list(ids)
    random.Random(seed).shuffle(ids)          # representative, not first-N
    sheet = montage(ids[:n], crops_dirs)
    if sheet is not None:
        import cv2
        cv2.imwrite(str(path), sheet)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="VLM field balance + verification")
    ap.add_argument("--in", dest="indir", nargs="+", required=True)
    ap.add_argument("--out", default="./field_report")
    ap.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    ap.add_argument("--montage-samples", type=int, default=16)
    ap.add_argument("--min-cell", type=int, default=5,
                    help="cross-tab cells at/below this count are flagged sparse")
    ap.add_argument("--thin-frac", type=float, default=0.05,
                    help="single-field buckets below this fraction are flagged")
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.indir]
    crops_dirs = [d / "crops" for d in in_dirs]
    out = Path(args.out)
    mont_dir = out / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(in_dirs)
    total = len(recs)
    flats = [flat(r) for r in recs]

    lines = ["# HARAMBLUR — field balance & verification\n",
             f"- Objects: **{total}**  ·  dirs: {', '.join(map(str, in_dirs))}\n",
             "Each bucket links a montage — open it to confirm the VLM labels "
             "are correct before trusting the numbers.\n"]

    # ---- 1. balance summary ------------------------------------------------
    lines.append("## Balance summary\n")
    lines.append("| field | balance (norm. entropy) | buckets | thinnest bucket |")
    lines.append("|---|---|---|---|")
    field_counts = {}
    for field in args.fields:
        counts = Counter(f.get(field, "unknown") for f in flats)
        field_counts[field] = counts
        thin = min(counts.items(), key=lambda kv: kv[1])
        lines.append(f"| {field} | {norm_entropy(counts):.2f} | {len(counts)} | "
                     f"`{thin[0]}` ({100*thin[1]/total:.1f}%) |")
    lines.append("")

    # ---- 2. per-field distributions + montages -----------------------------
    lines.append("## Distributions (with verification montages)\n")
    for field in args.fields:
        counts = field_counts[field]
        lines.append(f"### {field}")
        # bucket -> object ids (for montages)
        ids_by_bucket = defaultdict(list)
        for r, f in zip(recs, flats):
            ids_by_bucket[f.get(field, "unknown")].append(r["id"])
        for val, c in counts.most_common():
            pct = 100 * c / total
            flag = " ⚠️ thin" if pct < args.thin_frac * 100 else ""
            mname = f"{field}__{val}.jpg".replace("/", "_").replace(" ", "_")
            ok = save_montage(ids_by_bucket[val], crops_dirs,
                              mont_dir / mname, args.montage_samples)
            link = f" · [montage](montages/{mname})" if ok else ""
            lines.append(f"- `{val}`: {c} ({pct:.1f}%){flag}{link}")
        lines.append("")

    # ---- 3. cross-tabs: sparse cells = what to collect ---------------------
    lines.append("## Cross-tabs — sparse combinations to collect\n")
    sparse = []
    for a, b in DEFAULT_CROSS:
        if a not in args.fields and a not in DEFAULT_FIELDS:
            continue
        a_vals = [v for v, _ in field_counts.get(a, Counter(
            f.get(a, "unknown") for f in flats)).most_common()]
        b_counter = Counter(f.get(b, "unknown") for f in flats)
        b_vals = [v for v, _ in b_counter.most_common()]
        grid = Counter((f.get(a, "unknown"), f.get(b, "unknown")) for f in flats)

        lines.append(f"### {a} × {b}\n")
        lines.append("| " + a + " \\ " + b + " | " + " | ".join(b_vals) + " |")
        lines.append("|" + "---|" * (len(b_vals) + 1))
        for av in a_vals:
            row = [f"**{av}**"]
            for bv in b_vals:
                c = grid.get((av, bv), 0)
                cell = f"{c}" if c > args.min_cell else f"⚠️{c}"
                row.append(cell)
                if c <= args.min_cell:
                    sparse.append((a, av, b, bv, c))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Ranked gaps (sparsest combinations)\n")
    for a, av, b, bv, c in sorted(sparse, key=lambda x: x[4])[:25]:
        lines.append(f"- **{av} × {bv}** ({a}×{b}): {c} objects")

    out.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[field_report] {total} objects -> {out/'report.md'}")
    print(f"[field_report] {len(sparse)} sparse cells flagged (<= {args.min_cell})")


if __name__ == "__main__":
    main()
