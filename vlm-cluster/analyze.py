#!/usr/bin/env python3
"""
HARAMBLUR VLM-cluster — pairwise gap analysis.

The simple, general first read: count every value, and every pair of values,
then rank the sparsest combinations. Small cells = candidate gaps ("collect
more of this"). No hardcoded slices.

    CUDA_VISIBLE_DEVICES="" python analyze.py --in ./run2 --out ./run2/analysis \
        --thin 30 --top-montages 40

Outputs analysis.md (single-field distributions -> pairwise cross-tabs ->
ranked sparsest combinations with montage links), charts/, montages/, analysis.json.

Deeper analysis (3+ way mining, ad-hoc query, lift/surprise, curated slices) is
intentionally out of scope here — this gives initial direction first.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from cluster import flat, load_records, montage
from field_report import norm_entropy


def save_bar_chart(counts: Counter, order, title, path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    vals = [counts.get(b, 0) for b in order]
    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.6), 4))
    ax.bar([str(b) for b in order], vals, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("objects")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True

# Free-text fields never make good buckets — exclude from counting.
FREE_TEXT = {"clothing", "gender_cues", "activity", "distinctive_features",
             "short_caption"}
MAX_CARDINALITY = 25  # fields with more distinct values are treated as free-text


def mst_band(v) -> str:
    """Group Monk Skin Tone 1-10 into readable bands; honor lighting flag."""
    m = re.search(r"\d+", str(v))
    if not m:
        return "unknown"
    n = int(m.group())
    if n <= 3:
        return "light(1-3)"
    if n <= 6:
        return "medium(4-6)"
    if n == 7:
        return "tan(7)"
    if n <= 10:
        return "dark(8-10)"
    return "unknown"


def field_value(f: dict, field: str) -> str:
    v = f.get(field, "unknown")
    if field == "skin_tone_mst":
        # drop lighting-confounded readings so they don't fake a tone
        if f.get("skin_tone_confidence") == "uncertain_lighting":
            return "uncertain_lighting"
        return mst_band(v)
    return str(v) if v not in (None, "") else "unknown"


def detect_fields(flats) -> list:
    """Auto-pick categorical fields: present, not free-text, low cardinality."""
    keys = []
    seen = set()
    for f in flats:
        for k in f:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    fields = []
    for k in keys:
        if k in FREE_TEXT:
            continue
        vals = {field_value(f, k) for f in flats}
        if len(vals) <= MAX_CARDINALITY:
            fields.append(k)
    return fields


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def main():
    ap = argparse.ArgumentParser(description="pairwise gap analysis")
    ap.add_argument("--in", dest="indir", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thin", type=int, default=30,
                    help="cells at/below this count are flagged as gaps")
    ap.add_argument("--top-montages", type=int, default=40,
                    help="generate montages for the N sparsest combinations")
    ap.add_argument("--montage-samples", type=int, default=16)
    ap.add_argument("--fields", nargs="+", default=None,
                    help="override auto-detected fields")
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.indir]
    crops_dirs = [d / "crops" for d in in_dirs]
    out = Path(args.out) if args.out else in_dirs[0] / "analysis"
    charts_dir = out / "charts"
    mont_dir = out / "montages"
    charts_dir.mkdir(parents=True, exist_ok=True)
    mont_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(in_dirs)
    flats = [flat(r) for r in recs]
    total = len(recs)
    fields = args.fields or detect_fields(flats)
    print(f"[analyze] {total} objects · {len(fields)} fields: {fields}")

    # ---- single-field distributions ---------------------------------------
    single = {}
    for fld in fields:
        single[fld] = Counter(field_value(f, fld) for f in flats)

    # ---- pairwise cross-tabs + flattened cell list ------------------------
    pair_tables = {}   # (a,b) -> Counter[(av,bv)]
    cells = []         # (a, av, b, bv, count) for every occurring pair cell
    for a, b in combinations(fields, 2):
        grid = Counter((field_value(f, a), field_value(f, b)) for f in flats)
        pair_tables[(a, b)] = grid
        for (av, bv), c in grid.items():
            cells.append((a, av, b, bv, c))
    cells.sort(key=lambda x: x[4])  # sparsest first

    # ---- montages for the sparsest combinations ---------------------------
    ids_for = {}  # (a,av,b,bv) -> list of object ids
    for (a, av, b, bv, c) in cells[:args.top_montages]:
        ids = [r["id"] for r, f in zip(recs, flats)
               if field_value(f, a) == av and field_value(f, b) == bv]
        ids_for[(a, av, b, bv)] = ids
    montage_link = {}
    for key, ids in ids_for.items():
        a, av, b, bv = key
        name = safe_name(f"{a}={av}__{b}={bv}") + ".jpg"
        sheet = montage(ids[:args.montage_samples], crops_dirs)
        if sheet is not None:
            import cv2
            cv2.imwrite(str(mont_dir / name), sheet)
            montage_link[key] = f"montages/{name}"

    # ---- write report -----------------------------------------------------
    L = [f"# HARAMBLUR — pairwise gap analysis\n",
         f"- Objects: **{total}**  ·  fields analyzed: **{len(fields)}**",
         f"- Gap threshold: cells with **≤ {args.thin}** objects flagged ⚠️",
         f"- Dirs: {', '.join(map(str, in_dirs))}\n"]

    # 1. single-field distributions
    L.append("## 1. Single-field distributions\n")
    for fld in fields:
        counts = single[fld]
        save_bar_chart(counts, [v for v, _ in counts.most_common()], fld,
                       charts_dir / f"{safe_name(fld)}.png")
        L.append(f"### {fld}  ·  balance {norm_entropy(counts):.2f}")
        L.append(f"![{fld}](charts/{safe_name(fld)}.png)")
        for v, c in counts.most_common():
            flag = " ⚠️" if c <= args.thin else ""
            L.append(f"- `{v}`: {c} ({100*c/total:.1f}%){flag}")
        L.append("")

    # 2. ranked sparsest combinations (the actionable list)
    L.append("## 2. Sparsest combinations — collect more of these\n")
    L.append("| # | feature A | feature B | count | % | montage |")
    L.append("|---|---|---|---|---|---|")
    for i, (a, av, b, bv, c) in enumerate(cells[:max(args.top_montages, 60)], 1):
        link = f"[crops]({montage_link[(a, av, b, bv)]})" if (a, av, b, bv) in montage_link else "—"
        L.append(f"| {i} | {a}=`{av}` | {b}=`{bv}` | {c} | {100*c/total:.1f}% | {link} |")
    L.append("")

    # 3. pairwise cross-tab matrices
    L.append("## 3. Pairwise cross-tabs\n")
    for (a, b), grid in pair_tables.items():
        a_vals = [v for v, _ in single[a].most_common()]
        b_vals = [v for v, _ in single[b].most_common()]
        L.append(f"### {a} × {b}\n")
        L.append("| " + f"{a} \\ {b}" + " | " + " | ".join(b_vals) + " |")
        L.append("|" + "---|" * (len(b_vals) + 1))
        for av in a_vals:
            row = [f"**{av}**"]
            for bv in b_vals:
                c = grid.get((av, bv), 0)
                row.append(f"⚠️{c}" if c <= args.thin else str(c))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    out.joinpath("analysis.md").write_text("\n".join(L), encoding="utf-8")

    # machine-readable
    json.dump(
        {"total": total, "fields": fields, "thin": args.thin,
         "single": {k: dict(v) for k, v in single.items()},
         "pairs": {f"{a}|{b}": {f"{av}|{bv}": c for (av, bv), c in g.items()}
                   for (a, b), g in pair_tables.items()}},
        out.joinpath("analysis.json").open("w"), indent=2)

    n_gaps = sum(1 for c in cells if c[4] <= args.thin)
    print(f"[analyze] report -> {out/'analysis.md'}")
    print(f"[analyze] {n_gaps} combinations at/below thin={args.thin}")


if __name__ == "__main__":
    main()
