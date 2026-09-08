#!/usr/bin/env python3
"""
HARAMBLUR VLM-cluster — data-collection needs.

Turns the VLM descriptions into a plain-English shopping list: named, multi-
attribute slices (e.g. "dark-skin children", "men in Gulf/Arab attire"), each
with its count, share, a COLLECT/ok verdict, and a montage to verify.

    python collection_needs.py --in ./run2 --out ./run2/needs --min-count 30

Edit TARGET_SLICES below to match what you care about — each slice is just a
name + a predicate over the (flattened) description fields.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from cluster import flat, load_records, montage  # reuse tested helpers


# ---- helpers over a flattened record ------------------------------------

def mst(f):
    """Parse the Monk Skin Tone integer (1..10) from skin_tone_mst, or None."""
    m = re.search(r"\d+", str(f.get("skin_tone_mst", "")))
    n = int(m.group()) if m else None
    # ignore lighting-confounded readings
    if f.get("skin_tone_confidence") == "uncertain_lighting":
        return None
    return n if (n and 1 <= n <= 10) else None


def is_child(f):
    return (f.get("class") == "Child"
            or f.get("apparent_age_band") in {"infant", "toddler", "child", "preteen"})


def is_man(f):
    return f.get("class") == "Man"


def is_woman(f):
    return f.get("class") == "Woman"


def gulf_attire(f):
    return (f.get("garment_type") in {"thobe_robe", "abaya"}
            or f.get("head_covering") in {"ghutra_keffiyeh", "hijab"}
            or f.get("apparent_attire_region") == "gulf_arab")


# ---- the slices that matter (name, predicate) ---------------------------
# Skin-tone buckets ignore uncertain-lighting crops (mst() returns None there).
TARGET_SLICES = [
    ("Dark-skin children (MST 8-10)",       lambda f: is_child(f) and (mst(f) or 0) >= 8),
    ("Medium-skin children (MST 4-7)",      lambda f: is_child(f) and 4 <= (mst(f) or 0) <= 7),
    ("Light-skin children (MST 1-3)",       lambda f: is_child(f) and 1 <= (mst(f) or 0) <= 3),
    ("Dark-skin adults (MST 8-10)",         lambda f: not is_child(f) and (mst(f) or 0) >= 8),
    ("Infants / toddlers",                  lambda f: f.get("apparent_age_band") in {"infant", "toddler"}),
    ("Elderly people",                      lambda f: f.get("apparent_age_band") == "elderly"),
    ("Men in Gulf/Arab attire",             lambda f: is_man(f) and gulf_attire(f)),
    ("Women in Gulf/Arab attire",           lambda f: is_woman(f) and gulf_attire(f)),
    ("South Asian attire",                  lambda f: f.get("apparent_attire_region") == "south_asian"),
    ("African attire",                      lambda f: f.get("apparent_attire_region") == "african"),
    ("East Asian attire",                   lambda f: f.get("apparent_attire_region") == "east_asian"),
    ("Beardless men",                       lambda f: is_man(f) and f.get("facial_hair") in {"none", "stubble"}),
    ("Full-beard men",                      lambda f: is_man(f) and f.get("facial_hair") == "full_beard"),
    ("Gender-ambiguous people",             lambda f: f.get("gender_read_confidence") == "ambiguous"),
    ("Age-ambiguous people",                lambda f: f.get("age_read_confidence") == "ambiguous"),
    ("Back-facing people",                  lambda f: f.get("orientation") == "back"),
    ("Profile views",                       lambda f: f.get("orientation") == "profile"),
    ("Children in crowds",                  lambda f: is_child(f) and f.get("crowd") == "crowd"),
    ("People fully covered",                lambda f: f.get("clothing_coverage") == "fully_covered"),
    ("People in revealing/minimal dress",   lambda f: f.get("clothing_coverage") in {"revealing", "minimal"}),
]


def main():
    ap = argparse.ArgumentParser(description="VLM data-collection needs")
    ap.add_argument("--in", dest="indir", nargs="+", required=True)
    ap.add_argument("--out", default="./needs")
    ap.add_argument("--min-count", type=int, default=30,
                    help="slices below this many objects are flagged COLLECT")
    ap.add_argument("--montage-samples", type=int, default=16)
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.indir]
    crops_dirs = [d / "crops" for d in in_dirs]
    out = Path(args.out)
    mont_dir = out / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)

    recs = load_records(in_dirs)
    flats = [flat(r) for r in recs]
    total = len(recs)

    rows = []
    for name, pred in TARGET_SLICES:
        ids = [r["id"] for r, f in zip(recs, flats) if _safe(pred, f)]
        rows.append((name, ids))
    rows.sort(key=lambda r: len(r[1]))  # sparsest first = top priority

    lines = ["# HARAMBLUR — data-collection needs\n",
             f"- Objects analyzed: **{total}**  ·  target floor: **{args.min_count}** per slice",
             f"- Dirs: {', '.join(map(str, in_dirs))}\n",
             "Sparsest slices first. ⚠️ COLLECT = below the floor. Open each "
             "montage to confirm the slice is what the label says.\n",
             "| priority | slice | count | % | verdict | need |",
             "|---|---|---|---|---|---|"]
    for i, (name, ids) in enumerate(rows, 1):
        c = len(ids)
        pct = 100 * c / total if total else 0
        short = (i <= 0)
        if c < args.min_count:
            verdict = "⚠️ COLLECT"
            need = f"+{args.min_count - c}"
        else:
            verdict = "ok"
            need = "—"
        mname = name.lower().replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "") + ".jpg"
        ok = _montage(ids, crops_dirs, mont_dir / mname, args.montage_samples)
        label = f"[{name}](montages/{mname})" if ok else name
        lines.append(f"| {i} | {label} | {c} | {pct:.1f}% | {verdict} | {need} |")

    lines.append("\n## Top collection priorities\n")
    for name, ids in rows:
        if len(ids) < args.min_count:
            lines.append(f"- **{name}** — have {len(ids)}, want ≥{args.min_count} "
                         f"(collect ~{args.min_count - len(ids)} more)")

    out.joinpath("collection_needs.md").write_text("\n".join(lines), encoding="utf-8")
    flagged = sum(1 for _, ids in rows if len(ids) < args.min_count)
    print(f"[needs] {total} objects -> {out/'collection_needs.md'}")
    print(f"[needs] {flagged}/{len(rows)} slices below floor ({args.min_count})")


def _safe(pred, f):
    try:
        return bool(pred(f))
    except Exception:
        return False


def _montage(ids, crops_dirs, path, n, seed=0):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    sheet = montage(ids[:n], crops_dirs)
    if sheet is not None:
        import cv2
        cv2.imwrite(str(path), sheet)
        return True
    return False


if __name__ == "__main__":
    main()
