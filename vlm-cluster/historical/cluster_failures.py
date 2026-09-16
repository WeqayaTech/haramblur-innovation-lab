#!/usr/bin/env python3
"""
VLM-Failure-Dimensions — Object Layer Only (Pure Python Version)
No pandas, no mlxtend, no dependencies. Completely standalone.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
from itertools import combinations

# Strict focus on Object-level features
OBJECT_FEATURE_FIELDS = [
    "apparent_age_band", "apparent_attire_region", "facial_hair", "hair_length",
    "head_covering", "garment_type", "clothing_coverage", "build",
    "orientation", "visible_part", "skin_tone_mst", "pose", "occlusion"
]

def load_failure_records(in_dirs):
    files = []
    for d in in_dirs:
        files += sorted(Path(d).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found in {in_dirs}")
        
    recs, seen = [], set()
    for jf in files:
        for line in jf.read_text().splitlines():
            if not line.strip(): continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") in seen: continue
            seen.add(r["id"])
            recs.append(r)
    print(f"[Core] Loaded {len(recs)} high-confidence failure images.")
    return recs

def extract_multidim_insights(recs, min_percent=5.0):
    """
    Finds combinations of features (dimensions of 2 and 3) that appear 
    frequently together using pure Python dictionaries.
    """
    total_recs = len(recs)
    if total_recs == 0:
        return []

    combination_counts = Counter()

    for r in recs:
        obj = r.get("object", {})
        # Build clean string representations of features present in this item
        active_features = []
        for field in OBJECT_FEATURE_FIELDS:
            val = obj.get(field, "unknown")
            if val not in ("unknown", "none", ""):
                active_features.append(f"{field}_{val}")
        
        # Sort to ensure order doesn't create duplicate keys
        active_features.sort()
        
        # Extract pairs (2D) and triplets (3D) of features traveling together
        for size in (2, 3):
            for combo in combinations(active_features, size):
                combination_counts[combo] += 1

    # Filter by minimum threshold prevalence
    min_count = (min_percent / 100.0) * total_recs
    
    valid_clusters = []
    for combo, count in combination_counts.items():
        if count >= min_count:
            valid_clusters.append({
                "itemsets": combo,
                "percentage": (count / total_recs) * 100
            })
            
    # Sort highest prevalence first
    valid_clusters.sort(key=lambda x: x["percentage"], reverse=True)
    return valid_clusters

def write_markdown_report(out_dir, clusters, total_count):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# High-Confidence Model Failure Insights (Object Dimensions)",
        f"Analyzed **{total_count}** images where the model confidently (>0.9) failed.",
        "Below are the multi-dimensional **object attribute clusters** driving these errors, ranked by prevalence.\n",
        "## Top Multi-Dimensional Failure Drivers",
        "| Prevalence in Errors (%) | Feature Combination (The Blind-Spot Cluster) |",
        "|---|---|",
    ]
    
    for row in clusters[:25]:  # Top 25 patterns
        features = " + ".join([f"`{item}`" for item in row["itemsets"]])
        lines.append(f"| {row['percentage']:.1f}% | {features} |")
        
    lines.extend([
        "\n### How to read this:",
        "- If a row shows **42.5%** for `orientation_profile` + `occlusion_partial`, it means nearly half of your high-confidence errors are happening *specifically* on partially occluded profile shots.",
        "### Next Steps:",
        "1. Isolate images matching these exact combinations.",
        "2. Balance your training dataset by adding high-quality, correctly labeled samples that hit these exact multi-attribute intersections."
    ])
    
    (out_dir / "failure_insights.md").write_text("\n".join(lines))
    print(f"[Success] Multi-dimensional analysis compiled at: {out_dir / 'failure_insights.md'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, nargs="+")
    ap.add_argument("--out", default="./failure_analysis")
    ap.add_argument("--min-support-pct", type=float, default=5.0, help="Drop combinations rarer than this % of total errors")
    args = ap.parse_args()
    
    in_dirs = [Path(d) for d in args.indir]
    out_dir = Path(args.out)
    
    recs = load_failure_records(in_dirs)
    if not recs:
        return
        
    clusters = extract_multidim_insights(recs, min_percent=args.min_support_pct)
    
    if clusters:
        write_markdown_report(out_dir, clusters, len(recs))
    else:
        print("[!] No repeating multi-dimensional feature patterns found. Try lowering `--min-support-pct`.")

if __name__ == "__main__":
    main()