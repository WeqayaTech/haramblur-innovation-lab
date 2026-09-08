#!/usr/bin/env python3
"""
HARAMBLUR — Step 2 (gender axis): Label vs Model vs VLM for Woman/Man boxes.

Parallel to compare_child.py, but scoped to Woman/Man-labeled boxes (run
run_model_children.py with --classes Woman Man first) and asks the VLM for an
EXPLICIT, independent gender read (apparent_gender) — not just the age-based
inference used for the Child axis — plus the full feature description
(facial_hair, hair_length, garment_type, ...) so we can see WHY a mismatch
happened. This apparent_gender field is added only to THIS comparison pipeline
(describe.py's main schema is intentionally left unchanged).

    python compare_gender.py --in ./gender_eval \
        --yaml /workspace/open-images-v7/dataset.yaml \
        --filter misclassified --batch-size 16

Outputs (in --out, default <in>/compare):
  compare.jsonl  per box: model_pred, vlm_gender, vlm_age, category + fields
  summary.json   counts, accuracy, and the model-error vs label-error split
  montages/<category>.jpg  titled montage per category (to verify)
  report.md      headline table
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import Counter
from pathlib import Path

import cv2

import dataset_utils as du
from describe import MockDescriber, QwenDescriber, OBJECT_SCHEMA, fmt_dur
from describe_hardneg import HN_OBJECT_PROMPT
from label_errors import life_stage
from cluster import montage
from compare_child import load_done_all_shards, load_records_all_shards

# Extend the standard object schema with an explicit, independent gender read.
# (Only for this comparison pipeline — describe.py's OBJECT_SCHEMA is untouched.)
GENDER_SCHEMA = dict(OBJECT_SCHEMA)
GENDER_SCHEMA["apparent_gender"] = "woman|man|unclear"


def categorize(label, model_pred, vlm_gender, vlm_age_stage):
    """label/model_pred are 'Woman' or 'Man' (scope is Woman/Man boxes only).
    vlm_gender in {woman,man,unclear}; vlm_age_stage in {child,adult,ambiguous}
    (from label_errors.life_stage on apparent_age_band)."""
    if vlm_age_stage == "child":
        vlm_class = "Child"          # VLM thinks it's actually a child, not a
    elif vlm_gender == "woman":       # gender swap between the two adult classes
        vlm_class = "Woman"
    elif vlm_gender == "man":
        vlm_class = "Man"
    else:
        vlm_class = None

    if vlm_class is None:
        return "uncertain"
    if vlm_class == label:
        return "model_error"          # VLM sides with the label -> model wrong
    if vlm_class == model_pred:
        return "label_error"          # VLM sides with the model -> label wrong
    return "vlm_disagrees_both"        # e.g. VLM says Child while label/pred say adult


def main():
    ap = argparse.ArgumentParser(description="Woman/Man Label vs Model vs VLM compare")
    ap.add_argument("--in", dest="indir", required=True, help="step-1 out dir")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--filter", choices=["all", "misclassified", "high_conf_mismatch"],
                    default="misclassified")
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--montage-samples", type=int, default=20)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total parallel processes (one per GPU)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="this process's shard index, 0..num_shards-1")
    args = ap.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard-id must be in [0, --num-shards)")

    in_dir = Path(args.indir)
    crops_dir = in_dir / "crops"
    out = Path(args.out) if args.out else in_dir / "compare"
    mont_dir = out / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (out / "compare.jsonl" if args.num_shards == 1
             else out / f"compare.shard{args.shard_id}.jsonl")

    names = du.class_names(du.load_yaml(Path(args.yaml)))
    print(f"[gender-compare] classes={names}")
    if args.num_shards > 1:
        print(f"[gender-compare] shard {args.shard_id}/{args.num_shards} -> {jsonl.name}")

    results = [json.loads(l) for l in (in_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    # scope to Woman/Man labeled boxes only (run step 1 with --classes Woman Man)
    results = [r for r in results if r.get("gt_class") in ("Woman", "Man")]
    total_population = len(results)
    if total_population == 0:
        raise SystemExit("No Woman/Man boxes in results.jsonl — re-run "
                         "run_model_children.py with --classes Woman Man")

    if args.filter == "all":
        scope = results
    elif args.filter == "misclassified":
        scope = [r for r in results if r.get("status") == "misclassified"]
    else:  # high_conf_mismatch
        scope = [r for r in results if r.get("status") == "misclassified"
                 and r.get("pred_conf", 0) >= args.min_conf]
    print(f"[gender-compare] filter={args.filter} (min_conf={args.min_conf}): "
          f"{len(scope)} of {total_population} Woman/Man boxes selected for VLM")

    if args.num_shards > 1:
        scope = [r for r in scope
                 if zlib.crc32(r["id"].encode()) % args.num_shards == args.shard_id]
        print(f"[gender-compare] this shard: {len(scope)} boxes")

    done = load_done_all_shards(out)
    pending = [r for r in scope if r["id"] not in done
               and (crops_dir / f"{r['id']}.jpg").exists()]
    if done:
        print(f"[gender-compare] resuming — {len(done)} done")
    print(f"[gender-compare] {len(pending)} crops to judge (of {len(scope)} in scope)")

    if not pending:
        print("[gender-compare] nothing new for this shard; re-aggregating existing results")
    else:
        if args.mock:
            engine = MockDescriber()
            print("[gender-compare] MOCK engine")
        else:
            engine = QwenDescriber(model_id=args.model, device=args.device,
                                   max_pixels=args.max_pixels,
                                   max_new_tokens=args.max_new_tokens)
            print(f"[gender-compare] Qwen2.5-VL on {engine.device}")

        n = 0
        t0 = time.time()
        f = jsonl.open("a")
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start:start + args.chunk]
            crops, metas = [], []
            for r in chunk:
                im = cv2.imread(str(crops_dir / f"{r['id']}.jpg"))
                if im is None:
                    continue
                crops.append(im)
                metas.append(r)
            if not crops:
                continue
            objs = engine.describe_batch([(c, "person") for c in crops],
                                         args.batch_size,
                                         prompt_template=HN_OBJECT_PROMPT,
                                         schema=GENDER_SCHEMA)
            for r, obj in zip(metas, objs):
                gender = obj.get("apparent_gender", "unclear")
                age = obj.get("apparent_age_band", "unknown")
                age_stage = life_stage(age)
                label = r.get("gt_class")
                model_pred = r.get("pred_class")
                cat = categorize(label, model_pred, gender, age_stage)
                f.write(json.dumps({
                    "id": r["id"], "image": r["image"],
                    "label": label, "model_pred": model_pred,
                    "model_status": r.get("status"),
                    "vlm_gender": gender, "vlm_age": age,
                    "vlm_age_stage": age_stage,
                    "category": cat, "object": obj,
                }) + "\n")
                n += 1
            f.flush()
            el = time.time() - t0
            rate = n / el if el else 0
            print(f"[gender-compare] {n}/{len(pending)} · {rate:.2f}/s · "
                  f"ETA {fmt_dur((len(pending)-n)/rate if rate else 0)}", flush=True)
        f.close()

    # ---- aggregate ---------------------------------------------------------
    detected = [r for r in results if r.get("status") != "missed"]
    model_correct = sum(1 for r in detected if r.get("pred_class") == r.get("gt_class"))

    recs = load_records_all_shards(out)
    cats = Counter(r["category"] for r in recs)
    vlm_scanned = len(recs)

    summary = {
        "total_woman_man_boxes": total_population,
        "filter": args.filter, "min_conf": args.min_conf,
        "vlm_scanned": vlm_scanned,
        "categories_within_vlm_scanned": dict(cats),
        "label_vs_model_accuracy_full_population": round(
            model_correct / max(1, len(detected)), 4),
        "confident_errors_found": {
            "real_model_errors": cats.get("model_error", 0),
            "label_errors_gender_swapped": cats.get("label_error", 0),
            "vlm_disagrees_with_both": cats.get("vlm_disagrees_both", 0),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    by_cat = {}
    for r in recs:
        by_cat.setdefault(r["category"], []).append(r["id"])
    meaning = {
        "model_error": "label correct, model swapped the gender (model failure)",
        "label_error": "model correct, the LABEL has the wrong gender (data error)",
        "vlm_disagrees_both": "VLM disagrees with both (e.g. looks like a child)",
        "uncertain": "VLM gender unclear",
    }
    L = ["# HARAMBLUR — Woman vs Man Label vs Model vs VLM\n",
         f"- Total labeled Woman/Man boxes: **{total_population}**",
         f"- Filter: **{args.filter}** → VLM scanned **{vlm_scanned}** boxes",
         f"- Label-vs-Model accuracy (full population, detected only): "
         f"**{summary['label_vs_model_accuracy_full_population']*100:.1f}%**\n",
         "| category | count | % | meaning |", "|---|---|---|---|"]
    for cat, c in cats.most_common():
        L.append(f"| {cat} | {c} | {100*c/max(1,vlm_scanned):.1f}% | "
                 f"{meaning.get(cat, '')} |")
    L.append("\n## Montages (verify)\n")
    for cat, ids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        sheet = montage(ids[:args.montage_samples], [crops_dir],
                        title=f"{cat} (n={len(ids)})")
        if sheet is not None:
            cv2.imwrite(str(mont_dir / f"{cat}.jpg"), sheet)
            L.append(f"### {cat} ({len(ids)})")
            L.append(f"![{cat}](montages/{cat}.jpg)\n")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[gender-compare] categories: {dict(cats)}")
    print(f"[gender-compare] real model errors: {cats.get('model_error',0)} · "
          f"label errors (gender swapped): {cats.get('label_error',0)}")
    print(f"[gender-compare] -> {out/'report.md'} · {out/'summary.json'}")


if __name__ == "__main__":
    main()
