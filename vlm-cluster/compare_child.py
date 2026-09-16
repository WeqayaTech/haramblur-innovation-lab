#!/usr/bin/env python3
"""
HARAMBLUR — Step 2: Label vs Model vs VLM on child Val crops.

Takes Step 1 output (run_model_children.py -> child_eval/results.jsonl + crops/),
asks the VLM for an independent read of each child crop, then does a 3-way
comparison to separate MODEL errors from LABEL errors:

  Label  = Child (always, by selection)
  Model  = pred_class from results.jsonl
  VLM    = apparent_age_band -> child / adult / ambiguous

    python compare_child.py --in ./child_eval \
        --yaml /workspace/open-images-v7/dataset.yaml --batch-size 16

Outputs (in --out, default <in>/compare):
  compare.jsonl  per child box: model_pred, vlm_age, vlm_stage, category + fields
  summary.json   counts, label-vs-model accuracy, label-vs-VLM agreement,
                 and the model-error vs label-error split
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
from describe import (MockDescriber, QwenDescriber, OBJECT_SCHEMA, schema_str,
                      load_done, fmt_dur)
from describe_hardneg import HN_OBJECT_PROMPT
from label_errors import life_stage          # child/adult/ambiguous mapping
from cluster import montage

def load_done_all_shards(out_dir: Path) -> set:
    """Done ids across ALL compare*.jsonl files (single-run or sharded), so a
    shard resuming/restarting never redoes work another shard already did."""
    done = set()
    for jf in sorted(out_dir.glob("compare*.jsonl")):
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_records_all_shards(out_dir: Path):
    """Merge + dedupe records across all compare*.jsonl files for aggregation."""
    seen, recs = set(), []
    for jf in sorted(out_dir.glob("compare*.jsonl")):
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            recs.append(r)
    return recs


AGE_ONLY_PROMPT = (
    "Look at the person in this crop. Reply with a single JSON object: "
    '{"apparent_age_band": (infant|toddler|child|preteen|teenager|young_adult|'
    'adult|middle_aged|elderly|senior|unknown), "age_read_confidence": '
    '(clear|ambiguous)}. Judge from visible cues only. Return only the JSON.'
)


def categorize(model_pred, status, vlm_stage):
    if status == "missed":
        return "model_missed"
    if vlm_stage == "ambiguous":
        return "uncertain"
    if model_pred == "Child":
        return "correct" if vlm_stage == "child" else "uncertain"
    # model said Man/Woman (an adult class)
    if vlm_stage == "adult":
        return "label_error_adult"   # adult mislabeled Child; model + VLM agree
    if vlm_stage == "child":
        return "model_error"         # really a child; label + VLM agree, model wrong
    return "uncertain"


def main():
    ap = argparse.ArgumentParser(description="child Label vs Model vs VLM compare")
    ap.add_argument("--in", dest="indir", required=True, help="step-1 out dir")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--filter", choices=["all", "misclassified", "high_conf_mismatch"],
                    default="high_conf_mismatch",
                    help="which child boxes to send to the VLM: all crops; any "
                         "misclassified (model said not-Child); or only "
                         "misclassified ABOVE --min-conf (default; the cheap, "
                         "targeted scan for label errors)")
    ap.add_argument("--min-conf", type=float, default=0.8,
                    help="confidence floor for --filter high_conf_mismatch")
    ap.add_argument("--age-only", action="store_true",
                    help="shorter age-only VLM prompt (~2x faster)")
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
    # separate output file per shard so concurrent writers (one per GPU) never clash
    jsonl = (out / "compare.jsonl" if args.num_shards == 1
             else out / f"compare.shard{args.shard_id}.jsonl")

    names = du.class_names(du.load_yaml(Path(args.yaml)))
    print(f"[compare] classes={names}")
    if args.num_shards > 1:
        print(f"[compare] shard {args.shard_id}/{args.num_shards} -> {jsonl.name}")

    results = [json.loads(l) for l in (in_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    total_population = len(results)

    if args.filter == "all":
        scope = results
    elif args.filter == "misclassified":
        scope = [r for r in results if r.get("status") == "misclassified"]
    else:  # high_conf_mismatch — the cheap, targeted scan for label errors
        scope = [r for r in results if r.get("status") == "misclassified"
                 and r.get("pred_conf", 0) >= args.min_conf]
    print(f"[compare] filter={args.filter} (min_conf={args.min_conf}): "
          f"{len(scope)} of {total_population} child boxes selected for VLM")

    # data-parallel sharding: each process owns a disjoint set of ids (stable
    # crc32 hash — not Python's per-process-randomized hash())
    if args.num_shards > 1:
        scope = [r for r in scope
                 if zlib.crc32(r["id"].encode()) % args.num_shards == args.shard_id]
        print(f"[compare] this shard: {len(scope)} boxes")

    done = load_done_all_shards(out)
    pending = [r for r in scope if r["id"] not in done
               and (crops_dir / f"{r['id']}.jpg").exists()]
    if done:
        print(f"[compare] resuming — {len(done)} done")
    print(f"[compare] {len(pending)} child crops to judge (of {len(scope)} in scope)")
    if not pending:
        # Nothing NEW for this process, but other shards may have written
        # data since the last aggregation — fall through to re-aggregate
        # (report.md/summary.json/montages) instead of returning early.
        print("[compare] nothing new for this shard; re-aggregating existing results")
    else:
        if args.mock:
            engine = MockDescriber()
            print("[compare] MOCK engine")
        else:
            engine = QwenDescriber(model_id=args.model, device=args.device,
                                   max_pixels=args.max_pixels,
                                   max_new_tokens=args.max_new_tokens)
            print(f"[compare] Qwen2.5-VL on {engine.device}")
        prompt = AGE_ONLY_PROMPT if args.age_only else HN_OBJECT_PROMPT

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
                                         args.batch_size, prompt_template=prompt)
            for r, obj in zip(metas, objs):
                age = obj.get("apparent_age_band", "unknown")
                stage = life_stage(age)
                model_pred = r.get("pred_class")
                cat = categorize(model_pred, r.get("status"), stage)
                f.write(json.dumps({
                    "id": r["id"], "image": r["image"],
                    "label": r.get("gt_class", "Child"),
                    "model_pred": model_pred, "model_status": r.get("status"),
                    "vlm_age": age, "vlm_stage": stage,
                    "category": cat, "object": obj,
                }) + "\n")
                n += 1
            f.flush()
            el = time.time() - t0
            rate = n / el if el else 0
            print(f"[compare] {n}/{len(pending)} · {rate:.2f}/s · "
                  f"ETA {fmt_dur((len(pending)-n)/rate if rate else 0)}", flush=True)
        f.close()

    # ---- aggregate ---------------------------------------------------------
    # model-only stats (no VLM needed) come from the FULL labeled-Child
    # population, not just the filtered/VLM-scanned subset.
    detected = [r for r in results if r.get("status") != "missed"]
    model_correct = sum(1 for r in detected if r.get("pred_class") == "Child")

    # VLM-dependent stats only cover what was actually sent to the VLM
    # (compare.jsonl, accumulated across resumed runs/filters).
    # aggregate across ALL shard files on disk (works whether this run was
    # single-process, or one of several concurrent GPU shards)
    recs = load_records_all_shards(out)
    cats = Counter(r["category"] for r in recs)
    vlm_scanned = len(recs)
    vlm_child = sum(1 for r in recs if r["vlm_stage"] == "child")
    real_model_err = cats.get("model_error", 0)
    label_err = cats.get("label_error_adult", 0)

    summary = {
        "total_child_boxes": total_population,
        "filter": args.filter, "min_conf": args.min_conf,
        "vlm_scanned": vlm_scanned,
        "categories_within_vlm_scanned": dict(cats),
        "label_vs_model_accuracy_full_population": round(
            model_correct / max(1, len(detected)), 4),
        "label_vs_vlm_agreement_within_scanned": round(
            vlm_child / max(1, vlm_scanned), 4),
        "confident_errors_found": {
            "real_model_errors": real_model_err,
            "label_errors_adult_mislabeled_child": label_err,
            "note": f"counted within the {vlm_scanned} boxes the VLM actually "
                    f"reviewed (filter={args.filter}); raise --filter to 'all' "
                    f"or 'misclassified' to widen scope",
        },
    }
    total = vlm_scanned  # used below for montage report %s
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- montages per category + report ----------------------------------
    by_cat = {}
    for r in recs:
        by_cat.setdefault(r["category"], []).append(r["id"])
    L = ["# HARAMBLUR — child Label vs Model vs VLM\n",
         f"- Total labeled-Child boxes (Step 1): **{total_population}**",
         f"- Filter: **{args.filter}** (min_conf={args.min_conf}) "
         f"→ VLM scanned **{vlm_scanned}** boxes",
         f"- Label-vs-Model accuracy (full population, detected only): "
         f"**{summary['label_vs_model_accuracy_full_population']*100:.1f}%**",
         f"- Label-vs-VLM agreement (within scanned subset): "
         f"**{summary['label_vs_vlm_agreement_within_scanned']*100:.1f}%**\n",
         "| category | count | % | meaning |",
         "|---|---|---|---|"]
    meaning = {
        "correct": "model=Child, VLM=child (all agree)",
        "label_error_adult": "adult mislabeled Child (model+VLM say adult)",
        "model_error": "really a child, model said adult (model failure)",
        "model_missed": "model didn't detect the child",
        "uncertain": "VLM age ambiguous/teen/unknown",
    }
    for cat, c in cats.most_common():
        L.append(f"| {cat} | {c} | {100*c/total:.1f}% | {meaning.get(cat,'')} |")
    L.append("\n## Montages (verify)\n")
    for cat, ids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        title = f"{cat}  (n={len(ids)})"
        sheet = montage(ids[:args.montage_samples], [crops_dir], title=title)
        if sheet is not None:
            cv2.imwrite(str(mont_dir / f"{cat}.jpg"), sheet)
            L.append(f"### {cat} ({len(ids)})")
            L.append(f"![{cat}](montages/{cat}.jpg)\n")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[compare] categories: {dict(cats)}")
    print(f"[compare] real model errors: {real_model_err} · "
          f"label errors (adult→child): {label_err}")
    print(f"[compare] -> {out/'report.md'} · {out/'summary.json'}")


if __name__ == "__main__":
    main()
