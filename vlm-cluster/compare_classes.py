#!/usr/bin/env python3
"""
HARAMBLUR — unified Woman/Man/Child Label vs Model vs VLM comparison.

Step 1 (once, covers all 3 classes in one pass):
    python run_model_children.py --classes Woman Man Child --n 0 --out ./all_eval

Step 2 (this script): builds the full Label x Model confusion matrix (free,
no VLM), then VLM-verifies every mismatch direction (Woman<->Man<->Child) to
separate MODEL errors from LABEL errors. Child is determined purely by age
(gender-agnostic) using the project's puberty-based boundary: preteen and
younger = Child; teenager and up = Man/Woman (see label_errors.life_stage).

    python compare_classes.py --in ./all_eval \
        --yaml /workspace/open-images-v7/dataset.yaml --filter misclassified

Outputs (in --out, default <in>/compare):
  compare.jsonl   per box: vlm_gender, vlm_age, vlm_true_class, category, fields
  summary.json    raw confusion matrix + VLM-verified breakdown per direction
  montages/<label>_as_<pred>.jpg + montages/<category>.jpg
  report.md       full matrix -> per-direction VLM breakdown -> totals
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2

import dataset_utils as du
from describe import MockDescriber, QwenDescriber, OBJECT_SCHEMA, fmt_dur
from describe_hardneg import HN_OBJECT_PROMPT
from label_errors import life_stage
from cluster import montage
from compare_child import load_done_all_shards, load_records_all_shards

# Same extended schema as compare_gender.py: adds an explicit, independent
# gender read on top of the standard description fields.
GENDER_SCHEMA = dict(OBJECT_SCHEMA)
GENDER_SCHEMA["apparent_gender"] = "woman|man|unclear"


def vlm_truth(vlm_gender: str, vlm_age_stage: str) -> str | None:
    """Gender-agnostic Child determination: age decides Child first; gender
    only decides Woman vs Man for everyone past the puberty boundary."""
    if vlm_age_stage == "child":
        return "Child"
    if vlm_gender == "woman":
        return "Woman"
    if vlm_gender == "man":
        return "Man"
    return None


def categorize(label: str, model_pred: str, vlm_class: str | None) -> str:
    if vlm_class is None:
        return "uncertain"
    if vlm_class == label:
        return "model_error"          # VLM sides with the label -> model wrong
    if vlm_class == model_pred:
        return "label_error"          # VLM sides with the model -> label wrong
    return "vlm_disagrees_both"        # VLM picks a third class entirely


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description="unified 3-class Label vs Model vs VLM")
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
    print(f"[compare3] classes={names}")
    if args.num_shards > 1:
        print(f"[compare3] shard {args.shard_id}/{args.num_shards} -> {jsonl.name}")

    results = [json.loads(l) for l in (in_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    total_population = len(results)

    # ---- raw confusion matrix: free, instant, no VLM needed ---------------
    raw_matrix = defaultdict(lambda: Counter())
    for r in results:
        col = "missed" if r.get("pred_class") is None else r["pred_class"]
        raw_matrix[r["gt_class"]][col] += 1

    if args.filter == "all":
        scope = results
    elif args.filter == "misclassified":
        scope = [r for r in results if r.get("status") == "misclassified"]
    else:
        scope = [r for r in results if r.get("status") == "misclassified"
                 and r.get("pred_conf", 0) >= args.min_conf]
    print(f"[compare3] filter={args.filter} (min_conf={args.min_conf}): "
          f"{len(scope)} of {total_population} boxes selected for VLM")

    if args.num_shards > 1:
        scope = [r for r in scope
                 if zlib.crc32(r["id"].encode()) % args.num_shards == args.shard_id]
        print(f"[compare3] this shard: {len(scope)} boxes")

    done = load_done_all_shards(out)
    pending = [r for r in scope if r["id"] not in done
               and (crops_dir / f"{r['id']}.jpg").exists()]
    if done:
        print(f"[compare3] resuming — {len(done)} done")
    print(f"[compare3] {len(pending)} crops to judge (of {len(scope)} in scope)")

    if not pending:
        print("[compare3] nothing new for this shard; re-aggregating existing results")
    else:
        if args.mock:
            engine = MockDescriber()
            print("[compare3] MOCK engine")
        else:
            engine = QwenDescriber(model_id=args.model, device=args.device,
                                   max_pixels=args.max_pixels,
                                   max_new_tokens=args.max_new_tokens)
            print(f"[compare3] Qwen2.5-VL on {engine.device}")

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
                vclass = vlm_truth(gender, age_stage)
                label = r.get("gt_class")
                model_pred = r.get("pred_class")
                cat = categorize(label, model_pred, vclass)
                f.write(json.dumps({
                    "id": r["id"], "image": r["image"],
                    "label": label, "model_pred": model_pred,
                    "model_status": r.get("status"),
                    "vlm_gender": gender, "vlm_age": age,
                    "vlm_age_stage": age_stage, "vlm_true_class": vclass,
                    "category": cat, "object": obj,
                }) + "\n")
                n += 1
            f.flush()
            el = time.time() - t0
            rate = n / el if el else 0
            print(f"[compare3] {n}/{len(pending)} · {rate:.2f}/s · "
                  f"ETA {fmt_dur((len(pending)-n)/rate if rate else 0)}", flush=True)
        f.close()

    # ---- aggregate ----------------------------------------------------------
    recs = load_records_all_shards(out)
    cats = Counter(r["category"] for r in recs)
    vlm_scanned = len(recs)

    # per-direction (label -> model_pred) VLM-verified breakdown
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[(r["label"], r["model_pred"])].append(r)

    detected = [r for r in results if r.get("status") != "missed"]
    model_correct = sum(1 for r in detected if r.get("pred_class") == r.get("gt_class"))

    pair_summary = {}
    for (label, pred), members in by_pair.items():
        c = Counter(m["category"] for m in members)
        pair_summary[f"{label}->{pred}"] = {
            "scanned": len(members), **dict(c),
            "dominant_vlm_truth": Counter(
                m["vlm_true_class"] for m in members).most_common(1)[0][0]
            if members else None,
        }

    summary = {
        "total_boxes": total_population,
        "raw_confusion_matrix": {k: dict(v) for k, v in raw_matrix.items()},
        "label_vs_model_accuracy_full_population": round(
            model_correct / max(1, len(detected)), 4),
        "filter": args.filter, "min_conf": args.min_conf,
        "vlm_scanned": vlm_scanned,
        "categories_within_vlm_scanned": dict(cats),
        "per_direction_breakdown": pair_summary,
        "confirmed_totals": {
            "real_model_errors": cats.get("model_error", 0),
            "real_label_errors": cats.get("label_error", 0),
            "vlm_disagrees_with_both": cats.get("vlm_disagrees_both", 0),
            "uncertain": cats.get("uncertain", 0),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- montages: per (label->pred) direction, per overall category, AND
    # per (direction x category) — the granular "labeled_X_really_Y" views ---
    by_cat = defaultdict(list)
    for r in recs:
        by_cat[r["category"]].append(r["id"])

    by_pair_cat = defaultdict(list)   # (label, pred, category) -> [ids]
    for r in recs:
        by_pair_cat[(r["label"], r["model_pred"], r["category"])].append(r["id"])

    def make_montage(ids, name, title):
        sheet = montage(ids[:args.montage_samples], [crops_dir], title=title)
        if sheet is not None:
            cv2.imwrite(str(mont_dir / f"{name}.jpg"), sheet)
            return True
        return False

    # ---- report --------------------------------------------------------------
    classes = sorted(set(r["gt_class"] for r in results))
    L = ["# HARAMBLUR — Woman/Man/Child Label vs Model vs VLM\n",
         f"- Total boxes (all 3 classes): **{total_population}**",
         f"- Label-vs-Model accuracy (full population, detected only): "
         f"**{summary['label_vs_model_accuracy_full_population']*100:.1f}%**",
         f"- Filter: **{args.filter}** → VLM scanned **{vlm_scanned}** of the "
         f"mismatches\n",
         "## 1. Raw confusion matrix (Label x Model) — entire population, no VLM\n",
         "| label \\ pred | " + " | ".join(classes + ["missed"]) + " |",
         "|" + "---|" * (len(classes) + 2)]
    for lab in classes:
        row = [f"**{lab}**"]
        for col in classes + ["missed"]:
            row.append(str(raw_matrix.get(lab, {}).get(col, 0)))
        L.append("| " + " | ".join(row) + " |")

    # human-readable per-(direction,category) montages, e.g.
    # "labeled_Woman_really_Man" (confirmed label error) and
    # "really_Woman_but_model_said_Man" (confirmed model error)
    cat_link = {}  # (label, pred, category) -> markdown link or "—"
    for (label, pred, cat), ids in by_pair_cat.items():
        if cat == "label_error":
            name = safe_name(f"labeled_{label}_really_{pred}")
            title = f"Labeled {label}, really {pred} (confirmed label error, n={len(ids)})"
        elif cat == "model_error":
            name = safe_name(f"really_{label}_but_model_said_{pred}")
            title = f"Really {label}, model said {pred} (confirmed model error, n={len(ids)})"
        else:
            name = safe_name(f"{label}_as_{pred}__{cat}")
            title = f"{label} -> {pred}, {cat} (n={len(ids)})"
        ok = make_montage(ids, name, title)
        cat_link[(label, pred, cat)] = f"[{len(ids)}](montages/{name}.jpg)" if ok else str(len(ids))

    L.append("\n## 2. VLM-verified breakdown per mismatch direction\n")
    L.append("Each count below links its own montage — e.g. under Woman → Man, "
             "the **label_error** link shows confirmed `labeled_Woman_really_Man`, "
             "the **model_error** link shows confirmed `really_Woman_but_model_said_Man`.\n")
    L.append("| label → model | scanned | model_error | label_error | "
             "vlm_disagrees_both | uncertain |")
    L.append("|---|---|---|---|---|---|")
    for (label, pred), members in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        if label == pred or pred is None:
            continue
        L.append(
            f"| {label} → {pred} | {len(members)} | "
            f"{cat_link.get((label, pred, 'model_error'), 0)} | "
            f"{cat_link.get((label, pred, 'label_error'), 0)} | "
            f"{cat_link.get((label, pred, 'vlm_disagrees_both'), 0)} | "
            f"{cat_link.get((label, pred, 'uncertain'), 0)} |")

    L.append("\n## 3. Confirmed totals (across all directions)\n")
    t = summary["confirmed_totals"]
    L.append(f"- **Real model errors:** {t['real_model_errors']} "
             f"({100*t['real_model_errors']/max(1,vlm_scanned):.1f}% of scanned, "
             f"{100*t['real_model_errors']/max(1,total_population):.1f}% of all boxes)")
    L.append(f"- **Real label errors:** {t['real_label_errors']} "
             f"({100*t['real_label_errors']/max(1,vlm_scanned):.1f}% of scanned, "
             f"{100*t['real_label_errors']/max(1,total_population):.1f}% of all boxes)")
    L.append(f"- VLM disagrees with both: {t['vlm_disagrees_with_both']}")
    L.append(f"- Uncertain: {t['uncertain']}")

    L.append("\n## 4. Confirmed LABEL errors — every direction\n")
    L.append("Each is a real `label_error`: model was right, the dataset label was "
             "wrong. Click to see the crops.\n")
    label_error_pairs = sorted(
        [(l, p, len(ids)) for (l, p, c), ids in by_pair_cat.items() if c == "label_error"],
        key=lambda x: -x[2])
    for label, pred, n in label_error_pairs:
        name = safe_name(f"labeled_{label}_really_{pred}")
        L.append(f"- **Labeled `{label}`, really `{pred}`** — {n} images: "
                 f"[montages/{name}.jpg](montages/{name}.jpg)")

    L.append("\n## 5. Confirmed MODEL errors — every direction\n")
    L.append("Each is a real `model_error`: label was right, the model was "
             "wrong. Click to see the crops.\n")
    model_error_pairs = sorted(
        [(l, p, len(ids)) for (l, p, c), ids in by_pair_cat.items() if c == "model_error"],
        key=lambda x: -x[2])
    for label, pred, n in model_error_pairs:
        name = safe_name(f"really_{label}_but_model_said_{pred}")
        L.append(f"- **Really `{label}`, model said `{pred}`** — {n} images: "
                 f"[montages/{name}.jpg](montages/{name}.jpg)")

    L.append("\n## Category montages\n")
    for cat, ids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        if make_montage(ids, cat, f"{cat} (n={len(ids)})"):
            L.append(f"### {cat} ({len(ids)})")
            L.append(f"![{cat}](montages/{cat}.jpg)\n")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[compare3] categories: {dict(cats)}")
    print(f"[compare3] confirmed model errors: {t['real_model_errors']} · "
          f"confirmed label errors: {t['real_label_errors']}")
    print(f"[compare3] -> {out/'report.md'} · {out/'summary.json'}")


if __name__ == "__main__":
    main()
