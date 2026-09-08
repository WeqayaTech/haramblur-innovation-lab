#!/usr/bin/env python3
"""
Taxonomy-agnostic evaluator — score the VLM pipeline against human GT under any
number of label taxonomies in one pass.

Reads raw two-axis data on both sides (GT: numeric age + gender; VLM: age_band +
gender), pushes both through a chosen set of `Translation`s (see translation.py),
and reports per-taxonomy / per-head metrics so you can A/B different ways of
customizing the labels.

    python eval_taxonomy.py \
        --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
        --descs    /workspace/lagenda_eval/descriptions/descriptions.jsonl \
        --schemes  production_3class two_axis gender_only child_vs_adult

Each head yields a classification report: confusion (vs human GT), overall +
per-class accuracy, coverage (abstain rate), false-flag rate, Wilson 95% CIs.

Plus dataset-level diagnostics (independent of the chosen schemes):
  - child-vs-adult accuracy by true age (the boundary curve)
  - child/adult threshold sweep (<=9 / <=12 / <=17) with catch-kids + catch-adults
  - gender accuracy + coverage by true age
  - what age-group the VLM predicted, by true age (detected vs actual)
  - child-gender consistency (validates 'ignore gender for children')
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import translation as T


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - m) / d, (c + m) / d)


def ci(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:5.1f}% [{100*lo:4.1f}-{100*hi:4.1f}]"


# ---------------------------------------------------------------------------
# data loading -> rows with precomputed (gender, age-bucket) on both sides
# ---------------------------------------------------------------------------
class Row:
    __slots__ = ("id", "gt_g", "gt_b", "vlm_g", "vlm_b", "gt_age")

    def __init__(self, uid, gt_age, gt_gender, obj):
        self.id = uid
        self.gt_age = gt_age
        self.gt_g = T.norm_gender(gt_gender)
        self.gt_b = T.age_bucket_from_years(gt_age)
        self.vlm_g = T.norm_gender(obj.get("apparent_gender"))
        self.vlm_b = T.age_bucket_from_band(obj.get("apparent_age_band"))


def load(manifest_path, descs_path):
    man = {}
    for l in Path(manifest_path).read_text().splitlines():
        if l.strip():
            r = json.loads(l); man[r["id"]] = r
    des = {}
    for l in Path(descs_path).read_text().splitlines():
        if l.strip():
            r = json.loads(l); des[r["id"]] = r
    rows = []
    for uid, m in man.items():
        d = des.get(uid)
        if d is None:
            continue
        rows.append(Row(uid, m.get("gt_age"), m.get("gt_gender"), d.get("object", {})))
    return rows, len(man), len(des)


# ---------------------------------------------------------------------------
# per-head reports
# ---------------------------------------------------------------------------
def head_classification(rows, head, out):
    conf = defaultdict(Counter)          # gt_label -> Counter(pred_label incl (abstain))
    correct = scored = abstain = 0
    for r in rows:
        gt = head.label(r.gt_g, r.gt_b)
        if gt is None:                   # GT not applicable for this head
            continue
        pv = head.label(r.vlm_g, r.vlm_b)
        conf[gt][pv if pv is not None else "(abstain)"] += 1
        if pv is None:
            abstain += 1
            continue
        scored += 1
        correct += (pv == gt)

    determinable = sum(sum(c.values()) for c in conf.values())
    lines = [f"**Classification** — n(GT-determinable)={determinable}, "
             f"scored={scored}, abstained={abstain} "
             f"({100*abstain/max(1,determinable):.1f}% coverage gap)",
             f"- accuracy (of scored): **{ci(correct, scored)}**",
             f"- false-flag rate (would wrongly challenge a correct label): "
             f"**{ci(scored - correct, scored)}**", "",
             "| GT \\ pred | " + " | ".join(head.classes) + " | (abstain) |",
             "|" + "---|" * (len(head.classes) + 2)]
    for gt in head.classes:
        row = conf.get(gt, Counter())
        cells = " | ".join(str(row.get(c, 0)) for c in head.classes)
        lines.append(f"| **{gt}** | {cells} | {row.get('(abstain)', 0)} |")
    lines.append("")
    lines.append("Per-class accuracy (recall, of scored):")
    per_class = {}
    for cls in head.classes:
        row = conf.get(cls, Counter())
        tot = sum(v for k, v in row.items() if k != "(abstain)")
        cor = row.get(cls, 0)
        per_class[cls] = {"n": tot, "correct": cor}
        lines.append(f"- {cls}: {ci(cor, tot)}  (n={tot})")
    out.extend(lines); out.append("")
    return {"scored": scored, "correct": correct, "abstain": abstain,
            "accuracy": correct / scored if scored else None,
            "false_flag_rate": (scored - correct) / scored if scored else None,
            "per_class": per_class,
            "confusion": {g: dict(c) for g, c in conf.items()}}


# ---------------------------------------------------------------------------
# dataset-level diagnostics (scheme-independent)
# ---------------------------------------------------------------------------
def diagnostics(rows, out):
    out.append("## Diagnostics (scheme-independent)\n")

    # child-vs-adult accuracy by true age (puberty head)
    cva = T.get("child_vs_adult").heads[0]
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        gt = cva.label(r.gt_g, r.gt_b); pv = cva.label(r.vlm_g, r.vlm_b)
        if gt is None or pv is None or r.gt_age is None:
            continue
        b = int(r.gt_age // 5) * 5
        buckets[b][1] += 1
        buckets[b][0] += (gt == pv)
    out.append("**Child-vs-Adult accuracy by true age** (5-yr buckets):")
    for b in sorted(buckets):
        cor, tot = buckets[b]
        if tot:
            out.append(f"- age {b:>2}-{b+4}: {100*cor/tot:5.1f}%  (n={tot})")
    out.append("")

    # threshold sweep — overall accuracy + child/adult recall at each band-edge cutoff
    out.append("**Child/Adult threshold sweep** (VLM band edges):")
    sweep = {}
    sweep_labels = {"child_vs_adult_9": "under 10 (<=9)",
                    "child_vs_adult": "puberty (<=12)",
                    "child_vs_adult_18": "minor (<=17)"}
    for name in ("child_vs_adult_9", "child_vs_adult", "child_vs_adult_18"):
        h = T.get(name).heads[0]
        cor = tot = cc = ct = ac = at = 0   # overall, child correct/total, adult correct/total
        for r in rows:
            gt = h.label(r.gt_g, r.gt_b); pv = h.label(r.vlm_g, r.vlm_b)
            if gt is None or pv is None:
                continue
            tot += 1; cor += (gt == pv)
            if gt == "Child": ct += 1; cc += (pv == "Child")
            else:             at += 1; ac += (pv == "Adult")
        sweep[name] = {"label": sweep_labels[name], "n": tot,
                       "accuracy": cor / tot if tot else None,
                       "child_recall": cc / ct if ct else None,
                       "adult_recall": ac / at if at else None}
        out.append(f"- {sweep_labels[name]}: acc {ci(cor,tot)} · "
                   f"catch kids {ci(cc,ct)} · catch adults {ci(ac,at)}")
    out.append("")

    # gender accuracy + coverage by 5-yr true age
    gender_age = {}   # bucket -> [correct, committed, total]
    for r in rows:
        if r.gt_g not in ("man", "woman") or r.gt_age is None:
            continue
        b = int(r.gt_age // 5) * 5
        d = gender_age.setdefault(b, [0, 0, 0])
        d[2] += 1
        if r.vlm_g in ("man", "woman"):
            d[1] += 1
            d[0] += (r.vlm_g == r.gt_g)
    out.append("**Gender accuracy + coverage by true age** (5-yr buckets):")
    for b in sorted(gender_age):
        cor, com, tot = gender_age[b]
        acc = 100 * cor / com if com else 0
        cov = 100 * com / tot if tot else 0
        out.append(f"- age {b:>2}-{b+4}: acc {acc:5.1f}% (of committed) · "
                   f"commits {cov:5.1f}%  (n={tot})")
    out.append("")

    # what age-GROUP the VLM predicted, by 5-yr true age (detected vs actual)
    GROUPS = ["child", "teen", "adult", "senior"]
    vlm_group_age = {}   # bucket -> {group: count, ..., none, total}
    for r in rows:
        if r.gt_age is None:
            continue
        b = int(r.gt_age // 5) * 5
        d = vlm_group_age.setdefault(b, {g: 0 for g in GROUPS + ["none", "total"]})
        g = T._age_group_coarse(r.vlm_b)
        d["total"] += 1
        d[g if g in GROUPS else "none"] += 1
    out.append("**What age-group the VLM predicted, by true age** (5-yr buckets, row %):")
    for b in sorted(vlm_group_age):
        d = vlm_group_age[b]; t = d["total"] or 1
        parts = " ".join(f"{g}={100*d[g]/t:.0f}%" for g in GROUPS)
        out.append(f"- age {b:>2}-{b+4}: {parts}  (n={d['total']})")
    out.append("")

    # child-gender consistency: among GT children, does VLM gender agree, and
    # how often does the VLM even commit to a gender for a child?
    n_child = agree = committed = 0
    for r in rows:
        if r.gt_b not in T.CHILD_BUCKETS:
            continue
        n_child += 1
        if r.vlm_g in ("man", "woman"):
            committed += 1
            if r.vlm_g == r.gt_g:
                agree += 1
    out.append("**Child-gender consistency** (validates 'ignore gender for children'):")
    out.append(f"- GT children: {n_child}")
    if n_child:
        out.append(f"- VLM commits a gender on: {ci(committed, n_child)}")
    if committed:
        out.append(f"- of those, gender matches GT: {ci(agree, committed)}")
    out.append("")
    return {"child_gender": {"n_child": n_child, "committed": committed, "agree": agree},
            "threshold_sweep": sweep,
            "gender_by_age": gender_age,
            "vlm_group_by_age": vlm_group_age}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="gt jsonl with gt_age/gt_gender")
    ap.add_argument("--descs", required=True, help="describe.py descriptions.jsonl")
    ap.add_argument("--schemes", nargs="+",
                    default=["production_3class", "two_axis", "gender_only",
                             "child_vs_adult"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write machine-readable JSON here")
    args = ap.parse_args()

    rows, nman, ndes = load(args.manifest, args.descs)
    out = [f"# Taxonomy evaluation",
           f"GT records: {nman} · descriptions: {ndes} · **matched: {len(rows)}**\n"]
    results = {"matched": len(rows), "schemes": {}}

    for name in args.schemes:
        tr = T.get(name)
        out.append(f"## Scheme: `{name}`  ({len(tr.heads)} head"
                   f"{'s' if len(tr.heads) > 1 else ''})\n")
        results["schemes"][name] = {}
        for head in tr.heads:
            out.append(f"### head: `{head.name}`  classes={list(head.classes)}\n")
            cls = head_classification(rows, head, out)
            results["schemes"][name][head.name] = {"classification": cls}

    results["diagnostics"] = diagnostics(rows, out)

    report = "\n".join(out)
    print(report)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[eval] machine-readable results -> {args.out}")


if __name__ == "__main__":
    main()
