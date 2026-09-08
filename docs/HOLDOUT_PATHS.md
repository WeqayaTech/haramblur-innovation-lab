# `haramblur_holdout` — path reference (share this with the team)

Everything lives on the **RunPod network volume**, which persists across pods.
Mount point on any pod: `/workspace`. Root of this dataset:

```
/workspace/datasets/haramblur_holdout/
```

**This is a HOLDOUT TEST SET. Never train on it, never feed it to a labeling run
that produces training data.** It carries a `DO_NOT_TRAIN` marker at its root.

---

## The three things most people need

| What | Path |
|---|---|
| **Images** (11,494) | `labeling/full/images/` |
| **Labels for evaluation** | `labeling/full/labels_eval/` |
| **Ignore regions** (must be passed to the scorer) | `labeling/full/ignore/` |

Classes: `0 = Woman, 1 = Man, 2 = Child`.

**Two things that will silently break a scorer if missed:**
1. **Labels are SEGMENT POLYGONS**, not boxes — 26–48+ coordinates per row. A plain
   5-field YOLO box reader drops every row without error. Use `seg_boxes`.
2. **`images/` holds symlinks** into `../../images/<collection>/`. Fine on the volume;
   copy with `cp -rL` to dereference if moving off it.

---

## Full layout

```
/workspace/datasets/haramblur_holdout/
├── DO_NOT_TRAIN                  marker — read it
├── README.md                     dataset: source, census, dedup method, caveats
├── EVAL_VERIFICATION.html        ← the verification report (open in a browser)
├── SAM3_OUTPUT_HANDOFF.md        ← give this to anyone running ANOTHER VLM (Sol,
│                                   Claude…) on our detections: sidecar format, the
│                                   build_crop/PROMPT imports that make it apples-to-
│                                   apples, and our Gemini baseline to diff against
│
├── images/<collection>/          THE RAW IMAGES, as delivered (6 collections)
├── _archives/                    original zips + SHA256SUMS.txt (provenance)
│
├── clean/
│   ├── FINAL_unique.txt          the 11,494 unique images — the canonical list
│   ├── FINAL_unique_<coll>.txt   same, per collection
│   └── all.txt                   12,736 clean-but-not-deduped
│
├── census.json / census_files.jsonl    per-file: size, decoded w/h, format, sha256
├── FINAL_dedup.json              every near-duplicate cluster + why it merged
├── pdq_calibration.json          the threshold evidence
├── cluster_gallery_31.html       visual audit of the dedup decisions
│
└── labeling/
    ├── FULL_RUN.md               ← THE run record: results, findings, traps
    ├── README.md                 the 60-image pilot
    ├── tools/                    every script used (all re-runnable)
    │   ├── verify_eval_ready.py      completeness check
    │   ├── verify_report.py          builds EVAL_VERIFICATION.html
    │   ├── autolabel_sam_raw_allext.py   webp-patched labeler copy
    │   ├── final_analysis.py / completion_proof.py / make_ignore.py
    │   └── milestone_trace.py
    ├── test60/                   the pilot run (incl. gemini_crops/ = exact crops sent)
    └── full/                     ← THE PRODUCTION RUN
        ├── images/               11,494 symlinks (labeling input)
        ├── manifest.jsonl        collection + source path per staged image
        ├── raw/<stem>.txt        SAM3 YOLO labels
        ├── raw/<stem>.json       RAW SIDECARS: per-part mask polygons before YOLO
        │                         flattening, per-detection conf, suppressed dets
        ├── labels_eval/          classes 0-2  ← SCORE AGAINST THIS
        │                         (symlink to labels_std; also contains
        │                          _audit.jsonl + _emit_stats.json — glob *.txt)
        ├── labels_std/           the real directory
        ├── labels_unk3/          classes 0-3, class 3 = Unknown, single tree
        ├── ignore/               the 1,619 Unknown regions, one file per image
        ├── run/
        │   ├── verdicts_batch.jsonl   21,004 rows — full Gemini verdict, all 9
        │   │                          prompt fields, raw_text, blurriness,
        │   │                          person_px_height, sam_conf, prompt_sha
        │   ├── _audit.jsonl           every delete / relabel / unknown decision
        │   ├── _emit_stats.json       kept / deleted / relabeled / dropped
        │   ├── run_meta.json          full prompt text + sha + config
        │   │                          (RECONSTRUCTED 2026-08-20 — see note below)
        │   ├── cost_report.json       measured spend
        │   └── batch/jobs.json        Batch API ledger
        ├── traces/trace_{5000,10000,15000,20000}.html   stage-by-stage proof
        ├── final_analysis.txt    the headline numbers
        └── *.log                 sam3, batch submit/collect, emit, milestones
```

---

## Counts (verified)

```
11,494 images · 21,004 SAM3 detections · 1 parse failure · 0 API errors
labels_eval   16,139 rows — Woman 5,690 · Man 8,616 · Child 1,833
labels_unk3   17,758 rows — the same + 1,619 Unknown
ignore         1,619 regions across 704 images
Cost $6.78 (Gemini Batch API) · SAM3 2h54m on an L4
```

Per collection (unique images): `shiekhs` 2,897 · `women` 2,939 · `randoms` 2,023
· `men` 1,436 · `child` 1,247 · `women_hd` 952.

---

## Copies already outside the volume

In this repo, for people without pod access:

```
_holdout_review/EVAL_VERIFICATION.html    the verification report
_holdout_review/traces/trace_*.html       the four milestone traces
_holdout_review/cluster_gallery_31.html   dedup audit
docs/HOLDOUT_BENCHMARK_HANDOFF.md         verified commands to benchmark against it
docs/DATASET_REGISTRY.md                  registry row
CLAUDE.md → "THE HOLDOUT TEST SET"        the summary
```

---

## Tell the team these three caveats

1. **The labels are machine-generated** (SAM3 detects, Gemini 3.5 Flash-Lite judges) and
   have had **no human verification**. Measuring a model against them measures agreement
   with Gemini, not truth.
2. **Pass `ignore/` to the scorer.** Otherwise a model is penalised 1,619 times for
   correctly finding people the pipeline declined to gender.
3. **The 254 `randoms` gate-survivors are unadjudicated** — do not quote a
   false-positive rate from that arm until someone eyeballs them.

---

## Provenance note (read if you care where these labels came from)

`spotlight_batch.py` (the Batch API path) writes **neither `run_meta.json` nor
`cost_report.json`**, and stamps `prompt_sha` as an **empty string** on every verdict —
unlike `spotlight_run.py` (the sequential path), which writes both. So the production run
originally had no file recording which prompt produced these labels.

Both files were **reconstructed on 2026-08-20** and are flagged `"_reconstructed": true`.
The prompt identity is *derived, not assumed*: `spotlight_batch.py:45` imports
`PROMPT, PROMPT_VERSION` directly from `spotlight_run` and sends the constant verbatim at
line 216, so hashing that live constant gives the prompt the batch actually used —
**`spotlight-e1`, sha `4c85ffbf8bcb`**, which matches both the pilot's `run_meta.json`
and the `prompt_version` recorded per job in `run/batch/jobs.json`. Costs come from
`jobs.json`, written by the collector from the API's own token counts.

Rebuild at any time: `python3 labeling/tools/mk_runmeta.py` (it asserts on every
link in that chain and refuses to write if any breaks).

**Worth reporting upstream:** the Batch path losing `prompt_sha` and the run metadata is
a real provenance hole for any production run that uses it — including the OIV7 run.
