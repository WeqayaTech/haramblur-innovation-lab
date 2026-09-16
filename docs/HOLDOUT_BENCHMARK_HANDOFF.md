# `haramblur_holdout` — path reference + benchmark handoff

Dataset root on the RunPod volume:

```
/workspace/datasets/haramblur_holdout/
```

**This is a HOLDOUT TEST SET. Never train on it.** It carries a `DO_NOT_TRAIN` marker.

---

## Quick path reference

| What | Path |
|---|---|
| **Images** (11,494) | `labeling/full/images/` |
| **Labels for evaluation** | `labeling/full/labels_eval/` (classes 0=Woman, 1=Man, 2=Child) |
| **Ignore regions** (must pass to scorer) | `labeling/full/ignore/` (1,619 unknown-gender people) |
| **Labels with Unknown** | `labeling/full/labels_unk3/` (class 3 = Unknown) |
| **SAM3 raw output** | `labeling/full/raw/<stem>.txt` + `<stem>.json` (sidecars) |
| **Gemini verdicts** | `labeling/full/run/verdicts_batch.jsonl` (21,004 rows) |
| **Run record** | `labeling/FULL_RUN.md` |
| **Dedup list** | `clean/FINAL_unique.txt` (the canonical 11,494) |

**Two things that silently break a scorer if missed:**
1. Labels are **segment polygons** (26-48+ coords/row), not boxes — use `seg_boxes`.
2. `images/` holds **symlinks** — copy with `cp -rL` to dereference.

Per collection (filename prefix): `child__` 1,247 · `men__` 1,436 · `randoms__` 2,023
(person-free) · `shiekhs__` 2,897 (Gulf-dress) · `women__` 2,939 · `women_hd__` 952.

For running another VLM (Sol, Claude...) on the holdout detections, see
`docs/SAM3_OUTPUT_HANDOFF.md` (sidecar schema, the `build_crop`/`PROMPT` imports, and
the Gemini baseline to diff against).

## Full layout

```
/workspace/datasets/haramblur_holdout/
├── DO_NOT_TRAIN
├── README.md                     source, census, dedup method, caveats
├── EVAL_VERIFICATION.html
├── SAM3_OUTPUT_HANDOFF.md        → give to anyone running another VLM
├── images/<collection>/          raw images (6 collections)
├── _archives/                    original zips + SHA256SUMS.txt
├── clean/                        FINAL_unique.txt + per-collection lists
├── census.json / census_files.jsonl
├── FINAL_dedup.json / pdq_calibration.json / cluster_gallery_31.html
└── labeling/
    ├── FULL_RUN.md               run record: results, findings, traps
    ├── tools/                    all scripts (re-runnable)
    └── full/
        ├── images/               11,494 symlinks
        ├── raw/                  SAM3 labels + raw sidecars
        ├── labels_eval/          classes 0-2 (score against this)
        ├── labels_unk3/          classes 0-3
        ├── ignore/               1,619 Unknown regions
        ├── run/                  verdicts, audit, cost, batch ledger
        └── traces/               stage-by-stage proof HTMLs
```

---

## Benchmarking a model

Paste the block below into a fresh chat. **All flags verified against `--help` on the pod
2026-08-19.**

## Paste this into the new chat

> I want to benchmark model(s) against the **`haramblur_holdout`** evaluation set on the
> RunPod pod. Read `CLAUDE.md` → section **"THE HOLDOUT TEST SET"** first, then
> `/workspace/datasets/haramblur_holdout/labeling/FULL_RUN.md` on the pod.
>
> Connect with `ssh runpod`. Facts you must respect:
>
> - Ground truth lives at `/workspace/datasets/haramblur_holdout/labeling/full/`:
>   `images/` (11,494, symlinks — use `cp -rL` to move), `labels_eval/` (classes
>   0=Woman 1=Man 2=Child), `ignore/` (1,619 unknown-gender people as ignore regions,
>   on 704 images), `labels_unk3/` (same labels, class 3 = Unknown, one tree).
> - **Labels are SEGMENT POLYGONS (26-48+ coords/row), not boxes.** A 5-field YOLO box
>   reader silently drops every row — use the project's `seg_boxes` parser.
> - **Always pass `ignore/` to `map_eval.py --ignore-labels`**, or a model is penalised
>   1,619 times for correctly finding people the labeler declined to gender.
> - **Score per collection, never pooled.** Filename prefix is the collection:
>   `child__ men__ randoms__ shiekhs__ women__ women_hd__`. `randoms` (2,023 imgs) is the
>   negatives arm; `shiekhs` (2,897 imgs) is the Gulf-dress slice.
> - **None of the scorers has a `--filter-prefix`** — build per-collection symlink dirs
>   (recipe below) and point the scorer at those.
> - These are **machine labels (Gemini 3.5 Flash-Lite), not human-verified**. The **254
>   `randoms` gate-survivors are unadjudicated** — do not quote a false-positive rate
>   from that arm until they are eyeballed.
> - Dump at **conf floor 0.001** so thresholds sweep offline with no model re-run.
> - Fresh pods need `pip install opencv-python-headless` before the scorers import.

---

## 0. Confirm the set is intact (~5 min, re-runnable)

```bash
ssh runpod
cd /workspace/datasets/haramblur_holdout
python3 labeling/tools/verify_eval_ready.py      # expect: ALL CHECKS PASSED
```

## 1. Build per-collection slices (no scorer supports prefix filtering)

```bash
H=/workspace/datasets/haramblur_holdout/labeling/full
S=/workspace/holdout_eval/slices
for c in child men randoms shiekhs women women_hd; do
  mkdir -p $S/$c/images $S/$c/labels $S/$c/ignore
  for f in $H/images/${c}__*;      do ln -sf "$f" $S/$c/images/; done
  for f in $H/labels_eval/${c}__*; do ln -sf "$f" $S/$c/labels/; done
  for f in $H/ignore/${c}__*;      do ln -sf "$f" $S/$c/ignore/; done
  echo "$c: $(ls $S/$c/images | wc -l) images"
done
```

## 2. Dump predictions (Stage A) — floor 0.001

```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 run_ultralytics_labels.py \
    --engine ultralytics \
    --model /workspace/exp14/y26n_gradsupp/weights/best.pt \
    --images $H/images \
    --out /workspace/holdout_eval/y26n_gradsupp \
    --floor 0.001
```

Variants: `--engine mit --run-dir <YOLO-MIT run>` for the production checkpoint;
`--no-e2e` to read a YOLO26 one-to-many head; `--prompts`/`--distractors` for YOLOE.
Writes `labels/` (5-field YOLO) and `raw/` (per-image sidecars with conf) — **`map_eval`
reads `raw/`, not `labels/`.**

## 3. mAP with ignore regions — the headline number

```bash
python3 map_eval.py \
    --raw           /workspace/holdout_eval/y26n_gradsupp/raw \
    --gt-labels     $H/labels_eval \
    --ignore-labels $H/ignore \
    --expect-floor  0.001 \
    --out /workspace/holdout_eval/y26n_gradsupp/map_holdout.json
```

Per collection — same command against a slice:

```bash
python3 map_eval.py --raw <dump>/raw --gt-labels $S/shiekhs/labels \
    --ignore-labels $S/shiekhs/ignore --expect-floor 0.001 \
    --out <dump>/map_shiekhs.json
```

## 4. Negatives arm — false persons on person-free images

```bash
python3 eval_negatives_crowd.py --mode negatives \
    --images      $S/randoms/images \
    --pred-labels <dump>/labels \
    --class-names Woman Man Child \
    --out /workspace/holdout_eval/y26n_gradsupp/negatives_randoms
```

## 5. Offline threshold sweep (CPU only, no re-run)

```bash
python3 conf_sweep.py \
    --negatives randoms:<dump>/raw \
    --objective "1 - randoms.img_fp_rate" \
    --baseline 0.45 \
    --out /workspace/holdout_eval/y26n_gradsupp/charts
```

`--list-metrics` prints the metric catalogue. Write no-regression bounds as
`<= baseline.<arm>.<metric>` — a hand-copied literal rounded to 4dp makes the baseline
fail its own constraint and the run returns NO FEASIBLE. Constrain absolute counts, not
rates, where the denominator moves with the threshold.

**Unverified:** `conf_sweep.py`'s labeled arm (`--lagenda`) also takes `--lagenda-gt`
(a `gt.jsonl` of human age/gender rows) for its classification metrics. This holdout set
has **no `gt.jsonl`**, so the recall/gender metrics on that arm may need a small adapter,
or `--lagenda-gt-labels` alone may cover the detection-side metrics. Check before relying
on it; the negatives arm above needs no adapter.

## 6. Record results

One row per **exact run name + weights path** in `docs/MODEL_COMPARISON.md` — no invented
aliases. Holdout numbers are **not comparable** to Spotlight-val / LAGENDA / CrowdHuman
numbers; different dataset, never pool.

---

## What this set can and cannot tell you

**Can:** relative model ranking on a fresh, deduplicated, never-trained-on corpus; false
persons across 2,023 person-free images; the **first Gulf-dress measurement** the project
has; behaviour on deliberately hard slices (`long-hair-man`, `korean-man`).

**Cannot:** give a human-accurate ground-truth score. The labels are Gemini 3.5
Flash-Lite's output, so you are measuring one model against another model's judgement —
treat disagreements as candidates for adjudication, not proof of error. `shiekhs` is 7
identities and `women` is identity-concentrated (`jessica` 700), so effective sample size
for identity-level generalisation is far below the image count.
