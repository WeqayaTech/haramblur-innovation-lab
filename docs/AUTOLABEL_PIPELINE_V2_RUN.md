# Spotlight auto-labeling pipeline — production runbook

Lives on the pod at **`/workspace/autolabel_pipeline_v2/`**. Production code only —
no analysis or experiment tooling. Copied from `vlm-cluster` on 2026-07-27; that
directory remains the development source, this one is the frozen deployment.

## Files

| File | Stage | Purpose |
|---|---|---|
| `autolabel_sam_raw.py` | 1 | SAM3 → YOLO labels + raw mask sidecars (sharded, resumable) |
| `spotlight_run.py` | 2–5 | crops → Gemini verdicts → merge → **emits cleaned labels** |
| `api_describers.py` + `model_pricing.json` | — | Gemini client + verified rates |
| `build_training_set.py` | — | image reservation + provenance manifest |
| `estimate_cost.py` | — | crowding scan + cost projection |
| `.env` → symlink | — | API keys (single source of truth in vlm-cluster) |

Every script has `--selftest` and needs no data or network. Run them after any sync.

## Frozen configuration

| Setting | Value | Where |
|---|---|---|
| SAM3 prompts | `woman` `man` `child` | run command |
| SAM3 conf / NMS IoU | 0.4 / 0.7 | `autolabel_sam_raw.py` defaults |
| Crop padding | 25% | `PAD` |
| Min crop side (upscale) | 320 px | `MIN_CROP_SIDE` |
| Highlight | mask outline, two-tone green, 3 px | `OUTLINE_*` |
| Model | `gemini-3.5-flash-lite` | `--model` |
| Prompt | `spotlight-e1` (9 fields) | `PROMPT`, hash in `run_meta.json` |
| Child cutoff | `estimated_age <= 12` | `CHILD_AGE_MAX` |
| Box corrections | **never applied** (SAM3 geometry wins) | merge rules |
| Unknown gender | dropped from training | `--keep-unknown-gender` to override |
| Rates | $0.30 in / $2.50 out per 1M · Batch 50% off | verified 2026-07-27 |

## Step 0 — A/B the prompt before the full run (~$1, REQUIRED)

The validated results (EXP-2026-10) used the 21-field `spotlight-v2` prompt.
This folder ships the trimmed 9-field `spotlight-e1` prompt to hit the budget.
**It has not been validated yet.** Run the same gate we used for every prior
prompt change, and compare against these numbers:

| Metric | Must hold |
|---|---|
| TP-keep, crowd | ≥ 97% (v2 measured 98.48%) |
| TP-keep, LAGENDA | ≥ 97% (v2 measured 99.34%) |
| Gender, committed adults | ≥ 99% (v2 measured 99.05%) |
| Adults 18+ labeled Child | 0% (v2 measured 0/95) |

Use `vlm-cluster/pipeline_v1_eval.py` (the scoring harness) against the staged
benchmark sets for this — production code does not score.

## Step 1 — reserve images, with provenance

```bash
cd /workspace/autolabel_pipeline_v2
python3 build_training_set.py reserve --name pass10k \
    --source /workspace/datasets/pass_3k --n 10000 \
    --staging /workspace/train_set --seed 42
python3 build_training_set.py reserve --name crowd750 \
    --source /workspace/datasets/crowdhuman/Images --n 750 \
    --staging /workspace/train_set --seed 42
```

Writes `manifest.jsonl` (source path + sha1 per image) and `held_out_*.txt`.
**Before every eval run**, prove no leakage:

```bash
python3 build_training_set.py check --staging /workspace/train_set \
    --candidate <eval dir>        # exits 1 on any overlap, incl. renamed files
```

## Step 2 — SAM3 labeling (GPU)

Fresh pod first: `pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

```bash
export HF_HOME=/workspace/.cache/huggingface
python3 autolabel_sam_raw.py --input <images dir> --output <raw labels dir> \
    --classes "woman" "man" "child" --batch_size 1
```

Shard across N pods — same command everywhere, only the index changes:
`--shards 4 --shard-index 0|1|2|3`. Split is by stable path hash: disjoint,
complete, and each shard independently resumable. Progress shows real remaining
work plus live detections-per-image (your cost forecast).

~1.3 s/image on an L4 → 512k images ≈ 185 GPU-hours (~2 days on 4 pods).

## Step 3 — cost check before spending

```bash
python3 estimate_cost.py scan --labels <raw labels dir>
python3 estimate_cost.py project --labels <raw labels dir> \
    --calibration-run <a small completed run dir> --corpus-images 512892
```

## Step 4 — verify + emit

```bash
set -a; source .env; set +a
python3 spotlight_run.py --images <images dir> --raw-labels <raw labels dir> \
    --out /workspace/oiv7_clean --max-spend 500 --report-every 100
```

Hard-stops at the spend ceiling with everything saved; re-run to continue.
Watch from another shell: `watch -n 60 cat /workspace/oiv7_clean/status.json`

Outputs in `--out`:

| Artifact | Contents |
|---|---|
| `labels/*.txt` | **cleaned YOLO labels** — rejects removed, classes corrected, SAM3 polygons byte-identical |
| `labels/_audit.jsonl` | every delete and relabel, with reasons |
| `labels/_emit_stats.json` | kept / deleted / relabeled / dropped counts |
| `verdicts.jsonl` | one row per detection: full verdict, analysis fields, blurriness, person height, SAM3 conf, raw response |
| `status.json`, `cost_report.json` | live progress and spend |
| `run_meta.json` | full prompt text + sha + all config, frozen at run start |

Re-emit under a different policy for **free** (no API calls):

```bash
python3 spotlight_run.py --emit-only --raw-labels <raw> --out /workspace/oiv7_clean
```

## Step 5 — verify the output

- `_emit_stats.json` — deletion rate should be in the low single digits on ordinary
  photos; a spike means something is wrong upstream.
- Spot-check ~100 rows of `_audit.jsonl` against the images.
- Confirm counts: `ls labels/*.txt | wc -l` matches the labeled image count.

## Known gaps

1. **Batch API is not implemented** — sequential only. Batch halves the cost and is
   required at 500k scale (sequential ≈ 19 days for 1.25M detections).
2. **Gulf/traditional-dress bias is still untested** — no benchmark covers it; a
   hand-checked slice is the outstanding validation.
3. The `spotlight-e1` prompt awaits the Step 0 A/B.
4. Reconcile the first ~10k detections against the Google billing console to
   confirm the spend meter against a real invoice.
