# Code map — what exists, so nothing gets rewritten twice

One row per script. **Before writing a new tool, ctrl-F this file for the verb you need**
(gallery, sweep, mAP, crop, dedup, gate...) — there is very likely already one.

Convention: `vlm-cluster/` is the local dev copy of the shared tooling; the pod keeps a
separate copy at `/workspace/data_inspection_tools/vlm-cluster/` (sync before running) or
`/workspace/autolabel_pipeline_v2/` for the production-facing subset. Everything below is
`vlm-cluster/<file>` unless noted. All are CLI scripts (`--help` works); most ship a
`--selftest`/`--selftest`-equivalent that runs with no GPU/network/pod data — run it before
assuming a tool is broken.

Status legend: **Active** = current, reused by newer work · **Historical** = ran once for a
closed experiment, kept as reference/reproducibility, not meant to be extended · **Frozen
core** = imported by many other scripts, treat signature changes as breaking.

## Core primitives (imported everywhere — read before duplicating logic)

| File | Purpose | Status |
|---|---|---|
| `run_model_children.py` | `MitModel` (production YOLO-MIT wrapper), `match_boxes`/`iou` (the box-matching used by ~15 other scripts), `find_model_config`. | Frozen core |
| `translation.py` | Maps raw `(gender, age)` → any label taxonomy (`production_3class`, `two_axis`, `gender_only`, `child_vs_adult[_9/_18]`, `age_band`...). Ground truth and model reads both pass through this. `python3 translation.py` selftests. | Frozen core |
| `two_axis.py` | The EXP-2026-17 two-axis taxonomy (Woman/Man/GenderUnknown × Adult/Child/AgeUnknown) + `collapse()` blur-policy projection back to 4-class. Imported by the two-axis trainer/emit/decode so they can't drift apart. | Frozen core |
| `dataset_utils.py`, `cluster.py` | Shared helpers (image/label IO, clustering) used across the Track 4 diversity tools. | Active |

## Step 1 — VLM crop description (label a YOLO dataset with a VLM)

| File | Purpose | Status |
|---|---|---|
| `describe.py` | Crops each labeled box, sends to a VLM (`--engine qwen\|openai\|gemini\|claude`), emits raw ~20-field JSON/person to `descriptions.jsonl`. Frozen prompt; `{cls}`-neutral so labels don't leak into the read. Resumable. | Active |
| `api_describers.py` | Commercial-VLM engine classes (`OpenAIDescriber`/`GeminiDescriber`/`ClaudeDescriber`) with per-call token/USD tracking. `model_pricing.json` ships `null` rates — fill in before trusting a $ figure. `--selftest`. | Active |
| `describe_hardneg.py`, `annotate_focus.py` | Describe the model's confident misclassifications specifically (hard-negative focus). | Historical (val-set audit era) |

## Step 2 — score reads vs ground truth

| File | Purpose | Status |
|---|---|---|
| `eval_taxonomy.py` | Scores model reads vs GT under chosen taxonomy schemes; accuracy/confusion/coverage + diagnostics (by-age accuracy, threshold sweep, gender-by-age). CPU-only. | Active |
| `label_errors.py` | Confirms label errors from hard-negative descriptions. | Historical |
| `vlm_contradictions.py` | Flags a VLM's self-contradictions (gender read vs attire cues). | Historical |
| `compare_classes.py`, `compare_gender.py`, `compare_child.py` | Label vs Model vs VLM 3-way audits, val-set era (pre-EXP-framework). | Historical |

## Production-model / auto-labeler scoring (EXP-2026-02, 12, 13, 14, 16, 19...)

| File | Purpose | Status |
|---|---|---|
| `run_autolabel_on_manifest.py` | Scores production SAM3 auto-labeler's own YOLO output vs LAGENDA GT (EXP-2026-02). Also: `child_age_diagnostics` (implicit child/adult cutoff), `crowd_diagnostics`, dual-class NMS-leak split. `--selftest`. | Active |
| `run_ultralytics_labels.py` | THE Stage-A runner for any Ultralytics model (YOLO26/YOLOE incl. `--distractors`) OR production YOLO-MIT (`--engine mit`). Emits YOLO labels + log-raw JSON sidecars (conf ≥0.05). `--no-e2e` reads the one-to-many head. Exports `UltralyticsModel` for the flicker probe. `--selftest`. | Active |
| `eval_negatives_crowd.py` | Scores a model on PASS/object-set (FP counting) and CrowdHuman (recall/precision, occlusion split) via `seg_boxes` polygon parsing. | Active |
| `map_eval.py` | Self-contained COCO-style mAP (AP50/75/50-95, 101-pt interp, per-class+mean, `--ignore-labels`, `--stem-prefix` for multi-size arms) from raw sidecars. `--selftest`. | Active |
| `conf_sweep.py` | Confidence-threshold optimizer — replays log-raw sidecars offline (no re-run), `--objective`/`--constraint` as arithmetic over metric names, `--mode per-class`, `--pareto`, `--crosscheck` (proven number-for-number vs `eval_negatives_crowd.py`+`run_autolabel_on_manifest.py`), `--out` writes standalone SVGs, `--report` bundles HTML. `--selftest`. | Active |
| `percurve.py`, `exp19_summary.py` | EXP-2026-19 result readers: per-class AP at apparent sizes, Woman/Child end-to-end curves. Re-readable with no pod. | Active |
| `sam_fp_conf_report.py`, `conf_zoom_report.py` | EXP-2026-07: SAM3 confidence histograms + gate decision table; FP-vs-verified-TP separation. `--selftest`. | Historical |
| `make_object_negatives.py` | Builds the hand-verified doll/statue/toy negative set (EXP-2026-03). | Historical |

## Galleries / visual reports (HTML, self-contained, base64 thumbs)

| File | Purpose | Status |
|---|---|---|
| `build_error_gallery.py` | Wrong classifications, ambiguous/near-miss, missed detections, dual-class NMS leaks, unmatched-box CROWD tags. `--selftest`. | Active |
| `build_detect_gallery.py` | Whole-image VLM detection boxes vs GT (EXP-2026-06); model dropdown, `--engines` filter. `--selftest`. | Historical |
| `build_gallery_exp04.py` | EXP-2026-04 error galleries from existing Stage A outputs, no re-run. | Historical |
| `build_holdout_gallery.py` | Visual proof gallery for `haramblur_holdout`. | Active (holdout is a live benchmark) |
| `build_smallperson_gallery.py` | EXP-2026-19 `--report crowd\|synthetic` galleries; women pixelated by construction. `--selftest`. | Active |
| `trace_report.py` | Stage-by-stage HTML trace of the Spotlight pipeline on N images; `--seed` for unbiased sampling. | Active |
| `survivor_gallery.py` | EXP-2026-08 gate survivors (`--all` shows rejected crops too). | Historical |
| `shiekh_adjudicate_gallery.py` | Unredacted internal sheet for hand-adjudicating shaykh-context GT-women rows. | Historical (adjudication done 2026-08-27) |
| `tier_gallery.py` | Exposure-tier visual inspection (→ `T3_MEN_GALLERY.html` etc). | Active |
| `headtohead_report.py` | EXP-2026-09 evidence report: every headline claim paired with visual proof. | Historical |
| `pipeline_v1_report.py` | Team-facing pilot report for the Spotlight/pipeline-v1 arms. | Historical |
| `child_rescue_viewer.py` | Eyeball the children the `--child-by-age` rule brings back into training. | Active |
| `subset_verify.py` | Hard counts + redacted gallery for `deploy_compare` materialized subsets. `--selftest`. | Active |

## Whole-image commercial-VLM detection (EXP-2026-06/09)

| File | Purpose | Status |
|---|---|---|
| `vlm_detect_eval.py` | Sends whole images to a commercial VLM with one frozen detection prompt; scores vs PASS/object-set FP, CrowdHuman, LAGENDA. Emits AP + box-tightness. `--selftest`. | Historical (EXP-06/09 closed; SAM3 kept the detector seat) |
| `reparse_boxes.py` | Re-parses saved `raw_text` under any box convention (`xyxy_px`/`xyxy_1000`/`yxyx_1000`), rescoring with no API re-call; `diagnose` mode. `--selftest`. | Active (useful any time a VLM's box convention is in question) |
| `crowd_headtohead.py` | SAM3 vs Gemini Flash-Lite head-to-head on the same 100 crowd images (EXP-2026-09). `--selftest`. | Historical |
| `recompute_costs.py` | Re-multiplies stored `cost_report.json` token counts by current `model_pricing.json` rates, no API calls. `--selftest`. | Active |

## Spotlight pipeline (production labeling: detect → verify → merge → emit)

| File | Purpose | Status |
|---|---|---|
| `autolabel_sam_raw.py` | Patched copy of production `autolabel_sam.py`: byte-identical YOLO output + a raw sidecar (per-part mask polygons pre-flattening, raw conf, NMS-suppressed dets). Has the bbox-prefilter NMS perf fix and the `.webp`/uppercase-extension fix, neither of which is in the production script. | Active |
| `spotlight_run.py` | `build_crop` (mask-highlighted crop, pure fn of image+box+parts) + `PROMPT` (the frozen verdict schema) — imported by everything downstream, including cross-VLM comparisons, so crops stay byte-identical to what Gemini saw. Sequential runner. | Frozen core |
| `spotlight_batch.py` | Batch-API submit/collect version of the same pipeline; imports `PROMPT` from `spotlight_run` (sends it verbatim) — but does NOT write `run_meta.json`/`cost_report.json` and stamps an empty `prompt_sha` (known gap, see `mk_runmeta.py`). Submit sharding (`--shards/--shard-index`) + reservation rollback on upload failure. | Active |
| `parallel_emit.py` | Parallel (24-worker) emit from verdicts → cleaned YOLO labels; `--unknown-class N`; selftested byte-identical vs the serial reference. This is THE emit tool for anything corpus-scale. | Active |
| `gate_eval.py` | Scores the Flash-Lite verifier gate's TP-keep / FP-kill against ground-truth-verified crops (EXP-2026-08). No network needed for `--selftest`. | Active |
| `pipeline_v1_eval.py` | Runs/scores the full assembled pipeline v1 (`run`/`score`/`compare`/`selftest`). | Historical (superseded by production `parallel_emit.py`+`spotlight_batch.py` path) |
| `verify_labels.py`, `verify_datasets.py` | Post-hoc verification: emitted Spotlight labels are internally consistent; staged datasets on the volume match `DATASET_REGISTRY.md`'s expected state. | Active |
| `estimate_cost.py` | Accurate cost/time estimate for labeling a target corpus with Spotlight, before running it. | Active |
| `mk_runmeta.py`* | *(referenced in CLAUDE.md, lives under `labeling/tools/` per-dataset, not `vlm-cluster/`)* Reconstructs `run_meta.json`+`cost_report.json` for Batch-API runs that never wrote them, by hashing the live `PROMPT` constant. | Active |

## Crowd-scene track (SAM-outlined masks on human GT boxes)

| File | Purpose | Status |
|---|---|---|
| `crowd_sam_masks.py` | SAM-ViT-huge box-prompted masks on human GT boxes (fixes the box-outline-labels-the-wrong-neighbor failure in dense crowds). | Active (crowd set is parked, tool is ready) |
| `crowd_classify.py` | Runs the Spotlight verdict step on the crowd-mask crops. | Active (parked) |
| `crowd_trace.py` | Stage-by-stage trace for the crowd track. | Active (parked) |

## Training (custom losses on top of stock Ultralytics)

| File | Purpose | Status |
|---|---|---|
| `train_gradsuppress.py` | Cls-gradient-suppression trainer for YOLO26n on `labels_unk3` (class 3 supervises box/DFL/assignment, zeroed from BCE). Canonical-import main + `--export-plain` (see memory `ultralytics-custom-class-checkpoints` for the 3 pickling traps this guards against). | Active — current best-model lineage starts here |
| `train_gradsuppress_yoloe.py` | GS port to YOLOE's PE seg fine-tune. | Historical (YOLOE line closed, EXP-2026-15) |
| `train_twolabel.py` | Multi-label two-axis trainer (patches `TaskAlignedAssigner` to multi-hot per anchor). | Historical (EXP-2026-17 not adopted, but mechanism is validated — reusable if two-axis is revisited) |
| `train_small_object_patches.py` | `SmallObjectPatches` augmentation (paste-shrunk-objects-onto-grey trick) continuing a warm-restart fine-tune. This produced the current best model, `y26s_humanshaped_smallpatch_v1`. | Active — current best-model lineage |
| `wandb_mirror.py` | Mirrors a live Ultralytics run's metrics into W&B without touching the run. | Active |
| `track_small_object_progress.py` | Periodic progress tracker for a live training run against the small-object benchmark. | Active |

## EXP-2026-17 (two-axis) support

| File | Purpose | Status |
|---|---|---|
| `exp17_phase0_counts.py` | Phase-0 gate: streams `verdicts_batch.jsonl`, checks children keep a usable gender + AgeUnknown band size, before any GPU spend. `--selftest`. | Historical |
| `exp17_check_pairs.py` | Checks where two-axis label pairing breaks (three levels). | Historical |
| `twoaxis_report.py` | Per-axis scoring + HTML predicted-vs-label report with error-cause decomposition. | Historical |

## Small-person benchmark (EXP-2026-19)

| File | Purpose | Status |
|---|---|---|
| `build_small_person_set.py` | Builds the benchmark: `census`/`mine`/`synth`/`paste` subcommands (real-image, shrink-synthetic, and SAM-cut-paste-on-grey arms). Emits images/labels/ignore/persons.odgt the existing scorers read unchanged. `--selftest`. | Active |
| `verify_small_person_set.py` | Re-reads emitted files (never the manifest), writes `VERIFY.html` + `size_hist.svg`. `--selftest`. | Active |
| `scale_robustness.py` | Shrinks real images progressively, measures per-model degradation — the general-purpose version of the small-person question. | Active |

## Deployment ship-table (`deploy_compare` — the current flagship comparison tool)

| File | Purpose | Status |
|---|---|---|
| `deploy_compare.py` | `curate`/`materialize`/`score`/`selftest` subcommands. Customer-experience metrics (E-img, E-person, FB-area) per viewer audience, rect-union coverage, bootstrap CIs. THE ship-table tool. | Active |
| `deploy_charts.py` | Standalone SVGs from a `deploy_compare` score summary. | Active |
| `deploy_report.py` | One self-contained findings HTML: ship table, per-category mAP, exposure tables, escape decomposition, evidence galleries. Women always redacted. `--selftest`. | Active |
| `audience_charts.py` | Standalone per-audience threshold SVGs. | Active |
| `woman_threshold.py` | Single-threshold selection sweep for the woman/man class, per audience. | Active |
| `women_report_final.py` | Audience-parameterized (`--audience woman\|man`) coworker-facing final report generator (→ `WOMEN_REPORT_FINAL.html`/`MEN_REPORT_FINAL.html`). | Active |
| `report_to_md.py` | Converts a findings HTML report to Markdown for ClickUp/doc pasting. | Active |
| `threshold_decision_report.py` | Self-contained HTML threshold-decision writeup for coworkers. | Active |
| `lagenda_map_audit.py` | Audits/demonstrates the LAGENDA v2 mAP setup — trust-focused double-check tool. | Active |

## Video (EXP-2026-11 temporal/flicker)

| File | Purpose | Status |
|---|---|---|
| `video_flicker_probe.py` | Runs a model frame-by-frame over video, raw sidecar JSONL at conf floor 0.05 (all smoothing policies replay offline). `--engine ultralytics` for any candidate model. Resumable, `--mock` for plumbing tests. | Active |
| `flicker_metrics.py` | IoU track builder + flicker numbers (blur toggles/min, gap-cause split, conf straddle, jitter, class-flip matrix, exposure seconds) + `--render` annotated mp4. `--selftest`. | Active |

## Dataset build / provenance

| File | Purpose | Status |
|---|---|---|
| `build_lagenda_full.py` | Rebuilds LAGENDA v2 from the official CSV: `labels/` (all boxes), `gt.jsonl`, `persons.odgt`, `labels_3class/`+`ignore/`. `--exclude-list` drops train-contaminated images. `--selftest`. | Active |
| `build_training_set.py` | Assembles the production training image set with full provenance tracking. | Active |
| `build_overlay_labels.py` | Builds a training label dir as an overlay on an existing one, without copying it. | Active |
| `child_rescue_report.py` | How many children did a Spotlight emit throw away for having no gender (recoverable via `--child-by-age`). | Active |

## Track 4 — dataset diversity / balance

| File | Purpose | Status |
|---|---|---|
| `field_report.py`, `analyze.py`, `collection_needs.py` | VLM-description-based balance + gap analysis over a `describe.py` run. | Not run recently |
| `dataset-diversity-audit/diversity_audit.py` (separate directory, own README) | CV-only diversity spike: skin tone, scale, quality, optional CLIP view/dress. | Not run recently |

## Chart generation (`experiments/`, not `vlm-cluster/`)

| File | Purpose | Status |
|---|---|---|
| `experiments/make_charts.py`, `make_charts_exp02.py`, `..._exp03.py`, `..._exp04.py`, `..._exp09.py`, `..._exp10.py`, `..._exp12.py` | Generate standalone SVG charts from hardcoded eval numbers, one file per experiment; some also do the SVG→PNG Chrome-headless conversion for ClickUp. Pattern: copy the most recent one (`make_charts_exp12.py`) rather than starting from scratch. | Active (one-off per experiment, expected to keep growing — this is normal, not debt) |

---

**Maintenance rule:** when you add a new script, add one row here in the same pass (same PR/session) — don't let this drift back into prose. When a script is superseded, change its Status rather than deleting the row (the old tool is still often the correct reference for "how did we do X before").
