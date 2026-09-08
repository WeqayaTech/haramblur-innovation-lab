# HaramBlur — Innovation Lab (context for Claude Code)

**How to use this file:** this is a map, not a log. It stays short on purpose. When you need the
detail behind any line here, follow the pointer — don't re-derive it, and don't paste more
history back into this file. See "Where to find things" below before reading (or writing)
anything else.

## The goal (keep this in view for every decision)

> **HaramBlur helps a viewer lower their gaze (ghaḍḍ al-baṣar).** It blurs the **opposite adult
> gender** — adult women for male viewers, adult men for female viewers — so the viewer isn't
> shown the adults they're meant to look away from. The purpose is **Islamic/religious, not
> content-safety or moderation.** **Children are deliberately left unblurred** — they are not a
> gaze-lowering target.

**Two failure axes, both matter, lead with the first when framing results (owner correction,
2026-08-08):**

1. **False positives on non-people** (blurring books/statues/backgrounds) — the #1 user
   complaint and the actual reason the relabeling/retraining effort is funded. A product that
   blurs junk gets disabled, which protects nobody.
2. **Adults escaping the blur** — missed, mis-aged (read as Child), or mis-gendered. This is the
   *consequential* error direction, not "catch every child": over-blurring a ~12-year-old is
   acceptable, an adult reading as Child is not, because that adult then escapes the blur. Don't
   frame child recall as a "safety floor" — wrong connotation.

Every decision in this repo should be judged against both axes together, never one alone.

## What this project is

HaramBlur classifies each detected person as `{Woman, Man, Child}` (production classes
`{0,1,2}`) and blurs by gender. The model is trained on images labeled by an **automated
labeling pipeline** — if those labels are wrong, training quality suffers. This repo designs,
audits, and tests that labeling/verification system, then evaluates candidate detector/classifier
models against it.

**Known open weakness:** men in Gulf/traditional dress (thobe, ghutra) get misread as women.
Partially measured — see `docs/DATASET_REGISTRY.md` → `haramblur_holdout` (the `shiekhs` slice)
and `docs/EXPERIMENT_LOG.md` EXP-19. Hand-confirmation of the specific misread crops is still
open.

## Current state (edit this in place — do not append a new dated block, update the facts)

*Last updated: 2026-09-08.*

- **Best measured model:** `y26s_humanshaped_smallpatch_v1` —
  `/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt`. Best on every
  mAP/holdout/crowd-recall number measured so far. Not yet shipped.
- **Currently shipped in production:** `yolo11N-640` (`v11nclean2` export) —
  `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt`. Worst small-person recall and
  highest object-set FP rate of any model measured — the gap this whole model-eval line exists
  to close. Full comparison: `docs/MODEL_COMPARISON.md`.
- **Latest completed experiment:** EXP-2026-20 (humanshaped labeling policy + the model above).
  Full index: `docs/EXPERIMENT_LOG.md`.
- **Primary training data:** Spotlight-labeled full OIV7 train split (1.19M kept people) —
  `docs/SPOTLIGHT_PRODUCTION_RUN.md`. Primary eval sets: `haramblur_holdout` (QA/deploy
  comparisons, incl. the Gulf-dress slice), LAGENDA v2 + `fl1199` (human-labelled 3-way),
  `smallperson_v1` (small-person benchmark), CrowdHuman/PASS/object_set (detection FP/recall).
  All in `docs/DATASET_REGISTRY.md`.
- **Standing open threads** (see "Open items" at the bottom of this file for the full current
  list): Gulf-dress slice still not hand-confirmed at the crop level; API keys that passed
  through chat still need rotating; crowd-scene track (`crowd_1k`) still parked.

## Where to find things

**Read the relevant row, not the whole file — every doc below is small and single-purpose.**

| I want to know... | Go to |
|---|---|
| What experiments have been run, when, and what they found | `docs/EXPERIMENT_LOG.md` |
| Whether a tool/script for X already exists (check before writing a new one) | `docs/CODEMAP.md` |
| What's the current best model / how models compare | `docs/MODEL_COMPARISON.md` |
| What datasets exist, where on the volume, what they're good for | `docs/DATASET_REGISTRY.md` |
| The 3-component evaluation framework + scoreboard | `docs/COMPONENT_FRAMEWORK.md` |
| Deep pre-EXP-framework history (why the 3-pipeline split, early audits) | `docs/AUDIT_HISTORY.md` |
| The full production labeling run (what/how/traps) | `docs/SPOTLIGHT_PRODUCTION_RUN.md` |
| How to benchmark any model against the holdout set | `docs/HOLDOUT_BENCHMARK_HANDOFF.md` |
| Recurring pod/infra traps (quotas, threading, SSH) | this file's "Infrastructure" section, and Claude's own memory (`~/.claude/.../memory/`) — most are already captured there |
| Team-facing writeups already sent out | `experiments/*-TEAM-BRIEF.md`, `experiments/*-CLICKUP-READY.md`, `docs/REPORT_*.md` |

## The evaluation framework: three components

Every system is measured against one or more of three named components — never pool across them
or across datasets. Full rules + cross-experiment scoreboard: **`docs/COMPONENT_FRAMEWORK.md`**.

1. **Component 1 — Person Detection** (box every person; posters/photos of real people count,
   dolls/statues/mannequins historically didn't — **policy changed 2026-09-05, see EXP-20**:
   human-shaped objects now count as gaze-lowering targets too).
2. **Component 2 — Age group classification** (Adult vs Child; headline metric is the
   adult→Child leak rate — the direction that lets adults escape the blur).
3. **Component 3 — Adult gender classification** (Man vs Woman, decided by domain slice; the
   Gulf-dress slice is the standing gap, see "Current state" above).

## Repo structure

```
Innovation-lab/
├── CLAUDE.md                    # this file — map only, not a log
├── docs/
│   ├── EXPERIMENT_LOG.md        # every EXP, one row, sorted by date — START HERE for history
│   ├── CODEMAP.md               # every script's purpose — START HERE before writing new code
│   ├── MODEL_COMPARISON.md      # every measured model, exact run names + weights paths
│   ├── DATASET_REGISTRY.md      # every dataset on the volume, path + purpose + status
│   ├── COMPONENT_FRAMEWORK.md   # the 3-component evaluation framework + scoreboard
│   ├── AUDIT_HISTORY.md         # pre-EXP-framework history (val-set audits, pipeline design)
│   ├── SPOTLIGHT_PRODUCTION_RUN.md   # the full OIV7 production labeling run
│   ├── HOLDOUT_BENCHMARK_HANDOFF.md  # how to score any model against haramblur_holdout
│   ├── HOLDOUT_PATHS.md / LAGENDA_*.md / SAM3_OUTPUT_HANDOFF.md  # dataset-specific handoffs
│   ├── MODEL_EVAL_OVERVIEW.md / MODEL_EVAL_PROTOCOL.md  # method one-pagers
│   ├── TFJS_EXPORT_RECIPE.md / AUGMENTATION_CODE.md / TRAINING_CONFIG_HISTORY.md
│   └── PIPELINE_*.md / SPOTLIGHT_PIPELINE_OVERVIEW.md / AUTOLABEL_PIPELINE_V2_RUN.md
├── experiments/
│   ├── EXPERIMENT_TEMPLATE.md   # copy this to start a new EXP
│   ├── EXP-YYYY-NN-<slug>.md    # one full writeup per experiment (bar → method → result)
│   ├── EXP-YYYY-NN-POD-RUNBOOK.md / -TEAM-BRIEF.md / -CLICKUP-READY.md  # supplementary
│   └── make_charts*.py, assets/  # chart generation, one script per experiment
├── vlm-cluster/                 # the tools — see docs/CODEMAP.md for what each does
└── dataset-diversity-audit/     # Track 4: CV-only diversity spike (own README)
```

## Infrastructure (read before running anything)

- **GPU work runs on a RunPod pod**, not locally. Pods are ephemeral; the **network volume
  persists** at `/workspace`. This project's pod code copy: usually
  `/workspace/data_inspection_tools/vlm-cluster/` (sync from local `vlm-cluster/` before
  running — the two can drift). Production-facing subset also lives at
  `/workspace/autolabel_pipeline_v2/`.
- **Dataset convention:** one folder per dataset under `/workspace/datasets/<name>/`, with a
  `README.md` manifest and a row in `docs/DATASET_REGISTRY.md`. Never put the string
  `"negative"` in a dataset folder name — `autolabel_sam.py` silently skips such paths.
  Experiment-specific outputs (not reusable datasets) live under `/workspace/expNN/`.
- **All volume access is over SSH on the pod** — connect via the RunPod console (Web Terminal,
  or SSH over exposed TCP for VS Code Remote-SSH), attach the shared network volume to any pod
  in its datacenter, and transfer files with scp or heredoc. There is no local mount tool in
  this repo (removed 2026-09-08 — it was WebDAV-based and too slow for real use).
- **Picking a pod:** explicitly choose an L4 or A100 — Blackwell (sm_120) pods need a specific
  cu128 torch build (see memory `blackwell-pod-torch-cu128`). Fresh pods also ship a
  torch/transformers mismatch that breaks SAM3 import — reinstall
  `torch torchvision torchaudio` from the cu124 index before any GPU work.
- **`nproc`/`uptime`/`vmstat` report the HOST inside a RunPod container**, not your allocation —
  pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (and `cv2.setNumThreads`) or throughput craters.
- **RunPod can kill detached jobs on an SSH drop** (nohup/setsid don't always survive it) — hold
  the connection open (`ServerAliveInterval`) and checkpoint long runs.
- **Volume quota is invisible to `df`** and the Gemini File API has its own 20 GiB cap — both
  fail silently mid-run. See memory `runpod-volume-and-gemini-storage-caps`.
- CPU-only tools (`eval_taxonomy.py`, `translation.py`, `map_eval.py`, `conf_sweep.py`, the
  Track 4 report tools, everything under "Deployment ship-table" in `docs/CODEMAP.md`) run
  anywhere, no pod needed — most are `--selftest`-able with zero data.

Most other operational traps hit once and fixed are captured as Claude's own persistent memory
(`~/.claude/projects/.../memory/`) rather than duplicated here — check there for anything
pod/RunPod/training-specific that isn't listed above.

## Conventions (timeless — don't add dated entries here, that's what EXPERIMENT_LOG.md is for)

- **Starting a new experiment:** copy `experiments/EXPERIMENT_TEMPLATE.md` to
  `experiments/EXP-YYYY-NN-<slug>.md`. Write the question + pre-registered bar **before**
  running — don't move goalposts after seeing results. **Add its row to
  `docs/EXPERIMENT_LOG.md` in the same session you close it** — that file existing to fall
  behind (as it did for three weeks with EXP-18) is the failure mode it exists to prevent.
- **Before writing a new script, check `docs/CODEMAP.md`.** Reuse the frozen-core imports
  (`run_model_children.match_boxes`, `translation.py`, `spotlight_run.build_crop`+`PROMPT`)
  rather than reimplementing them — several bugs in this project's history were duplicated
  matching/scoring logic drifting out of sync with the original.
- **Every experiment needs a "verify it yourself" appendix**, written before running: verbatim
  code (not prose) for how the model was called, how matching was done, how samples were
  selected, and every threshold + its source line.
- **Say the full model name** ("Qwen2.5-VL-7B", not "Qwen") every time.
- **Be the harshest reader of your own result** — state explicitly what a result does NOT prove
  (e.g., a LAGENDA result says nothing about the Gulf-dress bias).
- **Don't select or validate a model using labels you don't already trust** (circular) — use
  external human-annotated ground truth (LAGENDA, CrowdHuman) for model selection; reserve our
  own hand-checked slices for what public benchmarks structurally can't cover.
- **Full run/directory names in comparison tables, never invented aliases.**
- **Charts:** standalone self-contained `.svg`, one graph per file, not a bundled HTML report
  (convert to PNG only for ClickUp).
- **Verify pod-only claims before acting on them** — a lot of state only exists on the volume
  and is recorded here from conversation, not committed to this repo.

## Open items (current — replace/remove lines as they close, don't let this list grow forever)

- **Gulf-dress bias: still not hand-confirmed at the crop level.** The measured 4.94% SAM3
  misread rate (`haramblur_holdout` `shiekhs` slice) and the 30,567 Woman→Man Spotlight
  corrections are both ready-made samples for this — see `docs/EXPERIMENT_LOG.md` EXP-19 and
  the holdout row in `docs/DATASET_REGISTRY.md`.
- **Rotate the API keys that passed through chat** during the production run (now carrying
  $500+ of billing history).
- **Crowd-scene track (`crowd_1k`) is parked** pending an owner decision on training impact;
  tooling is ready (`crowd_sam_masks.py`/`crowd_classify.py`/`crowd_trace.py`, see
  `docs/CODEMAP.md`).
- **EXP-2026-05 (MiVOLO V2 age-component experiment) has never been run** — the harness
  (`run_autolabel_on_manifest.py`/`eval_taxonomy.py`) is ready as-is.
- **`object_set` needs a relabel/re-adjudication** after the 2026-09-05 humanshaped policy
  change — its ground truth still scores "any detection = FP," which now double-counts
  correct human-shaped-object detections as errors.
- **EXP-2026-19's human eyeball pass on `VERIFY.html`** (crowd_small + paste_grey arms) still
  hasn't happened — local copies in `_smallperson_review/`, no pod needed.
- Video temporal-flicker fix (dual-threshold sustain / sticky class vote, designed in EXP-11)
  has not been built or replayed against any candidate model yet.
