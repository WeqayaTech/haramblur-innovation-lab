# HaramBlur — Innovation Lab

Experiment tracking and evaluation tooling for HaramBlur's person-detection / age-classification
/ gender-classification pipeline: the automated labeling pipeline that produces training data,
the models trained on it, and the benchmarks used to compare candidates.

**Start here: [`CLAUDE.md`](CLAUDE.md).** It's the map for this repo — mission, current best
model, and pointers to everything below. This README is just the quick-start.

## What's in this repo

This repo holds **source and documentation only** — code, experiment write-ups, and reference
docs. It does **not** hold data, model weights, or generated reports; those live on the shared
RunPod network volume (`/workspace/...`) and are reproducible from the tools here. See
`docs/DATASET_REGISTRY.md` for what's on the volume and `.gitignore` for what's deliberately
kept out of git.

| Where | What |
|---|---|
| `docs/EXPERIMENT_LOG.md` | Every experiment run, one row, sorted by date |
| `docs/CODEMAP.md` | Every script's purpose — check here before writing a new one |
| `docs/MODEL_COMPARISON.md` | Every model measured, exact run names + weights paths |
| `docs/DATASET_REGISTRY.md` | Every dataset on the volume, path + purpose + status |
| `experiments/EXP-YYYY-NN-*.md` | Full write-up per experiment (bar → method → result) |
| `vlm-cluster/` | The tools (labeling, scoring, training, reporting — see `docs/CODEMAP.md`) |

## Setup

GPU work and all volume access happen on a RunPod pod over SSH — there's no local data mount in
this repo. To work on the tooling itself:

```bash
git clone https://github.com/WeqayaTech/haramblur-innovation-lab.git
cd haramblur-innovation-lab
pip install -r vlm-cluster/requirements.txt
```

Most eval/scoring tools are CPU-only and self-test with zero data — a good first check:

```bash
python3 vlm-cluster/translation.py
python3 vlm-cluster/eval_negatives_crowd.py --selftest
python3 vlm-cluster/map_eval.py --selftest
```

Tools that call a commercial VLM (`describe.py --engine openai|gemini|claude`, `api_describers.py`)
read API keys from a local `.env` (not tracked — never commit it). On the pod, the same tools
source keys from the pod's own `.env`.

For actual dataset/GPU work, attach the RunPod network volume to a pod (L4 or A100 — avoid
Blackwell/sm_120, see `CLAUDE.md` → Infrastructure) and work over SSH; `docs/DATASET_REGISTRY.md`
has the full access pattern.

## Starting a new experiment

Copy `experiments/EXPERIMENT_TEMPLATE.md` to `experiments/EXP-YYYY-NN-<slug>.md`, write the
question and the pre-registered "good enough" bar *before* running anything, then add a row to
`docs/EXPERIMENT_LOG.md` once it's closed. Full conventions are in `CLAUDE.md`.
