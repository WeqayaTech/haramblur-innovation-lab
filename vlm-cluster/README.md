# HARAMBLUR — VLM description + clustering (slice discovery)

Replaces the brittle heuristic buckets with rich, accurate per-object
descriptions from a vision-language model, then clusters them to reveal the
natural slices in the dataset — including the rare ones you may need more of
("babies with chubby faces", "a man in a crowd").

Two stages, on purpose: the VLM is slow/expensive (run once, resumable);
clustering is cheap (re-tune freely).

```
describe.py  ──>  descriptions.jsonl + crops/      (Qwen2.5-VL, stage 1)
cluster.py   ──>  clusters.md + cluster_map.png + montages/   (stage 2)
```

## Why clusters, not "balanced" groups
We do NOT force balanced clusters. Clustering finds the *natural* groups; the
small ones are the signal — those are the under-represented slices. The report
sorts clusters smallest-first and flags any below `--small-frac`.

## Accuracy safeguards (VLM hallucinates otherwise)
- Output constrained to a fixed JSON schema; `"unknown"` always allowed.
- The YOLO class (Woman/Man/Child) is fed as a hint so the model anchors.
- Greedy decoding (`do_sample=False`); "describe only what is visible".
- Each object also gets a whole-image scene description (your chosen
  "crop + full-image scene" context), merged before embedding.
- **Verify**: every cluster ships a montage of its member crops — eyeball that
  the crops match the label.

## Install (in the POD)
```bash
pip install -r requirements.txt
# torch is usually already present from training; if not, install the CUDA build
# matching the pod before transformers/qwen-vl-utils.
```

## Full dataset on one GPU (resumable, with ETA)
Describe everything (`--max-crops 0` = no cap). Batched generation maximizes the
single GPU; the run is stop/resume-safe and prints live progress + ETA.

```bash
export HF_HOME=/workspace/hf-cache
PYTHONUNBUFFERED=1 python describe.py --yaml /workspace/open-images-v7/dataset.yaml \
  --split train --max-crops 0 --batch-size 16 --max-pixels 1003520 --out ./run_all
```
- **Stop anytime** (Ctrl-C). **Resume** by re-running the exact same command —
  already-described images are skipped *before* being decoded (fast at 500k).
- **Status line** each chunk: `done/total imgs (%) · crops · img/s · elapsed · ETA`.
- **Tune throughput:** raise `--batch-size` until GPU memory is ~full (watch
  `nvidia-smi`); 16–32 is typical on 24 GB. `--max-pixels` caps tokens/image to
  bound VRAM — lower it if you OOM, raise for more detail.
- **Check partial results anytime:** the reports read whatever exists so far:
  ```bash
  CUDA_VISIBLE_DEVICES="" python analyze.py --in ./run_all --out ./run_all/analysis
  ```
  then keep the describe run going to iterate over more data.

## Run — validation pass (~1.5k crops)
```bash
# Stage 1: describe (downloads Qwen2.5-VL-7B ~16GB on first run)
python describe.py --yaml /workspace/open-images-v7/dataset.yaml \
  --split train --max-crops 1500 --out ./run1

# Stage 2: cluster + report
python cluster.py --in ./run1 --out ./run1/clusters
```

Open `run1/clusters/clusters.md`.

## Analysis — start here (pairwise gaps)
`analyze.py` is the recommended first read: it counts every value and every
**pair** of values across all description fields, then ranks the sparsest
combinations as a "collect more of these" list. General and schema-agnostic — no
hardcoded slices.

```bash
CUDA_VISIBLE_DEVICES="" python analyze.py --in ./run2 --out ./run2/analysis \
  --thin 30 --top-montages 40
```
Open `run2/analysis/analysis.md`:
- **Single-field distributions** (+ `charts/*.png`) — is each field balanced?
- **Sparsest combinations** — ranked gap list, each linking a montage of example
  crops (`montages/*.jpg`) so you can verify the slice.
- **Pairwise cross-tabs** — full count matrix per field pair (thin cells ⚠️).

Tunables: `--thin N` (gap threshold), `--top-montages N`, `--fields a b c`
(restrict to specific fields). Multi-run: `--in ./run1 ./run2` (deduped by id).

> Deeper analysis (3+ way mining, ad-hoc slice queries, statistical "surprise",
> curated priority slices) is intentionally deferred — add as needed. The other
> scripts (`field_report.py` curated cross-tabs, `collection_needs.py` named
> slices, `cluster.py` embedding clusters) remain available as focused views.

## No-GPU dry run (validate plumbing)
```bash
python describe.py --yaml .../dataset.yaml --mock --max-crops 200 --out ./smoke
python cluster.py  --in ./smoke --embed mock --min-cluster-size 8
```

## Key flags
**describe.py**
- `--max-crops N` — cap new objects this run (`0` = all, no cap)
- `--max-images N` — cap images scanned (`0` = all)
- `--batch-size N` — crops/scenes per GPU generate call (raise to fill VRAM)
- `--max-pixels N` — cap pixels/image to bound VRAM (~1.0MP default; lower if OOM)
- `--chunk N` — images per status tick + write flush (default 64)
- `--pad 0.25` — context margin around each box
- `--mock` — synthetic descriptions, no GPU
- `--model` / `--device` — override VLM / device
- `--num-shards` / `--shard-id` — split across GPUs (one process each)
- resumable: re-run the same command; done images are skipped before decoding,
  and the status line shows `done/total · img/s · ETA`

**cluster.py**
- `--embed sbert|mock` (default sbert, `all-MiniLM-L6-v2`)
- `--min-cluster-size 15` — HDBSCAN granularity (smaller = more, finer clusters)
- `--small-frac 0.02` — clusters below 2% flagged as gaps
- `--montage-samples 12` — crops per cluster montage

## Outputs (stage 2, in `--out`)
- `clusters.md` — table (smallest-first), gap list, montages embedded
- `cluster_map.png` — 2D UMAP scatter colored by cluster
- `montages/cluster_<n>.jpg` — example crops per cluster (verification)
- `objects.csv` — per-object cluster assignment + canonical text
- `clusters.json` — machine-readable summary

## Tuning
- Schema fields & prompts: top of `describe.py` (`OBJECT_SCHEMA`, `SCENE_SCHEMA`).
- Cluster label/key fields: `KEY_FIELDS` in `cluster.py`.
- Too few/coarse clusters → lower `--min-cluster-size`.
