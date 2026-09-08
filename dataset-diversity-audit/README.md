# HARAMBLUR — Dataset Diversity Audit (spike)

One-time tool to quantify the variance of the **Woman / Man / Child** training
set (Open Images V7, YOLO format) and produce a confidence signal about coverage
and gaps. Built to run **inside the POD** where the volume is mounted at
`/workspace/open-images-v7/`.

## What it measures

Every labeled box (Woman/Man/Child — all are person types for HaramBlur) is
cropped and scored on:

| Dimension | How | Deps |
|---|---|---|
| **class_name** (Woman/Man/Child) | the real label — exact counts | core |
| **skin_tone** | ITA° over heuristic skin pixels → Fitzpatrick band | core |
| **scale / distance** | bbox area ÷ image area | core |
| **brightness / sharpness / resolution** | capture-quality stats | core |
| **view_angle** | frontal / profile / back (CLIP zero-shot) | optional |
| **dress_code** | modest / casual / revealing / … (CLIP zero-shot) | optional |

Per dimension it reports the distribution + an imbalance score (normalized
entropy and Gini), then rolls up to one **diversity confidence** number
(1.00 = perfectly balanced) and a ranked list of under-represented buckets.

> Ethnicity/nationality is **not** inferred per face (unreliable + sensitive).
> Skin-tone distribution is the diversity proxy instead.

## Install (in the POD)

```bash
pip install -r requirements-audit.txt
# torch is already present from training; open_clip installs on top.
# Skip CLIP entirely with --no-clip and you only need numpy/opencv/pyyaml/matplotlib.
```

## Run

```bash
# fast core-only pass on the val split
python diversity_audit.py \
  --yaml /workspace/open-images-v7/dataset.yaml \
  --split val --no-clip --out ./audit_val

# full audit including view-angle + dress-code (GPU recommended)
python diversity_audit.py \
  --yaml /workspace/open-images-v7/dataset.yaml \
  --split train --max-images 8000 --out ./audit_train
```

### Key flags
- `--split train|val` — which split (reads `train:`/`val:` from the yaml)
- `--max-images N` — sample N images (`0` = all; the train split is large)
- `--max-crops N` — hard cap on analyzed crops
- `--no-clip` — skip view-angle & dress-code (no torch needed)
- `--out DIR` — output directory

## Output (in `--out`)
- `report.md` — human-readable report with charts embedded
- `metrics.json` — raw distributions + scores (for diffing snapshots)
- `chart_<dimension>.png` — one bar chart per dimension
- `debug/` — example-crop montages per bucket + `index.md` (see below)

## Verifying accuracy (debug montages)
The estimated dimensions (`skin_tone`, `dress_code`, `view_angle`) are not
deterministic — always eyeball them. By default the tool saves up to 12 example
crops per bucket as a labeled contact sheet:

- `debug/index.md` — every bucket with its % and an embedded montage
- `debug/<dim>__<bucket>.jpg` — the crops, each captioned with its measured
  value: `ITA 35` (skin tone), `12.4%` (scale), brightness/sharpness numbers,
  `350px` (resolution), or `p=0.78` (CLIP confidence for dress/view)

Open `debug/index.md` and scan, e.g., `dress_code__revealing.jpg` to confirm the
crops really are what the label claims. Crops are reservoir-sampled, so they're
representative rather than just the first N.

```bash
--debug-samples 20   # more examples per bucket
--debug-samples 0    # disable (slightly faster, smaller output)
```

Low CLIP confidence (`p` near 1/num_buckets) on many crops in a bucket is a sign
the prompts need tuning for your taxonomy.

## Tuning
All buckets and the CLIP prompts are plain constants at the top of
`diversity_audit.py` — edit `DRESS_CODE_PROMPTS`, `SCALE_BANDS`,
`FITZPATRICK_BANDS`, etc. to match the taxonomy you care about.

## Caveats
- **Scale ≠ literal distance** — a close-up face and a far full body both move it.
- **Skin tone** is an estimate; crops with no visible skin are excluded.
- **CLIP dims** are zero-shot — directional, not ground truth. Tune prompts.
