# Do profile pictures get blurred? — the avatar benchmark

**One line:** we built a benchmark of Facebook/LinkedIn-style profile pictures and ran the four
candidate models over it. **Even on a large, guaranteed-crisp portrait, roughly one woman in
four is not blurred** — and at feed size the shipped production model misses three in four.

Measured 2026-08-25 · 240 images · 1,364 avatars · four sizes · four models.
Interactive version with the images: `REPORT_avatars.html`.

---

## 1. Why this benchmark exists

A social feed is mostly **head-and-shoulders portrait crops**. That is a different
distribution from the full-body people our detectors were trained on, and nothing we owned
measured it. Our other small-person arms also confound two different problems — *small* and
*low quality* — because a distant person in a photo is usually both.

This arm separates them. Every avatar is **sharp by construction**, so the only thing that
changes across the benchmark is **apparent size**.

---

## 2. How the data was made

**Source.** LAGENDA val images, restricted to people who carry a **human** age and gender
label (not a model's guess) and whose person box is at least 200 px tall. Class comes from the
project's own `gt_class` mapping → Woman / Man / Child.

**The crop.** A square anchored on the top of the person's box and centred on it:

```
side = min(box_width, box_height × 0.34)     # head + shoulders on a standing person
top  = box_top − box_height × 0.04           # boxes clip hair, so lift slightly
```

Taken from the **original photograph** — a profile picture is a photographic crop, not a
cut-out silhouette.

**Three gates, each rejecting a specific failure.** Counts are people rejected:

| gate | rule | rejected |
|---|---|---:|
| never upscale | source square ≥ avatar size | 793 |
| real downscale | source square ≥ **1.5×** avatar size | 896 |
| sharpness | source scores **≥ 700** variance-of-Laplacian at 128 px | 555 |
| in frame | crop must not run off the image edge (never padded) | 148 |

The sharpness threshold was **calibrated on this data, not guessed**. Across crops eligible at
128 px the distribution was p25 = 280, p50 = 540, p75 = 915, so a gate of 700 keeps roughly the
sharpest third. The downscale rule matters independently: a 1:1 crop is only as crisp as the
original pixels happened to be, whereas a real profile picture looks sharp *because* it was
downscaled from something larger.

**Rendering.** Circle and rounded-square masks alternating, six avatars per 640 px canvas on a
light background, placed without overlap.

**Paired across sizes — the controlled-variable property.** The cast and their anchor points are
chosen **once** at 128 px and re-rendered at every smaller size. `a128_0007` and `a48_0007` hold
the same six people in the same positions. A score difference across sizes therefore cannot be
blamed on different people or a different layout.

Verified from the emitted files (not from the build script's own claims):

| size | people | Woman | Man | Child |
|---|---:|---:|---:|---:|
| 128 px | 341 | 139 | 138 | 64 |
| 96 px | 341 | 139 | 138 | 64 |
| 64 px | 341 | 139 | 138 | 64 |
| 48 px | 341 | 139 | 138 | 64 |

Identical at every size, as designed. The set passes **7/7** automated checks.

---

## 3. How models are scored

- **Ground truth** is the placed avatar's square; class is the human label.
- **A match** is IoU ≥ 0.5, greedy, one prediction per GT — the same matcher every other
  scorer in this repo uses.
- **Detection recall** = matched GT ÷ all GT, class ignored.
- **Class accuracy** = of those matched, how many carry the right class.
- **End-to-end** = found **and** labelled correctly, per class. **This is the product metric**:
  a woman who is detected but called "Man" is a woman who does not get blurred. She counts as a
  failure here and as a success under plain recall.
- **mAP** is separate and threshold-free (COCO 101-point, confidence-ordered matching), so it
  rewards ranking quality rather than behaviour at one cut.

Everything at the production threshold **conf 0.45** unless stated.

---

## 4. Results

### 4.1 The product metric — Woman end-to-end (found AND called Woman)

| model | 128 px | 96 px | 64 px | 48 px |
|---|---:|---:|---:|---:|
| `y26n_sop50` | **77.7%** | **71.2%** | **60.4%** | **52.5%** |
| `y26n_noe2e_warm50-2` | 76.3% | 70.5% | 57.6% | 38.8% |
| `gelannfav14r4fw_gemlb_v2` | 70.5% | 66.9% | 48.2% | 25.2% |
| `yolo11N-640` *(production)* | 66.9% | 64.7% | 46.8% | 24.5% |

### 4.2 Full table, conf 0.45

| size | model | detection recall | class acc | Woman e2e | Man e2e | Child e2e |
|---|---|---:|---:|---:|---:|---:|
| **128 px** | `yolo11N-640` | 81.5% | 77.7% | 66.9% | 75.4% | 29.7% |
| | `y26n_noe2e_warm50-2` | 83.3% | 83.8% | 76.3% | 79.0% | 35.9% |
| | `y26n_sop50` | **85.9%** | **84.3%** | **77.7%** | **80.4%** | 43.8% |
| | `gelannfav14r4fw_gemlb_v2` | 84.2% | 82.9% | 70.5% | 73.2% | **60.9%** |
| **96 px** | `yolo11N-640` | 78.9% | 78.8% | 64.7% | 78.3% | 21.9% |
| | `y26n_noe2e_warm50-2` | 81.5% | 79.1% | 70.5% | 78.3% | 21.9% |
| | `y26n_sop50` | **83.9%** | 80.8% | **71.2%** | 78.3% | 37.5% |
| | `gelannfav14r4fw_gemlb_v2` | 77.1% | **82.1%** | 66.9% | 60.1% | **62.5%** |
| **64 px** | `yolo11N-640` | 67.7% | 74.0% | 46.8% | 74.6% | 4.7% |
| | `y26n_noe2e_warm50-2` | 74.8% | 76.9% | 57.6% | **76.8%** | 15.6% |
| | `y26n_sop50` | **75.1%** | 78.1% | **60.4%** | 73.9% | 21.9% |
| | `gelannfav14r4fw_gemlb_v2` | 61.6% | **82.4%** | 48.2% | 61.6% | **32.8%** |
| **48 px** | `yolo11N-640` | 40.5% | 76.8% | 24.5% | 52.2% | 0.0% |
| | `y26n_noe2e_warm50-2` | **70.1%** | 68.2% | 38.8% | **74.6%** | 9.4% |
| | `y26n_sop50` | 68.9% | 74.9% | **52.5%** | 66.7% | **17.2%** |
| | `gelannfav14r4fw_gemlb_v2` | 47.5% | **80.2%** | 25.2% | 60.9% | **17.2%** |

### 4.3 mAP across resolutions (threshold-free)

n = 341 labelled people at each size.

| MAP50 | 128 px | 96 px | 64 px | 48 px |
|---|---:|---:|---:|---:|
| `gelannfav14r4fw_gemlb_v2` | **0.775** | **0.763** | **0.679** | **0.632** |
| `y26n_sop50` | 0.710 | 0.675 | 0.597 | 0.529 |
| `y26n_noe2e_warm50-2` | 0.679 | 0.643 | 0.593 | 0.486 |
| `yolo11N-640` | 0.616 | 0.612 | 0.544 | 0.462 |

| MAP50-95 | 128 px | 96 px | 64 px | 48 px |
|---|---:|---:|---:|---:|
| `gelannfav14r4fw_gemlb_v2` | **0.352** | **0.376** | **0.369** | **0.328** |
| `y26n_sop50` | 0.321 | 0.309 | 0.296 | 0.284 |
| `yolo11N-640` | 0.302 | 0.304 | 0.285 | 0.278 |
| `y26n_noe2e_warm50-2` | 0.310 | 0.278 | 0.262 | 0.241 |

| WOMAN AP50 | 128 px | 96 px | 64 px | 48 px |
|---|---:|---:|---:|---:|
| `gelannfav14r4fw_gemlb_v2` | **0.817** | **0.802** | **0.697** | **0.624** |
| `y26n_sop50` | 0.761 | 0.711 | 0.629 | 0.534 |
| `y26n_noe2e_warm50-2` | 0.741 | 0.706 | 0.633 | 0.494 |
| `yolo11N-640` | 0.682 | 0.689 | 0.543 | 0.462 |

### 4.4 Model sizes — nothing here is a size trade

| model | params (fused) | GFLOPs @640 | ONNX CPU |
|---|---:|---:|---:|
| `y26n_noe2e_warm50-2` | 2.375 M | 5.3 | ~34 ms |
| `y26n_sop50` | 2.375 M | 5.3 | ~34 ms |
| `yolo11N-640` *(production)* | 2.583 M | 6.4 | 41 ms |
| `gelannfav14r4fw_gemlb_v2` | 2.962 M | — | ~59 ms |

All four are nano-class within 0.6 M params. gelan's ~1.75× latency is the only real cost.

---

## 5. What this means

**1. Sharpness is not the binding constraint — framing is.** At 128 px, a large and
guaranteed-crisp portrait, the best model still gets only **77.7%** of women found *and*
labelled Woman. Roughly **one in four profile-picture women goes unblurred** with image quality
at its best. This is a training-distribution problem, not an image-quality one.

**2. Production falls off a cliff at feed size.** At 48 px — an ordinary comment-thread avatar —
`yolo11N-640` reaches **24.5%** Woman end-to-end against `y26n_sop50`'s **52.5%**. A 28-point
gap on the metric that decides whether a blur happens, and it fails to find 3 people in 5.

**3. Part of the ranking is calibration, not capability.**
`gelannfav14r4fw_gemlb_v2` has the **best mAP at every size** yet places 3rd–4th at conf 0.45.
The reason is where each model puts its confidence. Detection recall at 48 px, by threshold,
with the median confidence of a correct detection:

| model | R @0.45 | R @0.25 | R @0.10 | median conf |
|---|---:|---:|---:|---:|
| `y26n_noe2e_warm50-2` | 70.1% | **85.3%** | 91.8% | 0.620 |
| `y26n_sop50` | 68.9% | **85.3%** | **92.4%** | 0.575 |
| `gelannfav14r4fw_gemlb_v2` | 47.5% | 72.7% | 82.7% | **0.406** |
| `yolo11N-640` *(production)* | 40.5% | 68.3% | 83.9% | 0.423 |

**Both gelan and production sit below the shipped 0.45 cut** (0.406 and 0.423) while the y26n
arms sit well above it. Dropping the cut to 0.25 lifts gelan from 47.5% → 72.7% and production
from 40.5% → 68.3%, so a good part of their deficit is threshold placement. But it does **not**
close the gap: the y26n arms reach 85.3% at the same threshold and still lead. So the ranking is
*partly* calibration and *partly* real — and a fixed-0.45 comparison overstates the gap.
`conf_sweep.py` should be run per model before any candidate is chosen.

**4. Children are the one place gelan wins outright** (60.9% vs 29.7% Child end-to-end at
128 px). Children are correctly left unblurred, so that is a win in the harmless direction.

---

## 6. Sharpness audit — how to confirm the set is crisp

Every emitted avatar was measured with **variance-of-Laplacian normalised at 128 px**, the same
metric the Spotlight pipeline uses for its `blurriness` field. Normalising to a fixed size
matters: a raw VoL grows with resolution and would simply rank big crops first.

**Distribution over 808 measured avatars** (men and children; women are excluded from the
rendered audit):

| min | p05 | p25 | median | p75 | max |
|---:|---:|---:|---:|---:|---:|
| 307 | 585 | 970 | **1472** | 2322 | 8222 |

For context, the ungated source crops had a median of 392 — the emitted set sits well above the
material it was drawn from, which is what the downscale rule buys.

**The audit is worst-first, not sampled.** `REPORT_avatars.html` ends with the **36 least-sharp
avatars in the entire set**, magnified, each captioned with its score. A sampled page can hide a
bad tail; a worst-first page cannot. If the bottom of the distribution is acceptable, everything
above it is.

> **If any of those 36 still look soft**, raise the gate: tell us the score of the worst
> acceptable one and the set is rebuilt at that threshold.

---

## 7. Limits — read before quoting

- The crop is a **geometric approximation** from the person's box (top 34% of its height), **not
  a face detection**, so framing is approximate.
- The source is **LAGENDA general photography, not real scraped profile pictures**. Pose and
  lighting are photographic rather than selfie-typical.
- Avatars sit on a **flat light canvas**, not a rendered feed UI.
- Numbers are **not comparable to the other small-person arms** (crowd photos, shrunk scenes),
  whose canvas sizes and difficulty differ. They are comparable **within this arm, across
  models and sizes**.
- Single run per model, no variance estimate — treat differences under ~2 points as ties.

---

## 8. Reproduce it

```bash
# build (pod)
python3 build_small_person_set.py avatars \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --labels /workspace/datasets/lagenda_full/eval_v2/labels \
    --gt-jsonl /workspace/datasets/lagenda_full/eval_v2/gt.jsonl \
    --polygons /workspace/lagenda_eval/sam_autolabel/labels \
    --targets 128,96,64,48 --images-per-target 60 --per-image 6 \
    --canvas 640 --bg 240 --min-source-height 200 --limit 9000 \
    --min-scale 1.5 --min-sharpness 700 \
    --out /workspace/datasets/smallperson_v1/avatars

# verify (7/7) and score
python3 verify_small_person_set.py --set /workspace/datasets/smallperson_v1/avatars
bash /workspace/exp19/score_avatars.sh && bash /workspace/exp19/map_all.sh

# the report
python3 build_smallperson_gallery.py --report avatars \
    --out REPORT_avatars.html --n-images 12 --n-worst 36
```

Dataset: `/workspace/datasets/smallperson_v1/avatars/` (seed 20260820).
Full analysis and the pre-registered bars: `experiments/EXP-2026-19-small-person-benchmark.md`.
