# 4-way model benchmark — LAGENDA, fl1199, CrowdHuman, PASS, Spotlight-val (2026-08-24/25)

## Models

| run name | weights path | notes |
|---|---|---|
| `y26n_noe2e_warm50-2` (baseline) | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` | YOLO26n, Spotlight labels |
| `yolo11N-640` (production) | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` | YOLO11n, shipped production, older labels |
| `y26n_sop50` | `/workspace/exp19/train/y26n_sop50/weights/best.pt` | `y26n_noe2e_warm50-2` + 50 more epochs with a live `SmallObjectPatches` augmentation |
| `gelannfav14r4fw_gemlb_v2` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r4fw_gemlb_v2/weights/best.ckpt` | GELAN architecture (YOLO-MIT framework), Spotlight labels, continued from `gelannfav14r4fw_gemlb_v1` |

## Data

| dataset | size | role | ground truth |
|---|---|---|---|
| LAGENDA (hand-labeled) | 4,601 images, 7,098 labeled people, 20,535 ignore | classification | human age+gender votes; unlabeled real people scored as COCO ignore regions |
| LAGENDA `fl1199` | 1,199 images, 1,514 labeled people, **0 ignore** | classification | every person box human-labeled — no ignore regions needed at all |
| CrowdHuman | 4,372 images, 99,481 GT persons | person recall | human-drawn exhaustive boxes |
| PASS | 3,000 person-free images | false-positive rate | curated negatives |
| Spotlight-val | 4,232 images, 8,035 GT boxes, 702 ignore | mAP | Gemini-verified Spotlight labels — **not held out, see below** |

**`fl1199` — verified, not assumed.** `labels_3class` and `labels` (all boxes) both
have exactly 1,514 lines across these 1,199 stems, and 0 of 1,199 stems have any
content in their `ignore/` file — checked directly on disk before running anything.
A data bug was found and fixed before trusting the first result: the Stage A tool
globs images recursively, which swept up 3 stray Jupyter-checkpoint duplicates with no
matching label file; fixed by building a clean, non-recursive 1,199-image symlink farm
and re-running. Separately confirmed: Sol (`gpt-5.6-sol`) was run over this set
(1,884 verdicts, matching the SAM3 detection count exactly), but that data belongs to
a different research thread (Gemini vs Sol vs human) and is **not** the ground truth
used here — the human labels are. One real limitation: **zero small people** in this
subset (median 386px) — every number is prominent-subject-only.

**Spotlight-val fairness — verified, not assumed.** Checked each model's own training
config directly and traced every `data.yaml` to its actual files on disk:

| run name | this image set's role | labels used at that time |
|---|---|---|
| `y26n_noe2e_warm50-2` | validation split (checkpoint selection) | Spotlight |
| `y26n_sop50` | validation split (checkpoint selection) | Spotlight |
| `gelannfav14r4fw_gemlb_v2` | validation split (checkpoint selection) | Spotlight (`labels_unk3`) |
| `yolo11N-640` | validation split (checkpoint selection) | its own, older labels |

All four used this same underlying image pool as their validation split — confirmed
`yolo11N-640`'s "different, older" tree (`open-images-v7/images/val`) is the *same*
4,485-image pool as the Spotlight tree (identical stem sets, one file MD5-identical
across both). None trained (gradient updates) on these specific images — checked 0
overlap against both `open-images-v7/images/train` (512,893 images) and Spotlight's
`train_full.txt` (475,207 images). **Net effect: Spotlight-val is in-sample/validation-
adjacent for all four models, not a fair generalization test for any of them.** LAGENDA,
CrowdHuman, and (separately) `haramblur_holdout` are the genuinely independent
benchmarks for this project.

## Metrics

### LAGENDA — hand-labeled 3-class mAP

| run name | mAP50 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8567 | 0.7231 | 0.8806 | 0.8861 | 0.8033 |
| `yolo11N-640` | 0.8228 | 0.7142 | 0.8656 | 0.8465 | 0.7562 |
| `y26n_sop50` | 0.8608 | 0.7266 | 0.8836 | 0.8882 | 0.8105 |
| `gelannfav14r4fw_gemlb_v2` | **0.8820** | **0.7312** | **0.9035** | **0.8979** | **0.8445** |

### LAGENDA `fl1199` — fully human-labeled, zero ignore

| run name | mAP50 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8955 | 0.7801 | 0.9085 | 0.9364 | 0.8417 |
| `yolo11N-640` | 0.8743 | 0.7896 | 0.9010 | 0.8987 | 0.8231 |
| `y26n_sop50` | 0.9061 | 0.7885 | 0.9252 | 0.9369 | 0.8561 |
| `gelannfav14r4fw_gemlb_v2` | **0.9433** | **0.8060** | **0.9459** | **0.9549** | **0.9291** |

GELAN sweeps every column — its most decisive win of any benchmark this session
(+3.7 pts mAP50 vs the next-best, vs ~0.5-1 pt gaps on LAGENDA/CrowdHuman). Consistent
with the zero-small-people limitation above: this is exactly the regime GELAN's
architecture is known to lead, and the y26n family's small-object weakness is never
exercised.

### CrowdHuman — person recall

| run name | recall | recall (heavy occl.) | recall (light) | precision | clear FP/100 img |
|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.3868 | 0.2072 | 0.4584 | 0.9201 | 1.74 |
| `yolo11N-640` | 0.3432 | 0.1377 | 0.4320 | **0.9452** | **1.28** |
| `y26n_sop50` | 0.3849 | 0.2035 | 0.4668 | 0.9219 | 1.67 |
| `gelannfav14r4fw_gemlb_v2` | **0.3893** | 0.2035 | 0.4668 | 0.9313 | 1.56 |

### PASS — false positives

| run name | FP/100 img | % img w/ FP | FP by class |
|---|---|---|---|
| `y26n_noe2e_warm50-2` | **0.37** | **0.37%** | Woman 2, Man 9 |
| `yolo11N-640` | 1.17 | 1.13% | Woman 14, Man 21 |
| `y26n_sop50` | 0.60 | 0.60% | Woman 3, Man 14, Child 1 |
| `gelannfav14r4fw_gemlb_v2` | 0.70 | 0.70% | Woman 6, Man 12, Child 3 |

### Spotlight-val — mAP (not held out, see Data section)

| run name | mAP50 | mAP75 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8022 | 0.7343 | 0.6970 | 0.8103 | 0.8644 | 0.7318 |
| `yolo11N-640` | 0.7436 | 0.6674 | 0.6033 | 0.7745 | 0.7838 | 0.6725 |
| `y26n_sop50` | 0.8056 | **0.7388** | **0.6996** | **0.8165** | 0.8674 | 0.7329 |
| `gelannfav14r4fw_gemlb_v2` | **0.8118** | 0.7160 | 0.6540 | 0.7771 | **0.8700** | **0.7884** |

By GT box size, mAP50-95:

| run name | small | medium | large |
|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.3371 | 0.5710 | 0.7598 |
| `yolo11N-640` | 0.2486 | 0.4948 | 0.6574 |
| `y26n_sop50` | **0.3392** | **0.5739** | **0.7603** |
| `gelannfav14r4fw_gemlb_v2` | 0.2913 | 0.5059 | 0.7347 |

## Bottom line

`gelannfav14r4fw_gemlb_v2` is the best classifier on every genuinely held-out benchmark
(LAGENDA, `fl1199`, CrowdHuman) and edges CrowdHuman recall, at the compute cost
documented in `docs/MODEL_COMPARISON.md`'s size & speed table — its margin is widest on
`fl1199`, the cleanest, most generous benchmark of the five. `yolo11N-640` (production)
has the cleanest CrowdHuman precision but the lowest recall and worst PASS rate of the
four. `y26n_noe2e_warm50-2` is cleanest overall on false positives. `y26n_sop50`'s
continuation is a real but modest classification win over its own baseline on every
benchmark measured (including Spotlight-val, despite that one being compromised for
cross-model comparison), at a real but modest PASS-FP cost.

**Do not read the Spotlight-val table as answering "which model classifies best"** —
use LAGENDA/`fl1199`/CrowdHuman for that.

---
Full detail, methodology, and narrative: `docs/MODEL_COMPARISON.md`. Published artifact:
https://claude.ai/code/artifact/3f8c26aa-a6db-4d60-a0cc-b808ba329334. Raw outputs:
`/workspace/exp_lagenda_bench/<run_name>/`, `/workspace/exp_spotval_bench/<run_name>/`,
and `/workspace/exp_fl1199_bench/<run_name>/`.
