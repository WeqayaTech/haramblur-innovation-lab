# Spotlight-val mAP — 4 model candidates

4,232 images · 8,035 GT boxes · 702 ignore (unknown-gender people) · conf 0.45 · IoU 0.7

| model | mAP50 | mAP75 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8022 | 0.7343 | 0.6970 | 0.8103 | 0.8644 | 0.7318 |
| `yolo11N-640` (production) | 0.7436 | 0.6674 | 0.6033 | 0.7745 | 0.7838 | 0.6725 |
| `y26n_sop50` | 0.8056 | **0.7388** | **0.6996** | **0.8165** | 0.8674 | 0.7329 |
| `gelannfav14r4fw_gemlb_v2` | **0.8118** | 0.7160 | 0.6540 | 0.7771 | **0.8700** | **0.7884** |

**Note:** this is the models' own training/checkpoint-selection validation split, not a
held-out set — all four saw this data (or an overlapping image pool) during their own
training. Read as in-sample fit, not generalization. See `docs/MODEL_COMPARISON.md` for
the full verification and the genuinely held-out benchmarks (LAGENDA, CrowdHuman).
