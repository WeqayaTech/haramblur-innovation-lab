# Training configuration history — every model, what it was actually trained with

Companion to `docs/MODEL_COMPARISON.md` (which records how each model *scored*). This one
records how each model was **trained**: epochs, image size, batch, optimizer, learning rate,
warm-start chain, and — the part that turned out to matter most — **augmentation**.

Compiled 2026-08-16 by reading every training config on the RunPod volume:
**69 Ultralytics `args.yaml` files** and **115 YOLO-MIT hydra runs**, plus the augmentation
source and `train.sh` command history. The augmentation implementations themselves are quoted
in **`docs/AUGMENTATION_CODE.md`**. Models are listed under their **exact run/directory
names**, as in `MODEL_COMPARISON.md`.

**Verify it yourself:**

```bash
# every Ultralytics run config
find /workspace -maxdepth 7 -name args.yaml -not -path "*/site-packages/*"
# every YOLO-MIT run: CLI overrides + fully resolved config
cat /workspace/YOLO-MIT/runs/v9MIT/<run>/.hydra/overrides.yaml
cat /workspace/YOLO-MIT/runs/v9MIT/<run>/.hydra/config.yaml
# the YOLO-MIT defaults those overrides apply on top of
cat /workspace/YOLO-MIT/yolo/config/task/train.yaml
# the full command history, one commented line per run
cat /workspace/YOLO-MIT/train.sh
# epochs actually completed (not just configured)
wc -l <run>/results.csv     # Ultralytics: rows - 1 = epochs
```

---

## 1. The headline: two frameworks, opposite augmentation philosophies

| | Ultralytics family (YOLOv8 / YOLO11 / YOLO26 / YOLOE) | YOLO-MIT family (v9 / gelan) |
|---|---|---|
| Mosaic | **on, 1.0** (stock default) in essentially every run | **off (`Mosaic: 0`)** in essentially every run |
| Scale/zoom | `scale 0.5` (stock) | `RandomZoom: 1` — always on, range (0.5, 1.5) |
| Colour | `hsv 0.015 / 0.7 / 0.4` | `RandomHSV` — **identical values** |
| Flip | `fliplr 0.5` | `HorizontalFlip 0.5` — identical |
| Random erasing | `erasing 0.4` in every config — **but detection ignores it** (classification-only, see `AUGMENTATION_CODE.md`) | none |
| Custom augs | none | `MaskSmallBoxes`, `SmallObjectPatches` |
| Class-loss weight | `cls 0.5` (stock) | **`BCELoss: 6`** — 12× |
| Warmup | **`warmup_epochs: 0`** (the one house-wide deviation) | `warmup.epochs: 0` + `optimizer.warmup_epochs: 0` |

Across all 69 Ultralytics runs the **only** consistent deviation from stock is
`warmup_epochs: 0`. All augmentation engineering in this project happened on the YOLO-MIT
side. This matters for interpretation: **augmentation is not what distinguishes the shipped
YOLO11 from the YOLO26n arms** — both ran stock Ultralytics augmentation with mosaic on.

---

## 2. The shipped model — `yolo11N-640`

`/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` (2.58M params, the
`v11nclean2` export family). **It is not one training run — it is a 3-stage fine-tune chain
totalling 155 epochs, all at 640 px**, each stage warm-starting the next:

| # | run directory | base checkpoint | epochs (**verified from `results.csv`**) | imgsz | batch | data |
|---|---|---|---|---|---|---|
| 1 | `/workspace/hgnet-yolo/cleaned_dataset_with_sheikhs/yolo11N-6402` | `yolo11n.pt` (COCO) | **80** | 640 | 64 | `open-images-v7/dataset.yaml` |
| 2 | `/workspace/hgnet-yolo/cleaned_dataset_with_sheikhs_negatives/yolo11N-640-negatives` | stage 1 `best.pt` | **25** | 640 | 64 | same + negatives |
| 3 | `/workspace/model_v2/dataset_v2/yolo11N-640` | stage 2 `best.pt` | **50** | 640 | 64 | `open-images-v7/dataset.yaml` |

Hyperparameters, all three stages identical and **stock apart from warmup**:

```
optimizer: auto   lr0: 0.01   lrf: 0.01   momentum: 0.937   weight_decay: 0.0005
warmup_epochs: 0  (default 3.0)   warmup_momentum: 0.8   warmup_bias_lr: 0.1
box: 7.5   cls: 0.5   dfl: 1.5   nbs: 64   patience: 100   cos_lr: false   amp: true
hsv_h 0.015  hsv_s 0.7  hsv_v 0.4   degrees 0  translate 0.1  scale 0.5  shear 0
perspective 0  flipud 0  fliplr 0.5  bgr 0  mosaic 1.0  close_mosaic 10
mixup 0  cutmix 0  copy_paste 0  erasing 0.4  auto_augment randaugment
```

`erasing` and `auto_augment` are recorded in every `args.yaml` but are **not applied to
detection training** — `v8_transforms()` never builds them; they exist only in
`classify_augmentations()`. Line-level proof in `docs/AUGMENTATION_CODE.md` §0.

The 416 branch forks off stage 2: `yolo11N-416-negatives` (50 ep @416) →
`/workspace/model_v2/dataset_v2/yolo11N-416` (50 ep @416).

> **Provenance caveat.** `model_v2/dataset_v2/yolo11N-640/args.yaml` records
> `name: yolo11N-6402` and `save_dir: /workspace/model_v2/dataset_v2/yolo11N-6402` — a
> directory that does not exist. The folder was renamed or the file copied, so the
> config↔weights binding at this path rests on convention, not on record. This compounds
> the already-open `v11nclean2` shipped-binary identity question.

---

## 3. Ultralytics runs — the full record

Everything not listed under "deviations" is the stock default for the installed
`ultralytics` version.

| run directory | base | epochs | imgsz | batch | deviations from stock |
|---|---|---|---|---|---|
| `hgnet-yolo/cleaned_dataset_nov_2/hgnetv2b0-relu{,2..5}` | `hgnet-yolo.yaml` | 80 | 320 | 64 | `warmup_epochs 0` |
| `hgnet-yolo/cleaned_dataset_nov_2/yolo-p6-relu` | self `best.pt` | 80 | 448 | 64 | `warmup_epochs 0`, `warmup_bias_lr 0.0` |
| `hgnet-yolo/cleaned_dataset_with_sheikhs/yoloN-p6-relu-640{,2}` | `yolo11n-p6.yaml` | 50 | 640 | 64 | none |
| `hgnet-yolo/cleaned_dataset_with_sheikhs/yolos-p6-hardswish-320{,-flip}` | `yolo11s-p6.yaml` | 50 | 320 | 64 | SGD `lr0 0.001`, `copy_paste 0.5`, `close_mosaic 5`, `warmup_epochs 0` |
| `hgnet-yolo/cleaned_dataset_with_sheikhs/yolos-p6-320-mixup` | self `last.pt` | 50 | 320 | 128 | SGD `lr0 0.0007`, `copy_paste 0.05` + `copy_paste_mode mixup`, `close_mosaic 5` |
| `hgnet-yolo/cleaned_dataset_with_sheikhs/yolos-35{2,4}` | `yolo11s.pt` / self | 80 / 50 | 352 / 320 | 64 | SGD `lr0 0.0007`, `copy_paste 0.5`/`0.05` + `mixup` mode, `close_mosaic 5` |
| `hgnet-yolo/cleaned_dataset_with_sheikhs/yolo11N-640` | `yolo11n.pt` | 50 | 640 | 64 | `warmup_epochs 0` |
| **`hgnet-yolo/cleaned_dataset_with_sheikhs/yolo11N-6402`** | `yolo11n.pt` | **80** | 640 | 64 | `warmup_epochs 0` — *shipped chain stage 1* |
| **`hgnet-yolo/cleaned_dataset_with_sheikhs_negatives/yolo11N-640-negatives`** | stage 1 | **25** | 640 | 64 | `warmup_epochs 0` — *stage 2* |
| `hgnet-yolo/cleaned_dataset_with_sheikhs_negatives/yolo11N-320-negatives` | stage 2 | 50 | 320 | 64 | `warmup_epochs 0` |
| `hgnet-yolo/cleaned_dataset_with_sheikhs_negatives/yolo11N-416-negatives` | stage 2 | 50 | 416 | 64 | `warmup_epochs 0` |
| **`model_v2/dataset_v2/yolo11N-640`** | 640-negatives | **50** | 640 | 64 | `warmup_epochs 0` — **SHIPPED** |
| `model_v2/dataset_v2/yolo11N-416` | 416-negatives | 50 | 416 | 64 | `warmup_epochs 0` |
| `model_v2/dataset_v2_random_nomosaic/yolo11N-640` | `dataset_v2/yolo11N-640` | 50 | 640 | 64 | **`mosaic 0.0`**, `warmup_epochs 0` |
| `model_v2/dataset_v2_random_rotate_nomosaic/yolo11N-416` | `dataset_v2/yolo11N-416` | 50 | 416 | 64 | **`mosaic 0.0`, `degrees 90`**, `warmup_epochs 0` |
| `model_v2/dataset_v2_with_new_val/yolo11M-416` | `yolo11m.pt` | 50 | 416 | 0.8 | **`mosaic 0.0`** |
| `segmentation-dataset/segmentation-balanced-dataset-trial1/yolo11N-640{,2}` | `yolo11n-seg.pt` / 640-negatives | 50 | 640 | 64 | none |
| `segmentation-dataset/segmentation-balanced-dataset-trial2/yolo11N-seg-640` | `yolo11n-seg.pt` | 50 | 640 | 64 | none |
| `old-volume/0.3.0/0.3.0/yolov8m-oiv7-120epoch-w-smallimages{,2..5}` | `yolov8m-*.pt` | 120 | 640 | 32 | **`mosaic 0.6`, `scale 0.2`, `translate 0`, `fliplr 0.4`, `bgr 0.1`, `patience 25`** |
| `old-volume/hgnet-yolo/**` (13 backbone probes) | `hgnet-yolo.yaml`, `mobilenet-yolo.yaml`, `yolo11n-short.yaml` | 80 | 320–640 | 64–128 | `warmup_epochs 0` |
| `YOLO26/yolo26/v1`, `v12`, `runs/detect/train` | `yolo26s.pt` | 50 / 80 | 320 | 64 / 16 | `warmup_epochs 0` |
| `YOLO26/runs/detect/yolo11/mv1_datav2` | self `last.pt` | 50 | 640 | 64 | `warmup_epochs 0`, `warmup_bias_lr 0.0` |
| `exp12/train/y26n_pilot` | `yolo26n.pt` | 3 | 640 | −1 | `freeze 11` |
| **`exp12/train/y26n_spotlight`** | self `last.pt` | **30** | 640 | 22 | `warmup_bias_lr 0.0` |
| `exp14/train/y26n_unk4` | `yolo26n.pt` | **30** | 640 | −1 | none (stock) |
| **`exp14/train/y26n_gradsupp`** | `yolo26n.pt` | **30** | 640 | −1 | none (stock) |
| `exp15/train/yoloe_n_gradsupp` | `yoloe-26n-seg.pt` | 30 | 640 | −1 | none (stock) |
| `exp16/train/y26n_noe2e` | `exp16/yolo26n_noe2e.yaml` | **30** | 640 | −1 | none (stock) |
| `exp17/train/y26n_twoaxis{,2}` | `yolo26n.pt` | 80 | 640 | 64 | none (stock) |
| `exp18/train/y26n_noe2e_warm50-2` | `exp16/train/y26n_noe2e/weights/best.pt` | 50 (**running**) | 640 | −1 | **`optimizer MuSGD`, `lr0 0.003`**, `warmup_epochs 0` |

Datasets: `open-images-v7/dataset.yaml` = 3 classes `{0 Woman, 1 Man, 2 Child}`, 467,745
train label files. `exp12/spotlight_oiv7_local.yaml` = same 3 classes, 475,206 train images.
`exp14/unk4_{b,c}.yaml` = 4 classes (+`3 Unknown`). `exp17/twoaxis.yaml` = 6 classes
(two-axis). So the corpora are the same size — **what changed is the label content**.

---

## 4. YOLO-MIT (v9 / gelan) runs

Defaults live in `/workspace/YOLO-MIT/yolo/config/task/train.yaml`; each run overrides on top.
The **current recipe**, verbatim from the resolved config of `gelansfav14_gemlb_v1`:

```yaml
image_size: [640, 640]
data_augment:
  Mosaic: 0            # OFF — in every gelan run
  MixUp: 0             CutMix: 0            CopyPaste: 0
  HorizontalFlip: 0.5  VerticalFlip: 0      RandomRotate90: 0
  RandomHSV: {h_gain: 0.015, s_gain: 0.7, v_gain: 0.4}
  RandomZoom: 1                    # always on; crop if <1, pad if >1, range (0.5,1.5)
  MaskSmallBoxes: {min_area_px: 0} # greys out boxes under N px² AND deletes their labels
  SmallObjectPatches: 0.05         # crops objects onto a grey canvas at 96px
close_aug: 0                       # disable Mosaic/MixUp/CutMix/VFlip/CopyPaste last N epochs
weighted: loss                     # loss-based resampling...
loss_component: cls                # ...driven by classification loss
mask_classes: {classes: [3], color: 114, use_polygon: true, drop_from_head: true}
optimizer: SGD nesterov, lr 0.0008, weight_decay 0.0005, momentum 0.937, warmup_epochs 0
scheduler: LinearLR 1 -> 0.01, warmup.epochs 0
loss: {BCELoss: 6, BoxLoss: 7.5, DFLoss: 1.5}, aux 0.25, matcher CIoU topk 10
ema: {enable: true, decay: 0.9999}
```

Every `*_gemlb_*` run is a warm start from a previous checkpoint, batch 32–64, 640 px,
`BCELoss: 6`, `mask_classes: [3]`, hand-tuned `lr` per run, `best_metric` = `cls/bal_acc`
or `map`. Configured epochs (from `.hydra/overrides.yaml`):

| run | model | epochs | batch | lr | `MaskSmallBoxes` | `weighted` | warm start from |
|---|---|---|---|---|---|---|---|
| `gelansfav14_gemlb_v1` | `gelan-sfav14` | 50 | 32 | 0.0008 | 0 | loss/cls | `gelansfav14_datav2_v4` |
| `gelannfav14_gemlb_v1` | `gelan-nfav14` | 80 | 64 | 0.005 | 30 | loss/cls | self `last.ckpt` (resumed) |
| `gelannfav14r4_gemlb_v1` | `gelan-nfav14r4` | 50 | 64 | 0.005 | 30 | loss/cls | `gelannfav14_gemlb_v1` |
| `gelannfav14r4_gemlb_v2` | `gelan-nfav14r4` | 15 | 64 | 0.001 | 0 | False/cls | `gelannfav14r4_gemlb_v1` |
| `gelannfav14w_gemlb_v1` | `gelan-nfav14w` | 50 | 32 | 0.005 | 0 | False/cls | `gelannfav14r4_gemlb_v1` |
| `gelannfav14w_gemlb_v2` | `gelan-nfav14w` | 15 | 32 | 0.0008 | 0 | loss/cls | `gelannfav14w_gemlb_v1` |
| `gelannfav14r3_gemlb_v3kd` | `gelan-nfav14r3` | 15 | 32 | 0.003 | 0 | loss/cls | + **distill**, teacher `gelansfav14_gemlb_v1`, `cls 0.25` |
| `gelannfav14r5_gemlb_v1` | `gelan-nfav14r5` | 50 | 64 | 0.005 | 30 | loss/cls | `gelannfav14_gemlb_v1` |
| `gelannfav14m/ml/mlc/mlcs/pb/pu/pub/pr/r4f_*` | (arch sweep) | 15–50 | 32 | 0.003–0.01 | 0 | mixed | chained |

**How the recipe evolved:**

| period | change |
|---|---|
| 2025-11 | v9n / v9-t / gelan-t probes at 320–448 px; **mosaic off from the very first run**; `close_aug 5` |
| 2025-12 | autolabel data; **`MaskSmallBoxes.min_area_px: 7000`** (masks anything under ~84×84 px); **FocalLoss** + `BCELoss 8`→`6` |
| 2025-12 → 2026-01 | `min_area_px` walked **7000 → 2048 → 1024 → 0**; FocalLoss used in **31 runs**, then abandoned (everything from 2026-04 is plain `BCELoss` at weight 6) |
| 2026-01 → 02 | `SmallObjectPatches: 0.05` added; `close_aug` → 0; `weighted` resampling introduced |
| 2026-04 | sfav13/14/15 arch sweep; recipe stable at batch 32 / 640 px |
| 2026-08 | Spotlight labels + `mask_classes: [3]` grey-masking; `min_area_px` briefly 30, then 0; one distillation run |

> **Quirk on record:** `gelanspda_datav2_focal_v2` was launched with
> `task.data.data_augment.Mosaic=4096`. Config scalars go straight to `Mosaic(prob=…)` and
> the guard is `random.random() >= prob`, so that value means **mosaic always on** —
> behaviourally identical to `1`. Almost certainly a typo; it is one of only three gelan runs
> that ever had mosaic effectively enabled (with `gelant_mosaic` and `v9n_fit`).

---

## 5. Why the shipped YOLO11 handles small people better than YOLO26n

The training *hyperparameters* do not explain it — both are stock Ultralytics augmentation
at 640 px with mosaic on. The **labels** do.

Measured 2026-08-16 on a random sample of **5,438 images present in both label sets**
(seed 20260816), comparing `/workspace/open-images-v7/labels/train` (old labels, what
`yolo11N-640` trained on) against `/root/oiv7_local/labels/train` (Spotlight `labels/`,
what `y26n_spotlight` trained on). Box heights are normalized × 640 = **pixels as the model
sees them at training resolution**. "drop%" = old boxes with no Spotlight counterpart, at
IoU 0.5 and 0.3:

| GT height @640 | old boxes | Spotlight boxes | net | drop%@IoU.5 | drop%@IoU.3 | added | `labels_unk3` class-3 |
|---|---|---|---|---|---|---|---|
| <32 px | 836 | 465 | **−44.4%** | 77.2% | 75.1% | 276 | 525 |
| 32–64 px | 1,747 | 1,674 | −4.2% | 49.0% | 46.7% | 743 | 731 |
| 64–100 px | 1,411 | 1,291 | −8.5% | 31.1% | 28.8% | 327 | 341 |
| **100–160 px** | 1,829 | 1,629 | −10.9% | **24.2%** | 22.6% | 224 | 274 |
| 160–320 px | 3,779 | 3,434 | −9.1% | 16.2% | 15.3% | 262 | 369 |
| ≥320 px | 7,670 | 7,197 | −6.2% | 7.8% | 7.5% | 180 | 205 |

Images with **no** labels at all: old **2.5%** → Spotlight **8.8%** (3.5× more pure-background
images).

**The gradient is monotone in object size, and it is not a matching artifact** — relaxing the
match to IoU 0.3 barely moves the drop rates (77.2 → 75.1 at the smallest band), so these are
genuine deletions, not boxes that merely shifted.

The mechanism is Spotlight's `dropped-unknown-gender` decision. Spotlight dropped 196,119
detections whose gender Gemini could not judge (median ~58 px, per the production-run record
in `CLAUDE.md`). Those people are **present as class 3 in `labels_unk3/` but absent entirely
from `labels/`**. Expressed as a share of every person Spotlight found at that size:

| GT height @640 | share of people dropped as unknown-gender |
|---|---|
| <32 px | **53%** |
| 32–64 px | 30% |
| 64–100 px | 21% |
| 100–160 px | 14% |
| 160–320 px | 10% |
| ≥320 px | 2.8% |

So a model trained on `labels/` is **actively taught that more than half of all tiny people,
and one in seven people around 100–160 px, are background.** The old labels contained no such
instruction. This is the same mechanism already suspected in `CLAUDE.md` §4.9995 hypothesis
(b) and recorded as a caveat in EXP-2026-12 — now quantified.

### Measured directly: the deficit is concentrated in smaller people

Detection recall split by GT person height, LAGENDA v2, all arms at conf 0.45, identical
scorer (`conf_sweep.py --report-metrics lagenda.det_recall_small|med|large`; results in
`/workspace/sweep/sizeband_<arm>/sweep.json`):

| arm | labels | 64–160 px (n=147) | ≥160 px (n=6,951) | gap between bands |
|---|---|---|---|---|
| `v11n_pt` (shipped `yolo11N-640`) | old | **83.0%** (122) | 95.8% (6,660) | −12.8 |
| `mit_prod` (gelan v9) | old | **89.1%** (131) | 96.4% (6,697) | −7.3 |
| `y26n_ft` (`y26n_spotlight`) | Spotlight `labels/` | **73.5%** (108) | 94.1% (6,544) | −20.6 |
| `y26n_gs_e2e` (`y26n_gradsupp`) | Spotlight `labels_unk3` + GS | **81.0%** (119) | 94.3% (6,554) | −13.3 |

Reading the shipped model against `y26n_spotlight`: the deficit is **−1.7 pts on large people
but −9.5 pts on 64–160 px people** — 5.7× larger in the smaller band. And gradient
suppression, which restores exactly the dropped-unknown supervision, recovers **7.5 of those
9.5 points** (73.5 → 81.0) while moving the large band by only +0.15. The two old-label models
lead the medium band; the two Spotlight-label models trail it.

**Honest limits of this measurement:**
- **The <64 px band is structurally empty on LAGENDA** — `det_recall_small` returns `[0, 0]`,
  because 6,951 of 7,098 human-labeled LAGENDA people (97.9%) are ≥160 px. LAGENDA cannot
  measure genuinely small people at all; the size bands as implemented live only in the
  LAGENDA scorer (`conf_sweep.py` `LagendaSet`), not the CrowdHuman one. Porting them to
  `CrowdSet` is the open work item for measuring the <64 px range.
- **n=147 in the 64–160 px band.** One person = 0.68 pts, so the 9.5-pt gap is 14 people.
  Direction is consistent across four arms and matches the label statistics above, but this
  band alone is not a conclusive test — treat it as corroboration, not proof.
- `woman_recall_e2e` in the same band (n=64) points the same way: `v11n_pt` 79.7%,
  `mit_prod` 78.1%, `y26n_gs_e2e` 73.4%, `y26n_ft` 71.9%.
- These numbers were only obtainable after fixing a crash in `print_table`
  (`conf_sweep.py:872` read an undefined `h`), which had prevented the size-stratified metrics
  from ever running.

Consistent supporting evidence already in the repo:

- `y26n_gradsupp` trains on `labels_unk3` with the unknown class supervising box/DFL/assignment
  (just not the classifier) — i.e. it puts exactly this supervision back — and gains
  **+3.7 crowd recall** over `y26n_spotlight` (32.8 → 36.5).
- `gelansfav14_gemlb_v1` grey-masks the unknowns instead (they become neither object nor
  background) and leads crowd recall at **46.2**.
- The shipped `yolo11N-640` also had **155 epochs across 3 stages including a dedicated
  negatives stage**, versus 30 epochs for every YOLO26n arm — a second, unseparated difference.

**Open, not yet answered:** the two candidate fixes have never been combined —
**grey-masked unknowns on YOLO26n**. Masking gained +6.6 crowd recall on the gelan
architecture; gradient suppression gained +3.7 on y26n; the combination is untested.

---

## 6. Other things that affect training — verified, and mostly not in any config

Augmentation, epochs, warm-start chains and model choice are only part of what determines a
run. Everything below was checked on the pod 2026-08-16. Several of these are **not
recoverable from `args.yaml`**, and none of them were deliberate choices.

### 6.1 The optimizer and learning rate in `args.yaml` are NOT what ran

Every Ultralytics run in this project used `optimizer: auto`, which discards the recorded
values:

```python
# ultralytics/engine/trainer.py:1113-1122
if name == "auto":
    LOGGER.info(f"'optimizer=auto' found, ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' ...")
    nc = self.data.get("nc", 10)
    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
    self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam
```

What actually ran, read from each run's own `results.csv` (`lr/pg*` columns) and its log:

| run | ultralytics | optimizer actually used | param groups |
|---|---|---|---|
| `yolo11N-6402` (shipped stage 1) | 8.3.225 | **SGD**, lr 0.01 | 3 |
| `model_v2/dataset_v2/yolo11N-640` (SHIPPED) | 8.3.235 | **SGD**, lr 0.01 | 3 |
| `exp12/train/y26n_spotlight` | 8.4.115 | **MuSGD(lr=0.01, momentum=0.9)** | 8 |
| `exp14/train/y26n_gradsupp`, `y26n_unk4` | 8.4.115 | **MuSGD(lr=0.01, momentum=0.9)** | 8 |
| `exp15/train/yoloe_n_gradsupp` | 8.4.115 | **MuSGD(lr=0.01, momentum=0.9)** | 8 |
| `exp16/train/y26n_noe2e` | 8.4.115 | **MuSGD(lr=0.01, momentum=0.9)** | 8 |
| `exp12/train/y26n_pilot` (3 ep) | 8.4.115 | **AdamW(lr=0.001429)** | — |
| `exp14/train/pilot_b`, `pilot_c` (3 ep) | 8.4.115 | **AdamW(lr=0.00125)** | — |
| `exp18/train/y26n_noe2e_warm50-2` | 8.4.115 | MuSGD(lr=0.003, mom=0.937) — *explicit, not auto* | 8 |

Three consequences:

1. **The shipped YOLO11 was trained with SGD; every YOLO26n arm was trained with MuSGD.**
   Nobody chose this — it falls out of `optimizer: auto` plus the version difference (MuSGD
   ships with the 8.4 / YOLO26 line). It is an uncontrolled variable in every
   shipped-vs-y26n comparison made so far.
2. **The 3-epoch pilots ran a different optimizer than the full runs they were meant to
   de-risk** — AdamW at lr ≈0.0013 vs MuSGD at lr 0.01, because the rule switches on
   `iterations > 10000`. Pilot behaviour does not predict full-run behaviour.
3. `auto` also silently sets **`warmup_bias_lr = 0.0`** whatever the config says — visible in
   every checkpoint's `train_args`, while `args.yaml` still records `0.1`.

### 6.2 Ultralytics version drift across the project

`8.3.225` (Nov 2025) → `8.3.235` (Dec 2025) → `8.4.115` (Aug 2026); `8.4.120` is installed
now. Defaults, the auto-optimizer rule and the fitness weights all changed across those
versions, so **every "stock defaults" statement is version-relative** and cross-era
comparisons silently inherit whatever changed in between. Recoverable per model with
`torch.load(ckpt)["version"]`.

### 6.3 `batch: -1` means the batch size was decided by free GPU memory

Every `exp1x` arm used AutoBatch (60% GPU memory target). Resolved sizes, from the logs:

| run | batch actually used |
|---|---|
| `exp12/train/y26n_pilot` | 40 |
| `exp12/train/y26n_spotlight` | 22 |
| `exp14/train/pilot_b`, `pilot_c` | 27 |
| `exp14/train/y26n_unk4` (arm B) | **22** |
| `exp14/train/y26n_gradsupp` (arm C) | **22** |
| `exp15/train/yoloe_n_gradsupp` | 19 |
| `exp16/train/y26n_noe2e` | 24 |
| `exp18/train/y26n_noe2e_warm50-2` | 24 |

EXP-2026-14's arm B vs arm C is clean (both 22). But **`yoloe_n_gradsupp` (19) vs
`y26n_gradsupp` (22)** and **`y26n_noe2e` (24) vs `y26n_gradsupp` (22)** were compared at
different batch sizes. Batch also drives gradient accumulation and rescales weight decay —
`y26n_noe2e` logged `decay=0.0005625`, not the configured `0.0005`. Because a rerun on a
different GPU picks a different batch, **`seed: 0` + `deterministic: true` does not make these
runs reproducible.**

### 6.4 `best.pt` is chosen by mAP50-95 alone

```python
# ultralytics/utils/metrics.py:1007-1010
def fitness(self) -> float:
    w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())
```

The checkpoint we ship is selected purely by **mAP50-95 on the val split** — not by the
object-set false-person rate, not by the adult→Child leak, not by woman recall. None of the
metrics the product is actually judged on. `conf_sweep.py` has since shown those axes trade
against each other, so the fitness-selected epoch need not be the best epoch for the product.
YOLO-MIT handles this better: `task.best_metric` is explicit, and the gelan runs deliberately
used `cls/bal_acc` or `map`.

Compounding it, the two families selected `best.pt` against **different val label
distributions** — the shipped chain against old `open-images-v7` val labels, the y26n arms
against Spotlight-verified val labels (already flagged as mildly optimistic in
`MODEL_COMPARISON.md`).

### 6.5 The label cache is keyed on summed file sizes

```python
# ultralytics/data/utils.py:139-147
def get_hash(paths: list[str]) -> str:
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())  # hash sizes
```

No content hash, no mtime. Editing labels **in place** while the total byte count happens to
match reuses the stale `.cache` silently and trains on the old labels. Switching to a
different label *directory* is safe, because the cache path follows the label dir. Not
hypothetical here: exp18's first launch died with
`No labels found in /root/oiv7_local/labels/train.cache`. **After any label re-emit, delete
the `.cache` sitting next to the labels.**

### 6.6 Restarts, and which checkpoint each chain hop starts from

`y26n_spotlight` was **resumed 3 times** (`exp12/log_resume{,2,3}_*.log`) after the documented
epoch-29 OOM. The gelan chains mix their starting points: some warm-start from `last.ckpt`
(`gelannfav14_gemlb_v1`, `gelannfav14r4_gemlb_v1`, `gelannfav14w_gemlb_v2`), others from
`best.ckpt` (`gelannfav14r4_gemlb_v2`, `gelannfav14w_gemlb_v1`, `gelannfav14m_gemlb_v1`).
Since `best.ckpt` is the metric-selected epoch and `last.ckpt` is not, the hops are not
equivalent — worth recording per hop rather than just "warm-started".

### 6.7 Seeding

Ultralytics: `seed: 0`, `deterministic: true` in every run — undercut by AutoBatch (§6.3).
YOLO-MIT: seeded from **`lucky_number: 10`** in `general.yaml` via
`seed_everything(cfg.lucky_number)` (`yolo/lazy.py:41`) — easy to miss, given the name.

### 6.8 Class balance and the split — measured

Train/val split (`exp12/train_full.txt` vs `val_full.txt`): 475,207 / 4,233, **overlap = 0**.
Clean.

Class balance, same 6,000 sampled images through each label set (seed 20260816):

| label set | boxes | empty imgs | Woman | Man | Child | Unknown | W:M |
|---|---|---|---|---|---|---|---|
| `open-images-v7/labels` (old) | 17,272 | 2.5% | 29.6% | 64.3% | 6.1% | — | **1 : 2.17** |
| Spotlight `labels/` | 15,706 | 17.3% | 28.4% | 66.6% | 5.1% | — | **1 : 2.35** |
| Spotlight `labels_unk3/` | 18,154 | 16.0% | 24.5% | 57.6% | 4.4% | **13.5%** | 1 : 2.35 |

Two readings that matter:

- **The ~2.3:1 male skew is inherited from the corpus, not created by Spotlight** (1:2.17 →
  1:2.35, a mild amplification). It therefore **cannot explain any difference** between the
  shipped model and the y26n arms — both label sets carry essentially the same imbalance. It
  is a shared property worth knowing about for its own sake (the model sees ~2.3× more men
  than women, and children are ~5% of boxes), not a candidate cause of the woman-recall gap.
- **`labels_unk3` carries 13.5% of all person boxes as class 3** — that is the exact quantity
  of supervision `labels/` discards, and it is consistent with the size-band result in §5.

Note the empty-image rate depends on how you sample: 17.3% unconditioned, but 8.8% in §5's
paired sample, which required the image to also have an old label file (biasing toward images
where the old labeler found someone).

### 6.9 What to record for future runs

The cheapest fix for most of the above is to capture, per run: the resolved optimizer line
from the log, the AutoBatch decision, `torch.load(ckpt)["version"]`, the `best_metric` used
for selection, and the exact label directory + its cache state. None of that is in
`args.yaml`.

## 7. Caveats

- **Deviation-from-stock is computed against the currently installed `ultralytics` 8.4.115
  defaults.** Older runs used older versions, so "stock" means "equal to today's default
  value". The absolute values quoted are read directly from each `args.yaml` and stand
  regardless.
- Ultralytics epochs marked **verified** were counted from `results.csv`; all others are the
  *configured* value and may differ if a run stopped early. YOLO-MIT epochs are configured
  values from `.hydra/overrides.yaml`.
- The label-size comparison uses normalized height × 640. It measures the label files as they
  exist on disk today; it does not re-derive them from the Spotlight verdict stream.
- `MaskSmallBoxes.min_area_px` is an **area** in px², so `7000` ≈ an 84×84 box — the
  2025-12-era gelan runs were masking out a large fraction of small people deliberately. The
  semantics of that knob did not change; only the values did.
