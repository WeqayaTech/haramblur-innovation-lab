# LAGENDA — dataset, ground truth, SAM3 output, and the fl1199 three-way set

Companion to `docs/SAM3_OUTPUT_HANDOFF.md` (which covers the holdout).
Everything verified on the volume 2026-08-23.

**The one thing that makes LAGENDA different from every other set we own: its age and
gender labels are HUMAN** (10 crowdsourced votes per person). The holdout's labels are a
model's judgement; LAGENDA's are people. That is why LAGENDA is the set we use to *select*
models, and the holdout is the set we use to *probe domains* we have no human labels for.

> Its **person boxes are still detector-derived** — the authors ran a YOLOv8 for the public
> release. So classification metrics here are human-grounded; detection metrics only
> measure agreement with *their* detector, a tier below CrowdHuman's human-drawn boxes.

---

## 1. Images

```
/workspace/lagenda_eval/lagenda_yolo/images/val/     4,899 images
```

There is **no images directory under `lagenda_full/`** — the v2 build emits labels only and
points back at this directory. Note the YOLO split convention: images are in
`images/val/`, not `images/`.

## 2. Two versions — use v2

| | v1 (legacy) | v2 (2026-08-10) |
|---|---|---|
| path | `/workspace/lagenda_eval/lagenda_yolo/` | `/workspace/datasets/lagenda_full/eval_v2/` |
| images | 4,899 | **4,601** |
| labeled persons | 5,000 (≈1/image) | **7,098** (+42%) |
| all person boxes | — | **27,633** (6.01/image) |
| train contamination | **present** | **excluded** |

**Use v2.** v1 sampled ~1 person per image, which is why historical "LAGENDA detection
recall 93–95%" numbers are *prominent-subject* recall, not people recall. LAGENDA is drawn
from Open Images and our models train on the OIV7 train split; v1 never excluded the overlap.

**Verified relationship: v2's 4,601 images are an exact subset of the 4,899 SAM3-labelled
images, minus 298 excluded as train-contaminated** (4,899 − 298 = 4,601).

## 3. The human ground truth (v2)

```
lagenda_full/eval_v2/
├── gt.jsonl          7,098 rows: {id, image, gt_age, gt_gender}  ← the human labels
├── labels/           4,601 files — EVERY person box, class 0, YOLO cxcywh
├── labels_3class/    4,601 — only persons WITH human labels, as {0 Woman, 1 Man, 2 Child}
├── ignore/           4,601 — every person box WITHOUT usable labels (20,535 boxes)
├── persons.odgt      CrowdHuman-format, all person boxes
└── BUILD_README.md   the authoritative build record
```

`gt.jsonl` ids are `<stem>_<N>` where **N is the line index in `labels/<stem>.txt`**.

**You must pass `ignore/` to `map_eval.py`.** Only 1.54 of the 6.01 people per image carry
human labels — **4 of every 6 real people are unlabeled**. Score without the ignore set and
a model is punished for correctly finding them. This is the same "absence of a label ≠
evidence of absence" trap as the holdout, but far more severe here.

Scorer wiring (from `BUILD_README.md`):

```bash
# 3-class mAP, human-labeled, with ignore semantics
map_eval.py --gt-labels .../eval_v2/labels_3class --ignore-labels .../eval_v2/ignore

# detection recall/precision over ALL person boxes
eval_negatives_crowd.py --mode crowd --gt-odgt .../eval_v2/persons.odgt

# classification against human age/gender
run_autolabel_on_manifest.py --gt-manifest .../eval_v2/gt.jsonl
```

## 4. SAM3 output — and its one big limitation

```
/workspace/lagenda_eval/sam_autolabel/labels/     4,899 × .txt   ← NO .json sidecars
```

Same format as the holdout's `.txt`: class id + a normalised **segment polygon**. Class ids
`0 = woman, 1 = man, 2 = child`. Line index = `det_index`.

**But these were produced by the production `autolabel_sam.py`, which does not write raw
sidecars.** So unlike the holdout, LAGENDA has **no `parts`, no per-detection `conf`, and no
NMS-suppressed detections** for the full set. Only the 500 images under
`/workspace/exp10/raw_full/lagenda/` have `.json` sidecars.

**Why that matters for VLM crops:** `build_crop(img, box, parts)` needs `parts` to draw the
mask outline. With `parts` empty it takes the **box-only fallback** — the exact configuration
EXP-2026-10 measured as worse.

To get holdout-quality crops over all of LAGENDA, **re-run the raw labeler** (~75 min on L4):

```bash
python3 /workspace/datasets/haramblur_holdout/labeling/tools/autolabel_sam_raw_allext.py \
    --input  /workspace/lagenda_eval/lagenda_yolo/images/val \
    --output /workspace/lagenda_eval/sam_raw \
    --classes "woman" "man" "child" --batch_size 1
```

## 5. The existing Gemini verdicts — NOT comparable to the holdout

```
/workspace/exp10/full_lagenda/verdicts.jsonl    1,331 verdicts / 150 images
```

**Three incompatibilities — do not pool with holdout numbers:**

1. **Different prompt.** `spotlight-v2`, sha `b30e8ec997f6` (21 fields). The holdout used
   `spotlight-e1`, sha `4c85ffbf8bcb` (9 fields).
2. **Different crops.** All 1,331 records have `has_poly: false` — box fallback, not mask.
3. **Different tool.** Written by `pipeline_v1_eval.py`, not `spotlight_run.py`.

## 6. How labels attach to the image

The VLM never returns geometry. SAM3's polygon is already in the original image's normalised
coordinates; the crop is a throwaway. Emit rewrites **only field 0** (the class id) and
passes every coordinate through byte-for-byte, or drops the line entirely.

Emitted label files have **fewer lines** than raw ones because deleted detections are removed,
so `det_index` indexes the **raw** file, not the emitted one.

## 7. Caveats for any LAGENDA number

- **Detection metrics measure agreement with LAGENDA's own YOLOv8**, not human boxes.
- **`occ_ratio` is 1.0 for every person** — occlusion breakdown is meaningless here (only
  valid on CrowdHuman).
- **4,983 images were excluded as train-contaminated** at v2 build time.
- **v1 numbers are prominent-subject recall**, not people recall.

---

# LAGENDA `fl1199` — the fully-labelled three-way set

## What the set is

1,199 LAGENDA images in which **every** person box carries a human label — no unlabeled
people, so an unmatched detection is a genuine false positive. That property holds nowhere
else in LAGENDA, where 4 of every 6 real people are unlabeled. This is the only three-way
we can run: **Gemini vs Sol vs hand-labelled**.

Selected by three independent routes (empty `ignore/`, labeled count == box count, and
`gt.jsonl` rows == box count), out of the 4,601-image v2 set.

```
1,199 images · 1,514 human-labelled people (1.263 per image)
   1 person 957 · 2 persons 191 · 3 persons 36 · 4+ persons 15   (max 6)
human labels : Woman 535 · Man 518 · Child 461     gender M 752 / F 762
age          : median 22 · children (0-12) 30.4% · teens (13-17) 10.2% · 50+ 21.9%
```

## Paths

| what | path |
|---|---|
| images (1,199 symlinks) | `fl1199/images/` |
| **SAM3 labels** | `fl1199/raw/<stem>.txt` |
| **SAM3 raw sidecars** | `fl1199/raw/<stem>.json` |
| staging manifest | `fl1199/manifest.jsonl` |
| **human ground truth** | `../eval_v2/gt.jsonl` |
| human boxes, 3-class | `../eval_v2/labels_3class/<stem>.txt` |
| all person boxes | `../eval_v2/labels/<stem>.txt` |

All under `/workspace/datasets/lagenda_full/`.

## SAM3 output (fresh, 2026-08-23)

```
1,199 / 1,199 images labelled · 0 missing · 0 orphans
1,884 detections (1.571 per image)
299 NMS-suppressed boxes
conf: min 0.401 · median 0.961 · max 0.990
SAM3 class mix: Woman 574 · Man 695 · Child 615
```

Made with the **webp-patched raw labeler**, so unlike LAGENDA's older labels it carries
full sidecars (29.1% of detections have multi-part masks).

## Building the crop — import ours

```python
import sys, hashlib; sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import build_crop, PROMPT, PROMPT_VERSION
assert hashlib.sha256(PROMPT.encode()).hexdigest()[:12] == "4c85ffbf8bcb"   # spotlight-e1
crop, scale = build_crop(img, det["box"], det["parts"])
```

## Gemini has NOT been run on this set

Only 39/1,199 images have Gemini verdicts (EXP-2026-10, different prompt). For a clean
three-way, run Gemini alongside Sol. Costs:

| arm | cost over 1,884 crops |
|---|---|
| Gemini 3.5 Flash-Lite (Batch) | **~$0.61** |
| GPT-5.6 Sol (Batch, 50% off) | **~$9** |

## Joining detections to human labels

SAM3 detections and human boxes are different box sets; match by IoU:

```python
sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
from run_model_children import match_boxes, iou
```

Human boxes: `eval_v2/labels/<stem>.txt` (YOLO cxcywh — convert to xyxy pixels before
matching). `gt.jsonl` ids are `<stem>_<N>` where N is the line index. Carry
`(image_stem, det_index)` on every row or the runs can't be diffed.

## fl1199 caveats

- **No small people.** 0 people below 64 px at model input; median 386 px. Any recall
  number here is prominent-subject recall.
- **"Fully labeled" means every box from LAGENDA's YOLOv8 has a human label**, not that
  every visible human is labeled — the 370 extra SAM3 detections prove that directly.
- **Boxes are detector-derived**, not human-drawn. The age/gender labels are human.
- **`occ_ratio` is 1.0** for every person — occlusion breakdown is meaningless.
- Contaminated images already excluded at v2 build.
