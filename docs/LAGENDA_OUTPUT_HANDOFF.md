# LAGENDA — dataset, human ground truth, and SAM3 output

Companion to `docs/SAM3_OUTPUT_HANDOFF.md` (which covers `haramblur_holdout`).
Everything verified on the volume 2026-08-23.

**The one thing that makes LAGENDA different from every other set we own: its age and
gender labels are HUMAN** (10 crowdsourced votes per person). The holdout's labels are a
model's judgement; LAGENDA's are people. That is why LAGENDA is the set we use to *select*
models, and the holdout is the set we use to *probe domains* we have no human labels for.

> Its **person boxes are still detector-derived** — the authors ran a YOLOv8 for the public
> release. So classification metrics here are human-grounded; detection metrics only
> measure agreement with *their* detector, a tier below CrowdHuman's human-drawn boxes.

---

## Short answer: has LAGENDA been labeled by our pipeline?

| stage | coverage | where |
|---|---|---|
| **Human age+gender** | **7,098 persons** over 4,601 images | `lagenda_full/eval_v2/gt.jsonl` |
| **SAM3 detections** | **4,899 images**, `.txt` polygons only — **no sidecars** | `lagenda_eval/sam_autolabel/labels/` |
| **SAM3 with raw sidecars** | only **500 images** | `/workspace/exp10/raw_full/lagenda/` |
| **Gemini / Spotlight verdicts** | only **1,331 detections over 150 images** | `/workspace/exp10/full_lagenda/verdicts.jsonl` |

So: **SAM3 has labeled essentially all of LAGENDA; Gemini has not.** Only a 150-image
slice from EXP-2026-10 has verdicts, and **it used a different prompt from the holdout run**
(see the compatibility warning below). If you want a Gemini pass over all of LAGENDA
comparable to the holdout, it has not been done.

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

**Verified relationship: v2's 4,601 images are an exact subset of the 4,899 SAM3-labelled
images, minus 298 excluded as train-contaminated** (4,899 − 298 = 4,601, and every v2 image
has a SAM3 label). So you can score v2 against the existing SAM3 output with no re-run —
just ignore the 298.

**Use v2.** v1 sampled ~1 person per image, which is why historical "LAGENDA detection
recall 93–95%" numbers are *prominent-subject* recall, not people recall — footnote that
wherever it is quoted. LAGENDA is drawn from Open Images and our models train on the OIV7
train split; v1 never excluded the overlap.

## 3. The human ground truth (v2)

```
lagenda_full/eval_v2/
├── gt.jsonl          7,098 rows: {id, image, gt_age, gt_gender}  ← the human labels
├── labels/           4,601 files — EVERY person box, class 0, YOLO cxcywh
├── labels_3class/    4,601 — only persons WITH human labels, as {0 Woman, 1 Man, 2 Child}
├── ignore/           4,601 — every person box WITHOUT usable labels (20,535 boxes)
├── persons.odgt      CrowdHuman-format, all person boxes
└── BUILD_README.md   the authoritative build record — read it
```

`gt.jsonl` ids are `<stem>_<N>` where **N is the line index in `labels/<stem>.txt`**.

**You must pass `ignore/` to `map_eval.py`.** Only 1.54 of the 6.01 people per image carry
human labels — **4 of every 6 real people are unlabeled**. Score without the ignore set and
a model is punished for correctly finding them, and the 3-class mAP is badly wrong. This is
the same "absence of a label ≠ evidence of absence" trap as the holdout, but far more severe
here.

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

Same format as the holdout's `.txt`: class id + a normalised **segment polygon** (~127
fields on a sample row — tens to hundreds of coordinates, *not* 4 box values). Class ids
`0 = woman, 1 = man, 2 = child`. Line index = `det_index`.

**But these were produced by the production `autolabel_sam.py`, which does not write raw
sidecars.** So unlike the holdout, LAGENDA has **no `parts` (per-part mask polygons), no
per-detection `conf`, and no NMS-suppressed detections** for the full set. Only the 500
images under `/workspace/exp10/raw_full/lagenda/` have `.json` sidecars.

**Why that matters if you want to feed crops to a VLM:** `build_crop(img, box, parts)`
needs `parts` to draw the mask outline. With `parts` empty it silently takes the
**box-only fallback** (`spotlight_run.py:153`) and outlines the rectangle instead — the
exact configuration EXP-2026-10 measured as worse, because the rectangle contains
neighbouring people and the model labels the wrong one.

To get holdout-quality crops over all of LAGENDA you must **re-run the raw labeler**
(~1.1 img/s on an L4, so ~75 min for 4,899):

```bash
python3 /workspace/datasets/haramblur_holdout/labeling/tools/autolabel_sam_raw_allext.py \
    --input  /workspace/lagenda_eval/lagenda_yolo/images/val \
    --output /workspace/lagenda_eval/sam_raw \
    --classes "woman" "man" "child" --batch_size 1
```

(Use that webp-patched copy; the production script silently skips `.webp` and uppercase
extensions.)

## 5. The existing Gemini verdicts — and why they are NOT comparable to the holdout

```
/workspace/exp10/full_lagenda/verdicts.jsonl    1,331 verdicts / 150 images
```

Results: `real_person` 1,264 · `depiction` 42 · `not_person` 23 · 2 parse failures.

**Three incompatibilities with the holdout run — do not pool these numbers:**

1. **Different prompt.** `spotlight-v2`, sha **`b30e8ec997f6`**, a 21-field schema
   (adds `occlusion`, `occlusion_percent`, `face_visible`, `pose`, …). The holdout used
   `spotlight-e1`, sha **`4c85ffbf8bcb`**, 9 fields.
2. **Different crops.** Every one of the 1,331 records has **`has_poly: false`** — all were
   built with the box fallback, not mask outlines.
3. **Different tool and schema.** Written by `pipeline_v1_eval.py`, not `spotlight_run.py`.
   Records key on `id` = `<stem>_<N>` and `image`, **not** `image_stem` + `det_index`.
   They do, usefully, carry `crop_origin`, `crop_scale` and `crop_wh`, so crop-space
   coordinates can be mapped back to the image.

## 6. How labels attach to the image

Identical to the holdout, and worth restating: **the VLM never returns geometry.** SAM3's
polygon is already in the original image's normalised coordinates; the crop is a throwaway
device for asking the question. Emit rewrites **only field 0** (the class id) and passes
every coordinate through byte-for-byte, or drops the line entirely. Nothing is ever mapped
back from crop space.

One consequence: emitted label files have **fewer lines** than the raw ones because deleted
detections are removed, so `det_index` indexes the **raw** file, not the emitted one.

## 7. Caveats to carry into any number you quote

- **Detection metrics measure agreement with LAGENDA's own YOLOv8**, not human boxes.
- **`occ_ratio` is 1.0 for every person** — LAGENDA has no visible-vs-full box distinction,
  so `recall_by_occlusion` reports everything as "light". That breakdown is meaningless
  here; it is only valid on CrowdHuman.
- **4,983 images were excluded as train-contaminated** at v2 build time (298 of our local
  4,899). Any v1 number predates that exclusion.
- **v1 numbers are prominent-subject recall**, not people recall.
