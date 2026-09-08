# LAGENDA `fl1199` — run Sol against SAM3 detections, scored on HUMAN labels

Everything is on the RunPod volume; nothing needs exporting. All facts below were
verified on the pod 2026-08-23.

**Why this set and not the holdout:** every person in these images has a **human**
age + gender label (10 crowdsourced votes each). The holdout can only tell you whether
two models *agree*; this set can tell you which one is **right**. It is the only
three-way we can run: **Gemini vs Sol vs hand-labelled**.

```
/workspace/datasets/lagenda_full/fl1199/
```

---

## 1. What the set is

1,199 LAGENDA images in which **every** person box carries a human label — no unlabeled
people, so an unmatched detection is a genuine false positive rather than someone the
annotators skipped. That property holds nowhere else in LAGENDA, where 4 of every 6 real
people are unlabeled.

Selected by three independent routes that agree exactly (empty `ignore/`, labeled count ==
box count, and `gt.jsonl` rows == box count), out of the 4,601-image contamination-excluded
v2 set.

```
1,199 images · 1,514 human-labelled people (1.263 per image)
   1 person 957 · 2 persons 191 · 3 persons 36 · 4+ persons 15   (max 6)
human labels : Woman 535 · Man 518 · Child 461     gender M 752 / F 762
age          : median 22 · children (0-12) 30.4% · teens (13-17) 10.2% · 50+ 21.9%
```

## 2. Paths

| what | path |
|---|---|
| images (1,199 symlinks) | `fl1199/images/` |
| **SAM3 labels** | `fl1199/raw/<stem>.txt` |
| **SAM3 raw sidecars** | `fl1199/raw/<stem>.json` |
| staging manifest | `fl1199/manifest.jsonl` |
| **human ground truth** | `../eval_v2/gt.jsonl` |
| human boxes, 3-class | `../eval_v2/labels_3class/<stem>.txt` |
| all person boxes | `../eval_v2/labels/<stem>.txt` |

`../eval_v2/` = `/workspace/datasets/lagenda_full/eval_v2/`.

## 3. The SAM3 output (fresh, 2026-08-23)

```
1,199 / 1,199 images labelled · 0 missing · 0 orphans
1,884 detections (1.571 per image)
299 NMS-suppressed boxes
conf: min 0.401 · median 0.961 · max 0.990
SAM3 class mix: Woman 574 · Man 695 · Child 615
```

This run was made specifically for this experiment with the **webp-patched raw labeler**, so
unlike LAGENDA's older labels it carries full sidecars. Sidecar shape (identical to the
holdout's): top level `image, width, height, classes, conf_threshold (0.4), nms_iou (0.7),
detections[], suppressed[]`; each detection `cls` (0=woman 1=man 2=child), `conf`,
`box` `[x1,y1,x2,y2]` in pixels, `parts` (per-part mask polygons before YOLO flattening).

**Why the sidecars matter here: 29.1% of detections (548) have multi-part masks** — a person
split into 2–6 visible pieces by occlusion (parts per detection: 1→1,336, 2→267, 3→135,
4→50, 5→32, 6→21). LAGENDA's older label set had no `parts`, which would silently force
`build_crop` into its box-only fallback and outline a rectangle full of neighbours.

## 4. Building the crop and prompt — import ours, don't reimplement

Identical to the holdout, and the same reasoning: if your crop or prompt differs, you are
measuring the harness rather than the model.

```python
import sys, hashlib; sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import build_crop, PROMPT, PROMPT_VERSION
assert hashlib.sha256(PROMPT.encode()).hexdigest()[:12] == "4c85ffbf8bcb"   # spotlight-e1
crop, scale = build_crop(img, det["box"], det["parts"])
```

`build_crop` is a pure function of image + box + parts: 25% padding, upscale so
`max(side) >= 320` (LANCZOS), and the detection's own mask parts drawn as a two-tone green
outline. **Only the crop is sent to the API** — one crop + the prompt per detection, never
the whole image. That is why the prompt opens "One candidate in this image is outlined in
green": the crop still contains neighbours, so the outline says which person to judge.

## 5. Gemini has NOT been run on this set — you need both arms

Only **39 of 1,199 images (3.3%)** have Gemini verdicts, from EXP-2026-10, and those used a
**different prompt** (`spotlight-v2`, sha `b30e8ec997f6`, 21 fields) with box-fallback crops.
**Do not reuse them.** For a clean three-way, run Gemini over `fl1199` with `spotlight-e1`
alongside Sol. It is nearly free:

| arm | cost over 1,884 crops |
|---|---|
| Gemini 3.5 Flash-Lite (Batch) | **~$0.61** |
| GPT-5.6 Sol (standard) | **~$18** |
| GPT-5.6 Sol (Batch, 50% off) | **~$9** |

Projected from our measured ~1,482 input + ~80 output tokens per detection.
Gemini arm: `spotlight_batch.py submit/poll/collect --images fl1199/images
--raw-labels fl1199/raw --out <run>`.

## 6. Joining a detection to its human label — the required step

SAM3 detections and human boxes are **different box sets**; they must be matched by IoU.
Reuse the project matcher rather than writing one:

```python
sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
from run_model_children import match_boxes, iou     # match_boxes(gt_boxes, dets, min_iou)
```

Human boxes come from `eval_v2/labels/<stem>.txt` (YOLO normalized **cxcywh** — convert to
xyxy pixels before matching). `gt.jsonl` ids are `<stem>_<N>` where **N is the line index in
that file**, which is how you reach `gt_age` and `gt_gender`.

Keys: SAM3 detections are `(image_stem, det_index)` where `det_index` is the line index in
`raw/<stem>.txt`. **Carry that key on every row of your output** or the runs can't be diffed.

## 7. What the numbers will split into

```
1,884 SAM3 detections
  ~1,514 expected to match a human-labelled person  -> TRUE 3-WAY (Gemini vs Sol vs human)
  ~370   SAM3-only detections                       -> 2-way + adjudication
```

SAM3 finds **+370 more people than LAGENDA's own detector** (1.24×), never fewer at the
count level. Those 370 are not waste — they are the most informative crops in the set,
because they answer whether SAM3's extra detections are real people LAGENDA's YOLOv8 missed
or genuine false positives. Neither VLM can be scored on them; a human has to look.

## 8. Caveats — read before quoting anything

- **This subset has no small people.** At 640 model input: **0 people below 64 px**, 96.1%
  are ≥160 px, median 386 px. That is *why* they are fully labeled — annotators labeled the
  prominent subjects. Any recall number here is prominent-subject recall and will not
  generalise. It is also 6× sparser than the rest of LAGENDA (1.26 vs 7.68 people/image).
- **"Fully labeled" means every box from LAGENDA's YOLOv8 has a human label**, not that
  every visible human is labeled — the 370 extra SAM3 detections prove that directly.
- **The boxes are detector-derived**, not human-drawn (the authors ran a YOLOv8 for the
  public release). The **age/gender labels are human**; the geometry is not.
- **`occ_ratio` is 1.0 for every person** in LAGENDA, so any occlusion breakdown is
  meaningless here — that analysis is only valid on CrowdHuman.
- Contaminated images (present in our OIV7 train split) were already excluded at v2 build.

## 9. How the label attaches back to the image

The VLM never returns geometry. SAM3's polygon is already in the original image's
normalised coordinates; the crop is a throwaway device for asking the question. Emit
rewrites **only field 0** (the class id) and passes every coordinate through byte-for-byte,
or drops the line. Nothing is mapped back from crop space — which is also why `build_crop`'s
returned `scale` is discarded by the production path.

Note that emitted label files have **fewer lines** than raw ones because deleted detections
are removed, so `det_index` indexes the **raw** file, not the emitted one.
