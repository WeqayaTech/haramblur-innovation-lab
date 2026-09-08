<!-- CLICKUP UPLOAD NOTES (delete this block after uploading):
  1. Paste this whole file into a ClickUp Doc (it renders Markdown: headings, tables, code, emoji).
  2. Charts are NOT embedded - ClickUp doesn't render SVG. Where you see a "INSERT IMAGE HERE"
     callout, drag the named PNG from experiments/assets/png/EXP-2026-03/ into that spot, then delete the callout line.
  3. All 3 PNGs live in ONE folder - experiments/assets/png/EXP-2026-03/ - in doc order:
     sam_fp_class_split.png, sam_fp_by_object_type.png, sam_crowd_recall_by_occlusion.png (rendered at 2x).
  4. Overlay adjudications are still pending (marked in the doc) - update the doc once done.
-->

# Experiment: Does the SAM3 auto-labeler invent people, and can it handle crowds?

**In one line:** part 2 of the SAM3 auto-labeler audit — we point the production labeler at the two
things LAGENDA couldn't test (images with *no* people, and crowds where *every* person is labeled)
to measure false positives and detection precision, scored **separately on each dataset**.

**Status:** Run complete (2026-07-13) — all three datasets scored; overlay adjudications pending ·
PASS 3,000 / objects 259 / CrowdHuman 500-image sample (11,259 persons) · Owner: Mostafa

---

# 1 · Executive summary

EXP-2026-02 concluded that SAM3 did a great job in detection people. This experiment shows that was
LAGENDA's framing, not SAM's reality. Measured on data LAGENDA couldn't provide:

- **SAM3 predicts people on human-shaped objects — badly.** 62% of verified person-free
  doll/statue/toy images gained at least one person label; on doll images it's 89%, averaging
  2.3 spurious labels per image, most often as "Woman." The failure scales cleanly with
  human-likeness (dolls 89% → bronze statues 65% → sculptures 53% → teddy bears 15%)
- **SAM3 predicts people on empty scenes (Human Free).** 1 in 18 person-free PASS images (5.4%) gains a person label, mostly "Man."
- **In real crowds, it misses 1 in 4 people.** Recall is 77% overall and degrades with
  occlusion (83% lightly occluded, down to 61% heavily occluded).

### Scorecard

| Dataset                        | Metric                                                  | Result                                                                         |
| ------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| PASS                           | false positives per 100 empty images                    | **8.4** and 5.4% of empty images get a person label                     |
| Open Images toys/statues/dolls | % of human-shaped-object images that get a person label | **62.2%** — dolls 88.8%, 2.3 FPs/image; Woman is the top spurious class |
| CrowdHuman                     | detection precision (boxes hitting a real person)       | **86.6%**                                                               |
| CrowdHuman                     | recall on lightly-occluded people                       | **82.7%**                                                               |
| CrowdHuman                     | duplicates per matched person (one person, many boxes)  | **1.1%**                                                                |

**Scope:** this is a **detection experiment** — false positives and precision. It does not touch
classification accuracy, gender, or age (those were EXP-2026-02 / EXP-2026-01, and CrowdHuman has
no age/gender labels anyway). Numbers are reported **per dataset** each dataset answers a different question.

---

# 2 · Plan and results

## 2.1 Context

EXP-2026-02 measured how well SAM handles *real, labeled people* and found detection recall of
98.7% and an age-axis classification problem. But every number there was **label-anchored**: it
only graded boxes that landed on a person LAGENDA had labeled. Two things stayed invisible:

- **False positives** — labels SAM writes where there's no person (the team has seen it fire on
  toys and mannequins). LAGENDA can't show this: every image has a person.
- **Detection precision and duplicates in crowds** — on LAGENDA an unmatched SAM box could always
  be a real but unlabeled person, so we could never call it a mistake. That left the result of "56% of
  images have 3+ more detections than labeled people" unresolved.

This experiment brings in data built exactly for those two questions, and scores each on its own.

## 2.2 The "person" ruling (this decides what counts as a false positive)

**A "person" is anything a viewer is meant to lower their gaze from in the Islamic sense — including a printed or photographic *depiction* of a real person** (a poster, billboard, magazine cover, photo-on-packaging). So:

- **Counts as a person (SAM firing on it is CORRECT, not a false positive):** a real human; a
  photo/poster/printed image of a real human.
- **Does NOT count as a person (SAM firing on it is a false positive):** a mannequin, doll, toy,
  statue, or sculpture — a human-*shaped* object that is not a depiction of a specific real person.

## 2.3  Datasets

The whole experiment depends on trusting each dataset's labels *for its specific job*. For a
false-positive test that means **exhaustive** annotation. That requirement drove the choices, and ruled two popular datasets out.

- **PASS** (~1.44M images, Oxford VGG) — for empty scenes. Purpose-built to contain no humans:
  filtered with RetinaFace + Cascade-RCNN *and* human verification, then cleaned again across
  versions (v2 removed 472 stragglers, v3 another 131).
- **CrowdHuman val** (4,370 images, ~23 people/image) — for crowds. Exhaustively annotated,
  **double-checked by different annotators**, every person carries head + visible-region +
  full-body boxes across mild-to-severe occlusion.
  - Subsampled to a random 500 images (seed 51)
- **Open Images subset, hand-verified** (~100–200 images) — for the human-shaped-object risk
  (toys/dolls/mannequins/statues). Sourced by filtering Open Images to `Doll` / `Teddy bear` /
  `Sculpture` (+ any mannequin-like class), then **manually verified.**

## 2.5 Method

Same two-stage shape as EXP-2026-02 (production code runs untouched), run **once per dataset**:

1. **Stage A (pod, GPU):** run the production `autolabel_sam.py` **unchanged** (frozen settings:
   prompts `"woman" "man" "child"`, conf 0.4, nms_iou 0.7) over each dataset's images.
2. **Stage B (CPU):** `run_autolabel_on_manifest.py`, extended with two modes:
   - **Negatives mode** (PASS, toys/mannequins): no ground-truth boxes. Every SAM detection is a
     false positive. Report FP count per image, % of images with ≥1 FP, and which prompt fired
     (woman/man/child). On the toy set, attribute each FP to the object type (doll/mannequin/…).
   - **Crowd mode** (CrowdHuman): exhaustive GT. Reuse `match_boxes`/`iou` unchanged, add
     precision (matched detections ÷ all detections), recall stratified by occlusion (from the
     CrowdHuman visible/full-body ratio), and duplicates-per-GT (how many SAM boxes hit one
     person — the fragmentation metric).
3. **Proof:** the `build_error_gallery.py` gallery, per dataset — FP montages (what did SAM fire
   on?), and for CrowdHuman, unmatched-detection and duplicate montages, plus the hand-checked
   sample separating poster/statue "FPs" from genuine hallucination.

## 2.6 Results — per dataset

### 2.6.1 PASS (empty scenes) — hallucination baseline

**Run 2026-07-13:** 3,000 person-free images (streamed sample of PASS.0.tar) · **251 false
person-labels** · **8.37 per 100 images** · **5.4% of images affected** ; when it fires, ~1.5 labels per affected image.

> 📊 **⬇︎ INSERT IMAGE HERE → `assets/png/EXP-2026-03/sam_fp_class_split.png`**  (drag this file from Finder into the doc)

| Prompt | Spurious labels |
| ------ | --------------- |
| Man    | 173 (69%)       |
| Woman  | 51              |
| Child  | 27              |

Roughly 1 in 18 genuinely person-free images still gains a person label. Note the class flip vs the object set: on empty scenes the dominant spurious class is *Man*, on dolls/statues it was *Woman*. What the "man" prompt fires on in empty scenes is answered by the overlay montage (100 written) — _adjudication pending_.

### 2.6.2 Open Images toys/statues/dolls — human-shaped-object false positives

**Run 2026-07-13:** 259 hand-verified person-free images · **373 false person-labels** · **62.2% of images got at least one**

> 📊 **⬇︎ INSERT IMAGE HERE → `assets/png/EXP-2026-03/sam_fp_by_object_type.png`**  (drag this file from Finder into the doc)

| Object type      | Images | Images with ≥1 FP | Total FPs           |
| ---------------- | ------ | ------------------ | ------------------- |
| Doll             | 80     | **88.8%**    | 181 (2.3 per image) |
| Bronze sculpture | 66     | 65.2%              | 73                  |
| Sculpture        | 80     | 52.5%              | 108                 |
| Teddy bear       | 33     | 15.2%              | 11                  |

By prompt: **Woman 180 · Man 140 · Child 53** — dolls and statues most often enter the training data as "woman" labels.

### 2.6.3 CrowdHuman — crowd precision, occlusion recall, duplicates

**Run 2026-07-13:** 500-image seed-51 sample · 11,259 GT persons · 12,667 SAM detections
(2,644 excluded by ignore regions, 10,023 scored).

> 📊 **⬇︎ INSERT IMAGE HERE → `assets/png/EXP-2026-03/sam_crowd_recall_by_occlusion.png`**  (drag this file from Finder into the doc)

| Metric                        | Result                                         |
| ----------------------------- | ---------------------------------------------- |
| Detection precision           | **86.6%** (8,683/10,023, CI 86.0–87.3%) |
| Recall, light occlusion       | **82.7%** (5,164/6,246)                  |
| Recall, partial occlusion     | **73.5%** (2,717/3,699)                  |
| Recall, heavy occlusion       | **61.0%** (802/1,314)                    |
| Recall, overall               | **77.1%** (8,683/11,259)                 |
| Duplicates per matched person | **1.1%** (99/8,683)                      |

**The headline: SAM silently misses people in crowds.** Nearly 1 in 4 real people go unlabeled (77.1% recall), and it degrades with occlusion exactly as you'd expect (82.7% → 73.5% → 61.0%).

## 2.7 Limitations (pre-registered)

- **Detection only.** No classification/gender/age here (CrowdHuman has no such labels).
- **Domain mismatch.** PASS, CrowdHuman and Open Images are general Western-ish photos, not the scraped imagery production runs on
- **The mannequin ruling is a judgment call**; a different ruling changes the toy-set FP count directly.
- **The toy/mannequin set is small (~100–200 images)** — wide confidence intervals; it tells us
  *whether* the failure mode is real and roughly how big, not a precise rate.

---

# 3 · Appendices

## Appendix A — datasets, access, and label-quality sources

- **PASS** — https://www.robots.ox.ac.uk/~vgg/data/pass/ · paper: arXiv 2109.13228. CC-BY,
  person-free by construction (face/human-detector filtered + human-verified + version cleanup).
- **CrowdHuman** — paper: arXiv 1805.00123 · exhaustively annotated, annotator double-check, head
  + visible + full-body boxes, ~23 people/image, 4,370 val. Use visible-region boxes.
- **Open Images V7** — non-exhaustive by design (image-level positive/negative labels; only
  positive categories boxed); Google's own docs flag person/fairness evals as potentially
  misleading. Used here ONLY as a source for the hand-verified object subset.
- **COCO** — documented missing person annotations (background and some foreground). Rejected as
  a negatives source for the same non-exhaustiveness reason.

## Appendix B — verify it yourself (method, written before running)

_Same standard as EXP-2026-02 Appendix B: the mechanism in verbatim code, quoted from
`vlm-cluster/eval_negatives_crowd.py` (new for this experiment; `--selftest` covers both modes
synthetically, no GPU/data). Box matching reuses `match_boxes`/`iou` from
`run_model_children.py:44/84` and polygon→box parsing reuses `seg_boxes` from
`run_autolabel_on_manifest.py:77` — all three already quoted in full in EXP-2026-02 Appendix B;
not repeated here._

### B.1 What counts as a false positive (negatives mode)

There is no matching step at all — the datasets are person-free by construction, so **every**
detection is an FP. The entire scoring is a count (`eval_negatives_crowd.py:99`, `run_negatives`):

```python
dets = seg_boxes(pred_dir / (img_path.stem + ".txt"), w, h)
if dets is None:            # no .txt -> Stage A hasn't run here -> excluded, counted separately
    n_not_processed += 1
    continue
classes = [class_names.get(d[0], f"class_{d[0]}") for d in dets]
fp_total += len(dets)       # every detection on a person-free image is a false positive
fp_by_class.update(classes)
```

The trust therefore lives in the *data*, not the code: PASS is person-free by construction
(detector-filtered + human-verified), and the object set is hand-verified per the §2.2 ruling.
**Object set as built (2026-07-13):** 259 Open Images validation candidates —
doll 80, sculpture 80, bronze sculpture 66, teddy bear 33 — reviewed image-by-image against the
§2.2 ruling via `contact_sheet.html`; **0 rejected** (no real people or printed depictions of
people found). Open Images has no `Mannequin`/`Statue`/`Figurine` class, so `Sculpture` +
`Bronze sculpture` serve as the photorealistic-human-form proxy. Second, free check at scoring
time: the FP overlays (images where SAM actually fired) are re-examined so any small background
person missed at thumbnail size can still be caught and excluded from the FP count.

### B.2 CrowdHuman ground truth parsing (crowd mode)

One JSON per line; `tag != "person"` or `extra.ignore == 1` becomes an **ignore region**; people
are matched on the **visible** box, with occlusion = vbox/fbox area ratio (`eval_negatives_crowd.py:162`, `load_odgt`):

```python
for gb in r.get("gtboxes", []):
    extra = gb.get("extra") or {}
    if gb.get("tag") != "person" or extra.get("ignore") == 1:
        ib = gb.get("fbox") or gb.get("vbox") or gb.get("hbox")
        if ib:
            ignores.append(_xywh_to_xyxy(ib))
        continue
    vbox = gb.get("vbox") or gb.get("fbox")
    fbox = gb.get("fbox") or vbox
    ratio = _area(v) / _area(fb)           # occlusion: visible / full-body area
    persons.append({"vbox": v, "occ_ratio": min(1.0, ratio)})
```

### B.3 Ignore regions, precision, and the unmatched-detection buckets (crowd mode)

A detection mostly inside an ignore region is excluded before scoring (standard CrowdHuman practice), by intersection-over-**detection**-area (`eval_negatives_crowd.py:198/238`):

```python
def _ioa(det_box, region):                 # how much of the DETECTION sits in the region
    inter = ...
    return inter / det_area

if any(_ioa(list(d[1:5]), ig) > IGNORE_IOA for ig in rec["ignores"]):   # 0.5
    ignored.append(d)
```

Precision = matched ÷ kept detections. Every kept-but-unmatched detection is bucketed by its best IoU against any GT person (`eval_negatives_crowd.py:260`):

```python
best = max((iou(v, d[1:5]) for v in vboxes), default=0.0)
if best >= DUP_IOU:        # 0.5  -> duplicate/fragment (extra box on a found person)
elif best >= PARTIAL_IOU:  # 0.1  -> partial overlap, ambiguous
else:                      #      -> clear false-positive CANDIDATE
```

### B.4 Every threshold, with its source line

| Parameter              | Value                                       | Where                                          | Note                                               |
| ---------------------- | ------------------------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| SAM3 conf / nms_iou    | 0.4 / 0.7                                   | `autolabel_sam.py` defaults                  | frozen production settings                         |
| match IoU (crowd)      | 0.5                                         | `--match-iou`                                | same as EXP-2026-02;**not calibrated**       |
| ignore exclusion IoA   | 0.5                                         | `IGNORE_IOA`, `eval_negatives_crowd.py:67` | standard practice;**picked, not calibrated** |
| duplicate threshold    | 0.5                                         | `DUP_IOU`, `:69`                           | **picked, not calibrated**                   |
| partial/clear-FP split | 0.1                                         | `PARTIAL_IOU`, `:70`                       | **picked, not calibrated**                   |
| occlusion bands        | ≥0.7 light / 0.3–0.7 partial / <0.3 heavy | `OCC_LIGHT, OCC_HEAVY`, `:65`              | vbox/fbox area ratio                               |
| PASS sample            | 3,000 imgs, seed 51                         | runbook Phase 1b                               | one tarball, random sample                         |
| object set             | ~100/class candidates → hand-verified      | `make_object_negatives.py`                   | seed 51; final counts recorded in B.1              |

### Pre-registered thresholds

| Parameter                        | Value                                              | Note                                                               |
| -------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| SAM3 conf / nms_iou              | 0.4 / 0.7                                          | frozen production settings, unchanged                              |
| detection match IoU (CrowdHuman) | 0.5                                                | same as EXP-2026-02; sensitivity-swept if the number looks fragile |
| CrowdHuman GT box                | visible-region                                     | pairs better with SAM masks than full-body                         |
| duplicate definition             | ≥2 SAM boxes matching one GT person at IoU ≥ 0.5 | the fragmentation metric                                           |
| occlusion bands                  | from CrowdHuman visible/full-body area ratio       | e.g. <0.3 heavy, 0.3–0.7 partial, >0.7 light                      |
