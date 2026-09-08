# SAM3 output handoff — for running another VLM (e.g. GPT-5.6 Sol) on our detections

Everything you need is already on the RunPod volume. Nothing needs to be exported or
copied. This document tells you where it is, what the fields mean, and — most importantly
— **how to build the crop and prompt so your run is directly comparable to ours.**

We already ran these exact detections through **Gemini 3.5 Flash-Lite**, so your output
can be diffed against ours row for row.

Dataset root (all paths below are relative to it):

```
/workspace/datasets/haramblur_holdout/
```

> This is a **holdout test set** (`DO_NOT_TRAIN` marker at the root). Evaluation only —
> never feed it into a training cycle.

---

## 1. Images

```
labeling/full/images/          11,494 images
```

These are **symlinks** into `../../images/<collection>/`. Fine to read in place; if you
copy them anywhere, use `cp -rL` to dereference.

The collection is the filename prefix: `child__`, `men__`, `randoms__`, `shiekhs__`,
`women__`, `women_hd__`. Keep results **split by collection — never pooled**; `randoms`
is a person-free negatives arm and `shiekhs` is the Gulf/traditional-dress slice.

## 2. SAM3 output

```
labeling/full/raw/<stem>.txt     flattened YOLO segment polygons
labeling/full/raw/<stem>.json    the raw sidecar  ← richer, use this
```

One pair per image, 11,494 of each. **21,004 detections total.**

### `<stem>.json` — the sidecar (verified 2026-08-20)

```json
{
  "image": "child__000096-google-com-child-678x452-images.jpg",
  "width": 678, "height": 452,
  "classes": ["woman", "man", "child"],
  "conf_threshold": 0.4,
  "nms_iou": 0.7,
  "detections": [ {"cls": 2, "conf": 0.9272,
                   "box": [x1, y1, x2, y2],
                   "parts": [[[x, y], [x, y], ...], ...]} ],
  "suppressed": [ ...same shape... ]
}
```

| field | meaning |
|---|---|
| `cls` | `0 = woman`, `1 = man`, `2 = child` — SAM3's own prompt-order class |
| `conf` | SAM3 detection confidence (floor 0.4) |
| `box` | `[x1, y1, x2, y2]` in **pixels** of the full image |
| `parts` | the person's mask as **separate polygons, before YOLO flattening** — the honest geometry for an occluded person split into multiple visible pieces |
| `suppressed[]` | detections NMS dropped. **Same key shape as `detections[]`** (`cls`, `conf`, `box`, `parts`). Not part of the 21,004 — available if you want to test whether a different verifier rescues them |

### `<stem>.txt` — the YOLO labels

One line per detection, `class_id` followed by a normalised **segment polygon** (tens to
hundreds of coordinates per row — *not* 4 box values). A plain 5-field YOLO box reader
silently drops every row and reports nothing; use a polygon-aware parser (the project's
`seg_boxes`).

### The join key

**`(image_stem, det_index)` where `det_index` is the line index in `<stem>.txt`**, which
is also the index into the sidecar's `detections[]`. Everything downstream — our verdicts,
our audit trail, our emitted labels — keys off this. **Your output must carry it** or the
two runs can't be compared.

## 3. Building the crop — import ours, don't reimplement

This is the one thing that decides whether the comparison is apples-to-apples. Do not
re-derive the crop; call the function we actually ran:

```python
import sys; sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import build_crop, PROMPT, PROMPT_VERSION

crop, scale = build_crop(img, det["box"], det["parts"])   # img = PIL.Image of the full image
```

`build_crop` (`spotlight_run.py:136`) is a pure function of image + box + parts, so it
reproduces byte-identically the crops Gemini saw:

- **25% padding** around the SAM3 box (`PAD = 0.25`)
- **upscale** so `max(side) >= 320` px, LANCZOS (`MIN_CROP_SIDE = 320`) — free, since the
  API bills small images the same
- the detection's **own mask parts** drawn as a two-tone outline,
  `(0,220,90)` over `(0,70,20)`, width 3

**A plain rectangle crop is not equivalent, and this is measured, not theoretical.**
EXP-2026-10 found box-outline crops failed because the rectangle contains neighbouring
people and the VLM labels the wrong one; switching to mask outlining plus the 320 px floor
took crowd TP-keep from 80.4% to 97.6%. If you crop differently, you will be measuring the
harness, not Sol.

## 4. The prompt

Same module, same import:

```python
from spotlight_run import PROMPT, PROMPT_VERSION      # "spotlight-e1"
import hashlib; hashlib.sha256(PROMPT.encode()).hexdigest()[:12]   # -> 4c85ffbf8bcb
```

Verify that sha before you spend anything — if it isn't `4c85ffbf8bcb`, the prompt has
changed and your run is not comparable to ours. The full frozen text plus every config
value is also in `labeling/full/run/run_meta.json`.

The prompt asks for one JSON object with nine fields: `verdict`
(`real_person` / `depiction` / `not_person`), `gender`, `age_group`, `estimated_age`,
`highlight_quality`, `confidence`, `apparent_race`, `head_covering`, `exposed_body_parts`.

Two rulings baked into it, worth knowing when you read results: a **photo or poster of a
real human counts as a person** (`depiction`, and we keep those), while a **doll,
mannequin, statue, toy, cartoon or drawing does not** (`not_person`, deleted). And gender
is judged from face and body, explicitly **not** from robes or clothing style.

## 5. Our Gemini baseline to diff against

```
labeling/full/run/verdicts_batch.jsonl     21,004 rows
```

One row per detection: `image_stem`, `det_index`, `box`, `sam_class_id`, `sam_conf`,
`n_parts`, `img_wh`, `blurriness`, `person_px_height`, and `v` — the parsed Gemini verdict
with all nine fields — plus `raw_text`, the unparsed response. Keep your raw responses
too; ours have already paid for themselves twice when a parse looked wrong.

Supporting files: `run/_audit.jsonl` (every delete / relabel / unknown decision with its
reason), `run/_emit_stats.json`, `run/cost_report.json`.

## 6. Read this before interpreting any disagreement

**Neither model is ground truth.** These labels are Gemini's judgement of SAM3's
detections, with no human verification. A Gemini-vs-Sol diff is an **agreement** analysis —
disagreements are candidates for human adjudication, not evidence that either model is
wrong. Say so in whatever you write up.

**Where the two are most likely to differ** — worth oversampling if you subset:

| stratum | n | why it's interesting |
|---|---:|---|
| SAM3 `Woman` misreads on `shiekhs` | 422 | the Gulf-dress bias; Gemini called 2,661/2,689 ghutra wearers `man` and **0** `woman` |
| relabels (Gemini overruled SAM3) | 607 | `Woman→Man` runs 2.48:1 over `Man→Woman` |
| `not_person` deletions | 3,246 | the false-positive gate doing its job |
| unknown-gender abstentions | 1,619 | where Gemini refused to commit |

**Cost, so it isn't a surprise.** Our entire Gemini run was **$6.78** using the Batch API
(21,004 detections, 31.1M input + 1.7M output tokens, $0.323/1k detections). Sol is
**$5.00 / $30.00 per 1M** against Flash-Lite's **$0.30 / $2.50** — roughly 30×. At our
measured token counts, all 21,004 detections project to **~$200**, or ~$100 on OpenAI's
Batch API. A stratified subset over the four strata above will answer the question for a
fraction of that.

**One layout trap:** `labeling/full/labels_eval/` is a symlink to `labels_std/`, and both
contain `_audit.jsonl` and `_emit_stats.json` beside the 11,494 `.txt` files — glob
`*.txt`, never `*`.

---

## Quick sanity check before you spend anything

```python
import sys, json, glob, os, hashlib
sys.path.insert(0, "/workspace/autolabel_pipeline_v2")
from spotlight_run import build_crop, PROMPT
from PIL import Image

H = "/workspace/datasets/haramblur_holdout/labeling/full"
assert hashlib.sha256(PROMPT.encode()).hexdigest()[:12] == "4c85ffbf8bcb"

f = sorted(glob.glob(H + "/raw/*.json"))[0]
sc = json.load(open(f)); stem = os.path.basename(f)[:-5]
det = sc["detections"][0]
img = Image.open(f"{H}/images/{sc['image']}")
crop, scale = build_crop(img, det["box"], det["parts"])
crop.save("/tmp/check.png")          # eyeball it: one person, green mask outline
print(stem, det["cls"], det["conf"], crop.size)
```

If `/tmp/check.png` shows a padded crop with one person outlined in green, you are
producing exactly what Gemini was shown.
