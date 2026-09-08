<!-- CLICKUP UPLOAD NOTES (delete this block after uploading):
  1. Paste this whole file into a ClickUp Doc (it renders Markdown: headings, tables, code, emoji).
  2. Charts are NOT embedded — ClickUp doesn't render SVG. Where you see a "📊 INSERT IMAGE HERE"
     callout, drag the named PNG from experiments/assets/png/EXP-2026-02/ into that spot, then delete the callout line.
  3. All 3 PNGs to drag live in ONE folder — experiments/assets/png/EXP-2026-02/ — in order:
     sam_confusion_matrix.png, sam_child_call_rate_by_age.png, sam_child_threshold_sweep.png (rendered at 2x).
  4. §3 appendices (verbatim code) can be collapsed into a ClickUp toggle if you want a shorter page.
-->

# Experiment: How accurate is the production SAM3 auto-labeler?

**In one line:** we ran the current production auto-labeler (`autolabel_sam.py`, built on Meta's
SAM3) against 5,000 people whose age and gender humans already verified, to measure — for the
first time — how much label noise it feeds into training.

**Status:** Done · 5,000 people (4,936 detected) · scored against the same LAGENDA answer key as
EXP-2026-01 · Owner: Mostafa

> **How to read this doc.** §1 is a 60-second executive summary. §2 is the detailed findings.
> §3 is the appendices — verbatim code and proofs for anyone who wants to check the mechanism
> themselves. You only need §1 to make a decision.

---

# 1 · Executive summary

### Bottom line

SAM3 does two of its three jobs well, and struggles with the third.

**It's reliable at finding people and at gender.** It detects 98.7% of the labeled people in the
set, and on adults it separates women from men almost perfectly (~99%). Neither of these is where
our risk sits.

**It's unreliable at telling children from adults, and the reason is age.** SAM3 has no real notion
of "how old is 12" — it just reacts to how child-like someone looks, and that look fades gradually
through the teens. It's confident on young kids, gets shaky around 10–14, and by the mid-teens it's
close to a coin flip. The net effect: about **1 in 8 adults it detects get labeled "Child."** Almost
all the errors are age; hardly any are gender.

**Why that matters for us.** HaramBlur blurs the opposite adult gender so a viewer can lower their
gaze, and leaves children alone. An adult wrongly labeled "Child" is an adult who won't be blurred,
so the viewer still sees them. Because the mistakes bunch up in the teens and early-20s, the people
slipping through are mostly young adults.

**A second, separate problem we couldn't measure here: false positives on non-people.** We've seen
SAM3 fire on toys, dolls and mannequins, and sometimes empty scenes, writing junk labels into
training. LAGENDA has no person-free images, so this run can't put a number on it — but it's real
and needs its own test (§2.7, §2.8).

**What we'd do about it.** This is the case for building the replacement pipeline. Detect people
with a single neutral "person" prompt (which also removes a labeling bug — see §2.4), then hand off
to a dedicated age model (MiVOLO) for the exact teen range where both SAM3 and our earlier
Qwen2.5-VL-7B verifier break down. §2.6 has the job-by-job scoreboard of which component wins
where, and what's still blocking each decision.

**Still untested, and the thing we care about most:** the Gulf/traditional-dress bias, where men in
thobe/ghutra get read as women. This dataset is general internet photos and says nothing about it.

### Scorecard (vs. bars we set before running)

| What we measured                                 | Result                   | Bar (pre-registered)   | Verdict                   |
| ------------------------------------------------ | ------------------------ | ---------------------- | ------------------------- |
| Detection recall (of labeled people)             | **98.7%**          | ≥ 90%                 | ✅ Pass                   |
| 3-class accuracy (Woman/Man/Child)               | **88.7%**          | ≥90% good · <80% bad | ⚠️ Middle — real noise |
| Gender accuracy (adults, gender-only errors)     | **~99%**           | —                     | ✅ Strong                 |
| Child recall (children correctly left unblurred) | **97.0%**          | ≥ 95%                 | ✅ Pass                   |
| Cross-class NMS leak (double-labeled people)     | **0.7%** (naive)¹ | < 1%                   | ✅ Pass                   |
| Box quality (matched IoU ≥ 0.75)                | **94.5%**          | —                     | ✅ Good                   |

_¹ Naive count; a mutual-IoU-filtered "confirmed" figure is pending and likely lower (§2.5, B.8)._

**What these numbers do and don't cover.** They measure how well SAM handles real, labeled people:
does it find them, and does it get the class right. They are not a measure of how clean the label
set is overall — anything SAM puts on a toy, an empty scene, or a duplicate box never shows up in
these numbers (§2.4, §2.7).

**One check worth mentioning.** We made sure the 88.7% isn't just an artifact of crowded photos —
a reviewer pointed out the matcher could grab the wrong box when, say, a parent is holding a child.
Only 3.6% of the errors fit that pattern, so the number holds (§2.5).

---

# 2 · Detailed findings

## 2.1 Context

Every label the HaramBlur model trains on comes from `autolabel_sam.py`. We have an architectural
critique of it (class assignment by comparing SAM3 per-prompt confidences is essentially
arbitrary; two class-prompts can both survive NMS for one person; fragmented masks produce loose
boxes — see `docs/AUDIT_HISTORY.md` → Track 1), but **no measurement**. Before investing in the
proposed replacement pipeline, we want a number: how often is the current method actually right?
That number also tells us how much mislabeled data is already in training.

EXP-2026-01 answered "can Qwen2.5-VL-7B *check* labels?" using the LAGENDA answer key. This
experiment points the same answer key at the thing that *makes* the labels.

## 2.2 Objective & success bars

When the production auto-labeler labels a photo of a person, how often does it (a) find the person
at all, and (b) give them the right class (Woman / Man / Child)?

Before running anything, we agreed what the numbers would mean — **written before the run, not
moved after**:

- **3-class accuracy (on people it detects):** ≥ **90%** = better than feared, lower priority to
  fix; < **80%** = a major training-noise source that justifies the replacement pipeline on data,
  not just critique.
- **Child recall:** ≥ **95%.** Children are the class we want to *leave unblurred*; low recall
  means we'd wrongly blur real children — tolerable per the product, but a quality signal.
  (Earlier drafts called this "safety" — wrong connotation; corrected.)
- **Detection recall (finding people at all, IoU ≥ 0.5):** ≥ **90%.** Below this, real people are
  silently unlabeled and YOLO training treats them as background.
- **Dual-class survivors (same person kept under two classes):** < **1%.** Above this, the
  cross-class NMS leak is a real dataset defect, not a theoretical one.

## 2.3 Method

1. **The answer key.** LAGENDA — the same 5,000-person, human-verified age+gender set used in
   EXP-2026-01. We convert each person's real age+gender into the production classes (Woman / Man
   / Child, child = under 13) with the same `translation.py` rules — so both experiments are
   scored on identical definitions.
2. **What we ran.** `autolabel_sam.py` exactly as it runs in production — same prompts, confidence,
   NMS, polygon code. **No fixes applied**: the moment we improve anything, the result stops
   describing production.
3. **How we compared.** Each SAM detection is matched to the human-verified person box it overlaps
   most (IoU ≥ 0.5). Matched → compare classes. Unmatched human box → the labeler missed a person.
4. **The exact code for every step above is in Appendix B** — matching, GT translation, sampling,
   thresholds — quoted verbatim so you can verify it, not just trust this summary.

## 2.4 Results

One reminder before the numbers: every result here starts from a real, labeled person and asks
whether SAM found them and got their class right. None of them say anything about labels SAM
invents on toys, empty scenes, or duplicate boxes — those land on nothing, so nothing grades them
(§2.7).

**Detection — 98.7%** (4,936/5,000, CI 98.4–99.0%), even across all three classes (Woman 98.7%,
Man 98.7%, Child 98.9%). Worth being precise about what this means: it's recall over the people
LAGENDA labeled, not everyone in the photo. LAGENDA doesn't mark every person, so what it really
says is that of the people we have a trusted label for, SAM found almost all of them. It doesn't
prove SAM finds everyone (§2.7).

**Classification — 88.7%** on the people it found (4,377/4,936, CI 87.8–89.5%). That lands between
the two bars we set: real, meaningful label noise, but not the "major noise source" the sub-80%
line would have meant.

> 📊 **⬇︎ INSERT IMAGE HERE → `assets/png/EXP-2026-02/sam_confusion_matrix.png`**  (drag this file from Finder into the doc)
> The errors are almost all age, not gender. Of the 536 adults SAM got wrong, only 38 (7%) were an
> actual Woman/Man mix-up; the other 498 (93%) were adults it called "Child." Left to itself, SAM
> tells women from men about 99% of the time — it's the age call that pulls the score down.

And its sense of "child" has no clean edge. It's a gradient that fades out through the teens:

> 📊 **⬇︎ INSERT IMAGE HERE → `assets/png/EXP-2026-02/sam_child_call_rate_by_age.png`**  (drag this file from Finder into the doc)

| True age         | SAM3 calls "Child" |
| ---------------- | ------------------ |
| 0–4             | 99.5%              |
| 5–9             | 99.3%              |
| 10–14           | 88.8%              |
| **15–19** | **36.3%**    |
| 20–24           | 5.8%               |
| 25–29           | 1.5%               |
| 30+              | 0.0%               |

That's the whole accuracy gap in one picture — the mistakes sit in the teen band, not spread
across every adult age. Our earlier Qwen2.5-VL-7B verifier (EXP-2026-01) fell apart in exactly the
same range. Two completely different models breaking at the same ages tells us teens are just hard,
not that one model is broken.

### Why moving the "child" cutoff changes the number

This part trips people up, so it's worth being explicit. SAM's behavior never changes when we move
the cutoff — the cutoff is *our* ruler, not SAM's. SAM makes one fixed call per person, "Child" or
"not Child," based only on the fade above. The cutoff is just the age where *we* decide the right
answer should flip from Child to Adult. So if accuracy shifts when we move it, that's us moving the
goalposts, not SAM doing anything different.

Take a real 16-year-old. SAM calls 16-year-olds "Child" about 36% of the time — that's fixed:

- cutoff **≤12** → right answer is **Adult**; SAM3's 64% "not Child" is right, its 36% "Child" is
  wrong (over-called a child).
- cutoff **≤17** → right answer is now **Child**; same output, flipped grade — the 36% "Child" is
  right, the 64% "not Child" is wrong (missed a child).

> 📊 **⬇︎ INSERT IMAGE HERE → `assets/png/EXP-2026-02/sam_child_threshold_sweep.png`**  (drag this file from Finder into the doc)

| Our cutoff (who*should* be "Child") | Overall accuracy | Child recall    | Adult recall    |
| ------------------------------------- | ---------------- | --------------- | --------------- |
| ≤9 (only ages 0–9 are "children")   | 83.6%            | 99.4%           | 80.2%           |
| **≤12 (current production)**   | 89.4%            | **98.1%** | 86.7%           |
| ≤17 (ages 0–17 are "children")      | 91.4%            | 81.2%           | **98.2%** |

- **Child recall** = of true children (age ≤ cutoff), the % SAM3 labeled Child. **Falls** as the
  cutoff rises (99.4 → 81.2): a higher line pulls teenagers into "child," and SAM3 rarely labels
  teens Child.
- **Adult recall** = of true adults (age > cutoff), the % SAM3 labeled not-Child. **Rises** as the
  cutoff rises (80.2 → 98.2): the problem teens move out of the "adult" group, so they stop
  counting against it.
- **Overall accuracy** rises (83.6 → 91.4) mostly as a **counting artifact** — true adults vastly
  outnumber teens, and ≤17 sits nearest SAM3's real flip point (~15–20), so it agrees with SAM3
  most often. **Higher accuracy here does not mean ≤17 is a better label definition.**

So ≤17 looks best on the accuracy column, but only because it quietly reclassifies the teens SAM
fails on as "children," where their mistakes stop counting. It also gives up ~17 points of child
recall versus ≤12. The direction that actually hurts us is an adult slipping through unblurred, so
of the three, the current ≤12 cutoff is the best balance. The accuracy number alone is not a reason
to move it.

**Child recall — 97.0%** (1,164/1,200, CI 95.9–97.8%). SAM correctly labels 97% of real children
as Child, so we leave them unblurred, which is what we want. That's above our 95% bar, and a bit
below the verifier's 99.6%.

**Double-labeling (the NMS leak) — 0.7%** (36/4,936, CI 0.5–1.0%). SAM runs once per class prompt
and only merges the results when the masks overlap by more than 0.7, so occasionally one person
ends up written under two classes at once. Appendix B.8 walks through the exact code. This 0.7% is
the loose count; the stricter, more trustworthy version is pending and will likely come in lower
(§2.5).

**Box quality — 94.5%** of matched boxes overlap the human box by 0.75 or more. There's a small
tail of loose boxes, which fits the fragmented-mask concern but doesn't prove it.

## 2.5 Robustness check (crowded / overlapping scenes)

These came out of looking through the error gallery by hand. Both re-score data we already had — no
new SAM run.

**Could the matcher be grabbing the wrong box?** In a scene like a parent holding a child, our
matcher might credit the wrong SAM box to a person. We checked: only 3.6% of the wrong-class errors
(20/559, CI 2.3–5.5%) had a correct-class SAM box sitting nearby that could have been the real
match. The other ~96% are genuine misreads, not matching slip-ups.

**Does it get worse in busy photos?** A little. Simple images (few extra detections) score 90.6%;
busy ones score 87.2% — a real but small ~3.5-point gap. Since even the simple images are only at
90.6%, crowding isn't the main story; the age problem is.

**One thing we couldn't fully explain:** 56% of images (2,748/4,899) have at least 3 more SAM
detections than labeled people. We confirmed this isn't our loader dropping boxes (0/4,899
mismatch). It's most likely real people LAGENDA never labeled — mass hallucination would be strange
for SAM — but it could partly be SAM splitting one person into several boxes. The only way to know
is to look at the gallery's crowd-tagged images (§2.8).

## 2.6 Component comparison — where each labeling job stands

The labeling pipeline (and the verification pipeline behind it) needs three abilities: find
people, tell women from men, and tell children from adults. Both completed experiments used the
same 5,000-person answer key and the same scoring harness, so their numbers compare directly.
This is the scoreboard so far:

| Job                            | Candidate                              | Measured result                                                     | Where it stands                                                                                                                                                                                                                            |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Person detection**     | SAM3                                   | 98.7% recall of labeled people (this experiment)                    | Recall is settled — no alternative is likely to beat it meaningfully. The missing half is precision: false positives on toys/empty scenes and duplicate boxes. Blocked on**negatives set** (§2.8).                                 |
| **Gender (adults)**      | SAM3                                   | ~99% on general photos (this experiment)                            | Strong — but untested on Gulf/traditional dress.                                                                                                                                                                                          |
|                                | Qwen2.5-VL-7B                          | 99.3% when it commits; abstains ~19% (EXP-2026-01)                  | Also strong, also untested on Gulf dress — and the earlier val-set audit suggests it shares the attire bias, so the two may fail*together* there. Decision blocked on the **Gulf-dress slice** (§2.8).                           |
| **Age (child vs adult)** | SAM3                                   | coin-flip on teens: 36% "Child" at 15–19 (this experiment)         | Not usable alone.                                                                                                                                                                                                                          |
|                                | Qwen2.5-VL-7B                          | worse in the same band: ~68% of teens read as "child" (EXP-2026-01) | Not usable alone.                                                                                                                                                                                                                          |
|                                | MiVOLO V2 (dedicated age/gender model) | **not yet measured**                                          | The main untested candidate. This harness reuses as-is, so evaluating it is cheap. If its teen curve looks like the other two, the answer is an ensemble plus a default-to-adult policy for the ambiguous band, not a better single model. |

In short: detection is a precision question, gender is a domain question, and age is the genuinely
unsolved one. The three experiments that would complete this table — Gulf-dress slice, MiVOLO
evaluation, negatives set — are all in §2.8, and any new candidate drops into the existing harness
for directly comparable numbers.

## 2.7 Limitations

- **Nothing about the Gulf/traditional-dress bias.** LAGENDA is general internet photos. Low
  gender-swap error here (~1%) says nothing about the scraped bias dirs (`arab_men`,
  `dataset_arab_traditional`, …) where the suspected Shaykh→Woman confusion is attire-triggered,
  not a general SAM3 weakness. **This is the single biggest open gap** (Track 1 item 1).
- **Detection recall only counts the people LAGENDA labeled — not everyone visible in the photo.**
  LAGENDA typically labels one verified person per image; a street scene with ten people might
  carry a single label. So 98.7% means "SAM rarely misses a person we have an answer for" —
  usually the clear, prominent subject. It says nothing about small background people, partial
  views, or crowds, because those aren't in the answer key. The same gap is why false positives
  can't be measured here: an extra SAM box could be a real unlabeled person or a mistake, and we
  can't tell which. Measuring either would need a dataset where *every* person is boxed
  (CrowdHuman/COCO-persons) or a hand-labeled slice. We did verify our loader isn't dropping any
  labels (0/4,899 mismatch between `gt.jsonl` and the label files) — the answer key is used in
  full; it's just inherently partial per image.
- **False positives on non-people are a known but unmeasured defect.** The team has observed SAM3
  labeling toys, dolls/mannequins — and occasionally person-free scenes — as people, injecting
  spurious labels that teach the model to "see" people (and potentially blur) where there are none.
  LAGENDA structurally cannot catch this: every image has ≥1 real person, so there is no
  person-free case to score. The pipeline's only current guard is a filename convention
  (`if "negative" in img_path: continue`, `autolabel_sam.py`) — fragile, since it depends on the
  file being named "negative". Quantifying this needs a dedicated negatives set (§2.8); note the
  Stage-0 redesign keeps SAM3 as the detector, so it does **not** automatically fix this — a
  confidence gate or a person-verifier step would.
- **Measured on LAGENDA's image style**, not the scraped-web imagery production actually runs on.
- **At frozen production settings** (conf 0.4, nms_iou 0.7) — says nothing about other settings.
- **Only the scoring cutoff was varied**, not SAM3's own concept of "child" (which is fixed by its
  prompt/weights; only different prompt wording, untested here, could move it).

## 2.8 Next steps

- **Track 1 item 1 (highest priority):** the Gulf/traditional-dress bias check this experiment
  deliberately did not cover — hand-checked slice, since gender is the reliable label there.
- **Measure the false-positive rate on a negatives set** — run the labeler on person-free images
  (toys, dolls/mannequins, objects, empty scenes, animals) and count spurious labels. This is the
  precision number LAGENDA structurally cannot give, and directly targets the toy/no-person noise
  the team observed (§2.7).
- **Resolve the 56%-crowded puzzle:** spot-check the gallery's CROWD frames. Extra boxes spread on
  distinct people → real crowds (harmless); clustered on the one GT person → duplicate/fragmented
  detection (a real defect to fix in Stage 0).
- **Feed into the Track 1 replacement design:** the dual-class leak and the teen-band failure
  directly motivate Stage 0 (single "person" prompt) and the MiVOLO ensemble member.
- **Optionally test alternate SAM3 prompts** (`"toddler"`, `"kid"`, `"teenager"`) to see if
  wording shifts the implicit age boundary — needs a new run.
- **EXP-2026-02b candidate:** repeat this harness against MSP60K/WIDER (messier, closer to
  production input).

---

# 3 · Appendices — proofs & reproducibility

## Appendix A — setup, commands, and the error gallery

- **System under test:** `/workspace/autolabel/autolabel_sam.py` (286 lines, pod volume —
  verified 2026-07-09), frozen at production settings taken from
  `/workspace/autolabel/run_autolabel.sh` + the script's defaults: HF `facebook/sam3` via
  transformers, prompts `"woman" "man" "child"` (→ class ids 0/1/2), `--conf 0.4`,
  `--nms_iou 0.7` (cross-class NMS on mask IoU), `--batch_size 1`, unmodified
  `mask_to_polygon` (all-contours concatenation). Output is YOLO **segment** polygons; we score
  the polygon's bounding extent.
- **Harness (two stages, so production code runs untouched):**

  - *Stage A (pod, GPU):* run the production `autolabel_sam.py` **unchanged** over the LAGENDA
    images — native output is YOLO label `.txt` files.
  - *Stage B (CPU):* `vlm-cluster/run_autolabel_on_manifest.py` — the scoring script. It reads
    SAM's label files and the human answer key (`gt.jsonl`), pairs each labeled person with the
    SAM box that overlaps them (reusing `match_boxes`/`iou`/`draw_match` from
    `run_model_children.py` rather than reimplementing them), and converts the human age+gender
    into the production Woman/Man/Child classes via `translation.py`. Outputs: `sam_matches.jsonl`
    (one verdict per person), `summary.json` (all metrics), and annotated debug images. Every rate
    in the summary carries a **Wilson 95% confidence interval** — the "give or take" range around
    a percentage given the sample size; the Wilson formula (reused from `eval_taxonomy.py`) is
    used because it stays accurate for rates near 0% or 100%, which several of ours are.
    `--selftest` runs the whole pipeline on tiny synthetic data, so the scoring logic can be
    checked without the pod, GPU, or dataset.
- **Metrics, in plain words:**

  - *Detection recall* — of all human-verified people, how many the labeler found (IoU ≥ 0.5).
  - *3-class accuracy* — on found people, how often the class matches the human answer.
  - *Child recall* — of all real under-13 children, how many it labeled Child.
  - *CI (confidence interval)* — the "give or take" range around a measured percentage, e.g.
    "97.0% (CI 95.9–97.8%)" means the true rate plausibly sits in that range given the sample
    size. Computed with the Wilson method, which stays reliable near 0% and 100%.
  - *Dual-class survivor rate* — how often one person ends up with two surviving labels of
    different classes (the cross-class NMS leak).
  - *IoU* — box-overlap score, 0–1; low matched-IoU tail = loose boxes.
- **Commands to re-run (as actually run, 2026-07-09, on an L4 pod):**

  ```bash
  # Stage A — production labeler, unchanged, ~1.08 it/s on an L4
  export HF_HOME=/workspace/.cache/huggingface   # SAM3 weights + HF token already cached here
  cd /workspace/autolabel
  python3 autolabel_sam.py \
      --input  /workspace/lagenda_eval/lagenda_yolo/images/val \
      --output /workspace/lagenda_eval/sam_autolabel/labels \
      --classes "woman" "man" "child" --batch_size 1
  # 4,899 images in 1h15m28s

  # Stage B — score it (CPU)
  cd /workspace/data_inspection_tools/vlm-cluster
  python3 run_autolabel_on_manifest.py \
      --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
      --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
      --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
      --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
      --out         /workspace/lagenda_eval/sam_autolabel/eval_full
  ```

  Charts: `experiments/make_charts_exp02.py` → `experiments/assets/sam_*.svg`; 2× PNGs for
  ClickUp (SVG can't embed there) in `experiments/assets/png/EXP-2026-02/sam_*.png`, rendered via
  Chrome headless at `--force-device-scale-factor=2`.
- **Visual error gallery:** `vlm-cluster/build_error_gallery.py` builds a single self-contained
  HTML file (embedded thumbnails) with wrong-classification cases (teen cases surfaced first),
  missed detections, dual-class-leak cases (split into "confirmed" vs "likely two different
  people"), and full-frame views of images with unmatched SAM detections (crowd-tagged). This is
  what surfaced the crowd/overlap methodology question. `--selftest` runs without pod/GPU data.
  Full commands in `EXP-2026-02-POD-RUNBOOK.md` Phase 5–6.
- **Format quirks discovered along the way (2026-07-09):** `autolabel_sam.py` writes YOLO *segment*
  polygons, not boxes — the scorer takes the polygon's bounding extent. `gt.jsonl` has no box
  field; the GT box is read from the LAGENDA YOLO label file next to each image, indexed by the
  trailing `_N` in each person's id. `run_autolabel.sh` also confirmed the `dataset_arab_traditional`
  invocation is malformed as written (`--output --"..."` and `--conf0.7` with no space) — recorded
  in `docs/AUDIT_HISTORY.md`, still unresolved, out of scope here.

## Appendix B — Verify it yourself (the mechanism, in code)

_Don't take the numbers on faith. Every claim above reduces to the code below. Snippets are quoted
verbatim from the repo with `file:line` so you can diff them against the source. Where a method has
a known weakness, it's called out — see also §2.7._

### B.1 How the labeler (SAM3) was called

Not an LLM with a chat prompt — SAM3 is a promptable segmentation model. It is run **once per
class name** ("woman", "man", "child"), and the highest-scoring prompt per region wins via
cross-class NMS. Verbatim from `/workspace/autolabel/autolabel_sam.py` (weights: HuggingFace
`facebook/sam3`):

```python
model = Sam3Model.from_pretrained("facebook/sam3").to(args.device)      # autolabel_sam.py:104
processor = Sam3Processor.from_pretrained("facebook/sam3")

inputs = processor(images=batch_images, text=batch_classes,             # one text prompt per class
                   return_tensors="pt").to(args.device)
outputs = model(**inputs)                                               # (under torch.autocast on cuda)
results = processor.post_process_instance_segmentation(
    outputs, threshold=args.conf, mask_threshold=0.5, target_sizes=target_sizes)
```

Frozen invocation (production defaults, unchanged):

```bash
python3 autolabel_sam.py --input <images> --output <labels> \
    --classes "woman" "man" "child" --batch_size 1        # conf=0.4, nms_iou=0.7 are the defaults
```

### B.2 How objects were compared / matched (the box-matching, in full)

**This is the step to scrutinise most** — it is pure box-overlap (IoU), greedy, with no
person-identity or size check. A GT box can in principle match a SAM box belonging to a *different*
nearby person if they overlap ≥ 0.5 (the limitation a reviewer caught; see B.7). Verbatim from
`vlm-cluster/run_model_children.py` (reused, not reimplemented, by the scorer):

```python
def iou(a, b):                                                         # run_model_children.py:84
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

def match_boxes(gt_boxes, dets, min_iou):                             # run_model_children.py:44
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for di, d in enumerate(dets):
            j = iou(gt, d[1:5])
            if j >= min_iou:                    # only pairs above the IoU floor survive
                pairs.append((j, gi, di))
    pairs.sort(key=lambda p: p[0], reverse=True)   # highest-IoU pairs claimed first
    used_gt, used_det, out = set(), set(), {}
    for j, gi, di in pairs:
        if gi in used_gt or di in used_det:        # each GT and each det used at most once
            continue
        used_gt.add(gi); used_det.add(di)
        out[gi] = (dets[di], j)
    return out
```

A SAM segment polygon is reduced to a box by its extent (min/max of the polygon points), in
`seg_boxes` — `run_autolabel_on_manifest.py:77`. The verdict per GT person is then just:
`gcls == pcls → correct`, `det is None → missed`, else `wrong_class`
(`run_autolabel_on_manifest.py:187`).

### B.3 How the ground-truth class was derived (age+gender → Woman/Man/Child)

The human answer key gives numeric age + M/F; both it and the label taxonomy pass through the same
rule, so nothing is hand-tuned per class. Verbatim:

```python
def gt_class(row):                                                    # run_autolabel_on_manifest.py:71
    return _PROD3.label(T.norm_gender(row.get("gt_gender")),
                        T.age_bucket_from_years(row.get("gt_age")))

def _prod3(gender, bucket):                                           # translation.py:129
    if bucket == UNKNOWN:      return None
    if bucket in CHILD_BUCKETS: return "Child"     # CHILD_BUCKETS = infant/toddler/child/preteen (age <=12)
    return _gender_label(gender)                   # else Woman/Man by gender

# numeric age -> band (translation.py:40)
_YEAR_EDGES = [(0,"infant"),(3,"toddler"),(9,"child"),(12,"preteen"),
               (17,"teenager"),(29,"young_adult"),(49,"adult"),(64,"middle_aged"),(79,"elderly")]
```

So "Child" = age ≤ 12; the child-cutoff sweep (≤9 / ≤12 / ≤17) just swaps `CHILD_BUCKETS` — see
`_CUTOFF_SCHEMES` at `run_autolabel_on_manifest.py:~347`.

### B.4 How samples were selected

No sampling in the scorer — **every** person in the 5,000-row LAGENDA manifest is scored (the
sample was fixed upstream by LAGENDA, the same set EXP-2026-01 used). Each `gt.jsonl` row is one
person; the GT box is looked up from the YOLO label file by the `_N` suffix in the id:

```python
def load_gt(path, limit):                                            # run_autolabel_on_manifest.py:106
    for line in open(path):
        r = json.loads(line)
        uid = r["id"]
        box_index = int(uid.rsplit("_", 1)[1])     # trailing _N = which box in the label file
        by_image[Path(r["image"]).name].append(
            {"id": uid, "box_index": box_index,
             "gt_age": r.get("gt_age"), "gt_gender": r.get("gt_gender")})
```

We sample **people (rows), not images** — 5,000 rows = 5,000 labeled people across 4,899 images.
Verified that each row maps 1:1 to a box in the label file (0/4,899 mismatch), so the GT count per
image is everything LAGENDA annotated — but LAGENDA does not label every person visible in a
frame, so an image's GT count is not a count of the people actually in it (see §2.7). (`--limit`
exists only for the 500-image pilot; the full run used none.)

### B.5 How balanced the evaluation data is

From this run's own counts (`n_gt_untranslatable = 0`, so all 5,000 map cleanly):

| GT class (production_3class) | count | share |
| ---------------------------- | ----- | ----- |
| Woman (female, age ≥ 13)    | 1,946 | 38.9% |
| Man (male, age ≥ 13)        | 1,854 | 37.1% |
| Child (age ≤ 12)            | 1,200 | 24.0% |

True-age distribution of the 4,936 detected people (5-yr bins, from `call_rate_by_age`'s `n`):
ages 0–19 are **heavily over-represented** (420 / 440 / 587 / 705 for the 0/5/10/15 bins) because
LAGENDA was deliberately over-sampled around the child/teen edge. This is worth keeping in mind:
the overall accuracy isn't representative of a normal age mix — it leans heavily on the hard young
ages, which makes the headline number pessimistic for adults. That's the reason the by-age curve,
not the single number, is what actually matters. We didn't break the Child class down by gender.

### B.6 Every threshold & configuration, with its source line

| Parameter                | Value | Where set                                        | Note / honesty flag                                                      |
| ------------------------ | ----- | ------------------------------------------------ | ------------------------------------------------------------------------ |
| SAM3 detection conf      | 0.4   | `autolabel_sam.py` `--conf` default          | production default; the 0.7 config (one scrape dir) is**untested** |
| SAM3 cross-class NMS IoU | 0.7   | `autolabel_sam.py` `--nms_iou` default       | a box survives unless it overlaps a kept box by >0.7 → the leak         |
| SAM3 mask threshold      | 0.5   | `autolabel_sam.py:~160`                        | hardcoded in`post_process_instance_segmentation`                       |
| GT↔pred match IoU floor | 0.5   | `run_autolabel_on_manifest.py` `--match-iou` | **picked by convention, not calibrated**                           |
| Child age cutoff         | ≤ 12 | `translation.py:35` `CHILD_BUCKETS`          | swept vs ≤9 / ≤17 in the results                                       |
| Dual-class "clip GT" IoU | 0.5   | `--dual-iou`                                   | naive leak flag                                                          |
| Dual-class mutual IoU    | 0.5   | `DUAL_MUTUAL_IOU_THRESHOLD`                    | **picked by intuition**; separates true dup from 2 people          |
| Near-miss IoU            | 0.3   | `NEAR_MISS_IOU_THRESHOLD`                      | **picked by intuition**; flags matcher artifacts                   |
| Crowd excess threshold   | 3     | `CROWD_EXCESS_THRESHOLD`                       | **picked by intuition**; "3+ more dets than GT = crowd"            |

### B.7 What was NOT done (so you don't over-trust the above)

- **The matcher was not made identity-aware.** We added two *post-hoc* checks (`near_miss_iou`,
  `dual_mutual_iou`) for patterns we found by eye; we did **not** re-derive matching from anything
  beyond box overlap, and we did **not** audit a random sample of *correct* matches for the same
  overlap bug — only the error buckets. The 88.7% could carry the same artifact in the safe
  direction. (Highest-value open follow-up.)
- **The four "intuition" thresholds in B.6 were never calibrated** against a hand-labeled set.
- **We did not confirm the scored polygon output is byte-for-byte what training consumes** — no
  check for a downstream conversion step between `autolabel_sam.py` and the training set.
- Reproduce end-to-end offline (no pod/GPU) with the synthetic self-tests:
  `python3 run_autolabel_on_manifest.py --selftest` and `python3 build_error_gallery.py --selftest`.

### B.8 The cross-class NMS leak, documented in full

**The issue in one sentence:** the production labeler runs SAM3 once per class prompt and then
de-duplicates across classes only when two masks overlap by **more than 0.7** — so two different
class prompts that both fire on the *same* person with mask-IoU in the 0.5–0.7 band **both
survive**, and that one person is written into the YOLO label file under **two contradictory
classes at once** (e.g. a `child` line and a `woman` line for the same body). This is Track 1
critique #2, now measured, not theorized.

**The exact code that causes it** (`/workspace/autolabel/autolabel_sam.py`, verbatim). First the
overlap it uses — **mask** IoU, computed over the segmentation masks, not boxes:

```python
def compute_iou(mask1, mask2):                          # autolabel_sam.py (top of file)
    intersection = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()
```

Then the cross-class NMS. `raw_detections` holds detections from **all three** class prompts mixed
together (each carries its own `class_id`); they are sorted by score globally and a candidate is
dropped **only** if it overlaps an already-kept detection by strictly more than `args.nms_iou`
(default **0.7**):

```python
raw_detections.sort(key=lambda x: x['score'], reverse=True)   # all classes, highest score first
final_detections = []
for det in raw_detections:
    keep = True
    for saved_det in final_detections:
        iou = compute_iou(det['mask'], saved_det['mask'])
        if iou > args.nms_iou:          # <-- 0.7. NOT ">= a lower bound". This is the leak.
            keep = False
            break
    if keep:
        final_detections.append(det)    # a 2nd class for the same person can land here
```

Each surviving detection is then written as its own YOLO segment line, class id included:

```python
annotation = f"{class_id} {poly_str}"   # one line per surviving detection
yolo_segments.append(annotation)
...
f.write("\n".join(yolo_segments))       # both the child-line AND the woman-line get written
```

**Illustrative example — the numbers here are invented to show the mechanism, not a logged case.**
A petite adult woman. SAM3's `"woman"` prompt returns a mask at score 0.61; its `"child"` prompt
returns a slightly different mask (hair/shoulder edges differ) at score 0.55. Their mask-IoU is
0.63. NMS keeps the woman mask first (higher score); when it checks the child mask, `0.63 > 0.7`
is **false**, so the child mask is **not** suppressed. The label file for that image now contains
both `0 <woman-polygon>` and `2 <child-polygon>` for the same person — contradictory supervision
the downstream model trains on.

**Do real cases exist? Partially verified.** The run flagged **36 real people (0.7%)** whose label
file genuinely contains two different-class lines overlapping them — those files exist on the pod,
and each case is recorded by id in `sam_matches.jsonl` and shown in the error gallery. What is
*not* yet verified is that any of the 36 is truly one person double-labeled: some are two
*different* nearby people (e.g. a man sitting behind a child — a case found by eyeballing the
gallery), which is precisely why the mutual-IoU split was added. Until the re-run with
`dual_class_survivors_confirmed` completes and at least one confirmed case is visually checked,
the honest status is: the code provably allows the leak, 36 candidate label files exist, and zero
same-person cases have been confirmed by eye.

**Why raising/lowering one number is the wrong fix.** Lowering `nms_iou` would suppress more true
duplicates but also delete genuinely-distinct adjacent people (two prompts on two real people can
overlap 0.4–0.6). The clean fix is the Stage-0 redesign already proposed in
`docs/AUDIT_HISTORY.md`: detect with a **single neutral `"person"` prompt** (no per-class prompts,
so nothing to cross-suppress), then classify each detected person once.

**How we measure it (scorer side).** For each GT person we collect every SAM detection whose box
clips the GT box (`dual_iou_thr = 0.5`) and record the set of distinct classes; >1 class = a flag.
The naive count was **0.7% of people (36/4,936)**. But two boxes both clipping one GT box can also
be *two different people* (the "man sitting behind a child" gallery case), so we added a
**mutual-IoU** check — do the two class boxes overlap *each other*, not just the GT box? — reported
as `dual_class_survivors_confirmed` (`DUAL_MUTUAL_IOU_THRESHOLD = 0.5`). The gallery splits these
into "confirmed" vs "likely two different people" so each can be eyeballed. Bear in mind the naive
0.7% is an upper bound; the confirmed number is the one to trust and is likely lower.
