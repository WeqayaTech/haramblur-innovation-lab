# Experiment: Can Qwen2.5-VL-7B (a vision AI) check our labels?

**In one line:** we tested whether an off-the-shelf vision AI (Qwen2.5-VL-7B) can look at a person in a photo
and correctly tell us their **gender** and whether they're a **child** — accurately enough to double-check the
labels our auto-labeling pipeline produces.

**Status:** Done · tested on 5,000 people · Owner: Mostafa

---

## The short version

- **Gender: excellent.** When the AI commits to a gender, it's right **99.3%** of the time. It only refuses
  to guess on about 1 in 5 adults. → Usable now, with a human handling the cases it's unsure about or escalating to a bigger model.
- **Child vs adult: catches kids reliably.** It correctly spots **99.6%** of real (under-13) children. → The
  safety-critical part works.
- **The one weak spot is teenagers.** It reads most 13–17-year-olds as "children."
- **Don't use it for exact age** (e.g. "is this person 35 or 55"). It's only reliable for the simple child-vs-adult split.
- **Where we draw the "child" line is our choice, but the model's behavior is fixed.** We also tried a younger cutoff (under-9) — it doesn't help. The model reads roughly under-18 as child-like no matter how we define it, so a stricter definition just makes the accuracy *look* worse without changing anything real.
- **Important Notes:**
  - this was tested on general internet photos, **not** on the Gulf/traditional-dress images.
  - What was tested is the **Person Classification** part **not Person Detection**

## Why we did this

Our training images are labeled automatically. If those labels are wrong, the model we train on them gets
worse. We want an automatic way to double-check labels before they go into training — so humans only
review the tricky cases instead of every single image. The question is simply: **is Qwen2.5-VL-7B good enough
to be that checker or possibly integrated into Autolabeling pipeline?**

## What we wanted to find out

Can Qwen2.5-VL-7B look at a person and get their **gender** and **child/adult** status right, compared to
labels that humans carefully assigned?

Before running anything, we agreed on what "good enough" means:

- **Gender:** right at least **99%** of the time (for adults — we don't ask about children's gender; see below).
- **Child vs adult:** right at least **97%** of the time.

## How we did it

1. **The answer key.** We used a public dataset called **LAGENDA** — 5,000 photos of people where humans have
   already recorded each person's real age and gender. That's our ground truth to check the AI against.
2. **We asked the AI blind.** For each person we showed Qwen2.5-VL-7B just the cropped photo and asked it to
   describe the age and gender. We deliberately did **not** show it any existing label. If you show an AI a label and
   ask "is this right?", it tends to just agree with you — so keeping it blind gives us its honest read.
3. **We compared.** We lined the AI's answer up against the human answer key and counted how often they
   matched — separately for gender and for age.
4. **What counts as a "child".** For this test, a child = **under 13**. Anyone 13 and older
   counts as an adult.
   1. The ideal Islamic definition of a child is rooted in the onset of puberty (Bulugh). However, because tracking biological changes is highly subjective and creates massive data curation difficulties, a chronological baseline must be established. Selecting **age 13** serves as an effective **statistical middle ground**. This threshold targets the median point of the average biological puberty window, which typically spans **ages 9 to 15**.
   2. Another approach would be to split them into tiers:
      * Tier 1: Absolute Child (Under 9 Years Old)
        * *Islamic Concept* : Non-discerning child ( *Ghayr Mumayyiz* ).
      * Tier 2: The Transition Zone (Ages 9 to 15)
        * *Islamic Concept* : Discerning child ( *Mumayyiz* ) moving into *Bulugh* .
      * Tier 3: Absolute Adult (15+ Lunar / ~14.5+ Solar Years)
        * *Islamic Concept* : Full legal/religious adult status achieved by default age limit.

One note on gender and kids: we deliberately **don't** score gender for children. It's genuinely hard to tell
a baby's gender from a photo, and it doesn't matter for our use. Interestingly, the AI agreed on its own — it
refused to guess a gender for about **70% of children**.


## What we found

**Gender (adults): 99.3% correct.**
When the AI committed to "man" or "woman," it was right 99.3% of the time — 21 wrong out of ~3,000. The only
catch is that it says "not sure" on about **19%** of adults. So it doesn't answer for everyone, but when it
does answer, you can trust it.

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/gender_confusion_matrix.png**  (drag this file from Finder into the doc)
> _Gender confusion matrix_

**Child vs adult: it catches children, but pulls teens to the child side.**
The clearest way to see it is this chart — each bar is a true age group, and it shows how the AI split that
group (green = matched the human answer, red = wrong):

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/prediction_by_gt_group.png**  (drag this file from Finder into the doc)
> _What the VLM predicted per true age group_

- **Real children (under 13):** correctly called "child" 99.6% of the time.
- **Teenagers (13–17):** called "child" 68% of the time — even though, by our cutoff, they're adults.
- **Adults and seniors:** correctly called "adult."

So the AI almost never *misses* a real child. Its mistake is **over-calling teenagers as children**. This
next chart shows the same thing across every age: near-perfect everywhere except a dip at ages 10–19.

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/age_accuracy_by_gt_age.png**  (drag this file from Finder into the doc)
> _Accuracy by true age_

Why it happens: a 13-year-old and a 12-year-old look almost identical — even people can't reliably tell them
apart. So a lot of this "error" is really just the fuzziness of where childhood ends, not the AI being bad.

**Exact age: not reliable.** When we asked it to sort people into finer buckets (child / teen / adult /
senior) it only got **73%** right — it mixes up teens with children and seniors with adults. So for anything
beyond the simple child-vs-adult split, don't rely on it.

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/age_confusion_matrix.png**  (drag this file from Finder into the doc)
> _Age confusion matrix_

## What we can decide from this

- **Turn on gender checking, with a safety net.** Let the AI auto-approve the gender labels it's confident
  about (~80% of them, at 99.3% accuracy) and send the ~20% it's unsure about to a human or a bigger model. That removes most of
  the manual gender-checking work.
- **Use it as a strong child safety-net.** It catches virtually every real child. It will also flag some teens
  as children — but that's the safe mistake for us.
- **Don't use it to guess exact ages.**
- **One decision we need to make:** where is *our* line for "child"? This still need to be investigated.
  * **Proposed Path:** Follow a strict, conservative model classifying anyone **under 9 years old** as a child.

## We checked the under-9 idea — here's what it actually changes

Since the plan is to call only **under-9** a child (so preteens and teens count as adults, and blurring a
12-year-old is acceptable), we re-scored the same 5,000 answers with that line. Three things came out:

- **Young children (9 and under) are still caught 99.9%** — the safety floor holds.
- **The headline accuracy *looks* worse (82% vs 86%), but that's bookkeeping, not the model getting worse.**
  Moving the line to 9 shifts preteens (10–12) into the "adult" column of our answer key — but the model still
  calls them "children." We created those disagreements by changing our own definition, not because
  Qwen2.5-VL-7B changed its answer.
- **Gender still works on the older kids: 99.3%** (it just says "not sure" on a few more of them). So pulling
  teens into the "adult" pool doesn't cost us any gender accuracy.

**The catch to be aware of:** the child line is *our* choice, but the model's behavior is fixed — it reads
roughly **under-18 as child-like**, and no relabeling on our side changes that. So setting our cutoff to 9
does **not** make the model treat a 12-year-old as an adult; under the under-9 line, only ~78% of the people
we now *call* adults (which includes preteens/teens) are labeled "adult" by the model. If we want
preteens/teens actually processed as adults, that needs a different rule (e.g. "process anyone the model isn't
clearly sure is a child"), not just a definition change. What holds at **any** cutoff: real young kids are
flagged, real adults (25+) are identified ~97%+, and gender is 99.3% whenever the model commits.

## Charts for the threshold decision

These three let you see the whole tradeoff without reading numbers.

**Gender is reliable at every age — it just abstains more on kids.** The green line (accuracy) sits at ~99%
from toddlers to seniors; the blue bars show it only *commits* on ~17% of the youngest but ~100% of adults.
So gender is never wrong-by-age — it's cautious-by-age.

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/gender_accuracy_by_age.png**  (drag this file from Finder into the doc)
> _Gender accuracy vs coverage by age_

**What the model calls each true age (detected vs actual).** It says "child" for almost everyone under ~15,
mixes through 15–24, and only reliably says "adult" from ~25+. This is the model's real, fixed boundary — and
it's why moving *our* label cutoff around doesn't change how the model behaves.

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/age_detected_vs_actual.png**  (drag this file from Finder into the doc)
> _What the VLM called each true age_

**Accuracy at each child cutoff.** The three measurable cutoffs (9, 12, 17). "Catch real kids" (green) stays
~98%+ at all of them; overall accuracy and "catch real adults" rise a little as the line moves up. Whatever
line we pick, the safety metric — catching real kids — holds.

> 📊 **⬇︎ INSERT IMAGE HERE → assets/png/child_threshold_sweep.png**  (drag this file from Finder into the doc)
> _Accuracy at each child cutoff_

## What this does NOT tell us

- **It doesn't test our known bias.** LAGENDA is general internet photos. It doesn't include many people in
  Gulf/traditional dress, which is exactly where our model tends to misread men as women. A great score here
  does **not** mean that bias is solved, we need a separate test on our own images for that.
- **It doesn't measure real error-catching yet.** We measured how often the AI is *right*. We have not yet
  measured how often it catches a specific *wrong* label from our actual pipeline, that needs our pipeline's
  real mistakes, which is the next experiment.

## What's next

- Test the same thing on our **Gulf/traditional-dress images** (the real bias check).
- Reduce the ~19% "not sure" rate on gender.
- Run the Same Test with **Bigger Model** as part of the escalation pipeline. 

---

## Appendix — the details (for anyone who wants them)

**Exactly what we asked the AI.** We showed it the cropped photo and this fixed instruction (same for every
photo — never changed mid-run). Note the `{cls}` slot: for this test it was always the word **`person`** — not
a prediction, not the real label, just "this is a person" (true for every crop, so it gives away nothing about
age or gender). The AI answers with a small JSON of attributes; the two we score are `apparent_age_band` and
`apparent_gender`.

```text
You are annotating a person-detection dataset. A detector labeled this crop 'person',
but it can be WRONG. Describe ONLY what is clearly visible; use 'unknown' when unsure.
Do not identify individuals.
Guidance:
- apparent_age_band: ALWAYS pick the single closest band as your best estimate — infant (<1),
  toddler (1-3), child (4-9), preteen (10-12, pre-puberty), teenager (13-17), young_adult (18-29),
  adult (30-49), middle_aged (50-64), elderly (65-79), senior (80+).
- apparent_gender: ALWAYS pick man or woman as your best estimate from visible cues, even if not
  fully certain; use 'unknown' ONLY when no gender cue is visible (fully covered figure, back
  turned, no face).
- (plus attribute fields: facial hair, head covering, garment, skin tone, etc. — used for
  diagnostics, not scored here.)
Return a single JSON object with EXACTLY these keys, then only the JSON.
```

**How the numbers were produced (to reproduce).**

- Sample: 5,000 people from LAGENDA, spread across all ages (with extra sampling around the child/teen edge).
- Two small scripts do the scoring: `vlm-cluster/translation.py` (turns the AI's raw age+gender into whatever
  label scheme we want to test) and `vlm-cluster/eval_taxonomy.py` (compares against the human answer key and
  prints the tables).
- Charts: `experiments/make_charts.py` → `experiments/assets/*.svg`.
- Command:
  ```
  python3 eval_taxonomy.py \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --descs    /workspace/lagenda_eval/descriptions/descriptions.jsonl \
    --schemes  gender_only child_vs_adult child_vs_adult_18 age_band
  ```

**Two terms used above, in plain words.**

- *"Not sure" / coverage:* how often the AI declines to answer instead of committing. High accuracy but lots
  of "not sure" still means a human has to handle the leftovers.
- *"Catches children" (recall):* out of all the real children, how many the AI correctly flagged as children.
