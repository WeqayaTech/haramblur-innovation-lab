# Proposal: dual-pipeline auto-labeler with VLM arbitration

**In one line:** two independent labeling pipelines vote on every person; where they agree we
accept the label for free, where they disagree a VLM (Gemini) makes the final call — cutting
VLM cost ~85–95% versus VLM-labeling everything, while specifically killing the false-positive
problem SAM3 has today.

**Status:** design proposal — asking for team input before building. Evidence base:
EXP-2026-01 through EXP-2026-06 (all numbers below are measured, sources in the experiment docs).

---

## Why change anything

- Our production labeler (SAM3) wrongly labels a person in **~62% of doll/statue/toy images**
  and hallucinates on **8.4 per 100 empty images** (EXP-2026-03). It matches human *form* and
  never asks "is this a real person?" Those false labels poison training.
- **93% of its classification errors are adults labeled Child** (EXP-2026-02) — the one error
  direction the product cannot tolerate (that adult escapes the blur).
- Commercial VLMs (EXP-2026-06 smoke test) are near-perfect at rejecting statues/dolls and at
  gender on general photos — but too weak at crowd detection (best 59.8% recall vs SAM3's 77%)
  and too expensive (~$4,200 per 500k images) to label everything.
- Conclusion we converged on: **use each system where it's strong** — SAM3 for recall and
  geometry, cheap local models as a second opinion, the VLM only as judge of disagreements.

## The design

### Pipeline A (exists today)
**SAM3** with the production woman/man/child prompts → mask-derived boxes + a class proposal.
SAM3 stays the source of geometry (its boxes come from masks — tight).

### Pipeline B (new, all free/local)
1. A **second person detector**, architecturally different from SAM3 and never trained on our
   labels: either a stock COCO/CrowdHuman-pretrained YOLO or NVIDIA LocateAnything (3B, open
   weights). Which one gets the seat is decided by a benchmark bake-off, not by opinion
   (see "Gates" below).
2. **MiVOLO V2** (dedicated age/gender specialist, human-labeled training data) classifies each
   Pipeline-B crop: age ≤ 12 → Child, else gender → Woman/Man.

Both pipelines independently output the same thing: a list of (box, class ∈ {Woman, Man, Child}).

### The vote
Match the two lists by box overlap (IoU ≥ 0.5, greedy mutual-exclusive — our existing
`match_boxes`). Every detection lands in exactly one bucket:

| Case | Meaning | Route |
|---|---|---|
| Matched box, same class | Both pipelines agree | **Accept** (free — the ~75–85% bulk) |
| Matched box, different class | Both see a person, disagree who | **Gemini crop** → verdict is final |
| Box in A only | SAM3 sees someone B doesn't (statue risk) | **Gemini crop** → real-person? + class |
| Box in B only | B sees someone SAM3 missed (crowd recovery) | **Gemini crop** → confirm + class |

Key property: a statue that fools SAM3 gets no matching Pipeline-B box — **the missing second
opinion IS the disagreement**, so every suspected false positive is routed to the one system we
measured to be near-perfect at rejecting them.

### The arbiter (Gemini 3.5 Flash, batch API)
One crop in → one JSON verdict out: `real_person / depiction / not_person`, gender, numeric age,
child-vs-adult, confidence. **The VLM never supplies geometry** — boxes always come from the
detectors; Gemini only judges existence and class. Its verdict is final and appended to the
label list. Islamic ruling is baked into the prompt: a **printed/photographic depiction of a
real person counts as a person** (posters are gaze-lowering targets); dolls/mannequins/statues/
cartoons are not; men in thobe/ghutra are men — judge by face, not robe.

### Policy rules that override the vote (the error-asymmetry layer)
1. **Every Child label is second-guessed.** Both pipelines share the teen blind spot (every
   model we've tested reads many 13–19s as children — SAM3, Qwen2.5-VL, Claude all measured
   doing it), so *agreement on Child is not trusted*: clearly-under-10 by MiVOLO's number →
   accept; anything reading 10–19 by either voter → **adult, by rule, no model call**
   (over-blurring a borderline 12-year-old is acceptable; an escaped 17-year-old is not);
   ambiguous → Gemini.
2. **2–5% random audit** of the accepted pool goes through Gemini anyway — not as a label
   source but as a thermometer: it measures the error rate of the "agreed" tier instead of
   assuming agreement means correct.
3. **One irreducible human task:** a small hand-checked **Gulf/traditional-dress slice**
   (~100–200 crops) to certify all three classifiers (SAM3, MiVOLO, Gemini prompt) on the bias
   that motivated this whole effort. No amount of model-vs-model agreement can detect a bias
   they might share; only this slice can.

### Final label assembly
`{0: Woman, 1: Man, 2: Child}` from exactly three sources: agreed labels (free) + Gemini
verdicts on disagreements (final) + nothing else. Deleted: everything Gemini calls
`not_person`. Dropped from training rather than guessed: confirmed persons with undeterminable
gender.

## Cost (measured token profiles × verified rates)

Per 500k mostly-person images (~4M detections): GPU passes are pod time we already pay;
Gemini sees ~10–15% of detections (disagreements + Child checks + audit) ≈ **$200–500 total**
(batch API). Versus ~$4,200 to VLM-label everything, or ~$175,000 for human annotation of the
same boxes.

## What we're explicitly NOT doing, and why

- **Not using the VLM as the main labeler** — measured crowd recall too low (best 59.8% vs
  SAM3's 77%); misses shrink the dataset, which is acceptable, but paying 10x for fewer labels
  isn't.
- **Not using our production YOLO as a voter** — it's trained on SAM3's labels, so it inherits
  SAM3's systematic errors and would agree with them (correlation trap). The student model is
  the *consumer* of this pipeline's output, not a judge of it.
- **Not letting MiVOLO near the "is this real?" question** — it's trained only on real people
  and has no reject class; it will happily assign an age and gender to a statue.
- **Not trusting any model in the 13–19 age band** — policy (default-to-adult) instead, because
  six models measured across three experiments all fail there the same way.

## Gates before building (all cheap, all pre-registered)

1. **EXP-2026-08 — detector bake-off:** stock person-YOLO vs LocateAnything on our four staged
   benchmarks (object set, PASS, CrowdHuman, LAGENDA). Seat goes to the lowest object-set
   false-person rate with crowd recall ≥ 60%. ~$0, an afternoon of pod time.
2. **EXP-2026-05 — MiVOLO's teen curve** on the LAGENDA harness (already planned): decides how
   wide the default-to-adult band must stay. Caveat pre-registered: MiVOLO's authors built
   LAGENDA, so add a WIDER/MSP60K cross-check.
3. **The Gulf-dress slice** — build it once (~100–200 crops), reuse it forever.

## Questions for the team

1. Do we agree with **default-to-adult for the whole 10–19 band**, even when both pipelines say
   Child? (Cost: some real 11–12-year-olds get labeled adult → blurred. Benefit: no teen adults
   escape the blur.)
2. **YOLO vs LocateAnything** for the second detector seat — anyone with experience/priors
   before the bake-off? Any other candidate we should add to EXP-2026-08?
3. Is **2–5% audit** the right rate, and who reviews the audit report cadence?
4. The **depictions-count-as-persons ruling** (posters/billboards labeled as people) — confirm
   this is settled policy for training data, since it flows into every label.
5. Who owns building the **Gulf-dress slice**, and what's the sourcing plan for those images?
6. Appetite check: this proposal optimizes for label *cleanliness* over dataset *size* (we
   delete/drop ambiguous cases rather than guess). Any concern about volume?
