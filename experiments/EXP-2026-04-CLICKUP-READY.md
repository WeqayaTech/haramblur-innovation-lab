# EXP-2026-04 — ClickUp-ready copy

> **Upload notes:** paste the text below into ClickUp, then drag in the 3 PNGs from
> `experiments/assets/png/EXP-2026-04/` at the `[IMAGE: ...]` placeholders:
> 1. `exp04_neg_fp_prompt_comparison.png`
> 2. `exp04_crowd_recall_prompt_comparison.png`
> 3. `exp04_crowd_precision_tradeoff.png`

---

# Experiment: Measured purely as a person detector, how does SAM3 do — and does the prompt matter?

**In one line:** first experiment under the three-component framework — we re-ran the production
SAM3 labeler with a single "person" prompt (instead of the production woman/man/child prompts) on
the same three detection datasets from EXP-2026-03, scored as **Component 1: Person Detection**
only, against EXP-2026-03's 3-prompt numbers.

**Status:** Run complete 2026-07-14 (one crowd image pending, see caveat) · PASS 3,000 / objects
259 / CrowdHuman 500-image sample · Owner: Mostafa

## Bottom line

**The three gendered production prompts were acting as an accidental specificity filter. Asked
simply for "person", SAM3 fires far more liberally everywhere — and it hurts more than it helps:**

- **Empty scenes got ~3× worse.** PASS false positives jumped from 8.4 to **27.4 per 100 images**
  (12.7% of empty images gain a person label, vs 5.4% with the production prompts).
- **Human-shaped objects got worse too:** 68.7% of doll/statue images gain a false person label
  (vs 62.2%), with +34% more spurious boxes (193 vs 144 per 100 images). Prompting does not fix
  the form-matching failure — it amplifies it. The person-verifier gate is mandatory either way.
- **The crowd trade is small and mixed:** lightly-occluded recall improved 82.7% → **84.7%**
  (+2.1 points — under the pre-registered 5-point adoption threshold, still failing the 90% bar),
  while precision slipped 86.6% → **84.3%**, just under the 85% bar. Heavy-occlusion recall didn't
  move at all (61%) — occlusion, not prompt phrasing, is the binding constraint.
- **Decision, per the pre-registered rules: do NOT adopt a person-prompt Stage 0.** The
  replacement pipeline keeps class-aware prompting (or gets a detector that isn't SAM3) and adds
  the person-verifier gate that both EXP-2026-03 and this run independently demand.

[IMAGE: exp04_neg_fp_prompt_comparison.png]

## Scorecard (bars pre-registered before running — identical to EXP-2026-03's)

| Dataset | Metric | Bar | 3-prompt (EXP-2026-03) | "person" prompt |
|---|---|---|---|---|
| PASS | FPs per 100 empty images | < 2 negligible · > 10 problem | 8.4 ⚠️ | **27.4** ❌ (~3×) |
| Object set | % images gaining a person label | > 25% = needs a gate | 62.2% ❌ | **68.7%** ❌ |
| CrowdHuman | detection precision | ≥ 85% | 86.6% ✅ | **84.3%** ❌ |
| CrowdHuman | light-occlusion recall | ≥ 90% | 82.7% ❌ | **84.7%** ❌ (+2.1) |
| CrowdHuman | duplicates per matched person | < 5% | 1.1% ✅ | **1.1%** ✅ |

Crowd numbers computed on 499 of 500 images (10,949 of 11,259 GT persons) — one image carrying
310 GT persons produced no label file; resolution pending. If it's a true zero-detection, overall
recall is ~76.3% instead of 78.5% (headline verdicts unaffected).

[IMAGE: exp04_crowd_recall_prompt_comparison.png]

[IMAGE: exp04_crowd_precision_tradeoff.png]

## Why we ran this

Under the new three-component framework (`docs/COMPONENT_FRAMEWORK.md`), Person Detection is its
own component. EXP-2026-03 measured detection *through* the three gendered classification prompts
— entangled. A clean Component-1 measurement needs the detector asked the detector's question
("person"), which requires a new model run — the old outputs can't be re-scored because the
prompts shaped what was detected. It also decides a real design question: should the replacement
pipeline's Stage 0 detect with "person" and classify downstream? **Answer: no.**

## What stayed frozen

Everything except the prompt: same production `autolabel_sam.py`, same checkpoint, conf 0.4, NMS
IoU 0.7, same three datasets byte-for-byte, same scorer and matching (IoU 0.5, CrowdHuman visible
boxes, ignore regions excluded), same "posters count as people, dolls/statues don't" ruling.
Run on an L4 — the same GPU class as the EXP-2026-03 baseline.

## What this does NOT tell us

- Conf 0.4 was frozen under the 3-prompt config; the "person" prompt's score distribution may
  differ, and YOLO outputs carry no confidences, so no post-hoc sweep. A conf sweep (Stage A at
  0.3/0.5) is the named follow-up if anyone wants to rescue the person prompt — though crowd
  recall has no headroom to give.
- Nothing about age or gender (Components 2–3): these datasets have no such labels. MiVOLO V2
  (EXP-2026-05) and the Gulf-dress slice remain the open items there.
- LAGENDA prominent-subject recall was not re-measured with the "person" prompt.
- The crowd clear-FP bucket (357 boxes) hasn't been hand-checked for posters/statues vs
  hallucinations (same protocol as EXP-2026-03; overlays are in `/workspace/exp04/eval/`).

## Verify it yourself

Full verbatim methods appendix (exact commands, matching function quoted in full, every
threshold with its source line) is in the repo doc:
`experiments/EXP-2026-04-sam-person-prompt-detection.md`, Appendix B.
