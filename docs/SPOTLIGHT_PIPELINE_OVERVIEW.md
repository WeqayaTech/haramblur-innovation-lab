# The Spotlight Pipeline — one-page overview

**What it does:** turns raw images into clean training labels. SAM3 finds every person;
each one is then put "in the spotlight" — outlined alone in a crop — and Gemini judges
exactly that person: real or not, man/woman/child, plus attributes for later analysis.
Fake detections get deleted, wrong labels get corrected, and every raw reading is kept.

**Why it exists:** today's labels come from SAM3 alone, which labels statues and dolls as
people (62% of human-shaped objects), hallucinates people in empty scenes, and mislabels
~10% of adults as children — errors that train the blur model to fail. Spotlight deletes
the fakes and fixes the labels for well under a cent per person.

---

## The five stages

| | Stage | What happens |
|---|---|---|
| 1 | **Detect** | SAM3 finds every person in the image: mask, box, first-guess label, confidence. Everything it sees is logged raw — including detections it later discards — so any analysis can be redone without re-running the GPU. |
| 2 | **Spotlight** | Each detection becomes its own crop: padded 25%, upscaled if the person is small, with the person's exact mask outlined in green. Context stays visible; in a crowd there is no doubt WHO is being judged. Sharpness and person-size are measured locally for free. |
| 3 | **Verify** | The crop goes to Gemini Flash-Lite (one cheap call per person). It answers one JSON: is this a real person / a photo of one / not a person at all; gender and age; a tighter box if needed; whether our outline even landed on the right thing; and ~14 analysis attributes (apparent race, garment, head covering, which body parts are exposed + clothing fit — from which awrah coverage is computed later under any standard — orientation, pose, skin tone, …). |
| 4 | **Merge** | Fixed rules combine both opinions: rejected detections are **deleted**; **Child only if the person looks ≤ 12** (anything older defaults to adult — the blur-safe direction); Gemini's gender wins when it commits; box corrections accepted only after sanity checks. |
| 5 | **Emit + Audit** | Clean YOLO training labels come out, plus a sidecar with every attribute and raw reading (filterable forever — e.g. "all men in thobes facing away from camera"). A 2–5% random sample goes to human audit as a permanent quality thermometer. |

## What gets recorded about every person

**Decision fields** (these determine the final label):

| Field | Values |
|---|---|
| verdict | real_person / depiction (photo-of-a-person — counts) / not_person (deleted) |
| gender | man / woman / unknown |
| age_group + estimated_age | child (≤12) / adult / unknown, plus a numeric age guess |
| box_correction | a tighter box, or "ok" |
| highlight_quality | did our green outline land on the right thing? (good / covers_wrong_object / covers_multiple_people) |
| verdict_confidence | high / low |

**Analysis fields** (for filtering and dataset analysis only — never change the label):

| Field | Values |
|---|---|
| apparent_race | the 7 FairFace groups — white / black / east_asian / southeast_asian / south_asian / middle_eastern_north_african (incl. Arab + North African) / hispanic_latino — plus central_asian_turkic (Kazakh, Uzbek, Uyghur, Turkish…, which FairFace splits), plus other / unknown. Judged from facial features + skin tone; an appearance guess, not an identity claim |
| occlusion + occlusion_percent | none / partial / heavy, plus 0–100 |
| face_visible | yes / no |
| orientation | frontal / three_quarter / profile / back |
| pose | standing / sitting / walking / running / lying / other |
| exposed_body_parts | list: face / hair / neck / shoulders / arms / hands / chest / midriff / back / legs / knees / feet — or none. Awrah coverage is computed later from this, per gender and standard |
| clothing_fit | loose / fitted / tight |
| garment_type | thobe_robe / cloak_bisht / abaya / dress / shirt_trousers / suit / sportswear / swimwear / other |
| head_covering | none / cap_hat / hijab / niqab / ghutra_keffiyeh / turban / helmet / other |
| facial_hair | none / stubble / moustache / short_beard / full_beard |
| gender_cues | short text: which cues drove the gender read (bias debugging) |
| skin_tone_mst (+ confidence) | Monk scale 1–10, with a lighting-reliability flag |

**Measured locally, free** (not asked of Gemini): blurriness (sharpness score),
person_px_height (person size in pixels) — plus SAM3's raw confidence and mask parts.

## What's measured so far (pilot scale)

- Keeps real people: **100%** on regular photos; crowds ~**98%** after the highlight fix
  (confirmation in progress).
- Gender on adults: **100%** so far; deletes **77%** of hallucinated detections on
  verified-empty scenes.
- Known weak spot: teenagers sometimes read as ≤ 12 (every model we've tested shares this);
  the >12-defaults-to-adult rule plus stored age estimates let us tighten policy without
  re-running anything.

## Cost

**~$1 per 1,000 people labeled** (~half that on batch API). Full validation experiment
(1,500 images): ~$14. Labeling a 500k-image corpus: roughly **$400–$3,200 on batch**
depending on how crowded the images are — versus ~$175,000 for human annotation.

## Status

Validation experiment (EXP-2026-10) in progress: mini end-to-end passed, full
500-images-per-dataset run pending the prompt A/B check. Still open before production:
the Gulf-dress bias slice (hand-checked), batch-API integration, and official price
verification. Details, evidence, and all pre-registered acceptance bars:
`docs/PIPELINE_V1_mask_highlight_labeler.md` + `experiments/EXP-2026-10-pipeline-v1-mask-highlight.md`.
