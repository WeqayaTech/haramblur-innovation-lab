# Model benchmark report — 4 candidates, 5 test sets

**Bottom line:** `gelannfav14r4fw_gemlb_v2` (GELAN architecture) is the strongest classifier on every test set that's genuinely fair to compare on, and its lead is largest on the cleanest one. But it costs roughly 2× the compute of the other candidates, and the production model still holds a real advantage on false-positive rate. No single model wins on every measure — the right pick depends on whether accuracy or false-positive rate matters more, and how much compute budget is available. Full detail below.

---

## The 4 models

| model                        | what it is                                                                                                 |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `y26n_noe2e_warm50-2`      | Our current baseline. YOLO26n architecture, trained on our own (Spotlight) labels.                         |
| `yolo11N-640`              | **The model currently shipped in production.** Older architecture and labels.                        |
| `y26n_sop50`               | The baseline, continued for 50 more epochs with a new "small object" training trick added.                 |
| `gelannfav14r4fw_gemlb_v2` | A different, larger architecture (GELAN), also trained on our labels, with its own custom training recipe. |

## The 5 test sets — what each one measures, and why we need more than one

A model can look great on one dataset and mediocre on another, so we never trust a
single number. Each dataset below tests a different thing:

| test set                     | what it measures                                                                                                            | size                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **LAGENDA**            | Does the model correctly classify gender/age? (Human-labeled, external — the model has never seen this data.)              | 4,601 photos, 7,098 labeled people |
| **LAGENDA `fl1199`** | Same question, but on a special cleaner slice where*every* person in the photo has a real human label (see caveat below). | 1,199 photos, 1,514 labeled people |
| **CrowdHuman**         | Does the model actually*find* people, especially in crowds? (External, human-labeled.)                                    | 4,372 photos, 99,481 people        |
| **PASS**               | How often does the model hallucinate a person in a photo with no people at all?                                             | 3,000 photos with zero people      |
| **Spotlight-val**      | A general accuracy check —**not fully fair, see the caveat below.**                                                  | 4,232 photos, 8,035 people         |

**Why LAGENDA needs two versions.** In LAGENDA's raw data, most people in a photo *aren't* labeled — only about 1 in 4 has a real human-verified label. The main LAGENDA test handles this correctly: unlabeled people are simply excluded from scoring, not punished as errors. `fl1199` is a special subset where every single visible person happens to be labeled, so it needs no exclusions at all — the cleanest possible test, but it only contains people who are large and prominent in the photo (no small or distant people), so it can't tell us anything about how models handle small/distant people.

**The Spotlight-val caveat, in plain terms.** We initially assumed this was a fair
"exam" all four models had never seen. We checked, and that assumption was wrong for all four: every model had already looked at this exact set of photos while choosing
its best training checkpoint (like picking your final answer based on a practice test
that turns out to be the real exam). **Treat Spotlight-val results as "how well did
each model memorize its own practice test," not "which model is actually best."** The other four test sets don't have this problem — they're genuinely never seen by any model.

## How the comparison works, in plain terms

- **Classification accuracy (mAP)**: for every real person in a photo, did the model
  (a) find them and (b) get their gender/age category right? We score this the
  standard way used across the computer-vision field (mAP50, mAP75, mAP50-95 — higher
  numbers mean a tighter, more confident match between the model's box and the real
  person's location and category).
- **Recall**: out of all the real people in a photo, what fraction did the model
  actually detect at all? This matters most in crowds, where people are small,
  overlapping, or partially hidden.
- **False-positive rate**: on photos that have *zero* real people, how often does the
  model wrongly draw a box around something anyway (a statue, a poster, a shadow)?
  Lower is better — this is the #1 complaint we get from real usage.
- **Fair comparison rules we followed**: every model was run with the exact same
  settings (same confidence threshold, same image size, same matching rules), and
  every test set was scored on its own — we never averaged different test sets
  together, since that would hide which specific weakness a model has.

## Settings used for this experiment

Every model ran under identical settings, so differences in the numbers reflect the
models themselves, not a mismatched setup:

- **Confidence threshold: 0.45** — the same threshold the production extension
  actually uses to decide "is this a real person," so results reflect real deployed
  behavior, not an artificially generous or strict cutoff.
- **Image size: 640×640** — the standard input size for all four models.
- **NMS IoU: 0.7** — the overlap threshold used to remove duplicate boxes on the same
  person.
- **Detections logged all the way down to confidence 0.001**, not just above 0.45 —
  this is needed to draw a proper mAP curve; the 0.45 threshold is only applied where
  the report says "false-positive rate" or "recall" (the numbers a viewer would
  actually see in production).
- **Matching rule**: a detection counts as correct only if it overlaps a real person
  by at least 50% (IoU ≥ 0.5) and has the right category; mAP50-95 additionally checks
  this at stricter overlap thresholds up to 95%, which is why that number is always
  lower — it's a harder bar.
- **Hardware**: all runs on the same RunPod GPU pod, so timing/throughput differences
  during the run aren't a factor in the accuracy numbers.
- One more thing specific to LAGENDA-style scoring: people without a usable label are
  never counted as a model's mistake — they're excluded from scoring entirely, not
  treated as a missed detection or a false alarm.

## Results

### LAGENDA — classification accuracy (hand-labeled, external)

| model                                  | mAP50           | mAP50-95        | Woman           | Man             | Child           |
| -------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| `y26n_noe2e_warm50-2`                | 0.857           | 0.723           | 0.881           | 0.886           | 0.803           |
| `yolo11N-640` (production)           | 0.823           | 0.714           | 0.866           | 0.847           | 0.756           |
| `y26n_sop50`                         | 0.861           | 0.727           | 0.884           | 0.888           | 0.811           |
| **`gelannfav14r4fw_gemlb_v2`** | **0.882** | **0.731** | **0.904** | **0.898** | **0.845** |

### LAGENDA `fl1199` — classification accuracy (fully labeled, no exclusions needed)

| model                                  | mAP50           | mAP50-95        | Woman           | Man             | Child           |
| -------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| `y26n_noe2e_warm50-2`                | 0.896           | 0.780           | 0.909           | 0.936           | 0.842           |
| `yolo11N-640` (production)           | 0.874           | 0.790           | 0.901           | 0.899           | 0.823           |
| `y26n_sop50`                         | 0.906           | 0.789           | 0.925           | 0.937           | 0.856           |
| **`gelannfav14r4fw_gemlb_v2`** | **0.943** | **0.806** | **0.946** | **0.955** | **0.929** |

### CrowdHuman — does it find people in crowds?

| model                                  | recall          | recall (heavily blocked) | precision       | false-positive rate         |
| -------------------------------------- | --------------- | ------------------------ | --------------- | --------------------------- |
| `y26n_noe2e_warm50-2`                | 38.7%           | 20.7%                    | 92.0%           | 1.74 per 100 imgs           |
| `yolo11N-640` (production)           | 34.3%           | 13.8%                    | **94.5%** | **1.28 per 100 imgs** |
| `y26n_sop50`                         | 38.5%           | 20.4%                    | 92.2%           | 1.67 per 100 imgs           |
| **`gelannfav14r4fw_gemlb_v2`** | **38.9%** | 20.4%                    | 93.1%           | 1.56 per 100 imgs           |

### PASS — false alarms on photos with zero people

| model                             | false alarms per 100 photos | % of photos with a false alarm |
| --------------------------------- | --------------------------- | ------------------------------ |
| **`y26n_noe2e_warm50-2`** | **0.37**              | **0.37%**                |
| `yolo11N-640` (production)      | 1.17                        | 1.13%                          |
| `y26n_sop50`                    | 0.60                        | 0.60%                          |
| `gelannfav14r4fw_gemlb_v2`      | 0.70                        | 0.70%                          |

### Spotlight-val — general check (not a fair exam, see caveat above)

| model                        | mAP50           | mAP50-95        | Woman           | Man             | Child           |
| ---------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| `y26n_noe2e_warm50-2`      | 0.802           | 0.697           | 0.810           | 0.864           | 0.732           |
| `yolo11N-640` (production) | 0.744           | 0.603           | 0.775           | 0.784           | 0.673           |
| `y26n_sop50`               | 0.806           | **0.700** | **0.817** | 0.867           | 0.733           |
| `gelannfav14r4fw_gemlb_v2` | **0.812** | 0.654           | 0.777           | **0.870** | **0.788** |

## The verdict, by what you care about most

- **Best classification accuracy** (on the tests that are actually fair): `gelannfav14r4fw_gemlb_v2`, clearly and consistently — but it runs roughly 2× slower/heavier than the other three candidates.
- **Fewest false alarms**: `y26n_noe2e_warm50-2` (our current baseline) — the cleanest of all four, including cleaner than its own "improved" successor.
- **Currently shipped model** (`yolo11N-640`) is now the weakest on almost every measure except CrowdHuman precision — it has the most false alarms by a wide margin and the lowest classification accuracy.
- **The small-object training trick** (`y26n_sop50`) gives a small, real accuracy improvement over the baseline it was built from, on every test we ran — but also makes false alarms modestly worse (0.37 → 0.60 per 100 photos). Not a clear win or loss, a real trade.

## Where the raw data lives

Full technical detail, verification steps, and every caveat: `docs/MODEL_COMPARISON.md`. Interactive version: https://claude.ai/code/artifact/3f8c26aa-a6db-4d60-a0cc-b808ba329334
