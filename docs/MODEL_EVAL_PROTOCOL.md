# How we evaluate a production-model candidate (team protocol)

One page: what data, why we trust it, how every model sees the *exact same* data, and how to
plug a new model in. Used by EXP-2026-12 (YOLO26 vs production YOLO-MIT); applies to any
future candidate. Component definitions live in `docs/COMPONENT_FRAMEWORK.md`.

## The four evaluation datasets (all on the volume, frozen, reused — never rebuilt)

| dataset | size | what it answers | why we trust the answer key |
|---|---|---|---|
| **LAGENDA sample** `/workspace/lagenda_eval/lagenda_yolo/` | **5,000 people in 4,899 images** (val split; manifest `gt.jsonl`) | detection recall on prominent subjects; age (Adult/Child) and gender vs ground truth (Components 2+3) | external public benchmark with **human-annotated apparent age + gender** per person; in use since EXP-2026-01; none of it appears in training |
| **CrowdHuman sample** `/workspace/datasets/crowdhuman/` + the 517-image sample | **517 images, 11,641 persons** | detection recall in crowds (stratified by occlusion), precision, duplicates (Component 1) | external benchmark, **exhaustively human-boxed** (every person, incl. occlusion info + ignore regions) — an unmatched detection is *provably* wrong here |
| **PASS subset** `/workspace/datasets/pass_3k/` | **3,000 images, zero people** | baseline hallucination rate (FPs/100 empty scenes) | dataset built to contain no humans; seed-51 sample; caveat: EXP-2026-08 found ~5% crop-level contamination (flag list under `/workspace/exp07_conf/`) |
| **Object set** `/workspace/datasets/object_set/` | **259 images of dolls/statues/toys, zero real people** | the "blurred books" symptom: false persons on human-shaped objects | **hand-verified image-by-image by the owner** (EXP-2026-03); expensive to rebuild — reuse only. Caveat: rates are upper bounds pending a re-verification pass (EXP-2026-04 §2.5) |

Rules: each dataset answers its own question; results are **never pooled**. LAGENDA labels
~1 person/image, so LAGENDA numbers are recall-side only (its "unmatched detections" are NOT
false positives — that's what PASS/objects/CrowdHuman measure).

## How "exact same data" is guaranteed (not just intended)

1. **One physical copy.** All arms read the same frozen directories above — no per-model
   copies of eval data, ever.
2. **Pinned membership.** The crowd sample's membership is the EXP-2026-03 SAM label stems,
   materialized once as a symlink dir (`/workspace/exp12/crowd_sample_imgs`, 517 links, 0
   broken). Every model is pointed at that dir, not at CrowdHuman itself.
3. **Complete coverage, verified per run.** The Stage A runner
   (`vlm-cluster/run_ultralytics_labels.py`) writes a label file for **every** processed image
   — an empty file when the model found nothing — so "not processed" and "found nothing" are
   distinguishable, and the scorers print `n_images_processed` / `n_gt_persons` /
   `n_images_not_processed_yet`. **The check:** these counts must be identical across arms
   (EXP-2026-12: 259 / 3,000 / 517+11,641 / 5,000 in both columns, `not_processed: 0`).
4. **Same operating point.** All arms run at the production settings: imgsz 640, conf 0.45,
   NMS IoU 0.7 (from the extension's `constants.mjs`). Raw sidecars additionally log every
   detection down to conf 0.05, so threshold sweeps replay offline without re-running any
   model.
5. **Same scorer, same matcher.** One scoring path for everyone:
   `eval_negatives_crowd.py` (Component 1) + `run_autolabel_on_manifest.py` (Components 2+3),
   both built on the same `match_boxes` (greedy mutual-exclusive, IoU ≥ 0.5). No model gets
   its own scorer.

## Plugging in a new model (the 3-step recipe)

1. **Dump labels.** If it loads via `ultralytics.YOLO`:
   `python3 run_ultralytics_labels.py --model <weights.pt> --map identity --images <dataset_dir> --out /workspace/expNN/<arm>/<dataset> --conf 0.45`
   (add `--prompts "woman,man,child"` for YOLOE-style promptable models). Any other
   framework: write a tiny adapter class with
   `.detect(img) -> [(cls, x1, y1, x2, y2, conf)]` — see `MitAdapter` in the same file
   (~20 lines wraps the production YOLO-MIT model). Class ids must be
   `{0: Woman, 1: Man, 2: Child}`.
2. **Run the four dumps** (lagenda / crowd_sample_imgs / pass_3k / object_set), then the four
   scorer commands — copy them verbatim from `EXP-2026-12-POD-RUNBOOK.md` Phase C, changing
   only the arm name in the paths.
3. **Check the coverage counts match** the table above, then read the summary JSONs. The
   headline numbers to put in the comparison table: object-set %-images-with-FP, PASS FPs/100,
   crowd recall/precision, LAGENDA detection recall, gender-on-adults, and the
   adult(GT≥20)→Child rate from `call_rate_by_age`.

## Current scoreboard (2026-08-04, conf 0.45)

| metric | YOLO-MIT v9 (production ckpt, 8.2M params) | YOLO26n-Spotlight (2.4M params) |
|---|---|---|
| Object set: images w/ false person | 53.7% | **12.7%** |
| PASS FPs/100 | 1.43 | **0.23** |
| Gender on adults | 87.2% | **90.7%** |
| Adult(≥20)→Child leak | 0.77% | **0.11%** |
| LAGENDA detection recall | **96.0%** | 94.1% |
| Crowd recall | **39.6%** | 32.8% |

Caveats that must travel with any quote of these numbers: models are not size-matched
(candidate 3.45× smaller); "production" = the team's v9 checkpoint pending the
shipped-binary identity question (EXP-2026-11); the **Gulf/traditional-dress gender slice is
untested for every model** — the biggest open gap in the lab.
