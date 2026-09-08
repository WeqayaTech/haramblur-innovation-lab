# How the three models handle small people — results anyone can read

**One line:** the shipped model is the best of the three on people who are reasonably
big in the frame, and the worst by far on people who are small. Below about 48 pixels
it stops working almost entirely, while both YOLO26n models keep going.

**Measured on:** `smallperson_v1/paste_grey` — 250 images we built for exactly this
question. Each image is 640×640 with six real people cut out of LAGENDA photos and
placed on a plain grey background at a known size. Because they never touch, every box
holds exactly one person and there is nothing to argue about. The *same* six people
appear at every size in the same spots, so a change down the table can only be caused
by size. 1,500 people total, 300 at each size (108 women, 107 men, 85 children).

Sizes are in pixels of the image the model actually sees, so "32 px" means the person
is 32 pixels tall to the model — about one twentieth of the frame height, i.e. a person
at the far end of a street.

## The models

| name in the tables | weights |
|---|---|
| `yolo11N-640` | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` — **what is shipped today** |
| `y26n_gradsupp` | `/workspace/exp14/train/y26n_gradsupp/weights/best_plain.pt` (read through its NMS head) |
| `y26n_noe2e_warm50-2` | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` |

All three at confidence 0.45, the setting the extension ships with.

## Table 1 — Does it find the person at all?

Percentage of real people the model puts a box on. 300 people at each size.

| model | 96 px | 64 px | 48 px | 32 px | 24 px |
|---|---:|---:|---:|---:|---:|
| `yolo11N-640` (shipped) | **85.0** | **64.7** | 35.0 | 3.3 | 0.3 |
| `y26n_gradsupp` | 70.7 | 49.3 | 44.0 | **16.3** | **4.7** |
| `y26n_noe2e_warm50-2` | 70.0 | 62.0 | **45.0** | **16.3** | 4.0 |

**The crossover is at roughly 48 px.** Above it the shipped model leads; below it the
shipped model collapses and the YOLO26n models are ~5× better. At 24 px the shipped
model found 1 person out of 300.

## Table 2 — Does a woman actually get blurred?

This is the product question. A woman is blurred only if the model **finds her AND
calls her a woman** — if it misses her, or finds her but says "man", she is not blurred.
This column combines both. 108 real women at each size.

| model | 96 px | 64 px | 48 px | 32 px | 24 px |
|---|---:|---:|---:|---:|---:|
| `yolo11N-640` (shipped) | **72.2** | 42.6 | 19.4 | 1.9 | 0.0 |
| `y26n_gradsupp` | 60.2 | 38.9 | **31.5** | **8.3** | 1.9 |
| `y26n_noe2e_warm50-2` | 57.4 | **49.1** | 23.1 | 7.4 | 1.9 |

Read the best number in the table plainly: even at 96 px, roughly **3 in 10 women are
not blurred**. At 48 px it is 7 in 10. Below that, essentially none of them are.

## Table 3 — Quality of the calls it does make (all sizes pooled)

| | `yolo11N-640` | `y26n_gradsupp` | `y26n_noe2e_warm50-2` |
|---|---:|---:|---:|
| people found (of 1,500) | 565 (37.7%) | 555 (37.0%) | **592 (39.5%)** |
| gender right, of adults found | 84.4% | 82.8% | **87.7%** |
| all-three-classes right, of people found | 67.4% | 65.2% | **70.6%** |
| **adult called a child** (adult escapes the blur) | **0.7%** | 1.7% | **0.7%** |
| child called an adult (child over-blurred — acceptable) | 86.4% | 84.5% | 78.5% |
| boxes on nothing (upper bound) | 2.6% | **0.9%** | 1.0% |

The gender step is **not** where the loss is: all three name 83-88% of the adults they
find correctly, at every size. It is the *finding* step that collapses. If we want small
women blurred, the work is in the detector, not the gender head.

Children are the weakest class everywhere — most children that are found get called an
adult. That direction over-blurs, which is the acceptable direction, but it means the
"leave children alone" behaviour degrades badly once people get small.

## Charts

- `assets/exp19_detection_by_size.svg` — Table 1 as a curve
- `assets/exp19_woman_e2e_by_size.svg` — Table 2 as a curve

## What this does NOT tell you

- **Grey background, no scene.** People are pasted on flat grey with no street, no
  crowd, no context — and real detectors use context. Treat these as *comparisons
  between models and across sizes*, not as the rate you would see in the wild. The
  real-image version of this measurement (`smallperson_v1/crowd_small`, 800 CrowdHuman
  images) is built but only partly scored — the pod went down mid-run.
- **The cut-outs are clean.** A shrunk photo has no motion blur, haze or focus loss, so
  if anything these numbers are *optimistic* for small people.
- **Two cells are meaningless and are excluded from any claim:** the shipped model's
  class accuracy at 32 px and 24 px rests on 10 and 1 detections respectively.
- **Nothing here is about video.** Production runs on frames with hysteresis; equal
  per-frame numbers can still look very different to someone watching.
- Single run per model, no variance estimate — treat gaps under ~1 point as ties.

## Where the numbers come from

Built by `vlm-cluster/build_small_person_set.py paste`, verified by
`vlm-cluster/verify_small_person_set.py`, scored by the project's own
`run_autolabel_on_manifest.py` against the set's `gt.jsonl` — no new scoring code, so
these are the same metric definitions used everywhere else in the project.
Classes come from LAGENDA's **human** age and gender labels; outlines come from the
SAM3 polygon that matched that person's human box at IoU ≥ 0.5.
Full method and pre-registered bars: `EXP-2026-19-small-person-benchmark.md`.
