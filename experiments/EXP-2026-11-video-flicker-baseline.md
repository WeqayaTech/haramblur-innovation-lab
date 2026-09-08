# EXP-2026-11 — How unstable is the production model on video? (flicker baseline)

**Status: RUN COMPLETE 2026-08-03 (results below). Render adjudication + the
model-identity question (v11nclean2 vs YOLO-MIT v9) still open.**

## The question

The extension runs the production YOLO-MIT model frame by frame. Users experience the
*temporal* behavior — blur popping in and out, boxes jittering, gender flipping — which none
of our image benchmarks measure. Before building any smoothing (hysteresis, EMA, class
votes, detect-every-N), this experiment measures what actually happens today: run the
frozen production checkpoint (`/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4`,
imgsz 640) over every frame of 4 videos and quantify each kind of instability separately.

This is a **baseline measurement, not a pass/fail experiment** — there are no adoption
bars here. The rule we pre-register instead: every future smoothing policy will be
evaluated by *offline replay over the sidecars this run produces* (same frames, same
detections, no model re-run), and its adoption bars will be written down before that
replay happens.

**Key design point (why no ground truth is needed):** people don't blink in and out of
existence between frames of continuous video. Detections are linked into tracks by IoU,
and every high-frequency inconsistency within a track *is* the flicker, self-measured.

## The videos (all Wikimedia Commons, direct-download URLs verified 2026-08-03)

| clip | scenario | source file | license | size |
|---|---|---|---|---|
| `interview` | single prominent person, talking head (the common browsing case) | Voices from Wikimania — Wikimedian Netha Hussain (2:09, 1080p) | CC BY 4.0 | 105 MB |
| `pedestrian_area` | crowd walking toward camera, classic test sequence (15 s, 1080p25) | Video Codec Test pedestrian area | CC0 | 11 MB |
| `street_crossing` | street scene, people waiting/crossing (15 s, 720p) | People waiting to cross the street | CC BY-SA 3.0 | 30 MB |
| `dubai_souk` | **the Gulf-dress slice on video**: walking tour of Naif Deira & Gold Souk, men in kanduras/ghutras, moving camera, crowds (probe first 3 min of 37 min, 480p transcode) | Dubai 4K Naif Deira & Gold Souk Day Walking Tour | CC BY 4.0 | 448 MB |

Exact `wget` lines are in `EXP-2026-11-POD-RUNBOOK.md`. **This experiment is fully
self-contained under `/workspace/innovation-lab/exp11_flicker/`** (owner decision
2026-08-03 — new volume layout for this line of work): `code/` (the tools + their two
imports, standalone), `videos/` (with README manifest + sha256s), `probe/` (the raw
sidecars — the reusable asset), `report_conf*/`. It deliberately does not touch
`/workspace/datasets/` or the shared vlm-cluster pod copy.

The `dubai_souk` arm doubles as the first-ever *temporal* look at the known thobe/ghutra
gender bias: if the Woman↔Man flip rate there is visibly higher than in the western street
clips, that's the bias expressing itself as flicker.

## Pre-registered metrics (implemented before the run — `vlm-cluster/flicker_metrics.py`)

All computed per video, never pooled, at a simulated production threshold (`--conf`,
default 0.1 — **the extension's real threshold is still unconfirmed, see caveat 1**).

1. **Existence flicker** — tracks are built by greedy frame-to-frame IoU association
   (`build_tracks`, flicker_metrics.py:74; class-agnostic — identity is spatial, class is
   an attribute). Per real track: gap count, gap-length histogram, and **blur on/off
   toggles per minute** (2 per gap).
2. **Gap cause** — a gap frame that still contains a *sub-threshold* detection overlapping
   the track (IoU ≥ 0.3) is a `threshold_gap`; otherwise `miss_gap`
   (`analyze_track`, flicker_metrics.py:117). This single split decides whether a
   dual-threshold (high-to-create / low-to-sustain) fix would work: threshold gaps it
   fixes, miss gaps it can only paper over via track coasting.
3. **Confidence** — per-track mean/std + fraction of frames within 0.05 above the
   threshold (the straddle band).
4. **Box jitter** — median consecutive-frame IoU within a track + center drift as a
   fraction of box diagonal. Decides EMA strength and blur padding.
5. **Class flicker** — flips per track-minute + the full transition matrix. Woman↔Man
   flips toggle blur for every viewer; adult↔Child flips are exposure events in the
   direction the product cares about.
6. **Exposure (the headline)** — for each track whose majority class is the viewer's blur
   target (Woman for a male viewer, Man for a female viewer): seconds visibly *unblurred*
   inside the track's lifetime, and distinct exposure events. Reported for both viewer
   genders.
7. **Phantom blips** — tracks alive < 0.3 s: blur popping onto nothing (the
   books/background FP symptom, in its temporal form).

The raw sidecars log **everything above conf 0.05** (well under any plausible production
threshold), per the log-raw-readings principle — so thresholds, hysteresis, votes and
every-N policies are all replayable offline afterwards.

Visual proof: `flicker_metrics.py --render` writes an annotated mp4 per clip — boxes,
class + conf labels, sub-threshold detections in thin gray, and the actual pixelation a
male viewer would see. Watching these is the "actually see the issue" deliverable.

## Results (run 2026-08-03, YOLO-MIT v9 checkpoint, production settings: conf 0.45, NMS 0.7)

**Bottom line: the instability is real, large, and comes from exactly two mechanisms —
threshold-slicing and gender oscillation. Detection misses and box jitter are non-issues.
No confidence threshold fixes it (swept 0.35/0.45/0.55): the cure is temporal policy, not
a better cutoff.**

At the production threshold (0.45), per clip:

| clip | blur toggles/min | class flips/min | gap cause (threshold:miss) | conf straddle | jitter IoU | exposure male-viewer | phantoms/min |
|---|---|---|---|---|---|---|---|
| dubai_souk (3.0 min, 60 fps) | 920.8 | 197.7 | **1367 : 14** | 0.164 | 0.949 | 45.1 s / 54 tracks | 134.0 |
| interview (2.15 min) | 114.2 | 40.9 | **121 : 2** | 0.108 | 0.950 | 10.6 s / 28 tracks | 21.4 |
| pedestrian_area (15 s) | 802.1 | 521.4 | **88 : 12** | 0.071 | 0.861 | 12.7 s / 30 tracks | 136.4 |
| street_crossing (16 s) | 646.1 | 58.4 | **65 : 18** | 0.155 | 0.970 | 10.0 s / 17 tracks | 112.9 |

The five findings:

1. **~90-99% of all dropouts are threshold gaps, in every scenario** — the model still
   detects the person; confidence just dips below 0.45 for a few frames. Detection misses
   (miss_gap) are a rounding error. A dual-threshold policy (create at 0.45, sustain
   lower) plus short coasting attacks nearly the entire existence-flicker mass. The gap
   histogram concentrates at 1-3 frames, exactly the regime hysteresis erases.
2. **No threshold escapes the trade.** 0.35→0.55 on dubai_souk: toggles 1348→754/min
   (never near zero), class flips 502→78/min, but straddle *rises* 0.17→0.25 (the
   confidence mass sits right around 0.45-0.6) and target tracks fragment 80→32. The
   knob rotates the failure; it doesn't remove it.
3. **Gender oscillation is generic, symmetric, and big: Woman↔Man flips are ~50/50 in
   every clip** (dubai 297:294, interview 45:42) — borderline people flip back and forth
   between frames. Even the single-person talking-head control flips ~41/min. This is
   the case for a majority-vote/sticky-class policy independent of the Gulf-dress
   question. **Child flips are near-zero everywhere** (≤2 at conf 0.45 except
   pedestrian's 1) — the age dimension is temporally stable; the consequential direction
   is not leaking via flicker.
4. **The single-person control (interview) still flickers** — 114 toggles/min and 78
   "real" tracks where ~1 person is on screen. So the instability is model-level, not a
   crowd-association artifact. (Caveat: the clip may contain b-roll/venue shots —
   confirm on the render before quoting the per-track numbers.)
5. **Box jitter needs nothing:** median consecutive-frame IoU 0.86-0.98, drift ≤2% of
   box diagonal. EMA is optional polish; blur padding alone suffices.

User-experience rollup at 0.45: a male viewer gets **45 s of unblurred woman-track time
in the 3-minute souk clip (405 exposure events)**; even the talking-head interview leaks
10.6 s. Read these as raw per-frame rates: production samples at 25 fps with its
1-positive/2-negative hysteresis and frame-diff cache, so the experienced baseline is
milder — simulating that exact production policy over these sidecars is the first replay,
before any new policy is scored. Note also the exposure denominator shrinks as the
threshold rises (tracks fragment), so exposure values are not comparable across conf
values.

## What this does NOT tell us (written before running)

1. **This is the Python checkpoint path, not the shipped extension** — but the extension's
   runtime settings were read from its source 2026-08-03 (`HaramBlurP-main/src/constants.mjs`
   MODELS + `processing2.js`) and the probe/analysis now match them: input 640 (default
   tier; 416/320 tiers exist), **scoreThreshold 0.45** (primary analysis conf), **NMS IoU
   0.7**, maxDetected 70. Still different: cadence (probe = native ~25-30 fps; extension
   samples at 25 fps full-tier, 15/20 on smaller tiers) and the extension's existing
   temporal policy — a 1-positive/2-negative consecutive-frame hysteresis on video blur
   state plus a frame-diff result cache (cacheSensitivity 0.5, skipFrames 20, skipTime
   1000 ms). So raw per-frame numbers here are the *model's* instability; the
   *user-experienced* baseline must be simulated by replaying that production policy over
   the sidecars (planned as the first replay). **Unresolved model-identity question: the
   extension's model folders are named `v11nclean2{,m,s}` and an optional 224×224
   second-stage person classifier can override class/score — whether the shipped detector
   is the YOLO-MIT v9 checkpoint probed here or a YOLOv11-nano export, and whether the
   classifier stage is active, is pending owner confirmation.** If it's v11n + classifier,
   this probe measures the wrong model and must be re-pointed.
2. **Track association is a measurement instrument, not ground truth.** In crowds, greedy
   IoU linking can swap identities or merge neighbors; per-track numbers in `dubai_souk`
   crowd segments carry that noise. The single-person `interview` arm is association-error-free
   by construction, which is why it's in the set.
3. **Exposure can't see what precedes the first detection** — a track starts existing when
   first detected, so "late first blur" latency is not measured here (it needs GT or manual
   annotation). Within-track exposure is fully measured.
4. **Class flips on the Dubai arm confound bias with crowd-association noise** — treat a
   high flip rate there as a flag to eyeball the render, not as a measured bias rate.
5. Four clips is a scenario sample, not a distribution — enough to see and size the issue,
   not to claim corpus-level rates.

## Appendix — verify it yourself

- **Model invocation:** the probe imports the unmodified production wrapper —
  `MitModel` (run_model_children.py:148) with `find_model_config` reading the run's own
  `.hydra/overrides.yaml`; weights `run_dir/weights/best.ckpt`; the only nonstandard
  setting is the *logging* floor `--floor-conf 0.05` (video_flicker_probe.py, argparse).
  Production-threshold behavior is recreated at analysis time by discarding
  sub-threshold detections (`build_tracks` cands filter, flicker_metrics.py:81).
- **Association:** greedy highest-IoU-first, mutually exclusive, min track IoU 0.3, track
  closed after 0.5 s unseen — `build_tracks` in full, flicker_metrics.py:74-113.
- **Thresholds table:** floor_conf 0.05 (probe argparse) · conf 0.1 default, sweep
  planned (metrics argparse) · track_iou 0.3 · max_gap 0.5 s · min real track 0.5 s ·
  phantom ≤ 0.3 s · straddle band 0.05 · gap-cause IoU 0.3 (`analyze_track` defaults).
- **Selftests:** both tools ship `--selftest` needing no GPU/data.
  `flicker_metrics.py --selftest` builds a synthetic 100-frame scenario and asserts
  every metric against hand-computed values (tracks=2, one 2-frame threshold-gap, 2 class
  flips Woman→Man→Woman, exposure 5/30 s in 2 events, 1 phantom). An end-to-end mock run
  (`video_flicker_probe.py --mock` on a synthetic mp4 → metrics → render) was verified
  locally 2026-08-03.
- **Not done:** no GT anywhere; no extension-fidelity check; no pooling across clips.
