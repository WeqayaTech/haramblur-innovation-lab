# EXP-2026-12 — Is YOLO26 a better production-model architecture than YOLO-MIT v9?

**In one line:** train Ultralytics YOLO26 on the Spotlight-cleaned labels and measure whether it
beats the production YOLO-MIT v9 architecture on every component, on temporal stability, and on
CPU speed — before anyone commits the extension to a new architecture.

**Status:** Phase B (head-swap fine-tune) STARTING 2026-08-03 · Owner: Mostafa
**Owner decision 2026-08-03:** training launched before Phase A, so **bar A1 is waived as a
spend gate** (recorded, not hidden). Phase A still runs later as the anchor measurement — the
production-model Component 1 rows and the pretrained-YOLO26 comparison keep their full value;
they just no longer gate the training money. All Phase B/C bars stand unchanged.

---

## The short version

_(written last, after the run — placeholder)_

## Why we did this

The extension ships a YOLO-MIT v9 checkpoint, chosen years ago partly because it is the
MIT-licensed reimplementation of YOLOv9. A retrain on the Spotlight-cleaned OIV7 labels is
already running on that same architecture. Ultralytics released **YOLO26** (weights shipped
January 2026, `ultralytics` 8.4.x): end-to-end **NMS-free** by design, DFL removed, and
vendor-claimed ~31–43% faster CPU ONNX inference than YOLO11n. For a browser extension that runs
the model on CPU/WebGPU every frame, NMS-free export (no NMS to reimplement in JS) and a real
CPU speedup would be product-level wins — *if* the accuracy holds on our components and the
flicker doesn't get worse. This experiment decides whether the Spotlight retraining effort
should target YOLO26 instead of (or alongside) YOLO-MIT v9.

**Non-accuracy gate, flagged up front: YOLO26 is AGPL-3.0** (code *and* weights; no MIT-licensed
equivalent exists, unlike v9). Shipping it in the extension requires either AGPL compliance for
the whole distributed product or an Ultralytics Enterprise license. **This is an owner/legal
decision that no benchmark number can settle — if the license answer is no, this experiment is
moot and should not run past Phase A.**

## What we wanted to find out

Can a YOLO26 model fine-tuned on the Spotlight-cleaned OIV7 labels match or beat the YOLO-MIT v9
architecture on Components 1–3, on video flicker, and on CPU latency?

Components measured (per `docs/COMPONENT_FRAMEWORK.md`): **Components 1, 2 and 3** — this is a
monolithic production candidate, decomposed by conditioning, plus the temporal axis from
EXP-2026-11. Scored per dataset, never pooled.

### Pre-registered bars (written 2026-08-03, before running anything)

**Phase A — go/no-go for training** (COCO-pretrained `yolo26n.pt`, person class only,
Component 1 only, vs the production YOLO-MIT v9 checkpoint measured on the same three datasets
with the same matcher and the same conf 0.45):

- **A1 (proceed condition):** pretrained YOLO26n crowd recall ≥ production YOLO-MIT's measured
  crowd recall, AND its object-set image-FP rate ≤ YOLO-MIT's. If it fails *both*, the
  architecture is rejected without spending training money.
- Phase A also produces the first-ever Component 1 rows for the production model itself
  (CrowdHuman / PASS / object set) — a scoreboard gap worth filling regardless of outcome.

**Phase B/C — the fine-tuned candidate.** Primary comparator: the YOLO-MIT retrain on the same
Spotlight labels if its checkpoint is available when we score; otherwise the shipped production
checkpoint (recorded which, in the results). Non-inferiority on everything + at least one strict
win, specifically:

- **LAGENDA detection recall ≥ 97%** (SAM3's 98.7% is the labeler anchor; a runtime model at
  conf 0.45 gets 1.7 pts of slack).
- **Gender on correctly-detected adults ≥ 98%**, and **adult(GT ≥ 18) → Child ≤ 1%** (the
  escape-the-blur direction; the Spotlight pipeline itself measured 0% at GT ≥ 18 in
  EXP-2026-10, but a distilled student model gets 1 point of slack). Child recall reported, not
  barred (it is the less important direction — see CLAUDE.md).
- **CrowdHuman: recall ≥ comparator − 1 pt and precision ≥ comparator − 1 pt.**
- **PASS: FPs/100 ≤ comparator.** **Object set: image-FP rate ≤ comparator** — this is the
  users-report-books-getting-blurred symptom; a new architecture must not regress it.
- **Flicker (the 4 EXP-2026-11 clips, identical replay machinery, production settings conf 0.45
  / NMS 0.7):** blur toggles/min AND Woman↔Man flips/min ≤ the comparator's on every clip.
  Reported per clip, never pooled.
- **CPU latency: end-to-end ONNX single-image ≥ 20% faster than the comparator's ONNX export at
  the same imgsz** (vendor claims ~31% vs YOLO11n; 20% leaves room for the claim shrinking on
  our hardware), OR strictly smaller exported model at equal-or-better accuracy. If neither,
  YOLO26 has no adoption case — accuracy parity alone doesn't justify an architecture change
  plus a license bill.

**Decision rule:** adopt-candidate = A1 pass AND all Phase B/C bars pass AND the license gate is
resolved yes. Any accuracy bar missed → stay on YOLO-MIT v9, record the deltas.

## How we plan to do it

1. **The answer keys.** Component 1: CrowdHuman 500-img seed-51 sample (exhaustive vbox GT,
   occlusion strata), PASS 3k verified-empty scenes, the 259-img hand-verified object set —
   all staged under `/workspace/datasets/` since EXP-2026-03. Components 2+3: the LAGENDA
   5,000-person human-checked sample (`/workspace/lagenda_eval/`). Temporal: the 4 EXP-2026-11
   clips under `/workspace/innovation-lab/exp11_flicker/videos/` (self-measured flicker, no GT
   needed). All reused, none rebuilt.
2. **Phase A (cheap, ~1 h GPU).** `yolo26n.pt` (COCO weights, person class only) via the new
   Stage A runner `vlm-cluster/run_ultralytics_labels.py --map coco-person`, AND the production
   YOLO-MIT v9 checkpoint via a `run_model_children.MitModel` label-dump over the same three
   Component 1 datasets. Both scored by the unchanged `eval_negatives_crowd.py`. Apply bar A1.
3. **Phase B (the spend).** Fine-tune `yolo26n` (and `yolo26s` if the n run is healthy) on the
   Spotlight-cleaned OIV7 train labels — the `run/oiv7_train/labels/` classes-0-2 variant
   (stock ultralytics cannot mask class-3 Unknown regions, so the `labels_unk3` masking trick
   the team retrain uses does not port; recorded as a comparability caveat). Standard
   `model.train()`, config frozen in the runbook before launch. Rough estimate: ~25–40 min/epoch
   on one A100 at 640px over 475k images → budget cap **$100 GPU, pre-registered**; if the run
   can't converge inside it, stop and report.
4. **Phase C (CPU + cheap GPU).** The fine-tuned checkpoint through the same Stage A runner
   (`--map identity` — it now speaks {0 Woman, 1 Man, 2 Child}) over all four datasets;
   Component 1 scored by `eval_negatives_crowd.py`, Components 2+3 by
   `run_autolabel_on_manifest.py` against the LAGENDA manifest (box lines parse fine — see
   appendix).
5. **Phase D (temporal).** `video_flicker_probe.py --engine ultralytics --model-path <best.pt>`
   over the 4 clips (raw sidecars at floor 0.05), `flicker_metrics.py` at conf 0.45 — numbers
   drop into the EXP-2026-11 table row-for-row.
6. **Phase E (deployability).** `model.export(format="onnx")` (end-to-end, output
   `(batch, max_det, 6)`, no NMS in the runtime); onnxruntime CPU latency vs the comparator's
   ONNX export, same machine, same imgsz, ≥100 warm iterations; a 20-image parity check
   (exported vs native detections). **Note: TF.js export is deprecated in ultralytics 8.4.83+**
   (emits LiteRT instead) — if the extension is tfjs-based, the deployment path is
   onnxruntime-web or a LiteRT.js migration; recorded as a product finding, not barred.

## What we found

_(to be filled after the run — per phase, per dataset, never pooled)_

### Phase B0 pilot — PASSED 2026-08-03 (plumbing proof; metrics are NOT results)

Run on an RTX 4090 pod (torch 2.6.0+cu124, ultralytics 8.4.115), 50k-image seeded subset
(seed 20260803), `freeze=11` (all backbone layers), killed after epoch 1 by design once every
check passed:

- **Head swap proven end-to-end:** the 80-class COCO head rebuilt to 3 classes automatically;
  `best.pt` answers `{0: 'Woman', 1: 'Man', 2: 'Child'}`. Epoch-1 val mAP50 0.504 /
  mAP50-95 0.381 — a healthy learning signal, and nothing more (frozen backbone, 10% of data,
  1 epoch; never goes in any scoreboard).
- **Data plumbing verified:** trains from explicit file lists (`train_full.txt` 475,207 /
  `val_full.txt` 4,233 / `pilot_50k.txt` 50,000 — lists exclude the ~37k unlabeled
  `negatives_*` images in the shared train dir, the recorded data choice), 17% of the pilot
  images are Spotlight verified backgrounds (empty label files, loaded as genuine negatives),
  1 corrupt 1×1 image auto-skipped. That corrupt file has a Roboflow-export name
  (`reports-before-march_..._rf.*.jpg`) — the shared `/workspace/open-images-v7/images/train/`
  dir contains at least some non-OIV7 files; flagged for the team, harmless to us.
- **Read-only guarantee verified, not assumed:** ultralytics label caches landed inside our
  symlink tree (`/workspace/exp12/train_tree/labels/*.cache`); the original Spotlight and
  open-images dirs gained nothing (the two caches found in `/workspace/open-images-v7/labels/`
  are dated Jun 18 — a prior team run, not ours).
- **The binding constraint is volume I/O, measured twice:** epoch 1 ran at 1.2 it/s at
  AutoBatch 40 (~830 ms/batch vs ~35–50 ms of GPU forward time — GPU ~80% idle); epoch 2, same
  data and config, ran **2.5× faster (3.0 it/s, 7:03)** purely because the 50k images were by
  then in the pod's RAM page cache — same-data A/B proof that storage, not compute, was the
  ceiling. The full 475k set (~90–125 GB) exceeds pod RAM, so the full run **stages all images
  to a ≥200 GB local container disk first**. Cached-rate projection: ~65 min/epoch → 30 epochs
  ≈ $20–25 on a 4090, inside the $100 cap. (The pilot in fact completed all 3 epochs in
  0.53 h: mAP50 0.504 → 0.573 → **0.626**, mAP50-95 0.510, still climbing at cutoff — a
  healthy floor for a frozen-backbone run on 10% of the data. Epoch 3 ran 3.2 it/s from page
  cache.)
- **Optimizer note:** `optimizer=auto` selects AdamW (lr≈0.0014), overriding YOLO26's
  advertised MuSGD default for this fine-tune; kept as-is and recorded.

### Phase B full fine-tune — COMPLETE 2026-08-04

`yolo26n` × 30 epochs × 475,207 Spotlight-labeled images (train_full.txt; val = the 4,233
Spotlight val set), RTX 4090 + 200 GB container disk with all images staged to local NVMe
(1.45 GB/s measured — the staging turned the projected ~80 h run into ~14 h). AutoBatch, AdamW
auto, imgsz 640, workers 8. **Total cost ≈ $11.**

- **Final (best.pt = epoch 28, selected by fitness): val mAP50 0.824 / mAP50-95 0.707.**
  Per class, mAP50: **Man 0.874 · Woman 0.853 · Child 0.745** (Child weakest — fewest
  instances, and the age axis is the known-hard one). These numbers are against Spotlight's
  own val labels — they measure "learned what the pipeline taught," NOT the experiment
  verdict; Phase C against human GT decides the bars.
- Curve: warmup dip epochs 3–4 (mAP50 0.42), recovery by 6, plateau ~0.82 from epoch 12;
  mosaic off at 21 per ultralytics default; epochs 29–30 added nothing (0.817/0.815 —
  best.pt correctly kept 28).
- **Ops incidents, for the record:** (1) killed at epoch 29 by a one-off CUDA OOM — a
  14.6 GiB allocation spike (target-dense batch + 12 h allocator fragmentation) vs ~8.2 GiB
  steady state; (2) first resume attempt RAM-killed — **RunPod pods report the HOST's 187 GB
  to `free`, but the container cgroup limit is the pod card's ~46 GB** (the RAM twin of the
  documented `nproc` trap); (3) a second resume was accidentally started while the first
  still lived — both killed, no checkpoint damage (neither reached an epoch boundary).
  Successful recipe: patch `workers` 8→4 inside last.pt's `train_args` (backup kept:
  `last_backup.pt`) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then
  `train(resume=True)` — finished 29–30 cleanly in 0.85 h.
- Pilot-phase find worth repeating: the shared train dir contains non-OIV7 strays (a corrupt
  1×1 Roboflow-named jpg, skipped) and one truncated JPEG decoded with a warning — harmless
  here, flagged for the dataset registry.

### Phase C scores — RUN 2026-08-04 (candidate arm scored; the YOLO-MIT anchor NOT yet run,
so bar verdicts vs "comparator" are pending; absolute reads below)

All at conf 0.45, match IoU 0.5, `best.pt`. Sample notes: crowd arm scored **517** images
(the exp03 label stems — the documented 500-img sample plus a 17-img top-up; every arm of
this experiment uses the same 517, cross-experiment comparisons vs SAM3's 500 are
approximate); object set is the 259-image set (a 260th non-image dir entry ignored).

- **PASS: 0.23 FPs/100 images** (7 total; 0.23% of images) — vs SAM3's 8.4/100, ~36× cleaner.
- **Object set: 12.7% of images gain a false person (23.9 FPs/100**, mostly Woman 34/Man 26)
  — vs SAM3's 62.2%/144. The Spotlight-cleaned training visibly transferred FP immunity,
  though not perfectly.
- **CrowdHuman: recall 32.8%** (light 40.0 / partial 28.0 / heavy 13.2) at **precision
  93.9%**, duplicates 2.0%, clear FPs 0.19/100. Low recall is confounded by two design
  facts: conf 0.45 (raw sidecars logged to 0.05 → offline sweep pending) and — the big one —
  **Spotlight training data deliberately dropped 12.4% of detections as unknown-gender
  (median ~58 px) and left them as unlabeled background, actively teaching the model NOT to
  detect small/distant people.** This is the first measured consequence of the
  classes-0-2-vs-labels_unk3 choice and directly validates the coworker's masking concern.
- **LAGENDA: detection recall 94.1%** (Woman 94.5 / Man 95.6 / Child 91.2) — below the 97%
  bar as written. **3-class on matched: 88.0%; gender on adults: 90.7%** — far below the 98%
  bar; gender errors are roughly symmetric (Man→Woman 89, Woman→Man 111). The nano model did
  NOT inherit its teacher pipeline's ~99% gender (EXP-2026-10) — capacity and/or training
  loss, to be dissected. **Age direction is the bright spot:** child-call rate by actual age
  = 96% at 0-4, 81% at 5-9, 42% at 10-14, **8.4% at 15-19, ~0.5% at 20-24, ~0% at 25+** —
  a much sharper implicit boundary than SAM3 (36% at 15-19) or Qwen2.5-VL-7B (~68% teens);
  the consequential adult(GT≥20)→Child rate is ~0.2% (3/1352 across 20+ buckets). Child
  recall (≤12, detected-only) 79.0% — the less-important direction, recorded not barred.
- Pathologies: dual-class survivors 1.3%; matched-IoU < 0.75 in 4.3% (boxes generally tight);
  crowded-vs-simple 3-class accuracy 86.0% vs 89.6%.

### Anchor run + scorecard — 2026-08-04. The production YOLO-MIT v9 model measured on
Components 1–3 for the first time, same machinery, same 517/3000/259/5000 samples, conf 0.45.

| metric | production YOLO-MIT v9 | YOLO26n-Spotlight | verdict |
|---|---|---|---|
| Object set: % images w/ false person | **53.7%** (89.6 FPs/100; W 132/M 76/C 24) | **12.7%** (23.9/100) | candidate **4.2× better** ✅ |
| PASS: FPs/100 empty images | 1.43 | 0.23 | candidate **6× better** ✅ |
| LAGENDA gender on adults | 87.2% | 90.7% | candidate +3.5 pts ✅ (both far below the 98% absolute bar — it was set unrealistically; the anchor now defines reality) |
| LAGENDA adult(GT≥20)→Child leak | 0.77% (21/2722) | **0.11%** (3/2699) | candidate **7× better** on the consequential direction ✅ |
| Child-call at age 15–19 (teen fade) | 26.6% | 8.4% | candidate sharper ✅ |
| LAGENDA 3-class on matched | 88.4% | 88.0% | tie |
| LAGENDA detection recall | 96.0% (Child 95.7) | 94.1% (Child 91.2) | candidate −1.9 pts ❌ |
| CrowdHuman recall | 39.6% (light 48.8/heavy 15.6) | 32.8% (40.0/13.2) | candidate −6.8 pts ❌ |
| CrowdHuman precision | 95.6% | 93.9% | −1.7 pts ❌ (marginal) |
| Child recall ≤12 (detected-only) | 92.1% | 79.0% | anchor better — the less-important direction, reported not barred |

**Fairness note (measured 2026-08-04):** both arms ran at identical settings (imgsz 640,
conf 0.45, NMS IoU 0.7, same images, same scorer), but the models are NOT size-matched —
**YOLO-MIT v9 = 8,207,129 params vs YOLO26n = 2,375,421 (candidate 3.45× smaller)**. The
mismatch is conservative for the candidate's wins (a third the capacity) and a possible
partial contributor to its recall losses; the size-matched arm is `yolo26s` (~9.5M), listed
as follow-up. Also: **the model-identity question is answered (owner, 2026-08-05): the shipped extension
runs `v11nclean2` — a YOLO11-class model — NOT the v9 GELAN checkpoint, which was never
deployed because it was too slow** (consistent with our timings: 136 ms vs yolo11n's ~44 ms).
So the "old prod" column below is really "the team's best v9 checkpoint (never shipped)";
the ACTUAL deployed baseline has still never been benchmarked on the components —
locating the `v11nclean2` weights and running them through the protocol is now the top
anchor task. Two knock-on effects: (a) EXP-2026-11's flicker baseline used the v9 checkpoint
and should be re-pointed (~20 min GPU, as that doc anticipated); (b) **the AGPL gate
reframes: YOLO11 is ALSO Ultralytics AGPL-3.0** — whatever license posture covers today's
shipped model covers YOLO26 identically, so the license question is about the status quo,
not a new barrier.

**Reading:** the candidate wins every product-pain axis — hallucinated people on
objects/backgrounds (the users' blurred-books symptom: the production model fires on 53.7% of
human-shaped-object images, nearly SAM3's 62.2%, confirming the trained-in symptom at model
level), gender (+3.5), and the escape-the-blur leak (7×) — and loses detection recall on
small/occluded people across every dataset. The recall loss pattern (crowd −6.8, Child
detection −4.5, LAGENDA −1.9) is exactly the signature predicted by the Spotlight
classes-0-2 data choice (12.4% unknown-gender people, median 58 px, trained as background).
**Whether recall recovers under `labels_unk3` masking is now the decisive open question, and
it is a DATA question, not an architecture one.**

**Still pending before adoption call:** offline conf sweep (raw sidecars, both arms, no GPU);
Phase D flicker replay; Phase E ONNX latency (the candidate's other claimed win);
the license gate.

### Three-way comparison — 2026-08-04 (adds the team's Spotlight retrain of the production
architecture, `gelansfav14_gemlb_v1`; same four datasets, same machinery, conf 0.45. Latency:
this pod's AMD EPYC 7352, ONNX Runtime CPU, 4 threads, 640×640, median of 60.)

| metric | old prod (datav2_v4) | team retrain (gemlb_s) | YOLO26n-Spotlight |
|---|---|---|---|
| params / ONNX size | 8.2M / 33.1 MB | 8.2M / 33.0 MB | **2.4M / 9.8 MB** |
| CPU latency (median) | 136.0 ms | 134.6 ms | **33.4 ms** (4.0×; also end-to-end — GELAN outputs still need JS NMS) |
| Object set: % images w/ false person | 53.7% | 18.2% | **12.7%** |
| PASS FPs/100 | 1.43 | 0.23 | 0.23 |
| Crowd recall | 39.6% | **46.2%** (heavy 27.0!) | 32.8% |
| Crowd precision / clear FPs per 100 | **95.6%** / 1.35 | 92.7% / 2.51 | 93.9% / **0.19** |
| LAGENDA detection recall | **96.0%** | 95.8% | 94.1% |
| LAGENDA 3-class on matched | 88.4% | **89.5%** | 88.0% |
| Gender on adults | 87.2% | 90.6% | 90.7% |
| Adult(GT≥20)→Child leak | 0.77% | **0.00%** (0/2737) | 0.11% |
| Teens 15–19 called Child | 26.6% | 13.9% | **8.4%** |
| Child recall ≤12 (detected-only) | **92.1%** | 85.8% | 79.0% |

**The three headline reads:**
1. **The Spotlight labels are the main story, architecture-independently.** Same GELAN
   architecture, warm-started, labels swapped: object FPs 53.7→18.2%, PASS 1.43→0.23, leak
   0.77→0.00%, gender +3.4. The labeling pipeline's value is now demonstrated on TWO
   architectures.
2. **Unknown-handling decides recall.** The team retrain (grey-masked unknowns, per coworker)
   GAINED crowd recall over old production (39.6→46.2, heavy occlusion 15.6→27.0) on the same
   labels where our unmasked run LOST it (→32.8). This isolates the data-handling variable
   far better than our earlier inference — masking is not just protective, it's a win.
   (Caveats: his warm start is a confound; label-variant confirmation still pending from him.)
3. **YOLO26n matches the retrain's gender/leak/FP quality at 4× the speed and 1/3.4 the
   size** — but trails everyone on small-person recall. The obvious synthesis experiment:
   **YOLO26 trained with proper unknown handling** (4-class `labels_unk3` or grey-mask) —
   if it recovers even half the recall gap while keeping its speed, it's the complete
   candidate. The team's still-training gemlb NANO variants (2.98M, ~45-50 ms on the
   laptop) are the direct like-for-like competitor — re-run this table when they land.

### The shipped-era baseline measured — 2026-08-05 (fourth column)

`/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` — Ultralytics YOLO11n, 2.59M
params, trained on the pre-Spotlight OIV7 labels; the closest available checkpoint to the
shipped `v11nclean2` (Dec-2025 best.pt vs the Nov-2025 TF.js export — exact identity
pending owner confirmation; shipped classes confirmed 0/1/2 = Woman/Man/Child via
dataset.yaml + extension source; the shipped metadata itself prints AGPL-3.0). Same four
datasets, same settings; ONNX 42.1 ms / 10.6 MB / raw output (browser still runs NMS).

| metric | shipped-era YOLO11n (old labels) | v9 GELAN-S (old labels, never shipped) | GELAN-S gemlb (Spotlight, masked) | YOLO26n (Spotlight, unmasked) |
|---|---|---|---|---|
| params / CPU ms | 2.59M / 42.1 (raw) | 8.2M / 136 | 8.2M / 135 | 2.38M / **33.4 (end-to-end)** |
| Object set: % images w/ false person | 54.1% | 53.7% | 18.2% | **12.7%** |
| PASS FPs/100 | 1.17 | 1.43 | 0.23 | 0.23 |
| Crowd recall / precision | 34.9 / 94.4 | 39.6 / 95.6 | **46.2** / 92.7 | 32.8 / 93.9 |
| LAGENDA detection recall | 95.3% | 96.0% | 95.8% | 94.1% |
| LAGENDA 3-class / gender on adults | 87.7 / 89.7 | 88.4 / 87.2 | 89.5 / 90.6 | 88.0 / 90.7 |
| Adult(GT≥20)→Child leak | 0.52% (14/2717) | 0.77% | **0.00%** | 0.11% |
| Teens 15–19 called Child | 15.0% | 26.6% | 13.9% | **8.4%** |

**What the fourth column changes:**
1. **The blurred-books bug is confirmed IN the deployed model class: 54.1% of
   doll/statue/toy images gain a false person** — statistically identical to the v9's 53.7%.
   Label era, not architecture, determines this failure: both old-label models ~54%, both
   Spotlight models 13–18%.
2. **YOLO26n's crowd-recall "regression" mostly evaporates against deployment reality:**
   32.8% vs the shipped-era 34.9% (−2.1) — the −6.8/−13.4 gaps were vs never-shipped and
   masked-retrain models. Vs what users run today, YOLO26n is ~parity on recall while being
   4× cleaner on objects, 5× on PASS, +1 gender, 5× lower leak, sharper teens, ~26% faster
   AND end-to-end.
3. The gemlb_s masked retrain beats the shipped model on recall by +11.3 pts but costs 3.2×
   the latency; the still-training gemlb NANOs are the arm that could deliver that recall at
   deployable speed — as would a masked/4-class YOLO26.

### ONNX export-bug follow-up — CLOSED 2026-08-05

Coworker reported his YOLO-MIT→ONNX exporter had a confidence-lowering bug and re-exported
`gelansfav14_gemlb_v1/weights/best.onnx` (Aug 5 09:48; `best.ckpt` unchanged at Aug 4 13:50
⇒ **all accuracy numbers unaffected** — accuracy always ran the native checkpoint).
**Parity check on the fixed export passed:** on 5 LAGENDA images, checkpoint top confidences
vs ONNX top anchor scores agree within ~±0.05 (0.92→0.96, 0.95→0.93, 0.66→0.72, 0.89→0.90,
0.94→0.93 — no systematic depression; ONNX outputs are in-graph-sigmoided probabilities).
Re-timed: 141.4 ms (vs 134.6 pre-fix on a different pod — cross-pod noise; speed conclusions
unchanged). **Open flags for the coworker:** (a) `best_web_model/` (the TF.js browser
export) is dated Aug 3 — BEFORE the fix — and needs re-exporting if it's ever deployed;
(b) `gelansfav14_datav2_v4/weights/best.onnx` presumably came from the same buggy exporter
generation — re-export before anyone quotes or ships it.

### Spotlight-val mAP, all four models, one pipeline — 2026-08-05 (team request)

Dataset: the Spotlight OIV7 val split — images `/workspace/open-images-v7/images/val/` via
the frozen `/workspace/exp12/val_full.txt` list, GT = the Gemini-verified labels at
`/workspace/spotlight/run/oiv7_val/labels/`.
4,232 val images (one of 4,233 lost to a symlink collision — identical set for all four),
`map_eval.py` over the raw sidecars (conf ≥ 0.05, COCO 101-point): yolo11N-640
**0.719 / 0.588** (mAP50 / mAP50-95) · gelansfav14_datav2_v4 **0.797 / 0.642** ·
gelansfav14_gemlb_v1 **0.836 / 0.695** · y26n_spotlight **0.799 / 0.685**. Framing recorded
in the brief: this measures alignment with the Spotlight labels (old-label models penalized
by construction); the 0.05 floor truncates the PR tail uniformly (ultralytics' own val gave
y26n 0.824/0.707 — the ~0.02 gap is the floor + interpolation differences, consistent
across columns). gemlb_s fits the labels best; y26n beats the old v9 at mAP50-95 at a third
the size.

**Phase E pre-answered by independent measurement (coworker, 2026-08-04, AMD Ryzen 5 Pro
5650U laptop, ONNX Runtime, 640×640, 3 classes):** yolo26n **33.7 ms CPU / 22.4 ms DML /
42.3 ms WebGPU** vs gelan-sfav14 (the production 8.2M model) **160.9 / 75.6 / 155.1 ms** —
**~4.8× faster CPU** — and faster than yolo11n (42.4 ms CPU) in every column. Third-party
numbers, not ours and not Ultralytics'; our own Phase E measurement remains to be run for
the record, but the ≥20% bar is clearly going to clear. Context from the same message: the
team is retraining the production architecture on new labels in two variants
(gelan-sfav14 ~34 h ETA, gelan-sfav14-fast 2.98M/8.5 GFLOPs ~36 h ETA), and an early
retrain shows map_small 3.7→12.5 on an internal val set — ask which label variant
(labels/ vs labels_unk3) that used; if masked, it starts answering the small-person data
question this experiment raised. Internal-val numbers are NOT comparable to this doc's
component benchmarks — his checkpoints get a third column via the protocol
(`docs/MODEL_EVAL_PROTOCOL.md`) when ready.

## What we can decide from this

_(after the run. The license gate decision belongs here too.)_

## What this does NOT tell us

Written before running, so the limits are honest:

- **The Gulf/traditional-dress slice remains untested** — same gap as every experiment before
  it. The `dubai_souk` flicker clip gives a temporal *peek* at thobe/ghutra behavior, not a
  measurement. The 30,567 Woman→Man Spotlight corrections are the ready-made probe sample and
  are a separate piece of work.
- **LAGENDA numbers are label-anchored (recall-side only)** — ~1 labeled person/image; they say
  nothing about precision there (that's what CrowdHuman/PASS/object set are for).
- **Both object-set rates are upper bounds** pending the EXP-2026-04 §2.5 re-verification pass
  (real people were later spotted in the set). Comparisons against the comparator on the *same*
  set are still fair; absolute rates are not.
- **PASS is contaminated at crop level** (EXP-2026-08 owner adjudication, ~5% flagged under
  `/workspace/exp07_conf/`) — we score the full set for comparability with the SAM3 rows AND
  with the flagged images excluded, and report both.
- **All YOLO26 speed/accuracy claims are vendor-published** (no independent benchmark found as
  of 2026-08-03) — which is exactly why Phase E measures latency ourselves instead of citing
  the docs.
- **Training-data difference vs the team retrain:** if the comparator was trained with
  `labels_unk3` masking and ours with the classes-0-2 variant, part of any delta is data
  handling, not architecture. Recorded, not hidden.

## What's next

- If adopted: `yolo26s` scale-up decision, onnxruntime-web integration spike, and the temporal
  smoothing policy replay (EXP-2026-11 follow-on) re-run over this model's sidecars.
- Either way: EXP-2026-13 (YOLOE) answers the open-vocabulary variant of the same question.

---

## Appendix — the details

### Frozen configuration (fill exact values in the runbook before launch; no mid-run changes)

| knob | value | source |
|---|---|---|
| ultralytics version | pinned `8.4.115` (2026-08-01) | runbook Phase 0 |
| Phase A weights | `yolo26n.pt` (COCO) | auto-download |
| working conf | **0.45** | extension production threshold (`HaramBlurP-main/src/constants.mjs`, confirmed EXP-2026-11) |
| NMS IoU | 0.7 (production; YOLO26 end-to-end ignores it) | same |
| imgsz | 640 | production + all prior experiments |
| log-raw floor | 0.05 | log-raw-readings principle |
| match IoU | 0.5 | `match_boxes` default, all prior experiments |
| train data | `/workspace/spotlight/run/oiv7_train/labels/` (+ `oiv7_val` as val) | Spotlight production run |
| GPU budget | $100 cap, pre-registered | this doc |

### Appendix — "verify it yourself" (written BEFORE running)

**How the model is called** — `vlm-cluster/run_ultralytics_labels.py`, `UltralyticsModel.detect`
(non-prompted branch loads `ultralytics.YOLO(model_path)`):

```python
def detect(self, img):
    res = self.model.predict(img, imgsz=self.imgsz, conf=self.floor,
                             iou=self.iou, device=self.device,
                             verbose=False)[0]
    ...
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        out.append((int(b.cls[0]), x1, y1, x2, y2, float(b.conf[0])))
```

**How label files are written** (the scorer boundary). Lines are 5-field on purpose —
`seg_boxes()` (run_autolabel_on_manifest.py:77) reads a 6-field line as a malformed polygon and
**silently drops it**, so confidences live only in the raw sidecars; and a zero-detection image
still writes an **empty** label file so "processed, found nothing" stays distinguishable from
"not processed" (the EXP-2026-04 lesson):

```python
lf.write_text("\n".join(lines) + ("\n" if lines else ""))
```

**Class mapping** — `run_ultralytics_labels.py`, `map_detection`:

```python
if mode == "identity":
    return cls, None                    # fine-tuned model: {0 Woman, 1 Man, 2 Child}
if mode == "coco-person":
    return (0, None) if cls == 0 else (None, "non_person_class")
```

Component 1 matching is class-agnostic (`match_boxes`, run_model_children.py — greedy
mutual-exclusive, IoU ≥ 0.5), so the Phase A `coco-person` class id carries no meaning and those
labels are never fed to a Component 2/3 scorer.

**How objects are matched** — unchanged `match_boxes`/`iou` from `run_model_children.py`, the
same functions every prior experiment quotes; CrowdHuman scored against **vbox** with
ignore-region handling exactly as `eval_negatives_crowd.py` documents (occlusion strata:
vbox/fbox ≥ 0.7 light, 0.3–0.7 partial, < 0.3 heavy).

**Flicker arm** — `video_flicker_probe.py --engine ultralytics` drives `UltralyticsModel`
through the *identical* `probe_video` loop and sidecar format as the EXP-2026-11 baseline;
`flicker_metrics.py` is untouched, so every metric definition is byte-identical to the baseline
row it's compared against.

**Selftests run 2026-08-03 (this machine, no GPU/network):**
`run_ultralytics_labels.py --selftest` (label format, empty-file semantics, distractor
exclusion, resume, and a round-trip through the real `seg_boxes` parser) and
`video_flicker_probe.py --selftest` — both pass.

**Honestly NOT done at design time:** no independent verification of any Ultralytics-published
number; the 20%-latency and 97%/98% accuracy bars are set by judgment anchored to prior
measurements (SAM3 98.7% LAGENDA recall, EXP-2026-10's 0% GT≥18 leak, vendor ~31% CPU claim),
not derived; the ultralytics train hyperparameters (epochs, batch, lr schedule) will be frozen
in the runbook at launch time rather than in this doc.
