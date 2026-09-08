# EXP-2026-02 — Pod runbook (you run, Claude reads outputs)

Claude has no pod access. Each phase is copy-paste blocks for the pod terminal
(web terminal or your own SSH). Blocks marked **→ paste back** produce output
Claude needs before writing the next phase's exact commands. Pod: **L4 or
A100** (not Blackwell/sm_120).

---

## Phase 1 — inspection (CPU, ~2 minutes, run first)

### 1a. The production labeler and its settings  → paste back

Confirmed location: `/workspace/autolabel/autolabel_sam.py` (alongside
`run_autolabel.sh`, which should hold the actual production invocations).

```bash
wc -l /workspace/autolabel/autolabel_sam.py
cat   /workspace/autolabel/autolabel_sam.py
cat   /workspace/autolabel/run_autolabel.sh
```

(Claude needs the full argparse/config section, the output-writing code, and
the shell script's invocation lines to freeze the production settings and know
how to point it at new images.)

### 1b. Confirm the answer key is intact and show its format  → paste back

```bash
ls /workspace/lagenda_eval/lagenda_yolo/
ls /workspace/lagenda_eval/lagenda_yolo/images* 2>/dev/null | head -5
head -c 2000 /workspace/lagenda_eval/lagenda_yolo/gt.jsonl; echo
wc -l /workspace/lagenda_eval/lagenda_yolo/gt.jsonl
```

### 1c. Confirm the tools dir + SAM3 weights exist  → paste back

```bash
ls /workspace/data_inspection_tools/vlm-cluster/ | head -40
find /workspace -iname "*sam3*" -not -path "*/.*" 2>/dev/null | head -20
```

**Stop here — send Claude all three outputs.** Claude then (a) adapts the
scorer's gt.jsonl loader if the field names differ, and (b) writes the exact
Stage A command with the production settings frozen.

---

## Phase 2 — sync the scorer (after Claude confirms Phase 1)

From your **local** machine (adjust host/port to your pod):

```bash
scp -P <POD_PORT> \
    ~/Desktop/HaramBlur/Innovation-lab/vlm-cluster/run_autolabel_on_manifest.py \
    root@<POD_IP>:/workspace/data_inspection_tools/vlm-cluster/
```

Then on the pod, verify it runs (no GPU, no data needed):

```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 run_autolabel_on_manifest.py --selftest   # → paste back if it fails
```

---

## Phase 3 — pilot (~500 images, GPU)

### 3a. Pre-flight (CPU, seconds)  → paste back

```bash
ls /workspace/lagenda_eval/lagenda_yolo/labels/
head -3 /workspace/lagenda_eval/lagenda_yolo/labels/val/dfc1605da5face39.txt
ls /workspace/lagenda_eval/lagenda_yolo/images/val | wc -l
python3 -c "from transformers import Sam3Model; print('sam3 import ok')"
```

(If the import fails: `pip install -r /workspace/autolabel/requirements.txt`.)

### 3b. Stage A — production labeler, settings FROZEN as production ran it

Frozen from `run_autolabel.sh` + the script's defaults: prompts
`"woman" "man" "child"` (class ids 0/1/2), `--conf 0.4` (default, what the
three real scrape runs used), `--nms_iou 0.7` (default), `--batch_size 1`.
Do NOT pass `--overwrite` — skip-existing is the resume mechanism.

```bash
cd /workspace/autolabel
python3 autolabel_sam.py \
    --input  /workspace/lagenda_eval/lagenda_yolo/images/val \
    --output /workspace/lagenda_eval/sam_autolabel/labels \
    --classes "woman" "man" "child" \
    --batch_size 1 \
    --max_images 500
```

Note the tqdm rate (images/s) — it sizes the full run.  → paste back the last
progress line.

### 3c. Stage B — score it (CPU)

```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 run_autolabel_on_manifest.py \
    --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
    --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --out         /workspace/lagenda_eval/sam_autolabel/eval_pilot
cat /workspace/lagenda_eval/sam_autolabel/eval_pilot/summary.json   # → paste back
```

(Unprocessed images are auto-excluded — `n_gt_not_processed_yet` in the
summary should be roughly 4,500 for a 500-image pilot.)

Also grab 5–10 files from `eval_pilot/overlays/` (download via VS Code) so we
can eyeball that matching looks right before the full run.

**Pilot gate (Claude checks):** counts reconcile, throughput is acceptable,
overlays sane. Only then:

## Phase 4 — full run

Stage A again **without `--max_images`** (already-labeled images are skipped,
so it just continues), then Stage B with `--out .../eval_full`.
→ paste back `eval_full/summary.json`; Claude writes up the experiment doc +
charts locally.

**Status: done** (2026-07-09) — see `EXP-2026-02-sam-autolabeler-accuracy.md`.

---

## Phase 5 — visual error gallery (optional, CPU-only)

A single self-contained HTML file with embedded image crops: wrong
classifications (teen-age 13-19 cases surfaced first — the headline finding),
missed detections, dual-class NMS-leak cases, and full-frame views of images
where SAM produced boxes not matched to any ground-truth person (so you can
eyeball whether they're real uncounted people or genuine false positives).

### 5a. Sync the new script

```bash
scp -P <POD_PORT> \
    ~/Desktop/HaramBlur/Innovation-lab/vlm-cluster/build_error_gallery.py \
    root@<POD_IP>:/workspace/data_inspection_tools/vlm-cluster/
```

Verify it (no data needed):
```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 build_error_gallery.py --selftest   # → paste back if it fails
```

### 5b. Build the gallery from the completed full run

```bash
python3 build_error_gallery.py \
    --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
    --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --out         /workspace/lagenda_eval/sam_autolabel/gallery.html
```

Takes a few minutes (reads/crops/encodes real images). → paste back the
printed counts line, e.g. `wrote .../gallery.html: {'wrong_class': 60, ...}`.

### 5c. Download and open it

In VS Code Remote (or `scp` the other direction), download
`/workspace/lagenda_eval/sam_autolabel/gallery.html` to your machine and open
it in any browser — no server needed, everything is embedded.

---

## Phase 6 — corrected accounting for crowded/multi-person images (2026-07-09)

The gallery surfaced a real methodology gap: LAGENDA only labels the "main"
person(s) per image, so (a) crowd photos made the unmatched-detection number
look worse than it is, and (b) in overlap scenes (e.g. a parent holding a
child) the box-matcher could in principle credit the wrong SAM box to a GT
person. Both scorer scripts now account for this — re-run Stage B (CPU-only,
uses the label files already on disk, no new SAM run) and rebuild the
gallery to get the corrected numbers:

```bash
cd /workspace/data_inspection_tools/vlm-cluster
# re-sync both updated scripts first (same scp pattern as Phase 2 / 5a)

python3 run_autolabel_on_manifest.py --selftest      # → paste back if it fails
python3 build_error_gallery.py --selftest            # → paste back if it fails

python3 run_autolabel_on_manifest.py \
    --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
    --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --out         /workspace/lagenda_eval/sam_autolabel/eval_full
cat /workspace/lagenda_eval/sam_autolabel/eval_full/summary.json   # → paste back

python3 build_error_gallery.py \
    --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
    --pred-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --out         /workspace/lagenda_eval/sam_autolabel/gallery.html
```

New in `summary.json` → `crowd_diagnostics`: `accuracy_3class_simple_images`
vs `accuracy_3class_crowded_images` (split instead of one blended number),
and `wrong_class_plausible_crowd_artifact` (what fraction of "wrong_class"
errors had a same-class SAM box sitting right next to the one we scored).
The gallery now has a new top section, "Ambiguous wrong-class", for exactly
those cases, and tags crowd images in the unmatched-detections section.

**2026-07-09, round 2:** the same confound applies to the dual-class-NMS-leak
metric — two SAM boxes both clipping one GT person's box doesn't prove one
physical person got double-labeled; it could be two *different* nearby people
(e.g. a man sitting behind a child), each legitimately boxed under their own
class. `pathologies.dual_class_survivors_naive` (old behavior) is now split
against `pathologies.dual_class_survivors_confirmed` (also requires the two
SAM boxes to overlap **each other**, not just the GT box, by IoU ≥ 0.5 —
i.e. plausibly the *same* object). The gallery's dual-class section is
correspondingly split into "Dual-class survivors, confirmed" and "Dual-class
flags, likely two different people". Re-sync + re-run the same two commands
above to get these numbers; thumbnail size/quality were also reduced for
easier ClickUp embedding (~220px/q62 for crops, ~480px/q58 for full frames).
