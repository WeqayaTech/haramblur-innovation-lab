# EXP-2026-10 — Pod runbook (you run, Claude reads outputs)

Pipeline v1 end-to-end: SAM3 frozen labels → mask-highlighted crops → Flash-Lite verdicts →
merge → per-dataset scores. **No GPU** — all SAM3 labels already exist on the volume. Keys from
the pod's `.env` (`set -a; source .env; set +a`).

Est. volume/cost: PASS ~40 crops (minutes) · LAGENDA ~1–2k crops (~30–60 min) · CrowdHuman
~13k crops (**~3.5–4h sequential — run in tmux/nohup; fully resumable, rerun the same command
to continue after any crash**). Total est. $10–15 at press-reported Lite rates.

---

## Phase 0 — sync + verify (minutes)

Copy to the pod's `/workspace/data_inspection_tools/vlm-cluster/`: **`pipeline_v1_eval.py`**
(new). Everything it imports is already there from EXP-2026-06/08/09.

```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 pipeline_v1_eval.py selftest          # stub engine, no network
# → paste back only if it fails

# the three frozen SAM3 label sets + images exist:
ls /workspace/lagenda_eval/sam_autolabel/labels | wc -l          # → paste back (~4-5k)
ls /workspace/lagenda_eval/lagenda_yolo/images/val | wc -l       # → paste back
ls /workspace/exp03/sam_labels/crowd | wc -l                     # → paste back
ls /workspace/exp03/sam_labels/pass  | wc -l                     # → paste back (~3000)
cat /workspace/exp03/sam_labels/crowd/$(ls /workspace/exp03/sam_labels/crowd | head -1) | head -2
# → paste back 2 label lines (confirms polygon format for mask rendering)
```

## Phase 0.5 — trace run: 4 images, every stage visualized (~$0.02, 2 min)

One self-contained HTML showing, per image: raw image + GT → SAM3 config + verbatim label
lines + rendered detections → the EXACT highlighted crop sent to Gemini + its raw JSON
response → the final merged labels (deleted = red, corrected boxes marked). **Eyeball this
before spending anything on the full arms** — it's the visual check that highlighting renders
sanely and verdicts look reasonable.

```bash
set -a; source .env; set +a

python3 pipeline_v1_eval.py trace \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --n 4 --out /workspace/exp10/trace

# optional: a crowd-scene trace too (busier images, no gender/age GT)
python3 pipeline_v1_eval.py trace \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --n 2 --max-dets 6 --out /workspace/exp10/trace_crowd
```

Download `/workspace/exp10/trace/trace.html` (and `trace_crowd/trace.html`), eyeball, and
only then proceed to Phase 1. If the highlight looks wrong (tint covering the wrong person,
outline off the body), stop and report — that's a rendering bug to fix before any full arm.

---

# PILOT — 25 images/arm + team report (CURRENT STEP; ~$1, ~30 min total)

Runs the identical pipeline at 1/20 scale, then builds the team-facing accuracy report.
**Sizing note:** PASS stays at 500 images — SAM3 only fires on ~5% of empty scenes, so 25
images would yield ~2 detections (statistically empty), while 500 images still cost only ~40
API calls. **Read pilot numbers as directional** (n=25 → wide CIs); the pre-registered bars
are formally judged at the full 500 scale.

Sync `pipeline_v1_report.py` (new) along with `pipeline_v1_eval.py`, then:

```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 pipeline_v1_report.py --selftest
set -a; source .env; set +a

# ── LAGENDA pilot (~50-100 crops) ──
python3 pipeline_v1_eval.py run \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --max-images 25 --seed 42 --out /workspace/exp10/pilot_lagenda
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pilot_lagenda \
    --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl

# ── CrowdHuman pilot (~650 crops, ~12 min) ──
python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --max-images 25 --seed 42 --out /workspace/exp10/pilot_crowd
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pilot_crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt

# ── PASS (500 imgs ≈ ~40 crops — see sizing note) ──
python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/pass_3k \
    --sam-labels /workspace/exp03/sam_labels/pass \
    --max-images 500 --seed 42 --out /workspace/exp10/pilot_pass
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pilot_pass --gt none

# ── the team report (one self-contained HTML: scores + evidence crops) ──
python3 pipeline_v1_report.py \
  --arm "lagenda:/workspace/exp10/pilot_lagenda:/workspace/lagenda_eval/lagenda_yolo/images/val:/workspace/lagenda_eval/sam_autolabel/labels" \
  --arm "crowd:/workspace/exp10/pilot_crowd:/workspace/datasets/crowdhuman/Images_sample500:/workspace/exp03/sam_labels/crowd" \
  --arm "pass:/workspace/exp10/pilot_pass:/workspace/datasets/pass_3k:/workspace/exp03/sam_labels/pass" \
  --note "PILOT — 25 imgs/arm (PASS 500), directional numbers; bars judged at full scale" \
  --out /workspace/exp10/pilot_report.html
```

**PILOT v1 OUTCOME (2026-07-23):** crowd TP-keep 80.4% — the 35%-alpha tint obscured
small/distant people (owner-confirmed visually). Variant recovery test on the 76 failures:
outline-only 45/76, **outline + `--min-crop-side 320` = 68/76 → adopted (projects 97.9%)**.
All subsequent runs — pilot v2 below AND the full Phases 1–3 — must add:
`--highlight-style outline --min-crop-side 320`.

## Trace v2 — 5 random images per dataset under the ADOPTED config (~$0.10, ~5 min)

The stage-by-stage documentation of the final pipeline (outline + upscale-320 is now the
DEFAULT in `pipeline_v1_eval.py` — no flags needed). Three self-contained HTMLs, one per
dataset; PASS's trace picks only images where SAM3 fired (the interesting ones):

```bash
set -a; source .env; set +a

python3 pipeline_v1_eval.py trace \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --n 5 --seed 7 --out /workspace/exp10/trace2_lagenda

python3 pipeline_v1_eval.py trace \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --n 5 --max-dets 25 --seed 7 --out /workspace/exp10/trace2_crowd
# --max-dets 25 traces WHOLE crowd images; any detections beyond the cap are
# drawn gray "not traced" in Stage 4 (never red) so the cap can't read as deletion

python3 pipeline_v1_eval.py trace \
    --images /workspace/datasets/pass_3k \
    --sam-labels /workspace/exp03/sam_labels/pass \
    --n 5 --seed 7 --out /workspace/exp10/trace2_pass
```

Download the three `trace.html` files (rename per dataset when attaching). Seed 7 ≠ the
pilot's seed 42, so these are fresh random images, not ones already discussed.

## Raw-mask trace — SAM3 re-run with raw readings logged (GPU: L4, ~15 min total)

The definitive rendering path: `autolabel_sam_raw.py` (patched copy, production untouched —
EXP-2026-07 pattern) re-runs SAM3 on the 5 trace images and logs the RAW readings per image
(`<stem>.json`: per-part mask polygons BEFORE the YOLO single-polygon flattening, raw scores,
plus what NMS suppressed) alongside byte-identical YOLO output. The trace then renders
highlights from the exact parts — zero heuristics.

```bash
# ── on an L4 pod (NOT Blackwell) ──
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
export HF_HOME=/workspace/.cache/huggingface

# sync autolabel_sam_raw.py into /workspace/autolabel/ (+ updated pipeline_v1_eval.py
# into vlm-cluster), then materialize the same 5 seed-7 trace images:
cd /workspace/data_inspection_tools/vlm-cluster
python3 - <<'EOF'
from pathlib import Path
import shutil
from vlm_detect_eval import select_images
from pipeline_v1_eval import seg_polys
src = Path('/workspace/datasets/crowdhuman/Images_sample500')
sam = Path('/workspace/exp03/sam_labels/crowd')
out = Path('/workspace/exp10/trace_raw_imgs'); out.mkdir(parents=True, exist_ok=True)
picked = []
for p in select_images(src, 0, 7):
    if seg_polys(sam / f"{p.stem}.txt", 1, 1):
        picked.append(p)
    if len(picked) == 5: break
for p in picked: shutil.copy2(p, out / p.name)
print("copied:", [p.name for p in picked])
EOF

cd /workspace/autolabel
python3 autolabel_sam_raw.py --input /workspace/exp10/trace_raw_imgs \
    --output /workspace/exp10/raw_labels \
    --classes "woman" "man" "child" --batch_size 1
ls /workspace/exp10/raw_labels/          # 5x .txt + 5x .json → paste back
python3 -c "import json; r=json.load(open(sorted(__import__('glob').glob('/workspace/exp10/raw_labels/*.json'))[0])); print(r['image'], len(r['detections']),'dets,', sum(len(d['parts'])>1 for d in r['detections']),'multi-part,', len(r['suppressed']),'nms-suppressed')"
# → paste back (proves multi-part masks + suppressed dets were captured)

# ── trace from the RAW parts (CPU + API, ~$0.08; any pod) ──
cd /workspace/data_inspection_tools/vlm-cluster
set -a; source .env; set +a
python3 pipeline_v1_eval.py trace \
    --images /workspace/exp10/trace_raw_imgs \
    --raw-labels /workspace/exp10/raw_labels \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --n 5 --max-dets 25 --seed 7 --out /workspace/exp10/trace5_crowd_raw
```

Crop captions show "RAW mask: N part(s)"; Stage 1 shows the raw JSON instead of the flattened
YOLO lines. Compare against trace3/trace4 — same images, heuristic vs exact rendering.

## Pilot v2 — confirmation under the winning config (~$1.1, ~25 min)

```bash
set -a; source .env; set +a
# crowd: seed with the 76 variant-C verdicts already paid for (resume-by-id)
mkdir -p /workspace/exp10/pilotC_crowd
cp /workspace/exp10/fail_outline_up/verdicts.jsonl /workspace/exp10/pilotC_crowd/verdicts.jsonl

python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --max-images 25 --seed 42 --highlight-style outline --min-crop-side 320 \
    --out /workspace/exp10/pilotC_crowd
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pilotC_crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt

python3 pipeline_v1_eval.py run \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --max-images 25 --seed 42 --highlight-style outline --min-crop-side 320 \
    --out /workspace/exp10/pilotC_lagenda
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pilotC_lagenda \
    --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl

python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/pass_3k \
    --sam-labels /workspace/exp03/sam_labels/pass \
    --max-images 500 --seed 42 --highlight-style outline --min-crop-side 320 \
    --out /workspace/exp10/pilotC_pass
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pilotC_pass --gt none

python3 pipeline_v1_report.py \
  --arm "lagenda:/workspace/exp10/pilotC_lagenda:/workspace/lagenda_eval/lagenda_yolo/images/val:/workspace/lagenda_eval/sam_autolabel/labels" \
  --arm "crowd:/workspace/exp10/pilotC_crowd:/workspace/datasets/crowdhuman/Images_sample500:/workspace/exp03/sam_labels/crowd" \
  --arm "pass:/workspace/exp10/pilotC_pass:/workspace/datasets/pass_3k:/workspace/exp03/sam_labels/pass" \
  --note "PILOT v2 — outline + min-crop-side 320 (the adopted config)" \
  --out /workspace/exp10/pilotC_report.html
# → paste back the three score JSONs; download pilotC_report.html for the team
```

Read pilot v2 against: crowd TP-keep ≥ 97% (the blocker), LAGENDA gate stays ~26/26, PASS
kill ≥ 65% (outline may keep more dolls than tint did — watch this), leak cases eyeballed.

→ paste back the three score JSONs + cost reports, and download the report for the
team. **Confirmation gate: full 500-image arms (Phases 1–4 below) run only after the team
signs off on the pilot v2 report — and add `--highlight-style outline --min-crop-side 320`
to every Phase 1–3 run command.** The full runs REUSE the pilot verdicts automatically —
resume-by-id means pointing the full run at the same `--out` dir only pays for the new
detections (use `/workspace/exp10/pilot_*` as the full-run out dirs, or copy verdicts.jsonl
over before starting).

---

## Phase 1 — PASS arm first (tiny; validates the whole plumbing for pennies)

```bash
set -a; source .env; set +a

python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/pass_3k \
    --sam-labels /workspace/exp03/sam_labels/pass \
    --max-images 500 --seed 42 --out /workspace/exp10/pass

python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pass --gt none
# → paste back the score JSON + cost_report.json
```

Sanity: only images where SAM3 fired produce crops (~40 detections expected from ~500 PASS
images). Every kill is a poisoned training label prevented.

## Phase 2 — LAGENDA arm + the highlight-bias QC (~1h)

```bash
python3 pipeline_v1_eval.py run \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --max-images 500 --seed 42 --out /workspace/exp10/lagenda

python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/lagenda \
    --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl
# → paste back score JSON + cost

# QC arm — SAME first 200 images (same seed ⇒ shared detection ids), PLAIN crops:
python3 pipeline_v1_eval.py run \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
    --max-images 200 --seed 42 --plain-crops --out /workspace/exp10/lagenda_plain

python3 pipeline_v1_eval.py compare \
    --run-a /workspace/exp10/lagenda/verdicts.jsonl \
    --run-b /workspace/exp10/lagenda_plain/verdicts.jsonl
# → paste back (bar 6: keep-flips < 2%, label-flips < 3%)
```

## Phase 3 — CrowdHuman arm (the long one, ~3.5–4h; tmux)

```bash
tmux new -s exp10   # or: nohup ... &

python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --max-images 500 --seed 42 --out /workspace/exp10/crowd
# resumable — if it dies (503 storm etc.), rerun the same command

python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt
# → paste back score JSON + cost
```

## Phase 4 — survivor skims (human, ~20 min)

- PASS survivors: listed in `/workspace/exp10/pass/score_none.json` — for each, is the crop a
  real human/depiction (gate right, benchmark contaminated) or true junk (gate wrong)?
- Crowd clear-FP survivors: same question (posters count as people per the standing ruling).
→ paste back tallies. Then Claude fills the scorecard, writes the verdict, and updates the
pipeline doc's gate table.

---

# FULL RUN — the confidence program (raw labels; supersedes Phases 1–4 and Pilot v2)

Owner decision 2026-07-23: fresh raw labels via `autolabel_sam_raw.py` (see the doc's
full-run decision note). Crowd arm runs FIRST so a TP-keep failure stops spending early.
Total: ~1.5–2 h GPU (Stage A) + ~4 h API wall time (~$9.2) + ~2 h human steps.

## F1 — Stage A: materialize subsets + raw-label all three datasets (GPU, ~1.5–2 h)

```bash
# subsets (seed 42 = the pre-registered sample; crowd uses Images_sample500 as-is)
cd /workspace/data_inspection_tools/vlm-cluster
python3 - <<'EOF'
from pathlib import Path
import shutil
from vlm_detect_eval import select_images
for name, src, n in (("full_lagenda_imgs", "/workspace/lagenda_eval/lagenda_yolo/images/val", 500),
                     ("full_pass_imgs", "/workspace/datasets/pass_3k", 500)):
    out = Path(f"/workspace/exp10/{name}"); out.mkdir(parents=True, exist_ok=True)
    picked = select_images(Path(src), n, 42)
    for p in picked: shutil.copy2(p, out / p.name)
    print(name, len(picked))
EOF

export HF_HOME=/workspace/.cache/huggingface
cd /workspace/autolabel
python3 autolabel_sam_raw.py --input /workspace/datasets/crowdhuman/Images_sample500 \
    --output /workspace/exp10/raw_full/crowd --classes "woman" "man" "child" --batch_size 1
python3 autolabel_sam_raw.py --input /workspace/exp10/full_lagenda_imgs \
    --output /workspace/exp10/raw_full/lagenda --classes "woman" "man" "child" --batch_size 1
python3 autolabel_sam_raw.py --input /workspace/exp10/full_pass_imgs \
    --output /workspace/exp10/raw_full/pass --classes "woman" "man" "child" --batch_size 1
for d in crowd lagenda pass; do echo "$d: $(ls /workspace/exp10/raw_full/$d/*.json | wc -l) sidecars"; done
# → paste back
```

## F2 — the arms (CPU + API; crowd FIRST, in tmux; all resumable)

```bash
cd /workspace/data_inspection_tools/vlm-cluster
set -a; source .env; set +a

# CROWD (~10.5k dets, ~3 h, ~$6.4) — check its score BEFORE running the rest
python3 pipeline_v1_eval.py run \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --raw-labels /workspace/exp10/raw_full/crowd \
    --max-images 0 --seed 42 --out /workspace/exp10/full_crowd
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/full_crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt
# → paste back; GATE: tp_keep_rate >= 0.97 or STOP here

# LAGENDA (~3.2k dets, ~1 h, ~$2)
python3 pipeline_v1_eval.py run \
    --images /workspace/exp10/full_lagenda_imgs \
    --raw-labels /workspace/exp10/raw_full/lagenda \
    --max-images 0 --seed 42 --out /workspace/exp10/full_lagenda
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/full_lagenda \
    --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl

# PASS (~50 dets, minutes)
python3 pipeline_v1_eval.py run \
    --images /workspace/exp10/full_pass_imgs \
    --raw-labels /workspace/exp10/raw_full/pass \
    --max-images 0 --seed 42 --out /workspace/exp10/full_pass
python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/full_pass --gt none

# QC arm: 200 LAGENDA images PLAIN crops (~1.3k dets, ~$0.8) + compare
python3 pipeline_v1_eval.py run \
    --images /workspace/exp10/full_lagenda_imgs \
    --raw-labels /workspace/exp10/raw_full/lagenda \
    --max-images 200 --seed 42 --plain-crops --out /workspace/exp10/full_lagenda_plain
python3 pipeline_v1_eval.py compare \
    --run-a /workspace/exp10/full_lagenda/verdicts.jsonl \
    --run-b /workspace/exp10/full_lagenda_plain/verdicts.jsonl
# → paste back all score JSONs + cost reports + the compare
```

## F3 — the team report

```bash
python3 pipeline_v1_report.py \
  --arm "lagenda:/workspace/exp10/full_lagenda:/workspace/exp10/full_lagenda_imgs:/workspace/exp10/raw_full/lagenda" \
  --arm "crowd:/workspace/exp10/full_crowd:/workspace/datasets/crowdhuman/Images_sample500:/workspace/exp10/raw_full/crowd" \
  --arm "pass:/workspace/exp10/full_pass:/workspace/exp10/full_pass_imgs:/workspace/exp10/raw_full/pass" \
  --note "FULL RUN — 500 imgs/dataset, fresh raw labels, outline+320 config" \
  --out /workspace/exp10/full_report.html
```

## F4 — human steps (~2 h total, parallelizable)

- PASS + crowd FP survivor skims (listed in the score JSONs / report).
- Accepted-tier audit: ~100 random KEPT labels eyeballed (shared-bias check):
```bash
python3 - <<'EOF'
import random
from pathlib import Path
from pipeline_v1_eval import load_verdicts, merge
kept = []
for arm in ("full_lagenda", "full_crowd"):
    for r in load_verdicts(Path(f"/workspace/exp10/{arm}")):
        m = merge(r)
        if m["keep"] and m["final_class"] not in (None, "Unknown"):
            kept.append((arm, r["id"], m["final_class"]))
random.Random(7).shuffle(kept)
Path("/workspace/exp10/audit_sample.txt").write_text(
    "\n".join(f"{a}\t{i}\t{c}" for a, i, c in kept[:100]) + "\n")
print("100 audit ids -> /workspace/exp10/audit_sample.txt (crops findable in full_report.html by id)")
EOF
```
- EXP-2026-09 Phase 5 adjudication (still open — locks the gender-conflict rule).
- Gulf-dress slice (Track 2 — build + run + adjudicate; the one non-negotiable extra).

## F5 — closeout

- Fill the doc scorecard + short version; update `docs/COMPONENT_FRAMEWORK.md` + team brief.
- Verify Flash-Lite rates (Google pricing page) → fill model_pricing.json →
  `python3 recompute_costs.py` → all cost reports become verified dollars.

## Wrap-up checklist

- [ ] All three score JSONs + cost reports + QC compare pasted back
- [ ] Survivor skim tallies pasted back
- [ ] Doc scorecard + short version filled; pipeline doc gates V1/V2 updated
- [ ] Component scoreboard updated where applicable
