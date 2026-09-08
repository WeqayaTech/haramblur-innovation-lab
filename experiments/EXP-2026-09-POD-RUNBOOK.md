# EXP-2026-09 — Pod runbook (you run, Claude reads outputs)

Same pattern as EXP-2026-02/03/04/06: copy-paste blocks; **→ paste back** marks output Claude
needs. **No GPU needed** — the SAM3 arm reuses the frozen EXP-2026-03 labels, Flash-Lite is
API-billed, scoring is CPU. Any CPU pod with the volume works.

API keys come from the pod's existing `.env` at
`/workspace/data_inspection_tools/vlm-cluster/.env` (same as EXP-2026-06/08) — loaded with
`set -a; source .env; set +a`. **Never paste real key values into chat or into any file that
syncs back to the repo.** (Rotation of the four EXP-2026-06 keys is already a pending item.)

Expected cost: well under $1 (Flash-Lite, ~100 images; EXP-2026-08's 1,424 gate crops cost
$0.76 total, and this is ~100 calls with larger outputs).

---

## Phase 0 — sync code + verify prerequisites (minutes)

Copy from local `vlm-cluster/` into the pod's `/workspace/data_inspection_tools/vlm-cluster/`:

- `crowd_headtohead.py` (**new** — the head-to-head scorer + gallery)

These should already be on the pod from EXP-2026-06/08 — verify, don't assume:
`vlm_detect_eval.py`, `reparse_boxes.py`, `api_describers.py`, `model_pricing.json`,
`describe.py`, plus EXP-2026-03's `eval_negatives_crowd.py`, `run_autolabel_on_manifest.py`,
`run_model_children.py` (crowd_headtohead imports from all three).

```bash
cd /workspace/data_inspection_tools/vlm-cluster
pip install -q google-genai pillow opencv-python-headless

python3 crowd_headtohead.py --selftest          # no data, no network
python3 vlm_detect_eval.py --selftest
python3 reparse_boxes.py --selftest
# → paste back only if any fails

# data still where the registry says
ls /workspace/datasets/crowdhuman/ | head            # Images  Images_sample500  annotation_val.odgt
find /workspace/datasets/crowdhuman/Images_sample500 -name '*.jpg' | wc -l    # 500
ls /workspace/exp03/sam_labels/crowd/ | wc -l        # → paste back (SAM label count)
```

## Phase 1 — freeze the 100-image subset + SAM coverage check (CPU, seconds)

```bash
cd /workspace/data_inspection_tools/vlm-cluster

python3 crowd_headtohead.py --make-subset \
    --source-images /workspace/datasets/crowdhuman/Images_sample500 \
    --n 100 --seed 42 --subset-out /workspace/exp09/subset100

# every subset image must already have a frozen EXP-2026-03 SAM label file
missing=0
while read f; do
  [ -f "/workspace/exp03/sam_labels/crowd/${f%.jpg}.txt" ] || { echo "MISSING $f"; missing=$((missing+1)); }
done < /workspace/exp09/subset100/subset_list.txt
echo "missing SAM labels: $missing"          # → paste back
```

If `missing` is anything but 0, **stop and paste back** — we'll plan a targeted Stage A top-up
(that would need an L4 + the fresh-pod torch fix) rather than silently scoring a partial arm.

## Phase 2 — the Flash-Lite run (CPU + API, ~10–30 min)

```bash
cd /workspace/data_inspection_tools/vlm-cluster
set -a; source .env; set +a                      # loads the keys, same as EXP-2026-08
echo "google key present: ${GOOGLE_API_KEY:+yes}${GEMINI_API_KEY:+yes}"   # prints yes, never the value

python3 vlm_detect_eval.py --engine gemini --model gemini-3.5-flash-lite \
    --images /workspace/exp09/subset100 --gt crowdhuman \
    --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --max-images 0 --seed 42 --box-prompt gemini --max-tokens 8000 \
    --out /workspace/exp09/lite_crowd
# --max-images 0 = ALL images in the dir (the subset is the sample; don't re-subsample)

cat /workspace/exp09/lite_crowd/cost_report.json     # → paste back
grep -c parse_ok /workspace/exp09/lite_crowd/detections.jsonl          # should be 100
grep -c '"parse_ok": false' /workspace/exp09/lite_crowd/detections.jsonl || true   # → paste back
```

Don't bother pasting this phase's `summary.json` — its scores are wrong-axis until Phase 3.

## Phase 3 — box-convention diagnose + rewrite (CPU, seconds)

Flash-Lite was stable y-first in EXP-2026-06; verify on THIS run before rewriting:

```bash
python3 reparse_boxes.py --run /workspace/exp09/lite_crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt
# → paste back the whole diagnose table (goes into the experiment doc appendix §2)

python3 reparse_boxes.py --run /workspace/exp09/lite_crowd \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --write yxyx_1000
```

(If the diagnose table says a different convention wins, write THAT one and flag it — don't
force yxyx_1000 against the data.)

## Phase 4 — the head-to-head scoring (CPU, ~1 min)

```bash
python3 crowd_headtohead.py \
    --images /workspace/exp09/subset100 \
    --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --lite-detections /workspace/exp09/lite_crowd/detections.jsonl \
    --out /workspace/exp09/headtohead

cat /workspace/exp09/headtohead/summary.json         # → paste back (the main result)
```

Sanity expectations: `images_scored` = 100, `images_missing_*` = 0, `gt_persons` ≈ 2,000–2,500.

## Phase 5 — adjudication (human, ~30–45 min)

Download `/workspace/exp09/headtohead/gallery.html` (one self-contained file) and open locally.

1. **Section 1 (label disagreements):** tally every crop as
   `SAM right / Lite right / both wrong / can't tell`, and separately note each crop where the
   disagreement is adult↔Child (the direction that matters). → paste back the tallies.
2. **Section 2 (agreement sample, 30 crops):** quick skim — are the agreed labels right? Note
   any that are jointly wrong. → paste back a one-line verdict + count.
3. **Sections 3/4 (found-by-only-one samples):** skim for a qualitative sentence each — what
   kind of person does each model miss?

Then Claude fills the scorecard in `EXP-2026-09-sam3-vs-flashlite-crowd.md`, makes charts, and
updates the Component 1 scoreboard in `docs/COMPONENT_FRAMEWORK.md`.

---

# Part 2 — best-setup arm (OUTCOME 2026-07-23: CONCLUDED — no further runs needed)

**Phase 6 ran and Phase 7 is unnecessary.** The `high` media-resolution run produced input
tokens byte-identical to Part 1 (152,794); a 4-call diagnostic (doc §Part 2) showed
`low`=275 / `medium`=547 / `high`=default=1,095 tokens — **Flash-Lite's default is already the
API's maximum resolution**, so Part 1's configuration was already "best." The Phase 6 output
(`/workspace/exp09/lite_crowd_hires/`) is kept as a run-to-run variance replicate (recall
49.2→50.0, precision 93.2→91.9). Original instructions below for the record.

Pre-registered in the experiment doc §Part 2. **Lite-only** — the SAM3 low-conf arm was dropped
before running (owner decision, recorded in the doc; SAM3's frozen production settings ARE its
best measured setup per EXP-2026-04/07). **No GPU needed anywhere.** Sync the updated
`crowd_headtohead.py`, `api_describers.py`, and `vlm_detect_eval.py` to the pod first, then
re-run the selftests (Phase 0 block) — the media-resolution code path is new.

## Phase 6 — Flash-Lite at high media resolution (CPU + API, ~10–30 min, <$1)

```bash
cd /workspace/data_inspection_tools/vlm-cluster
set -a; source .env; set +a

python3 vlm_detect_eval.py --engine gemini --model gemini-3.5-flash-lite \
    --images /workspace/exp09/subset100 --gt crowdhuman \
    --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --max-images 0 --seed 42 --box-prompt gemini --max-tokens 8000 \
    --media-resolution high --out /workspace/exp09/lite_crowd_hires

cat /workspace/exp09/lite_crowd_hires/cost_report.json    # → paste back
# input tokens/image vs Part 1's 1,528 = the resolution actually changed; if
# it's identical, the API ignored the knob for this model — stop and paste back

python3 reparse_boxes.py --run /workspace/exp09/lite_crowd_hires \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt
# → paste diagnose table, then rewrite under the winner (expected yxyx_1000):
python3 reparse_boxes.py --run /workspace/exp09/lite_crowd_hires \
    --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --write yxyx_1000
```

## Phase 7 — score L+ against the same frozen SAM3 arm (CPU, ~1 min)

```bash
cd /workspace/data_inspection_tools/vlm-cluster

python3 crowd_headtohead.py \
    --images /workspace/exp09/subset100 \
    --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --sam-labels /workspace/exp03/sam_labels/crowd \
    --lite-detections /workspace/exp09/lite_crowd_hires/detections.jsonl \
    --out /workspace/exp09/h2h_part2 --crops 250

cat /workspace/exp09/h2h_part2/summary.json      # → paste back
```

SAM3's numbers must be identical to Part 1's (same labels, same scorer — a differing SAM3 row
means something is wrong). The Lite row is the L+ result: compare recall/precision/matched-IoU
and parse-failure count against Part 1's 49.2% / 93.2% / 0.842 / 4. The gallery from this run
also shows whether hi-res changed WHAT Lite sees (more small/distant people) vs just box quality.

## Wrap-up checklist

- [ ] `summary.json`, diagnose table, cost report, tallies pasted back
- [ ] Part 2: hi-res cost report + h2h_part2 summary pasted back; SAM3 row identical to Part 1
- [ ] Experiment doc scorecards (Part 1 + Part 2) + "short version" filled
- [ ] `docs/COMPONENT_FRAMEWORK.md` scoreboard updated
- [ ] Charts (`make_charts_exp09.py`, SVG→PNG) + ClickUp copy if the result is worth posting
- [ ] Regenerate the API key used in Phase 2 if it was ever pasted anywhere
