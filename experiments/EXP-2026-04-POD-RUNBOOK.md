# EXP-2026-04 — Pod runbook (you run, Claude reads outputs)

Same pattern as EXP-2026-02/03: copy-paste blocks for the pod terminal; blocks marked
**→ paste back** produce output Claude needs. Pod: **L4 or A100** (not Blackwell/sm_120).

This experiment reuses EXP-2026-03's staged datasets unchanged — there is **no dataset phase**.
The only Stage A difference from EXP-2026-03 is `--classes "person"` instead of
`--classes "woman" "man" "child"`. Outputs go under `/workspace/exp04/`.

---

## Phase 0 — verify the frozen pieces (CPU, minutes)

```bash
mkdir -p /workspace/exp04/{sam_labels/{objects,pass,crowd},eval}

# labeler still there, and its frozen defaults haven't drifted since EXP-2026-02/03
ls /workspace/autolabel/autolabel_sam.py
grep -n "conf\|nms\|default" /workspace/autolabel/autolabel_sam.py | grep -i "add_argument\|default" | head -20
# → paste back (goes into the experiment doc's B.1/B.5 — confirms conf 0.4 / nms 0.7)

# datasets untouched since EXP-2026-03
echo "objects: $(find /workspace/datasets/object_set -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' | wc -l)"
echo "pass:    $(find /workspace/datasets/pass_3k -type f | wc -l)"
echo "crowd:   $(find /workspace/datasets/crowdhuman -name '*.jpg' | wc -l)  odgt: $(wc -l < /workspace/datasets/crowdhuman/annotation_val.odgt)"
# → paste back (expect ~259 / 3000 / CrowdHuman val counts matching EXP-2026-03)

# scorer present + self-test
cd /workspace/data_inspection_tools/vlm-cluster
python3 eval_negatives_crowd.py --selftest        # → paste back only if it fails
```

(If `eval_negatives_crowd.py` is missing on the pod — new pod, old volume is fine — scp it from
local `vlm-cluster/` as in EXP-2026-03 Phase 0.)

## Phase 1 — Stage A: same labeler, single "person" prompt (GPU, ~2h total)

Only the prompt changes. Estimates from the L4's measured ~1.08 it/s (3-prompt); one prompt may
run faster: objects ~3 min · PASS ~46 min · CrowdHuman ~68 min. Each run is resumable — rerun the
same command to continue; don't pass `--overwrite`.

```bash
export HF_HOME=/workspace/.cache/huggingface
cd /workspace/autolabel

python3 autolabel_sam.py --input /workspace/datasets/object_set \
    --output /workspace/exp04/sam_labels/objects \
    --classes "person" --batch_size 1

python3 autolabel_sam.py --input /workspace/datasets/pass_3k \
    --output /workspace/exp04/sam_labels/pass \
    --classes "person" --batch_size 1

python3 autolabel_sam.py --input /workspace/datasets/crowdhuman/Images_sample500 \
    --output /workspace/exp04/sam_labels/crowd \
    --classes "person" --batch_size 1
# → paste back each run's final tqdm line
```

(`Images_sample500/` is the exp03 seed-51 sample dir — confirmed on-volume in Phase 0. Running on
the full `Images/` (4,370) would cost ~10× the GPU time and not be comparable.)

```bash
```

## Phase 2 — Stage B: score each dataset separately (CPU, minutes)

Unchanged scorer; `--class-names Person` maps class id 0 for display.

```bash
cd /workspace/data_inspection_tools/vlm-cluster

python3 eval_negatives_crowd.py --mode negatives \
    --images /workspace/datasets/object_set \
    --pred-labels /workspace/exp04/sam_labels/objects \
    --out /workspace/exp04/eval/objects --overlays 60 --class-names Person
cat /workspace/exp04/eval/objects/summary.json          # → paste back

python3 eval_negatives_crowd.py --mode negatives \
    --images /workspace/datasets/pass_3k \
    --pred-labels /workspace/exp04/sam_labels/pass \
    --out /workspace/exp04/eval/pass --overlays 60 --class-names Person
cat /workspace/exp04/eval/pass/summary.json             # → paste back

python3 eval_negatives_crowd.py --mode crowd \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --pred-labels /workspace/exp04/sam_labels/crowd \
    --out /workspace/exp04/eval/crowd --overlays 60 --class-names Person
cat /workspace/exp04/eval/crowd/summary.json            # → paste back
```

Optional while the pod is up (feeds the EXP-2026-03 closeout's sensitivity item too):

```bash
python3 eval_negatives_crowd.py --mode crowd \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --pred-labels /workspace/exp04/sam_labels/crowd \
    --out /workspace/exp04/eval/crowd_iou04 --overlays 0 --match-iou 0.4 --class-names Person
python3 eval_negatives_crowd.py --mode crowd \
    --images /workspace/datasets/crowdhuman/Images_sample500 \
    --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --pred-labels /workspace/exp04/sam_labels/crowd \
    --out /workspace/exp04/eval/crowd_iou06 --overlays 0 --match-iou 0.6 --class-names Person
cat /workspace/exp04/eval/crowd_iou0*/summary.json      # → paste back
```

## Phase 3 — overlays (human eyeball)

Download `/workspace/exp04/eval/*/overlays/` (small annotated JPEGs) and compare side-by-side
with EXP-2026-03's: does the "person" prompt still box the dolls? Same clear-FP tally protocol
for CrowdHuman (real unlabeled person / poster-statue / genuine hallucination) if the precision
number moved meaningfully.

## Phase 4 — write-up

Claude fills the scorecard in `EXP-2026-04-sam-person-prompt-detection.md` (§1), applies the
pre-registered comparative decision rules, updates the Component 1 scoreboard in
`docs/COMPONENT_FRAMEWORK.md`, and generates comparison charts.
