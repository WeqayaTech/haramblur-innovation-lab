# EXP-2026-13 — pod runbook (YOLOE promptable candidate + distractor arm)

Copy-paste, phased. Phases A/B/D/E run on an **L4** (~2–3 h total); Phase C (fine-tune, only if
the Q1 gate passes) needs an **A100**. Bars are pre-registered in the experiment doc; nothing
here may change them. Same house rules as the EXP-2026-12 runbook (nohup + hostname-stamped
logs on the volume, OMP pinning, no "negative" in dir names, empty-label-file semantics).

## Phase 0 — setup + the prompt-order assertion

Identical setup + dataset verification as EXP-2026-12 Phase 0 (run it; the two experiments
share every input, and the crowd sample symlink dir can be reused —
`/workspace/exp12/crowd_sample_imgs`). Then the one YOLOE-specific check, **which must pass
before anything else runs** — class ids must follow prompt order or every downstream number is
mislabeled:

```bash
python3 - <<'EOF'
from ultralytics import YOLOE
m = YOLOE('yoloe-26s-seg.pt')
vocab = ["woman", "man", "child", "statue", "mannequin", "doll"]
try:
    m.set_classes(vocab)
except TypeError:
    m.set_classes(vocab, m.get_text_pe(vocab))
names = m.model.names if hasattr(m.model, 'names') else m.names
print(names)
assert [names[i] for i in range(6)] == vocab, "PROMPT ORDER != CLASS IDS — ABORT"
print("prompt-order assertion OK")
EOF
```

## Phase A — zero-shot, production vocabulary (4 datasets)

```bash
HOST=$(hostname); cd /workspace/data_inspection_tools/vlm-cluster
mkdir -p /workspace/exp13
for DS in crowd:/workspace/exp12/crowd_sample_imgs \
          pass:/workspace/datasets/pass_3k \
          objects:/workspace/datasets/object_set \
          lagenda:/workspace/lagenda_eval/lagenda_yolo/images/val; do
  name=${DS%%:*}; dir=${DS#*:}
  nohup python3 -u run_ultralytics_labels.py --model yoloe-26s-seg.pt \
      --prompts "woman,man,child" \
      --images "$dir" --out /workspace/exp13/yoloe_s/$name --conf 0.45 \
      > /workspace/exp13/log_yoloeS_${name}_$HOST.log 2>&1 &
  wait
done
```

(⚠ verify the LAGENDA images dir name on the pod first, as in the EXP-2026-12 runbook.)
If the s-run is healthy, repeat with `yoloe-26m-seg.pt` → arm `yoloe_m`.

Score — Component 1 (same three `eval_negatives_crowd.py` commands as the EXP-2026-12 runbook,
`--pred-labels /workspace/exp13/yoloe_s/<name>/labels`, out `/workspace/exp13/eval/yoloe_s/…`;
PASS reported full-set AND with the `/workspace/exp07_conf/` flagged images excluded), then
Components 2+3:

```bash
python3 run_autolabel_on_manifest.py \
    --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
    --pred-labels /workspace/exp13/yoloe_s/lagenda/labels \
    --out         /workspace/exp13/eval/yoloe_s/lagenda
```

**Apply the Q1 gate** (doc §bars): LAGENDA detection ≥ 95% AND adult gender ≥ 95%. Record
pass/fail before looking at anything else.

## Phase B — distractor arm (same 4 datasets)

Same loop with `--distractors "statue,mannequin,doll"`, arm `yoloe_s_distract`, then the same
scoring. The two numbers that decide (doc bars B1/B2):

- object-set image-FP rate: Phase B vs Phase A (need ≥ 30% relative drop);
- LAGENDA + crowd recall: Phase B vs Phase A (each may lose ≤ 1 pt).

For adjudicating *what* the distractors absorbed: every excluded detection is in
`/workspace/exp13/yoloe_s_distract/<ds>/raw/*.json` with `"excluded": "distractor"` — sample
40, eyeball crops, tally statue/mannequin/doll vs real-person-stolen. A real person absorbed by
`mannequin` is the failure mode; count it explicitly.

## Phase C — fine-tune (ONLY if the Q1 gate passed; A100; $100 cap)

Same data tree + yaml as EXP-2026-12 Phase B (build once, share). Two arms:

```bash
HOST=$(hostname)
# full fine-tune
nohup python3 -u -c "
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
YOLOE('yoloe-26s-seg.pt').train(data='/workspace/exp12/spotlight_oiv7.yaml',
    trainer=YOLOEPESegTrainer, epochs=30, imgsz=640, batch=-1, device=0,
    project='/workspace/exp13/train', name='yoloe_s_full')
" > /workspace/exp13/log_train_full_$HOST.log 2>&1 &
```

⚠ the linear-probe arm follows the recipe in the YOLOE docs (freeze backbone+neck+cv3
internals) — copy the exact freeze list from docs.ultralytics.com/models/yoloe at run time into
the experiment doc appendix, don't reconstruct it from memory. Score both arms with the Phase A
scoring commands (`--map identity`, since a fine-tuned model speaks 0/1/2 natively).

## Phase D — export sanity (CPU)

```bash
python3 - <<'EOF'
from ultralytics import YOLOE
m = YOLOE('yoloe-26s-seg.pt')
try:
    m.set_classes(["woman", "man", "child"])
except TypeError:
    m.set_classes(["woman", "man", "child"], m.get_text_pe(["woman", "man", "child"]))
m.export(format='onnx')     # vocabulary baked in; fixed-class detector from here on
EOF
```

Checks (doc §Phase D): (a) run the ONNX over the 259 object-set images + 20 LAGENDA images and
assert every emitted class id ∈ {0,1,2} (the out-of-vocabulary export bug class — ultralytics
issue #23250); (b) native-vs-exported parity on 20 held-out images (≥ 95% of boxes at
IoU ≥ 0.9, same class, |Δconf| ≤ 0.05). If the distractor arm was adopted, export WITH the
6-word vocabulary instead and drop classes 3–5 client-side — record which export shipped.

## Phase E — flicker (ONLY if Q1 gate passed)

```bash
HOST=$(hostname)
nohup python3 -u video_flicker_probe.py --engine ultralytics \
    --model-path yoloe-26s-seg.pt --prompts "woman,man,child" \
    --videos /workspace/innovation-lab/exp11_flicker/videos \
    --out /workspace/exp13/flicker_probe \
    > /workspace/exp13/log_flicker_$HOST.log 2>&1 &
```

Then `flicker_metrics.py` at conf 0.45, per clip, into the EXP-2026-11 comparison table. The
`dubai_souk` Woman↔Man flip rate is the number to call out (prompted vs trained gender on Gulf
dress — a peek, not a measurement; say so wherever it's quoted).

## Wrap-up

Fill the experiment doc per phase; scoreboard rows in `docs/COMPONENT_FRAMEWORK.md`; charts
via a new `make_charts_exp13.py` after numbers exist; the distractor-vocabulary adoption
decision + the license note go in "What we can decide".
