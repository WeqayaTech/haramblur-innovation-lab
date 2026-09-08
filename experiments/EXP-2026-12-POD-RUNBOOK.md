# EXP-2026-12 — pod runbook (YOLO26 production candidate)

Copy-paste, phased. Phases A/C/D/E run on an **L4**; Phase B (training) needs an **A100**.
Read the experiment doc first — bars are pre-registered there; nothing here may change them.

House rules that bit us before, applied throughout: **no tmux → `nohup` + a log on the volume,
hostname-stamped**; **pin OMP threads** (`nproc` reports HOST cores); **never put "negative" in
a directory name** (autolabel_sam.py skips such paths — we keep the convention everywhere);
after any interruption, remember label-file semantics: *empty file = processed-found-nothing,
missing file = not processed*.

---

## Phase 0 — setup + verification (every fresh pod)

```bash
cd /workspace/data_inspection_tools/vlm-cluster
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8       # RunPod nproc lies (host cores)
pip install "ultralytics==8.4.115"               # pinned in the experiment doc
```

Sync the two changed tools from local before anything (pod copy is separate from local):
`run_ultralytics_labels.py` (new) and `video_flicker_probe.py` (gained `--engine ultralytics`).
Then:

```bash
python3 run_ultralytics_labels.py --selftest      # expect: selftest OK
python3 video_flicker_probe.py --selftest
```

Verify every input exists before spending GPU time (pod-only claims must be verified, not
assumed):

```bash
ls /workspace/datasets/crowdhuman/Images | wc -l          # images present?
ls /workspace/datasets/crowdhuman/annotation_val.odgt
ls /workspace/datasets/pass_3k | wc -l                    # ~3000
ls /workspace/datasets/object_set | wc -l                 # ~259
ls /workspace/exp03/sam_labels/crowd/*.txt | wc -l        # the frozen 500-img sample
ls /workspace/lagenda_eval/lagenda_yolo/gt.jsonl
ls -d /workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4
ls -d /workspace/innovation-lab/exp11_flicker/videos      # 4 clips (Phase D)
ls -d /workspace/spotlight/run/oiv7_train/labels          # Phase B labels
```

Materialize the **same crowd sample EXP-2026-03 scored** (image list = the frozen SAM label
stems — this is what makes every crowd number comparable):

```bash
mkdir -p /workspace/exp12/crowd_sample_imgs
for f in /workspace/exp03/sam_labels/crowd/*.txt; do
  s=$(basename "$f" .txt)
  ln -s /workspace/datasets/crowdhuman/Images/"$s".jpg /workspace/exp12/crowd_sample_imgs/
done
ls /workspace/exp12/crowd_sample_imgs | wc -l   # RECORD this count (expect ~500)
find /workspace/exp12/crowd_sample_imgs -xtype l | head   # broken links = missing images — investigate before proceeding
```

## Phase A — pretrained YOLO26n + the production YOLO-MIT anchor (Component 1, ~1 h L4)

Six Stage A runs (2 engines × 3 datasets). `HOST=$(hostname)` stamps every log.

```bash
HOST=$(hostname); cd /workspace/data_inspection_tools/vlm-cluster
for DS in crowd:/workspace/exp12/crowd_sample_imgs \
          pass:/workspace/datasets/pass_3k \
          objects:/workspace/datasets/object_set; do
  name=${DS%%:*}; dir=${DS#*:}
  nohup python3 -u run_ultralytics_labels.py --model yolo26n.pt --map coco-person \
      --images "$dir" --out /workspace/exp12/y26n_coco/$name --conf 0.45 \
      > /workspace/exp12/log_y26n_${name}_$HOST.log 2>&1 &
  wait
  nohup python3 -u run_ultralytics_labels.py --engine mit \
      --images "$dir" --out /workspace/exp12/mit_prod/$name --conf 0.45 \
      > /workspace/exp12/log_mit_${name}_$HOST.log 2>&1 &
  wait
done
```

Score (CPU, same scorer as EXP-2026-03/04 — unchanged):

```bash
for ARM in y26n_coco mit_prod; do
  python3 eval_negatives_crowd.py --mode negatives \
      --images /workspace/datasets/object_set \
      --pred-labels /workspace/exp12/$ARM/objects/labels \
      --out /workspace/exp12/eval/$ARM/objects --overlays 40
  python3 eval_negatives_crowd.py --mode negatives \
      --images /workspace/datasets/pass_3k \
      --pred-labels /workspace/exp12/$ARM/pass/labels \
      --out /workspace/exp12/eval/$ARM/pass --overlays 40
  python3 eval_negatives_crowd.py --mode crowd \
      --images /workspace/exp12/crowd_sample_imgs \
      --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
      --pred-labels /workspace/exp12/$ARM/crowd/labels \
      --out /workspace/exp12/eval/$ARM/crowd --overlays 40
done
```

**STOP here. Apply bar A1** (doc §bars): YOLO26n crowd recall ≥ MIT's AND object-set
image-FP rate ≤ MIT's — else the experiment ends, cheaply. Also paste the PASS numbers twice:
full set, and with the ~5% flagged images (list under `/workspace/exp07_conf/`) excluded.

## Phase B — fine-tune on Spotlight labels (A100, budget cap $100 — pre-registered)

⚠ **VERIFY the OIV7 image root first** — `/workspace/spotlight/DATA_README.md` has the path
reference. Ultralytics derives label paths by replacing `/images/` with `/labels/`, so build a
symlink tree pairing the Spotlight labels with the images, then a `spotlight_oiv7.yaml`:

```yaml
path: /workspace/exp12/train_tree
train: images/train      # symlinks -> OIV7 train images
val: images/val          # symlinks -> OIV7 val images
names: {0: Woman, 1: Man, 2: Child}
```

(labels/train -> `/workspace/spotlight/run/oiv7_train/labels/`, labels/val -> the val emit.
Classes-0-2 variant, NOT `labels_unk3` — stock ultralytics can't mask class-3 regions; this is
the recorded comparability caveat vs the team retrain.)

**Phase B0 — frozen-backbone pilot (~1-2 h, run FIRST).** The head-swap itself is automatic:
ultralytics sees the 80→3 class-count mismatch, rebuilds the head's classification branch with
3 outputs, and transfers every other pretrained weight. The pilot freezes the backbone
(`freeze=N`; ⚠ read N — the backbone layer count — from the model yaml
`ultralytics/cfg/models/26/yolo26.yaml` at run time, don't guess) and trains on a 50k-image
subset for a few epochs. Its job is to prove the yaml, the symlink tree, and the 3-class head
end-to-end BEFORE the paid run — its metrics are plumbing signals, not results, and never go
in the scoreboard:

```bash
HOST=$(hostname)
nohup python3 -u -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='/workspace/exp12/spotlight_oiv7_50k.yaml',
    epochs=3, imgsz=640, batch=-1, device=0, workers=8, freeze=FILL_N,
    project='/workspace/exp12/train', name='y26n_pilot')
" > /workspace/exp12/log_pilot_y26n_$HOST.log 2>&1 &
```

Sanity-check the pilot checkpoint speaks 3 classes (`YOLO(best).names` →
`{0: Woman, 1: Man, 2: Child}`) and produces boxes on a few LAGENDA images. Then the real run:

```bash
HOST=$(hostname)
nohup python3 -u -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='/workspace/exp12/spotlight_oiv7.yaml',
    epochs=30, imgsz=640, batch=-1, device=0, workers=8,
    project='/workspace/exp12/train', name='y26n_spotlight')
" > /workspace/exp12/log_train_y26n_$HOST.log 2>&1 &
```

Freeze whatever hyperparameters actually launch into the experiment doc's appendix BEFORE
looking at results. Watch the first epoch's img/s; if the projected cost blows the $100 cap,
stop and report — don't quietly shrink the run. RunPod volume-quota trap applies (a full volume
looks like a hang — check the console, `dd` probe, run the `.txt`/`.json` orphan pair-check
after any interruption).

## Phase C — score the fine-tuned checkpoint (all four datasets)

Same six-command pattern as Phase A with
`--model /workspace/exp12/train/y26n_spotlight/weights/best.pt --map identity`, output arm
`y26n_ft`, PLUS the LAGENDA arm:

```bash
nohup python3 -u run_ultralytics_labels.py \
    --model /workspace/exp12/train/y26n_spotlight/weights/best.pt --map identity \
    --images /workspace/lagenda_eval/lagenda_yolo/images/val \
    --out /workspace/exp12/y26n_ft/lagenda --conf 0.45 \
    > /workspace/exp12/log_y26nft_lagenda_$(hostname).log 2>&1 &
```

⚠ verify the LAGENDA images dir name against `run_autolabel_on_manifest.py`'s header docs on
the pod before running. Then Components 2+3 (CPU):

```bash
python3 run_autolabel_on_manifest.py \
    --gt-manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl \
    --gt-labels   /workspace/lagenda_eval/lagenda_yolo/labels/val \
    --images      /workspace/lagenda_eval/lagenda_yolo/images/val \
    --pred-labels /workspace/exp12/y26n_ft/lagenda/labels \
    --out         /workspace/exp12/eval/y26n_ft/lagenda
```

## Phase D — flicker (the EXP-2026-11 clips, identical replay)

```bash
HOST=$(hostname)
nohup python3 -u video_flicker_probe.py --engine ultralytics \
    --model-path /workspace/exp12/train/y26n_spotlight/weights/best.pt \
    --videos /workspace/innovation-lab/exp11_flicker/videos \
    --out /workspace/exp12/flicker_probe \
    > /workspace/exp12/log_flicker_$HOST.log 2>&1 &
```

Then `flicker_metrics.py` exactly as the EXP-2026-11 runbook ran it (conf 0.45), one report per
clip, dropped row-for-row into that doc's table.

## Phase E — export + CPU latency (CPU box is fine)

```bash
python3 - <<'EOF'
from ultralytics import YOLO
m = YOLO('/workspace/exp12/train/y26n_spotlight/weights/best.pt')
m.export(format='onnx')          # end-to-end: (batch, max_det, 6), NMS baked in
EOF
```

Latency: onnxruntime CPU, same machine and imgsz for both models, ≥100 warm iterations after
20 warmup, report median ms — candidate ONNX vs the comparator's ONNX export (⚠ if no YOLO-MIT
ONNX export exists on the volume, produce one with the YOLO-MIT repo's own export path and
record exactly how). Parity check: 20 held-out images, exported-vs-native boxes.

## Wrap-up

- Fill the experiment doc per phase; update `docs/COMPONENT_FRAMEWORK.md` scoreboard rows.
- Charts: `experiments/make_charts_exp12.py` (write after numbers exist, same SVG→PNG pattern).
- The license-gate decision (AGPL / Enterprise) goes in "What we can decide" — owner call.
