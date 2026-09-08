# TF.js export recipe for YOLO26-family checkpoints (validated 2026-08-11)

Turns any ultralytics `.pt` into the extension's TF.js graph-model folder format
(`model.json` + shards + `metadata.yaml` — same structure as the shipped `v11nclean2`).
First validated on `y26n_gradsupp`; works for any YOLO26/YOLO11 checkpoint.

**Why the pinned version:** TF.js export was deprecated in ultralytics 8.4.83+ (emits LiteRT,
which TF.js cannot load). `8.4.82` is the last version that knows YOLO26 AND still has the
real TF.js exporter — the same pipeline that produced `v11nclean2` (which used 8.3.226).

## The recipe (fresh pod, CPU is fine, ~10 min total)

```bash
# 0. If the checkpoint came from a custom trainer (gradsupp arms), make it stock-loadable first:
python3 /workspace/data_inspection_tools/vlm-cluster/train_gradsuppress.py \
    --export-plain <run>/weights/best.pt          # -> best_plain.pt

# 1. Isolated venv with the pinned exporter
python3 -m venv /root/tfjsenv
source /root/tfjsenv/bin/activate
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q "ultralytics==8.4.82" onnx

# 2. First export attempt — lets ultralytics auto-install its pinned TF deps
#    (tensorflow 2.19, onnx2tf, tensorflowjs...). EXPECTED to fail at the last
#    step with the tensorflow_decision_forests/protobuf error.
yolo export model=<weights.pt> format=tfjs imgsz=640

# 3. The one required workaround: tensorflowjs unconditionally imports
#    tensorflow_decision_forests (only USED for forest models — irrelevant here)
#    whose bundled ydf ships protobuf-6 gencode while TF pins protobuf 5.
#    Remove it and make the import optional:
pip uninstall -y -q tensorflow_decision_forests ydf
python3 - <<'EOF'
from pathlib import Path
f = Path('/root/tfjsenv/lib/python3.11/site-packages/tensorflowjs/converters/tf_saved_model_conversion_v2.py')
s = f.read_text()
s = s.replace('import tensorflow_decision_forests',
              'try:\n  import tensorflow_decision_forests\nexcept Exception:\n  tensorflow_decision_forests = None')
f.write_text(s)
print('patched')
EOF

# 4. Re-run — completes in ~30 s
yolo export model=<weights.pt> format=tfjs imgsz=640
deactivate
```

Output: `<weights>_web_model/` next to the checkpoint.

## Post-export checklist

1. `metadata.yaml` — if the checkpoint carried a cosmetic 4th class name (gradsupp plain
   exports do), delete the `3: Unknown` line; the head is 3-channel and only emits ids 0-2.
2. **The graph has TWO outputs** — `Identity:0` (the real one: `[1, 300, 6]` =
   `[x1, y1, x2, y2, conf, class_id]`, final detections, 640-px coords) and a `TopKV2:0`
   byproduct to IGNORE.
3. **Integration contract differs from v11nclean2**: input is the same
   (`[1, 640, 640, 3]` NHWC, 0-1 floats) but the output is post-NMS — the JS skips the
   decode+NMS path entirely and just filters `conf >= 0.45`:

```javascript
const out = await model.executeAsync(input);      // pick the Identity:0 tensor
const data = await out.data();                    // [1, 300, 6] flattened
for (let i = 0; i < 300; i++) {
  const [x1, y1, x2, y2, conf, cls] = data.slice(i * 6, i * 6 + 6);
  if (conf >= CONF_THRESHOLD) boxes.push({ x1, y1, x2, y2, conf, cls });
}
```

4. **If the OLD (1,7,8400) + JS-NMS contract is required instead:** use the documented flag
   `model.export(format=..., end2end=False)` (also `model.val(..., end2end=False)` to score
   that branch). Validated — produces `(1,7,8400)`. Setting `model.model[-1].end2end = False`
   by hand works too but the flag is first-class.

   **How YOLO26's two heads actually train** (verified in ultralytics 8.4.115 source,
   2026-08-11 — worth knowing before anyone proposes retraining "without the NMS head"):
   - `init_criterion` selects **`E2ELoss`** for YOLO26 (NOT `E2EDetectLoss`, which is
     YOLOv10's and sums the branches equally — an easy misread).
   - `E2ELoss` weights the branches on a schedule: one-to-many **0.8 → 0.1** (linear over
     epochs, `final_o2m = 0.1`), one-to-one **0.2 → 0.9**.
   - **But `Detect.forward` detaches the features feeding the one-to-one head**
     (`x_detach = [xi.detach() ...]  # detach keeps one2one out of the backbone`), so o2o
     gradients only ever touch `one2one_cv2/cv3`. The backbone, neck and dense head are
     trained **solely** by the o2m loss.
   - Net effect: the decay acts as an extra LR damping on the trunk late in training — it
     does NOT redirect the trunk toward the NMS-free head (a common claim; architecturally
     impossible here). A pure non-e2e run (weight 1.0 throughout) might refine the dense
     head slightly more; untested, and no public precedent exists for training YOLO26 non-e2e.
   - Vendor numbers, measured WITH this schedule: YOLO26n **40.9 mAP (o2m+NMS) vs 40.1
     (e2e)** — the NMS path is the more accurate one already (+0.6-0.8 AP across scales;
     docs: "if maximum accuracy is your top priority, fall back… using end2end=False").
   - **MEASURED on our data (y26n_gradsupp, Spotlight val 4,233 imgs, 2026-08-11) — the
     dense head is the BETTER branch, retraining is unnecessary:**

     | branch | mAP50 | mAP50-95 | P | R |
     |---|---|---|---|---|
     | e2e (NMS-free) | 0.827 | 0.711 | 0.811 | 0.740 |
     | **o2m + NMS** | **0.833** | **0.715** | 0.803 | **0.761** |

     +0.6 mAP50 / +0.4 mAP50-95, matching the vendor's published +0.6-0.8. Recall gains
     **+2.1 pts** for 0.8 of precision — the right trade for this product (small/occluded
     people). Per class mAP50: Woman 0.855→0.864, Man 0.875→0.881, Child 0.752→0.754.
   - **`val(end2end=False)` fails with `KeyError: 'feats'`** — `fuse()` strips the one2many
     head for inference, so the flag arrives too late. Set it on the model FIRST:
     ```python
     m = YOLO(ckpt); m.model.model[-1].end2end = False; m.model.end2end = False
     r = m.val(data=..., imgsz=640)
     ```
   - **Decision rule:** run that val on both branches before assuming anything. o2m ≥ e2e →
     export and ship, no retrain (the observed case). o2m materially worse → retrain non-e2e
     via a copied `yolo26.yaml` with `end2end: False` (parse_model forwards the key to the
     head; `init_criterion` then picks plain `v8DetectionLoss`) — fresh weights required.
   - Either branch shipped must carry its OWN benchmark row: the two heads disagree on
     borderline detections (observed: e2e Child @0.494 vs o2m Woman @0.722 on the same
     person).
5. **Parity check (run it — 2 min):** the original .pt vs the exported SavedModel (the
   direct precursor of the tfjs folder; the pb→tfjs step only repackages identical weights)
   on ~30 real images, identical pre-resized 640 inputs. Pass bar: class agreement ~100%,
   mean |Δconf| < 0.01, mean box IoU > 0.98. **First run (y26n_gradsupp, 2026-08-11) was
   perfect: 179/179 detections matched, 179/179 classes, mean |Δconf| 0.0000, mean IoU
   0.9998.** A conversion defect looks categorically different (systematic conf shift /
   missing detections). The comparison script lives in the session transcript; residual
   untested surface = the browser TF.js runtime itself, covered by the team's test deploy.
6. Before customer rollout (not needed for test deploys): the checkpoint's own benchmark
   row in the `docs/MODEL_COMPARISON.md` context.

## Three export variants, and which artifact each produces

| source checkpoint | export command | output | JS work |
|---|---|---|---|
| e2e checkpoint (normal YOLO26 training) | `format=tfjs` | **2 tensors**; use `Identity:0` = `(1,300,6)` final detections + a TopK byproduct to ignore | delete decode+NMS, filter conf, read rows |
| e2e checkpoint | `format=tfjs end2end=False` | `(1,7,8400)` raw | **none — existing NMS path unchanged** |
| **natively non-e2e checkpoint** (trained from a `end2end: False` yaml) | `format=tfjs` | **1 tensor**, `Identity:0` = `(1,7,8400)` raw; metadata already says `end2end: false`, `nms: false`, 3 classes — nothing to hand-edit | **none** |

The third row is the cleanest artifact (single output, correct metadata out of the box) but
requires having trained non-e2e — which EXP-2026-16 showed is not worth doing for accuracy.
For an existing e2e checkpoint, row 2 gives the same production contract for free.

Shipped so far: `y26ngs1` (row 1, y26n_gradsupp e2e), `y26nraw1` (row 3, y26n_noe2e).

## Known failure modes (all hit and solved 2026-08-11)

- `pip install tensorflowjs` outside a venv → resolver backtracks for 20+ min. Use the venv
  recipe; ultralytics' AutoUpdate installs a consistent set.
- `ModuleNotFoundError` chain (tf_keras → tensorflow_decision_forests → tensorflow_hub) when
  hand-assembling with `--no-deps` — avoided entirely by letting ultralytics install.
- `protobuf gencode 6.31 vs runtime 5.29` VersionError → the step-3 patch above.
- Custom-trainer checkpoints (`GSDetectionModel` unpickling error) → `--export-plain` first.
