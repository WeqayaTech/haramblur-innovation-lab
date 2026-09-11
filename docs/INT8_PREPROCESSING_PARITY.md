# INT8 vs fp32 — is the process exact? Pre-processing parity, verified (2026-09-10)

**Question this answers** (reviewer, 2026-09-10): *"exactly the same pre-processing? the results on 640 px are unusually bad — a 7–10% hit is not small."*

**Short answer.** Inference pre-processing is byte-identical for the fp32 and INT8 TFLite rows —same runner, same letterbox, same `/255`, same float32 input tensor, same NMS, same scorer — and
the fp32 TFLite row reproduces the `.pt` checkpoint within ±0.005, so the TFLite path itself is faithful. The one place the pipeline is *not* identical is calibration-time letterboxing
(`scaleup=False` vs the predictor's `scaleup=True`), and it is minor. The large 640 loss has a different, now-measured cause: the exporter quantizes the **final output tensor** with one per-tensor int8 scale (0.0042) shared by normalized box coordinates and class scores, so **every box edge snaps to a 2.7 px grid at 640** (1.7 px at 416, 1.3 px at 320) and scores collapse to a handful of levels. Details, verbatim code, and a dig-deeper plan below.

Versions (the ones that produced every export and ran every TFLite row): ultralytics **8.4.146**,
ai-edge-litert **2.2.0**, litert_torch 0.9.4, ai-edge-quantizer 0.9.0, Python 3.11, `device=cpu`.
All snippets are copied from the installed package, not paraphrased.

---

## 1. Stage-by-stage comparison

| stage              | fp32 TFLite row                                             | INT8 TFLite row                                                       | calibration (INT8 export only)                       | identical?                                                           |
| ------------------ | ----------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| image load         | `run_ultralytics_labels.py` → ultralytics `predict()`  | same                                                                  | ultralytics val dataloader                           | same decoder (cv2, BGR)                                              |
| resize / pad       | `LetterBox(640, auto=False, scaleup=True, center=True)`   | same                                                                  | `LetterBox((640,640), scaleup=False, center=True)` | **inference: same. calibration: no upscaling of small images** |
| color / range      | BGR→RGB,`/255` → float32 `[1,3,640,640]`              | same                                                                  | `batch["img"].float() / 255.0`, NCHW               | same                                                                 |
| model input tensor | float32, no quantization                                    | float32, no quantization (verified)                                   | —                                                   | same                                                                 |
| graph              | fp32                                                        | int8 weights (per-channel) + int8 activations, float32 output dequant | —                                                   | *this is the only intended difference*                             |
| output tensor      | float32`[1,7,8400]`, normalized xywh + 3 scores           | float32`[1,7,8400]` (dequantized from int8, scale 0.0042)           | —                                                   | shape same;**INT8 values are on a 0.0042 grid**                |
| denormalize        | `x[:, [0,2]] *= 640; x[:, [1,3]] *= 640`                  | same                                                                  | —                                                   | same                                                                 |
| NMS                | ultralytics, per-class, IoU 0.7, conf floor 0.001           | same                                                                  | —                                                   | same                                                                 |
| scoring            | `map_eval.py`, COCO 101-pt, IoU .50:.95, 702 ignore boxes | same                                                                  | —                                                   | same                                                                 |

---

## 2. Inference pre-processing — the code that ran

**Runner** (`vlm-cluster/run_ultralytics_labels.py`, `UltralyticsModel`). One code path for every
Ultralytics-loadable file; the only difference between rows is the path passed to `YOLO()`:

```python
self.model = YOLO(model_path)                       # .pt, _fp32.tflite, _int8.tflite alike

def detect(self, img):
    res = self.model.predict(img, imgsz=self.imgsz, conf=self.floor,   # imgsz=640, floor=0.001
                             iou=self.iou, device=self.device,          # iou=0.7, device=cpu
                             verbose=False)[0]
```

**Predictor pre-processing** (`ultralytics/engine/predictor.py`, lines 161–218):

```python
def preprocess(self, im):
    if not isinstance(im, torch.Tensor):
        im = self.pre_transform(im)
        im = torch.from_numpy(im[0]).unsqueeze(0) if len(im) == 1 else torch.from_numpy(np.stack(im))
        im = im.to(self.device)                       # transfer as uint8, then reorder on device
        im = im.permute(0, 3, 1, 2)                   # BHWC to BCHW, (n, 3, h, w)
        if im.shape[1] == 3:
            im = im.flip(1)                           # BGR to RGB
        im = im.contiguous()
        im = (im.half() if self.model.fp16 else im.float()).div_(255)  # uint8 -> 0.0-1.0

def pre_transform(self, im):
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(
        self.imgsz,
        auto=same_shapes and self.args.rect
             and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
        stride=self.model.stride,
    )
    return [letterbox(image=x) for x in im]
```

For a `.tflite` file `self.model.format != "pt"` and it is not dynamic, so `auto=False`: a square
640×640 letterbox, centered, `scaleup=True` (the `LetterBox` defaults —
`ultralytics/data/augment.py` line 1658: `new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True`). Identical for the fp32 and INT8 files.

**Backend I/O** (`ultralytics/nn/backends/litert.py`, `LiteRTBackend.forward`, lines 47–96):

```python
im = im.cpu().numpy()                                  # BCHW, float [0,1]
if self.nhwc:                                          # False for litert-torch exports (NCHW graph)
    im = im.transpose(0, 2, 3, 1)
h, w = im.shape[1:3] if self.nhwc else im.shape[2:4]   # 640, 640
details = self.input_details[0]
if details["dtype"] in {np.int8, np.int16}:            # NOT taken: our input tensor is float32
    scale, zero_point = details["quantization"]
    im = (im / scale + zero_point).astype(details["dtype"])
self.interpreter.set_tensor(details["index"], im)
self.interpreter.invoke()
...
    if output["dtype"] in {np.int8, np.int16}:         # NOT taken: our output tensor is float32
        x = (x.astype(np.float32) - zero_point) * scale
    if x.ndim == 3 and not self.end2end:               # taken: [1, 7, 8400] raw head
        x[:, [0, 2]] *= w                              # normalized xywh -> pixels
        x[:, [1, 3]] *= h
```

The docstring of that method states the design choice that matters here: *"Box and pose keypoint
coordinates are exported normalized to [0, 1] (so INT8 quantization preserves class score
resolution) and denormalized here by the input image size."*

---

## 3. Calibration pre-processing — the code that ran at export

`ultralytics/engine/exporter.py`, `get_int8_calibration_dataloader`:

```python
LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
cfg = deepcopy(self.args)
cfg.imgsz = max(self.imgsz)                            # calibration letterbox at the export size (640/416/320)
split = self.args.split or "val"                       # our exports: split=train (fixed 500-image list)
data = check_det_dataset(self.args.data, split=self.args.split)
dataset = build_yolo_dataset(cfg, data[split], self.args.batch, data, mode="val", fraction=cfg.fraction)
...
return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)
```

`ultralytics/data/dataset.py`, line 319 (the `mode="val"` transform):

```python
transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
```

`ultralytics/utils/export/litert.py`, lines 104–105 (what the quantizer actually sees):

```python
for batch in calibration_dataset:
    imgs = batch["img"].cpu().float() / 255.0          # uint8 NCHW -> float32 0-1, same range as inference
```

**The one difference:** calibration letterboxes with `scaleup=False` — images smaller than 640 px are padded, not upscaled — while inference upscales them. Same centering, same padding value, same
`/255`, same NCHW layout. Effect: for small images the activation statistics were gathered at a different scale than inference sees. Real, but second-order; it cannot produce a per-tensor output grid, which is what section 4 measures.

---

## 4. Measured facts (y26n_humanshaped_v2 @640, train-calibrated INT8 vs fp32 TFLite)

Probe (`ai_edge_litert.interpreter`, same letterbox as above built by hand):

```python
from ai_edge_litert.interpreter import Interpreter
import numpy as np
B = "/workspace/exports/y26n_humanshaped_v2_calib500/sz640/y26n_humanshaped_v2_"
its = {k: Interpreter(model_path=B + k + ".tflite") for k in ("fp32", "int8")}
for it in its.values(): it.allocate_tensors()
for k, it in its.items():
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    print(k, i["dtype"], i["shape"], i["quantization"], o["dtype"], o["shape"], o["quantization"])
td = its["int8"].get_tensor_details()
for t in td:
    if t["dtype"] == np.int8 and list(t["shape"]) == [1, 7, 8400]:
        q = t["quantization_parameters"]; print("OUTPUT int8", q["scales"], q["zero_points"], q["quantized_dimension"])
```

| fact                                                               | value                                                                                                                                   |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| input tensor, both files                                           | `float32 [1,3,640,640]`, quantization `(0.0, 0)`                                                                                    |
| output tensor, both files                                          | `float32 [1,7,8400]`, quantization `(0.0, 0)`                                                                                       |
| int8 graph: conv/fc weights                                        | **94 per-channel, 0 per-tensor** (kernel sizes 1×1 ×55, 3×3 ×39)                                                              |
| int8 graph: activations                                            | 466 int8 tensors (458 per-tensor, 8 per-channel); only 2 float32 tensors exist — input and`output_dequant`                           |
| **int8 output tensor** (`serving_default_output_0_output`) | **scale 0.0042, zero-point −128, per-tensor** → 256 levels over ≈ [0, 1.07], shared by cx, cy, w, h *and* the 3 class scores |
| coordinate granularity on a real image (00fd93edd4717826.jpg)      | fp32: 1,012 distinct coordinate values · INT8:**244, minimum step 0.0040 normalized = 2.6–2.7 px at 640**                       |
| score granularity, same image                                      | fp32: continuous · INT8:**6 distinct values**, step 0.0041; top anchor 0.953 (fp32) → 0.86 (INT8)                               |

Same anchor, side by side (normalized cx, cy, w, h | score):

```
fp32  [0.50, 0.57, 0.66, 0.87]  0.953
int8  [0.52, 0.56, 0.66, 0.86]  0.86
```

**Interpretation.** The whole network is W8A8 and the last tensor is quantized like any other
activation. Because coordinates are normalized so that scores keep resolution (the LiteRT
backend's own rationale), one 8-bit scale has to serve both: a 0.0042 step is 0.4% of a score
*and* 2.7 px of a box edge at 640. mAP50 tolerates 2.7 px; mAP50-95 does not. The step is fixed in
normalized units, so its pixel cost grows with input size — which is why 640 is worst and 320
nearly free — and models with tighter fp32 boxes (y26s) lose the most. The same output-tensor
quantization also explains the score collapse (confidence redistribution, class flips on
borderline people) and the 7× rise in same-class duplicate boxes (jittered twins fall below the
NMS 0.7 merge threshold).

What this does **not** rule out: additional loss from int8 activations *inside* the head (the DFL
box-distribution logits and the attention `BatchMatMul` buffers are int8 too). Section 5 separates
those.

---

## 5. How to dig deeper — ablations that isolate each suspect

Ordered by cost. Each is a one-flag or one-recipe re-export of the same 9 cells, scored by the
same `map_eval.py` protocol, so the numbers drop straight into the existing matrix.

| # | ablation                                                                       | what it isolates                                                                                                                                  | how                                                                                                                                                                |
| - | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | **`quantize=w8a32`** — int8 weights, fp32 activations, no calibration | is the loss all activations/output? (weights alone should be ≈ fp32)                                                                             | `yolo export … format=tflite quantize=w8a32 device=cpu` (supported by this exporter, no `data=` needed); same 2.9 MB / 10.2 MB files                          |
| 2 | **`quantize=w8a16`** — int8 weights, int16 activations                | is 8-bit activation resolution the problem? (int16 → output step 0.00002)                                                                        | `quantize=w8a16 data=… split=train`; expect ≈ fp32 accuracy, slower on CPU                                                                                     |
| 3 | **keep output + head float** (mixed precision)                           | is it the*output tensor* specifically?                                                                                                          | `ai_edge_quantizer` per-op recipe excluding the last convs / concat / dequant from int8 (regex on op names from `get_tensor_details()`); rest stays W8A8       |
| 4 | **calibrate with the predictor's letterbox** (`scaleup=True`)          | the one pre-processing difference                                                                                                                 | monkeypatch`LetterBox(..., scaleup=True)` in the calibration transform, re-export, compare                                                                       |
| 5 | **range selection**: percentile / MSE instead of min-max                 | outlier-widened activation ranges                                                                                                                 | quantizer calibration algorithm option; check the output scale drops below 0.0042                                                                                  |
| 6 | **box-edge error histogram** vs fp32 per size, on matched detections     | direct evidence of the grid: expect a comb at multiples of 0.004 in normalized coords at every size, i.e. a*pixel* error that scales with imgsz | replay the existing raw dumps (`/workspace/quant_matrix_calib500/<run>/{fp32tflite,}_sz<SZ>_spotval/raw`), pair boxes at IoU ≥ 0.7, histogram Δcx/Δcy/Δw/Δh |
| 7 | **NMS isolation**                                                        | how much of the mAP50-95 loss is duplicate-box precision rather than localization                                                                 | re-run NMS at IoU 0.5 / class-agnostic on the same raw INT8 outputs (no re-export), re-score                                                                       |
| 8 | **per-precision threshold re-sweep**                                     | scores are on a 0.0042 grid and shifted (0.95 → 0.86): 0.45 is not the same operating point for INT8                                             | `conf_sweep.py` on the INT8 dumps (already possible offline)                                                                                                     |

Ablations 1–3 will settle the reviewer's question. Prediction, stated before running: **1 and 2
recover to within ±0.5 pt of fp32 at every size; 3 recovers most of the 640 gap on its own.** If 1
does *not* recover, the loss is in the weights after all and everything above is wrong — that is
the falsifying outcome to look for.

---

## 6. Verify it yourself

```bash
# the runner (same call for .pt / _fp32.tflite / _int8.tflite)
python3 /workspace/data_inspection_tools/vlm-cluster/run_ultralytics_labels.py --engine ultralytics \
  --model <file> --imgsz 640 --device cpu --images /workspace/exp12/images_val4232 --out <out> --floor 0.001
# the scorer
python3 /workspace/data_inspection_tools/vlm-cluster/map_eval.py --raw <out>/raw \
  --gt-labels /workspace/spotlight/run/oiv7_val/labels --ignore-labels /workspace/spotlight/run/oiv7_val/ignore_unk \
  --expect-floor 0.001 --out <out>_map.json
# the source lines quoted above (installed package on the pod)
E=/root/venvs/export_tflite/lib/python3.11/site-packages/ultralytics
sed -n 161,218p $E/engine/predictor.py
sed -n 319p     $E/data/dataset.py
sed -n 104,105p $E/utils/export/litert.py
sed -n 47,96p   $E/nn/backends/litert.py
grep -n "def get_int8_calibration_dataloader" -A 40 $E/engine/exporter.py
```

Related: `docs/MODEL_COMPARISON.md` ("INT8 quantization accuracy matrix — train-calibrated" and
the FP16 subsection), `docs/DEPLOYMENT_MODELS_AND_PATHS.md`, memory `int8-calibration-recipe-flaw`.
