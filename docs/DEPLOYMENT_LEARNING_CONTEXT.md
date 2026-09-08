# Context: Model deployment for HaramBlur — learning session

> Paste this whole file into a chat session. Goal: **I want to learn and understand the
> deployment option space deeply, not make a decision yet.** Explain concepts, trade-offs,
> and how things actually work under the hood. Assume I'm technical (I run ML training and
> evaluation pipelines) but new to browser/mobile ML deployment specifics.

## What the product is

HaramBlur is a browser extension (with future iOS/Android apps planned, shape undecided)
that helps Muslim users lower their gaze: it detects people on web pages/videos and blurs
adults of the opposite gender. It runs a small YOLO-family object detection model
(3 classes: Woman / Man / Child) **entirely on-device, in the browser** — privacy is a hard
requirement; images never leave the user's machine.

## How deployment works today (the shipped extension)

- **Manifest V3 Chrome/Firefox extension.** All ML inference runs in an **offscreen
  document** (not the content script, not the service worker). The content script scans the
  DOM for images/video frames and sends them over a long-lived message port; the offscreen
  document runs the model and returns boxes + classes; the content script applies blur via
  pure CSS (`filter: blur()` on elements, `backdrop-filter` rect overlays for region blur).
- **Runtime: TensorFlow.js 4.22**, backend priority webgpu → webgl → wasm → cpu. WASM is
  SIMD-only; **threads are deliberately disabled** because spawning blob workers violates
  the extension's CSP. The manifest CSP includes `wasm-unsafe-eval`.
- **Models shipped in the package**: three TFJS exports of the same YOLO11n checkpoint at
  input sizes 640/416/320 (~10.5 MB each). A startup **device benchmark** (30 timed
  inferences) picks the tier: ≤60 ms → 640, ≤100 ms → 416, else 320. WASM backend forces
  320; pure-CPU disables object detection entirely.
- Models are XOR/AES-encrypted at build time and cached in **IndexedDB** after first load.
- **NMS runs in JavaScript** (the exported graph has `nms: false`); the detector code
  hardcodes the raw output layout `[1, 4+C, N]` and class indices 0=Woman, 1=Man, 2=Child.
- A separate NSFW classifier is **fetched remotely at runtime** (Cloudflare Pages URL) and
  cached in IndexedDB — but with no version key, so it's never re-fetched after caching.
- The extension has a hidden dev **compare mode**: run two models side-by-side on the same
  frames, collect disagreements, and log per-model latency. Also a dormant "second-stage
  classifier" code path (crop each detected person, run a small 224px classifier) that is
  fully wired but currently has no model assigned.
- Temporal behavior on video: frames sampled at 15–25 fps depending on tier, a
  1-positive/2-negative frame hysteresis for blur on/off, and a frame-diff cache that skips
  inference when the frame barely changed.

## The new model and why deployment is now a question

We retrained on a new, much cleaner label set and the winning candidate is a fine-tuned
**YOLO26n** ("y26n_gradsupp": 2.38M params, 9.8 MB ONNX, measured 33.8 ms on a server CPU
at 4 threads, 640×640). Two properties matter for deployment:

1. **It's end-to-end / NMS-free** — NMS is baked into the exported graph, so the JS
   postprocessing shrinks. But the output tensor shape differs from what the current
   detector code expects.
2. **Ultralytics (the training framework) deprecated TF.js export** in its current
   versions. It exports ONNX, CoreML, and TFLite/LiteRT natively — but not TF.js. So
   shipping this model to the browser means either **migrating the runtime**
   (onnxruntime-web or LiteRT.js) or a multi-hop conversion chain back to TF.js.

A cautionary tale we already lived: a teammate's ONNX export had a confidence-value bug that
silently corrupted outputs. So "every export artifact needs a parity check against the
original checkpoint" is a lesson we've paid for.

## The option space I want to learn about

### A. Browser ML runtimes (main topic)
- **onnxruntime-web**: how the WASM backend works (SIMD, threads, cross-origin isolation
  requirements), what the WebGPU and WebNN execution providers are and their maturity,
  bundle-size implications, how it compares operationally to TF.js.
- **LiteRT.js** (Google's new web runtime, TFLite lineage): what it is, maturity, WebGPU
  story, whether "same artifact as Android" is a real advantage.
- **TF.js today**: is it really in maintenance mode? What breaks when converting a modern
  ONNX/PyTorch graph (e.g. with baked-in NMS: TopK, NonMaxSuppression ops) back to TFJS?
- How much slower should I expect in-browser WASM inference vs native ONNX Runtime on the
  same hardware? What actually dominates (memory copies, no threads, kernel quality)?
- Constraints specific to **MV3 extensions**: offscreen documents, CSP (`wasm-unsafe-eval`),
  why threaded WASM is hard there, whether WebGPU is usable from an offscreen document.

### B. Quantization & model packaging
- fp16 vs dynamic int8 vs static int8 for a small YOLO in WASM/WebGPU: real speed/size
  gains, and what accuracy risks look like for detection models (we care most about false
  positives on statues/dolls and borderline gender flips — quantization shifts borderline
  cases).
- Multi-resolution tiers (640/416/320) vs a single model with dynamic input size.

### C. Model delivery infrastructure
- Remote model hosting with a **version manifest** (version, URL, sha256, min app version),
  client-side caching (IndexedDB / app storage), staged percentage rollouts, kill switches.
  How do real products (e.g. Chrome's component updater, app ML model delivery services)
  structure this? What are the pitfalls (cache invalidation, hash verification, offline)?
- Trade-off: bundled models (store review covers them, works offline instantly) vs remote
  (update without review cycles — very valuable on iOS).

### D. Mobile deployment (apps not built yet — I want to understand the shapes)
- **iOS**: Core ML export path from Ultralytics (including NMS-as-pipeline), the Apple
  Neural Engine, fp16/palettization. Crucially: what a **Safari Web Extension** can and
  cannot do (memory limits, no good in-extension ML → native messaging to a host app), vs
  a **custom WKWebView browser** (native inference on frames; WKURLSchemeHandler tricks
  like serving pre-blurred image bytes). Why system-wide content filtering is impossible
  on iOS.
- **Android**: LiteRT/TFLite with GPU/NNAPI delegates vs ONNX Runtime Mobile; int8
  calibration; custom WebView browser vs Firefox-extension route vs the ambitious option —
  a system-wide screen filter (accessibility service / MediaProjection): feasibility,
  battery, Play Store policy risk.
- The idea of one shared JS "front half" (DOM scanning, blur logic, temporal smoothing)
  across extension + webview apps, with per-platform native inference behind a message-port
  contract — is this how others structure cross-platform ML products?

### E. Verification & rollout discipline
- Golden-set parity gates for export artifacts (fixed image set, reference outputs from the
  PyTorch checkpoint, tolerance-based comparison) — standard practice? tooling?
- Using the extension's built-in compare mode as an A/B soak before rollout.
- Privacy-preserving telemetry for a privacy-sensitive user base: what's reasonable to
  collect (model version, backend chosen, median inference ms, blur-toggle rate as a
  flicker proxy) and how to aggregate it without identifying users.

### F. One adjacent fact worth knowing
Our measurements show most user-visible "flicker" (blur turning on/off on video) comes from
confidence dipping below the threshold on people who are still detected — a **temporal
policy** problem (dual thresholds with sustain, short coasting through gaps, sticky class
votes), not a model problem. That policy is pure JS and platform-independent, so it's a
deployment-relevant lever that's independent of any runtime/model choice.

## How I'd like the chat session to go

Teach me these topics area by area (A → F is a sensible order, but follow my questions).
Prefer explaining *how things work and why* over giving me a recommendation. Where my
framing above contains a misconception, correct it. Concrete numbers, real API names, and
"here's what actually bites people in production" detail are all welcome. Licensing (AGPL)
is deliberately out of scope.
