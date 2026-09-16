# Fresh-pod bring-up (RTX PRO Blackwell / sm_120)

RunPod pods are ephemeral — every new pod resets the environment. The default
PyTorch on these images is built for CUDA ≤12.4 (max compute capability sm_90),
but the **RTX PRO 4000 Blackwell is sm_120**, so you get:

    RuntimeError: CUDA error: no kernel image is available for execution on the device

Run these steps once per new pod.

## 1. Swap to a Blackwell-ready PyTorch (CUDA 12.8)

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`torchaudio` is removed on purpose — the stale cu124 build throws
`libtorchaudio.so: undefined symbol` against the new torch, and we don't use audio.

Verify (must print capability `(12, 0)` and `True`):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
nvidia-smi   # confirm 2x Blackwell, driver >= 570
```

## 2. Install remaining deps

```bash
pip install opencv-python-headless Pillow PyYAML
export HF_HOME=/workspace/hf-cache
```

For the Spotlight pipeline specifically, use `pod_setup.sh` after this.

## One-liner version

```bash
pip uninstall -y torch torchvision torchaudio && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
pip install opencv-python-headless Pillow PyYAML && \
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

## Notes

- Long runs die if the SSH session closes. Use `nohup ... &` (not tmux — not
  installed on fresh pods). See CLAUDE.md → Infrastructure.
- To make this permanent, bake the torch swap into a custom RunPod template image.
