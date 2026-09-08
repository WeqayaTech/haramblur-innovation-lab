#!/usr/bin/env bash
# Spotlight pipeline — fresh-pod setup. Run FIRST on every new pod.
#
#   bash /workspace/autolabel_pipeline_v2/pod_setup.sh
#
# Fixes the things that bite every new pod, in the order they must happen:
#   1. The pipeline's python deps are missing (requirements_clean.txt).
#   2. RunPod images ship torch 2.4.1 with transformers 5.x -> SAM3 import
#      dies on `cannot import name 'DTensor'`. Newer torch fixes it. This runs
#      AFTER the deps install, so an old torch pinned by that file is corrected.
# Also points HF_HOME at the volume so SAM3 weights are not re-downloaded,
# and verifies the pipeline actually imports before you start a 20-hour job.
set -u
# BASH_SOURCE, not $0: when this file is SOURCED (the recommended way, so the
# thread-pinning exports reach your shell) $0 is the shell name and dirname
# would drop us in the wrong directory, making every selftest below fail.
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

echo "=== 1/6  GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "!! no GPU visible — this pod cannot run SAM3"; }
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" in
  *5090*|*"PRO 6000"*|*"PRO 4500"*|*"PRO 4000"*|*B200*|*B300*)
    echo "!! WARNING: this looks like a Blackwell card (sm_120)."
    echo "!! The cu124 wheels below do NOT support it — expect kernel errors."
    echo "!! Use an Ada/Ampere pod (L4, 4090, A4500, RTX 4000 Ada) instead." ;;
esac

echo "=== 2/6  python deps ==="
# Must run BEFORE the torch step: this file can pin an old torch, and the
# step below then corrects it. Running it after would re-break SAM3's import.
REQS=/workspace/data_inspection_tools/vlm-cluster/requirements_clean.txt
if [ -f "$REQS" ]; then
    echo "installing $REQS"
    pip install -q -r "$REQS" || { echo "!! pip install failed — stop here"; exit 1; }
else
    echo "!! MISSING $REQS"
    echo "!! Every pod needs it before any pipeline code runs. Stop here."
    exit 1
fi

echo "=== 3/6  torch (the DTensor fix) ==="
TORCH_V=$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
case "$TORCH_V" in
  2.4*|2.3*|none)
    echo "torch=$TORCH_V -> upgrading"
    pip install -q -U torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 ;;
  *) echo "torch=$TORCH_V -> ok" ;;
esac

echo "=== 4/6  workspace dirs ==="
mkdir -p /workspace/spotlight/{raw,run,logs}
echo "logs -> /workspace/spotlight/logs (tail these from any pod)"

echo "=== 5/6  environment ==="
export HF_HOME=/workspace/.cache/huggingface
grep -q 'HF_HOME' ~/.bashrc 2>/dev/null || \
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo "HF_HOME=$HF_HOME  (SAM3 weights cached on the volume)"
# Thread pinning. `nproc` reports the HOST's core count inside RunPod
# containers (seen: 255 on an A100 SXM whose container had 32 vCPU), so torch
# sizes its thread pool from that and every worker spawns hundreds of threads.
# Measured on 2026-07-28: 8 workers unpinned -> GPU 6%, 10-32 s/img; the same
# 8 pinned to 4 threads -> GPU 100%, 2.5 s/img. Same code, 10x throughput.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
grep -q 'OMP_NUM_THREADS' ~/.bashrc 2>/dev/null || \
    echo 'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4' >> ~/.bashrc
echo "threads pinned to 4/worker (nproc=$(nproc); on RunPod this is often the"
echo "  HOST's core count, not your container's vCPU - do not size anything by it)"
echo "  NOTE: this export does not reach your current shell if you ran"
echo "  'bash pod_setup.sh'. Prefer 'source pod_setup.sh', or re-export before launching."
df -h /workspace | tail -1

echo "=== 6/6  verify the pipeline imports ==="
python3 -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
from transformers import Sam3Processor, Sam3Model   # the import that fails on old torch
print('SAM3 import OK')" || { echo "!! SAM3 still broken — stop here"; exit 1; }
for s in spotlight_run spotlight_batch verify_labels estimate_cost; do
    python3 "$s.py" --selftest >/dev/null 2>&1 \
        && echo "  ok   $s" || echo "  FAIL $s (sync it from the dev copy)"
done
python3 autolabel_sam_raw.py --help 2>/dev/null | grep -q shard \
    && echo "  ok   autolabel_sam_raw (sharding present)" \
    || echo "  FAIL autolabel_sam_raw — missing --shards, re-sync it"

echo
echo "READY. Start a shard (pick an index no other pod is using):"
echo
echo "  export HF_HOME=/workspace/.cache/huggingface"
echo "  cd /workspace/autolabel_pipeline_v2"
echo "  SHARD=0"
echo "  nohup python3 autolabel_sam_raw.py \\"
echo "      --input /workspace/open-images-v7/images/train \\"
echo "      --output /workspace/spotlight/raw/oiv7_train \\"
echo "      --classes \"woman\" \"man\" \"child\" --batch_size 1 \\"
echo "      --shards 32 --shard-index \$SHARD \\"
echo "      > /workspace/spotlight/logs/label_\$SHARD.log 2>&1 &"
echo
echo "  tail -f /workspace/spotlight/logs/label_\$SHARD.log"
