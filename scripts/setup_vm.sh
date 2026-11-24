#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$HOME/venvs/pdp}

sudo mkdir -p /data/rpc /data/runs
sudo chmod 1777 /data /data/rpc /data/runs

PACKAGES=(python3.10-venv python3.10-dev build-essential unzip tmux)
if ! dpkg -s "${PACKAGES[@]}" >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq "${PACKAGES[@]}"
fi
[[ -d $VENV ]] || python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -q -r "$REPO/requirements.txt"
pip install -q -e "$REPO"

export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
start=$(date +%s)
(cd "$REPO/pdp" && python -c "import models.modeling_deformable_detr as m; print('kernel:', m.MultiScaleDeformableAttention is not None, m.KERNEL_LOAD_ERROR or '')")
echo "kernel build/load took $(( $(date +%s) - start )) s"
(cd "$REPO" && python -m pytest -q)
df -h /
