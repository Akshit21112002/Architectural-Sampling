#!/usr/bin/env bash
# =============================================================================
# Environment setup for the RVD best-of-n VLM evaluation scripts.
#
# DEFAULT environment = Qwen2.5-VL and Qwen3-VL  (transformers 4.57.0).
# For Qwen3.5-VL: keep this same env and only upgrade transformers to 5.16.1
#                 (see the bottom of this file / the README).
#
# Tested with: torch 2.5.1+cu121  (CUDA 12.1 wheels)
# Usage:  bash setup_env.sh
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-rvd}"

# ---- 1. create + activate a fresh env (conda shown; venv works too) ----------
# conda create -y -n "${ENV_NAME}" python=3.10
# conda activate "${ENV_NAME}"
#
# or with venv:
# python3.10 -m venv "${ENV_NAME}" && source "${ENV_NAME}/bin/activate"

# ---- 2. PyTorch 2.5.1 + CUDA 12.1 -------------------------------------------
pip install --upgrade pip
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# ---- 3. default transformers stack (Qwen2.5-VL / Qwen3-VL) ------------------
pip install "transformers==4.57.0"
pip install "accelerate>=0.34" datasets pillow matplotlib
pip install peft

# ---- 4. sanity check --------------------------------------------------------
python - <<'PY'
import torch, transformers
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY

echo
echo "Default env ready for Qwen2.5-VL and Qwen3-VL (transformers 4.57.0)."
echo "For Qwen3.5-VL, run:  pip install -U 'transformers==5.16.1'"
