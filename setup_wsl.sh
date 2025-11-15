#!/usr/bin/env bash
# Setup and run dLNk GPT training in WSL2 (Ubuntu)
# - Installs Python env and ML dependencies
# - Starts training using local JSONL and 8-bit if available
set -euo pipefail

# Config
WIN_PROJECT_PATH='C:\\Users\\atSine\\Downloads\\dlnkgpt'
WSL_PROJECT_PATH='/mnt/c/Users/atSine/Downloads/dlnkgpt'
DATA_FILE_WSL="${WSL_PROJECT_PATH}/training_data_1.1m_final.jsonl"
USE_8BIT_DEFAULT="1"

log() { echo "[setup_wsl] $*"; }

# Ensure we're in WSL
if ! grep -qi microsoft /proc/version; then
  echo "This script must be run inside WSL (Ubuntu)." >&2
  exit 1
fi

# Move to project directory
cd "$WSL_PROJECT_PATH"
log "Project path: $(pwd)"

# Update apt and install base packages
log "Installing base packages..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip build-essential git

# Create and activate venv
if [ ! -d .venv ]; then
  log "Creating Python venv (.venv)"
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Install CUDA-enabled PyTorch for CUDA 12.1 (matches +cu121 in Windows log)
log "Installing PyTorch (CUDA 12.1 wheel)"
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install ML dependencies
log "Installing ML dependencies (transformers, datasets, peft, accelerate, bitsandbytes)"
pip install transformers datasets peft accelerate bitsandbytes

# Quick GPU check
log "nvidia-smi (from WSL)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "nvidia-smi not found; GPU should still be accessible via PyTorch if drivers are correct on Windows."
fi

python - <<'PY'
import torch
print('[setup_wsl] Torch CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[setup_wsl] GPU:', torch.cuda.get_device_name(0))
PY

# Prepare environment for training
export DATA_FILE="$DATA_FILE_WSL"
export USE_8BIT="${USE_8BIT_DEFAULT}"
# Optional: export HF_TOKEN if available in WSL env
# export HF_TOKEN="${HF_TOKEN:-}"

log "Starting training with DATA_FILE=${DATA_FILE} USE_8BIT=${USE_8BIT}"
python ./train_local.py | tee train_run_wsl.log

log "Training process finished (or interrupted). Review train_run_wsl.log for details."

