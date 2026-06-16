#!/usr/bin/env bash
set -euo pipefail

if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "Warning: this helper is written for Ubuntu 22.04; detected ${PRETTY_NAME:-unknown}." >&2
  fi
fi

if [[ "${EUID}" -eq 0 ]]; then
  sudo_cmd=()
else
  sudo_cmd=(sudo)
fi

echo "[deps] installing base build tools"
"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y \
  bash \
  build-essential \
  ca-certificates \
  coreutils \
  curl \
  git \
  python3 \
  python3-pip \
  python3-venv

echo "[deps] checking NVIDIA driver and CUDA toolkit"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install a working NVIDIA driver first." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: nvcc not found.

Install NVIDIA CUDA Toolkit for Ubuntu 22.04, then make sure nvcc is in PATH.
For example, after installing CUDA under /usr/local/cuda:

  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

EOF
  exit 1
fi

echo "[deps] versions"
python3 --version
gcc --version | head -n 1
nvcc --version | tail -n 1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || nvidia-smi

echo "[deps] OK"
