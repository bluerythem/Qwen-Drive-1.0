#!/usr/bin/env bash
# Rebuild the whole environment on a fresh Linux x86_64 machine with an NVIDIA GPU.
#
#   git clone git@github.com:bluerythem/Qwen-Drive-1.0.git -b interactive-demo qwen-drive
#   cd qwen-drive
#   ./bootstrap.sh [--nuscenes /path/to/v1.0-mini.tar] [--no-cuda-toolchain]
#   source env.sh && python app.py          # http://127.0.0.1:7860
#
# Idempotent: every step checks whether it has already been done. Nothing needs sudo.
# What it fetches and why is written up in SETUP_NOTES.md.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
NUSCENES_TAR=""
CUDA_TOOLCHAIN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --nuscenes) NUSCENES_TAR="$2"; shift 2 ;;
    --no-cuda-toolchain) CUDA_TOOLCHAIN=0; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

say "uv (Python package manager, no root needed)"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

say "Python 3.12 virtual environment"
[ -d .venv ] || uv venv --python 3.12 .venv       # downloads a managed CPython if the box lacks 3.12
PIP="uv pip install --python .venv/bin/python"

say "torch 2.8.0 + CUDA 12.8 wheels"
.venv/bin/python -c "import torch; assert torch.__version__.startswith('2.8.0')" 2>/dev/null \
  || $PIP torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

say "Python dependencies"
$PIP transformers==5.14.1 accelerate==1.12.0 safetensors==0.8.0 numpy==2.2.6 pillow==12.0.0 \
     opencv-python matplotlib==3.10.7 tqdm==4.67.1 pyarrow "huggingface_hub[cli]" \
     flash-linear-attention==0.5.1 ninja nvidia-cuda-cccl-cu12 gradio nuscenes-devkit python-docx

say "flash-attn and causal-conv1d (prebuilt wheels: torch 2.8, CUDA 12, cp312, cxx11abi TRUE)"
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
CC=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
.venv/bin/python -c "import flash_attn" 2>/dev/null || $PIP "$FA"
.venv/bin/python -c "import causal_conv1d" 2>/dev/null || $PIP "$CC"

say "model weights (13 GB from Hugging Face; skipped if present)"
if [ ! -f Qwen-Drive-1.0-4B/model.safetensors ]; then
  .venv/bin/hf download Qwen/Qwen-Drive-1.0-4B --local-dir Qwen-Drive-1.0-4B --exclude "assets/*" --exclude ".DS_Store"
fi

if [ "$CUDA_TOOLCHAIN" = 1 ]; then
  say "CUDA 12.8 nvcc for the perception kernels (micromamba, user-local)"
  # The perception mode JIT-builds two CUDA extensions and needs an nvcc matching torch's
  # CUDA 12.8. Systems rarely have one, so a private toolchain is assembled: nvcc from
  # conda-forge, the remaining headers (cusparse.h, cublas...) from torch's own pip wheels.
  if [ ! -x cuda128/bin/nvcc ]; then
    [ -x bin/micromamba ] || { mkdir -p bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba; }
    ./bin/micromamba create -y -q -p "$ROOT/cuda128" -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8
  fi
  if [ ! -d cuda-home/include ]; then
    rm -rf cuda-home && mkdir -p cuda-home/include
    ln -s "$ROOT/cuda128/bin" cuda-home/bin
    ln -s "$ROOT/cuda128/nvvm" cuda-home/nvvm
    ln -s "$ROOT/cuda128/targets/x86_64-linux/lib" cuda-home/lib64
    ln -s "$ROOT/cuda128/targets/x86_64-linux/lib" cuda-home/lib
    for d in "$ROOT"/cuda128/targets/x86_64-linux/include "$ROOT"/.venv/lib/python3.12/site-packages/nvidia/*/include; do
      [ -d "$d" ] || continue
      for f in "$d"/*; do ln -sfn "$f" "cuda-home/include/$(basename "$f")"; done
    done
  fi
fi

if [ -n "$NUSCENES_TAR" ]; then
  say "nuScenes mini + map expansion, and the derived scenes"
  # v1.0-mini.tar needs a (free) nuScenes account, so it is supplied by hand. The map
  # expansion is served from Motional's public bucket. Both are non-commercial licence.
  mkdir -p data/nuscenes
  [ -d data/nuscenes/v1.0-mini ] || tar xf "$NUSCENES_TAR" -C data/nuscenes v1.0-mini samples
  if [ ! -d data/nuscenes/maps/expansion ]; then
    mkdir -p data/nuscenes/maps
    curl -L -o /tmp/map-expansion.zip https://motional-nuscenes.s3.amazonaws.com/public/v1.0/nuScenes-map-expansion-v1.3.zip
    unzip -q -o /tmp/map-expansion.zip -d data/nuscenes/maps && rm /tmp/map-expansion.zip
  fi
  # shellcheck disable=SC1091
  source env.sh
  [ -f data/nuscenes_scenes.jsonl ] || python tools/nuscenes_to_scenes.py --root data/nuscenes --output data/nuscenes_scenes.jsonl
  [ -d data/nuscenes_perception ] || python tools/nuscenes_perception_frames.py
  python tools/nuscenes_lane_change.py --scenes data/nuscenes_scenes.jsonl
  python tools/nuscenes_multilane.py --scenes data/nuscenes_scenes.jsonl
fi

say "done"
cat <<MSG
  source env.sh && python app.py            # interactive app on http://127.0.0.1:7860
  ./run_demo.sh                             # planning demo on the bundled WOD-E2E scene
  ./run_perception_demo.sh                  # perception demo (needs the CUDA toolchain)
$( [ -z "$NUSCENES_TAR" ] && echo "  Re-run with --nuscenes /path/to/v1.0-mini.tar to enable the nuScenes scenes and the Trion tab." )
MSG
