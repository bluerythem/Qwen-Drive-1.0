# Source this before running anything:  source env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUDA_HOME is a hand-built prefix (symlinks) because this box has no system
# CUDA toolkit -- only an unusable CUDA 9.0. nvcc 12.8 came from micromamba
# (cuda128/), the extra headers (cusparse.h, cublas...) from torch's pip
# nvidia-* wheels. Only the perception mode needs it; it JIT-builds
# src/qwen_drive_perception/ops/ on first use.
export CUDA_HOME="$ROOT/cuda-home"
# Build the JIT kernels only for the GPU that is present (e.g. 12.0 for an RTX PRO 6000
# Blackwell, 8.9 for an RTX 4090). Override by exporting TORCH_CUDA_ARCH_LIST first.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  [ -n "$_cap" ] && export TORCH_CUDA_ARCH_LIST="$_cap"
  unset _cap
fi

export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"   # .venv/bin puts ninja on PATH
export PYTHONPATH="$ROOT/src"
