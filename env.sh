# Source this before running anything:  source env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUDA_HOME is a hand-built prefix (symlinks) because this box has no system
# CUDA toolkit -- only an unusable CUDA 9.0. nvcc 12.8 came from micromamba
# (cuda128/), the extra headers (cusparse.h, cublas...) from torch's pip
# nvidia-* wheels. Only the perception mode needs it; it JIT-builds
# src/qwen_drive_perception/ops/ on first use.
export CUDA_HOME="$ROOT/cuda-home"
export TORCH_CUDA_ARCH_LIST="12.0"          # RTX PRO 6000 Blackwell (sm_120)

export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"   # .venv/bin puts ninja on PATH
export PYTHONPATH="$ROOT/src"
