#!/usr/bin/env bash
# BEV perception (3D detection + occupancy + map seg) on the six bundled frames.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
python scripts/run_perception.py \
    --vlm Qwen-Drive-1.0-4B \
    --model Qwen-Drive-1.0-4B/perception \
    --frames data/demo/perception \
    --output outputs/perception_demo
python scripts/visualize_perception.py \
    --frames data/demo/perception \
    --predictions outputs/perception_demo
