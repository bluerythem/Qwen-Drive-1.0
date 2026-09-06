#!/usr/bin/env bash
# Planning + VQA demo on the bundled WOD-E2E scenes.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
python scripts/demo.py \
    --model Qwen-Drive-1.0-4B \
    --planner Qwen-Drive-1.0-4B/planner-rl \
    --scenes data/demo/planning_scenes.jsonl \
    --image-archive data/demo/frames.parquet \
    --plot demo.png "$@"
