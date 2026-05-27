#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/serve_1-3b.yaml}"
MODEL_PATH="${2:-}"

ARGS=(src.serve.launch_vllm --config "$CONFIG")
if [ -n "$MODEL_PATH" ]; then
  ARGS+=(--model_path "$MODEL_PATH")
fi

docker run --rm --gpus all --ipc=host \
  --env-file .env \
  -v "$PWD:/workspace" -w /workspace \
  -p 8000:8000 \
  pretrainer:latest "${ARGS[@]}"

