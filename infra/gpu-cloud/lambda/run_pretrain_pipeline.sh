#!/usr/bin/env bash
# Full pretrain pipeline on a GPU VM (run inside repo root or via bootstrap).
set -euo pipefail

DATASET_VERSION="${DATASET_VERSION:-v1}"
TOKENIZER="${TOKENIZER:-gpt2}"
RAW_JSONL="${RAW_JSONL:-data/raw/wikitext.jsonl}"
CONFIG="${PRETRAIN_CONFIG:-configs/pretrain_1-3b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_1-3b}"
MAX_ROWS="${MAX_ROWS:-0}"   # 0 = full WikiText-103 train split

DOCKER="${DOCKER:-docker}"
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

run_in_docker() {
  $DOCKER run --rm --gpus all --ipc=host \
    --env-file .env \
    -v "$PWD:/workspace" \
    -w /workspace \
    pretrainer:latest "$@"
}

echo "[pipeline] 1/3 prepare jsonl (WikiText-103)"
run_in_docker src.data.prepare_jsonl \
  --dataset Salesforce/wikitext \
  --config wikitext-103-raw-v1 \
  --split train \
  --out "$RAW_JSONL" \
  ${MAX_ROWS:+--max_rows "$MAX_ROWS"}

echo "[pipeline] 2/3 tokenize + shard"
run_in_docker src.data.tokenize_and_shard \
  --input "$RAW_JSONL" \
  --tokenizer "$TOKENIZER" \
  --out_dir data \
  --dataset_version "$DATASET_VERSION" \
  --shard_tokens 5000000

echo "[pipeline] 3/3 pretrain"
run_in_docker src.infra.run_job --spec infra/gpu-cloud/job_pretrain.yaml

echo "[pipeline] done. Checkpoints: $OUTPUT_DIR/final/"
