#!/usr/bin/env bash
# Run (1) continued pretraining from our current checkpoint and
# (2) SFT fine-tuning on Alpaca, inside the GPU VM.
set -euo pipefail

DOCKER="${DOCKER:-docker}"
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

run_in_docker() {
  $DOCKER run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --env-file .env \
    -v "$PWD:/workspace" \
    -w /workspace \
    pretrainer:latest "$@"
}

echo "[combo] 1/3 download SFT dataset (alpaca-cleaned)"
run_in_docker src.data.download_sft_alpaca --out-dir data/sft --max-train "${SFT_MAX_TRAIN:-20000}" --max-eval "${SFT_MAX_EVAL:-1000}"

if [ ! -f "data/manifests/v1.json" ]; then
  echo "[combo] (prep) dataset manifest missing; building WikiText shards"
  DATASET_VERSION="${DATASET_VERSION:-v1}"
  TOKENIZER="${TOKENIZER:-gpt2}"
  RAW_JSONL="${RAW_JSONL:-data/raw/wikitext.jsonl}"

  run_in_docker src.data.prepare_jsonl \
    --dataset Salesforce/wikitext \
    --config wikitext-103-raw-v1 \
    --split train \
    --out "$RAW_JSONL"

  run_in_docker src.data.tokenize_and_shard \
    --input "$RAW_JSONL" \
    --tokenizer "$TOKENIZER" \
    --out_dir data \
    --dataset_version "$DATASET_VERSION" \
    --shard_tokens 5000000
fi

echo "[combo] 2/3 continue pretraining from current checkpoint"
run_in_docker src.infra.run_job --spec infra/gpu-cloud/job_pretrain_continue_from_current.yaml

echo "[combo] 3/3 SFT fine-tune"
SFT_SPEC="infra/gpu-cloud/job_sft_from_pretrained.yaml"
CONTINUE_FINAL="runs/pretrain_continue_from_current/final"
if [ -d "$CONTINUE_FINAL" ]; then
  echo "[combo] SFT base model: $CONTINUE_FINAL"
  SFT_CFG_TMP="configs/.sft_from_continued.yaml"
  sed "s|model_name_or_path: checkpoints/pretrain_1-3b/final|model_name_or_path: ${CONTINUE_FINAL}|" \
    configs/sft_from_pretrained.yaml >"$SFT_CFG_TMP"
  run_in_docker src.train.sft --config "$SFT_CFG_TMP" --output_dir runs/sft_from_pretrained
  rm -f "$SFT_CFG_TMP"
else
  echo "[combo] SFT base model: checkpoints/pretrain_1-3b/final (continue output not found)"
  run_in_docker src.infra.run_job --spec "$SFT_SPEC"
fi

echo "[combo] done"

if [ "${AUTO_TERMINATE:-1}" = "1" ]; then
  echo "[combo] auto-terminate enabled"
  "$(dirname "$0")/remote_terminate_instance.sh"
fi

