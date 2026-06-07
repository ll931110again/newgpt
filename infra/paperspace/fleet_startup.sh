#!/usr/bin/env bash
# Remote Paperspace Gradient notebook startup (runs inside the notebook VM).
set -euo pipefail

ROOT="${WORKSPACE:-/notebooks}"
cd "$ROOT"

CANON="${FLEET_CANONICAL_INIT:-checkpoints/canonical}"
if [[ ! -f "$CANON/model.safetensors" && ! -f "$CANON/pytorch_model.bin" ]]; then
  if [[ -f checkpoints/canonical/model.safetensors ]]; then
    CANON="checkpoints/canonical"
  elif [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "[paperspace] Fetching canonical checkpoint from Kaggle dataset..."
    python3 -m pip install -q kaggle
    mkdir -p /tmp/pretrainer-bundle
    kaggle datasets download -d "${KAGGLE_CHECKPOINT_DATASET:-linhvuongnguyen/pretrainer-bundle}" \
      -p /tmp/pretrainer-bundle --unzip 2>/dev/null || true
    if [[ -f /tmp/pretrainer-bundle/checkpoints/canonical/model.safetensors ]]; then
      mkdir -p checkpoints/canonical
      cp /tmp/pretrainer-bundle/checkpoints/canonical/* checkpoints/canonical/
      CANON="checkpoints/canonical"
    fi
  fi
fi

echo "[paperspace] Installing dependencies..."
python3 -m pip install -q -U pip
python3 -m pip install -q torch transformers accelerate datasets tokenizers safetensors pyyaml tqdm numpy

if [[ -f requirements.txt ]]; then
  python3 -m pip install -q -r requirements.txt
fi

TRAIN_CONFIG="${FLEET_TRAIN_CONFIG:-configs/pretrain_continue_from_current.yaml}"
OUTPUT_DIR="${FLEET_OUTPUT_DIR:-/notebooks/fleet-output}"
mkdir -p "$OUTPUT_DIR"

echo "[paperspace] Fleet worker=${FLEET_WORKER_ID:-unknown} rank=${FLEET_RANK:-0}/${FLEET_WORLD_SIZE:-1}"
if [[ -f "$CANON/model.safetensors" || -f "$CANON/pytorch_model.bin" ]]; then
  export FLEET_CANONICAL_INIT="$CANON"
  echo "[paperspace] Canonical init=$CANON"
fi
echo "[paperspace] Training config=$TRAIN_CONFIG output=$OUTPUT_DIR"

python3 -m src.train.pretrain \
  --config "$TRAIN_CONFIG" \
  --output_dir "$OUTPUT_DIR"

# Copy final model into artifact-friendly path for orchestrator sync.
if [[ -d "$OUTPUT_DIR/final" ]]; then
  mkdir -p /notebooks/artifacts
  rm -rf /notebooks/artifacts/final
  cp -R "$OUTPUT_DIR/final" /notebooks/artifacts/final
fi

echo "[paperspace] Done."
