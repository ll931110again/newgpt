#!/usr/bin/env bash
# Remote Lightning AI Studio startup (runs inside the Studio VM).
set -euo pipefail

ROOT="${LIGHTNING_REPO_ROOT:-/teamspace/studios/this_studio/pretrainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="${WORKSPACE:-$(pwd)}"
fi
cd "$ROOT"

CANON="${FLEET_CANONICAL_INIT:-checkpoints/canonical}"
if [[ ! -f "$CANON/model.safetensors" && ! -f "$CANON/pytorch_model.bin" ]]; then
  if [[ -f checkpoints/canonical/model.safetensors ]]; then
    CANON="checkpoints/canonical"
  elif [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "[lightning] Fetching canonical checkpoint from Kaggle dataset..."
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

echo "[lightning] Installing dependencies..."
python3 -m pip install -q -U pip
python3 -m pip install -q torch transformers accelerate datasets tokenizers safetensors pyyaml tqdm numpy

if [[ -f requirements.txt ]]; then
  python3 -m pip install -q -r requirements.txt
fi

TRAIN_CONFIG="${FLEET_TRAIN_CONFIG:-configs/pretrain_continue_from_current.yaml}"
OUTPUT_DIR="${FLEET_OUTPUT_DIR:-$ROOT/runs/lightning-output}"
mkdir -p "$OUTPUT_DIR"

MANIFEST="${ROOT}/data/manifests/v1.json"
if [[ ! -f "$MANIFEST" && -n "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "[lightning] Fetching data shards from Kaggle dataset..."
  python3 -m pip install -q kaggle
  mkdir -p /tmp/pretrainer-bundle
  kaggle datasets download -d "${KAGGLE_CHECKPOINT_DATASET:-linhvuongnguyen/pretrainer-bundle}" \
    -p /tmp/pretrainer-bundle --unzip 2>/dev/null || true
  if [[ -f /tmp/pretrainer-bundle/data/manifests/v1.json ]]; then
    mkdir -p "$ROOT/data/manifests" "$ROOT/data/shards"
    cp /tmp/pretrainer-bundle/data/manifests/v1.json "$ROOT/data/manifests/v1.json"
    cp /tmp/pretrainer-bundle/data/shards/* "$ROOT/data/shards/" 2>/dev/null || true
  fi
fi

echo "[lightning] Fleet worker=${FLEET_WORKER_ID:-unknown} rank=${FLEET_RANK:-0}/${FLEET_WORLD_SIZE:-1}"
if [[ -f "$CANON/model.safetensors" || -f "$CANON/pytorch_model.bin" ]]; then
  export FLEET_CANONICAL_INIT="$CANON"
  echo "[lightning] Canonical init=$CANON"
fi
echo "[lightning] Training config=$TRAIN_CONFIG output=$OUTPUT_DIR"

python3 -m src.train.pretrain \
  --config "$TRAIN_CONFIG" \
  --output_dir "$OUTPUT_DIR"

mkdir -p "$ROOT/artifacts"
if [[ -d "$OUTPUT_DIR/final" ]]; then
  rm -rf "$ROOT/artifacts/final"
  cp -R "$OUTPUT_DIR/final" "$ROOT/artifacts/final"
fi

echo "[lightning] Done."
