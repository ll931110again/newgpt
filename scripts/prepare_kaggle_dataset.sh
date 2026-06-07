#!/usr/bin/env bash
# Stage src/configs/data for Kaggle dataset upload.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="$ROOT/infra/kaggle/dataset"

mkdir -p "$DS/src" "$DS/configs" "$DS/data/raw" "$DS/data/manifests" "$DS/data/shards"

rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/src/" "$DS/src/"
rsync -a --exclude '__pycache__' "$ROOT/configs/" "$DS/configs/"
if [[ -f "$ROOT/data/raw/train.jsonl" ]]; then
  cp "$ROOT/data/raw/train.jsonl" "$DS/data/raw/train.jsonl"
fi
if [[ -f "$ROOT/data/manifests/v1.json" ]]; then
  cp "$ROOT/data/manifests/v1.json" "$DS/data/manifests/v1.json"
  rsync -a "$ROOT/data/shards/" "$DS/data/shards/"
fi
cp "$ROOT/infra/kaggle/requirements-kaggle.txt" "$DS/requirements-kaggle.txt"

echo "[dataset] Staged bundle at $DS"
