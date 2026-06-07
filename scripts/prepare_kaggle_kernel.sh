#!/usr/bin/env bash
# Bundle repo sources into infra/kaggle/ for `kaggle kernels push`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/infra/kaggle"

cd "$ROOT"

echo "[prepare] Staging training code into $STAGE"

rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/src/" "$STAGE/src/"

rsync -a \
  --exclude '__pycache__' \
  "$ROOT/configs/" "$STAGE/configs/"

mkdir -p "$STAGE/data/raw" "$STAGE/data/manifests" "$STAGE/data/shards"
if [[ -f "$ROOT/data/raw/train.jsonl" ]]; then
  cp "$ROOT/data/raw/train.jsonl" "$STAGE/data/raw/train.jsonl"
fi
if [[ -f "$ROOT/data/manifests/v1.json" ]]; then
  cp "$ROOT/data/manifests/v1.json" "$STAGE/data/manifests/v1.json"
  rsync -a "$ROOT/data/shards/" "$STAGE/data/shards/"
fi

# Package marker for local download path resolution.
echo "pretrainer-kaggle-$(date -u +%Y%m%d)" > "$STAGE/.bundle_version"

echo "[prepare] Staged: src/, configs/, data/raw/train.jsonl"
