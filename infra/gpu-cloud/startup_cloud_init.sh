#!/usr/bin/env bash
set -euo pipefail

# Cloud-init friendly bootstrap for a GPU VM:
# - installs docker + nvidia container toolkit (if missing)
# - clones repo (or pulls an image)
# - runs a job spec using docker
#
# This script intentionally does NOT embed secrets.
# Provide env vars via your VM provider UI or a local .env file (ignored by git).

REPO_URL="${REPO_URL:-https://github.com/your-org/pretrainer.git}"
REPO_REF="${REPO_REF:-main}"
WORKDIR="${WORKDIR:-/opt/pretrainer}"

JOB_KIND="${JOB_KIND:-pretrain}"      # pretrain|sft|dpo|eval|serve
JOB_SPEC="${JOB_SPEC:-/opt/pretrainer/infra/gpu-cloud/job_${JOB_KIND}.yaml}"

DOCKER_IMAGE="${DOCKER_IMAGE:-pretrainer:latest}"
ENV_FILE="${ENV_FILE:-/opt/pretrainer/.env}"

echo "[bootstrap] starting"

if ! command -v git >/dev/null 2>&1; then
  apt-get update && apt-get install -y git
fi

mkdir -p "$(dirname "$WORKDIR")"
if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"
git fetch --all --tags
git checkout "$REPO_REF"

if [ ! -f "$ENV_FILE" ]; then
  echo "[bootstrap] env file not found at $ENV_FILE (ok)."
  echo "[bootstrap] copy infra/gpu-cloud/env.example to $ENV_FILE and fill values, or set env vars in provider UI."
fi

echo "[bootstrap] building docker image"
docker build -t "$DOCKER_IMAGE" .

echo "[bootstrap] running job kind=$JOB_KIND spec=$JOB_SPEC"
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file "$ENV_FILE" \
  -v "$WORKDIR:/workspace" \
  -w /workspace \
  "$DOCKER_IMAGE" src.infra.run_job --spec "$JOB_SPEC"

echo "[bootstrap] done"

