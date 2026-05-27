#!/usr/bin/env bash
# Run a pretrain job spec in Docker, then auto-terminate this Lambda instance.
# Start the watcher in the background so teardown still runs if this script is killed
# after the docker container starts (e.g. SSH disconnect).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

JOB_SPEC="${1:-infra/gpu-cloud/job_pretrain_continue_from_current.yaml}"
LOG="${TRAIN_LOG:-/tmp/pretrainer_train.log}"

chmod +x "$SCRIPT_DIR"/remote_terminate_instance.sh "$SCRIPT_DIR"/remote_watch_and_teardown.sh

if ! pgrep -f remote_watch_and_teardown.sh >/dev/null 2>&1; then
  echo "[launch] starting background teardown watcher"
  nohup "$SCRIPT_DIR/remote_watch_and_teardown.sh" >> /tmp/pretrainer_teardown.log 2>&1 &
  echo $! > /tmp/pretrainer_teardown.pid
fi

DOCKER="${DOCKER:-docker}"
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

RESUME_FROM="${RESUME_FROM:-}"
if [ -z "$RESUME_FROM" ] && [ -d "runs/pretrain_continue_from_current" ]; then
  RESUME_FROM="$(ls -d runs/pretrain_continue_from_current/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
fi

echo "[launch] training -> ${LOG}"
if [ -n "$RESUME_FROM" ]; then
  echo "[launch] resume_from=${RESUME_FROM}"
fi
: >"$LOG"

if [ -n "$RESUME_FROM" ]; then
  $DOCKER run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --env-file .env \
    -v "$PWD:/workspace" \
    -w /workspace \
    pretrainer:latest \
    src.train.pretrain \
    --config configs/pretrain_continue_from_current.yaml \
    --output_dir runs/pretrain_continue_from_current \
    --resume_from "$RESUME_FROM" 2>&1 | tee -a "$LOG"
else
  $DOCKER run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --env-file .env \
    -v "$PWD:/workspace" \
    -w /workspace \
    pretrainer:latest src.infra.run_job --spec "$JOB_SPEC" 2>&1 | tee -a "$LOG"
fi

echo "[launch] training finished"
