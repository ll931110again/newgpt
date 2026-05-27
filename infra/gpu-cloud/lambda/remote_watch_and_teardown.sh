#!/usr/bin/env bash
# Run on the GPU VM (nohup). Waits for training to exit, then terminates this Lambda instance.
# Survives laptop disconnect — does not require SSH from your machine.
#
# Usage (on VM):
#   nohup ./infra/gpu-cloud/lambda/remote_watch_and_teardown.sh >> /tmp/pretrainer_teardown.log 2>&1 &
#
# Env:
#   WATCH_PROCESS_PATTERN  default: src.train.pretrain
#   WAIT_FINAL_PATH        optional: wait until this file exists after process exits
#   POLL_SECS              default: 60
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PRETRAINER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$ROOT"

PATTERN="${WATCH_PROCESS_PATTERN:-src.train.pretrain}"
POLL_SECS="${POLL_SECS:-60}"
FINAL_PATH="${WAIT_FINAL_PATH:-}"

echo "[watch] monitoring process pattern: ${PATTERN}"
echo "[watch] poll interval: ${POLL_SECS}s"
if [ -n "$FINAL_PATH" ]; then
  echo "[watch] will wait for final artifact: ${FINAL_PATH}"
fi

while pgrep -f "$PATTERN" >/dev/null 2>&1; do
  progress="$(grep -oE '[0-9]+/20000' /tmp/pretrainer_continue_pretrain.log 2>/dev/null | tail -1 || true)"
  if [ -n "$progress" ]; then
    echo "[watch] training progress: ${progress}"
  else
    echo "[watch] training still running..."
  fi
  sleep "$POLL_SECS"
done

echo "[watch] training process ended"

if [ -n "$FINAL_PATH" ]; then
  echo "[watch] waiting for ${FINAL_PATH}..."
  while [ ! -f "$FINAL_PATH" ]; do
    if [ -d "runs/pretrain_continue_from_current" ]; then
      latest="$(ls -d runs/pretrain_continue_from_current/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
      if [ -n "$latest" ] && [ -f "${latest}/config.json" ]; then
        echo "[watch] final/ missing; latest checkpoint: ${latest}"
        break
      fi
    fi
    sleep "$POLL_SECS"
  done
  if [ -f "$FINAL_PATH" ]; then
    echo "[watch] final artifact ready"
  fi
fi

# Brief pause so S3 checkpoint callbacks can finish
sleep "${TEARDOWN_DELAY_SECS:-120}"

exec "$SCRIPT_DIR/remote_terminate_instance.sh"
