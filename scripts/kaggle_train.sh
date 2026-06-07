#!/usr/bin/env bash
# Kaggle training workflow — safe to suspend your machine between commands.
#
# Training runs on Kaggle's GPUs; your laptop only pushes code and polls for results.
#
# Usage:
#   ./scripts/kaggle_train.sh start    Push kernel and exit (detach — close/suspend OK)
#   ./scripts/kaggle_train.sh resume   Wait for remote run + download model (after wake)
#   ./scripts/kaggle_train.sh status   One-shot status + recent logs
#   ./scripts/kaggle_train.sh download Download outputs if training already finished
#   ./scripts/kaggle_train.sh wait     Block until done + download (push + poll + download)
#
# Legacy:
#   ./scripts/kaggle_train.sh                full wait flow (push + poll + download)
#   KAGGLE_DETACH=1 ./scripts/kaggle_train.sh   push and exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/kaggle_common.sh"

kaggle_load_env

if [[ $# -eq 0 ]]; then
  if [[ -n "${KAGGLE_DETACH:-}" ]]; then
    set -- start
  else
    set -- wait
  fi
fi

cmd="$1"
shift || true

case "$cmd" in
  start|detach)
    kaggle_push_and_run
    ;;
  resume)
    phase="$(kaggle_run_phase "$(kaggle_fetch_logs 2>/dev/null || true)")"
    if [[ "$phase" == "completed" ]]; then
      echo "[kaggle] Training already complete; downloading..."
      kaggle_download_outputs
    elif [[ "$phase" == "failed" ]]; then
      echo "[kaggle] Last run failed. Recent logs:" >&2
      kaggle_fetch_logs | tail -60
      exit 1
    else
      if ! kaggle_wait_for_completion; then
        exit $?
      fi
      kaggle_download_outputs
    fi
    echo "[kaggle] Done."
    ;;
  status)
    kaggle_show_status
    ;;
  download)
    kaggle_download_outputs
    ;;
  wait|run|all)
    kaggle_push_and_run
    if ! kaggle_wait_for_completion; then
      exit $?
    fi
    kaggle_download_outputs
    echo "[kaggle] Done."
    ;;
  help|-h|--help)
    grep '^#' "$0" | head -16 | sed 's/^# \?//'
    ;;
  *)
    echo "Unknown command: $cmd (try: start | resume | status | download | wait)" >&2
    exit 1
    ;;
esac
