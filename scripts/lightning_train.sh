#!/usr/bin/env bash
# Lightning AI remote fleet worker.
#
# Usage:
#   ./scripts/lightning_train.sh start
#   ./scripts/lightning_train.sh status
#   ./scripts/lightning_train.sh download
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lightning_common.sh"

lightning_load_env

cmd="${1:-status}"
shift || true

case "$cmd" in
  start)
    lightning_push_and_run
    ;;
  status)
    lightning_show_status
    ;;
  download)
    lightning_download_outputs
    ;;
  *)
    echo "Unknown command: $cmd (try: start | status | download)" >&2
    exit 1
    ;;
esac
