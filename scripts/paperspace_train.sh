#!/usr/bin/env bash
# Paperspace remote fleet worker — launch Gradient notebook and sync outputs.
#
# Usage:
#   ./scripts/paperspace_train.sh start
#   ./scripts/paperspace_train.sh status
#   ./scripts/paperspace_train.sh download
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/paperspace_common.sh"

paperspace_load_env

cmd="${1:-status}"
shift || true

case "$cmd" in
  start)
    paperspace_push_and_run
    ;;
  status)
    paperspace_show_status
    ;;
  download)
    paperspace_download_outputs
    ;;
  *)
    echo "Unknown command: $cmd (try: start | status | download)" >&2
    exit 1
    ;;
esac
