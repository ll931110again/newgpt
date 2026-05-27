#!/usr/bin/env bash
# Poll Lambda until an instance type has capacity, then provision.
#
# Usage:
#   ./poll_provision.sh gpu_1x_a100_sxm4
#   LAMBDA_FALLBACK_80GB=h100 ./poll_provision.sh   # uses auto + h100 fallback
set -euo pipefail

INSTANCE_TYPE="${1:-auto}"
INTERVAL="${LAMBDA_POLL_INTERVAL_SEC:-60}"

export LAMBDA_INSTANCE_TYPE="$INSTANCE_TYPE"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[poll] Waiting for capacity: $INSTANCE_TYPE (every ${INTERVAL}s)"
while true; do
  if OUTPUT="$("$SCRIPT_DIR/provision.sh" 2>&1)"; then
    echo "$OUTPUT"
    exit 0
  fi
  echo "$OUTPUT" | tail -n 3
  echo "[poll] retrying in ${INTERVAL}s..."
  sleep "$INTERVAL"
done
