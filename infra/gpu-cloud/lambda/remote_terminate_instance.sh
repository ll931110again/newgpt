#!/usr/bin/env bash
# Terminate this Lambda instance via API. Intended to run ON the GPU VM (no SSH from laptop).
#
# Requires in /workspace/.env (or env):
#   LAMBDA_API_KEY
# And one of:
#   .lambda_instance_id, LAMBDA_INSTANCE_ID
#
# Disable with: AUTO_TERMINATE=0
set -euo pipefail

ROOT="${PRETRAINER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ "${AUTO_TERMINATE:-1}" != "1" ]; then
  echo "[teardown] AUTO_TERMINATE=0; skipping instance terminate"
  exit 0
fi

if [ -z "${LAMBDA_API_KEY:-}" ]; then
  echo "[teardown] LAMBDA_API_KEY not set; cannot terminate instance" >&2
  exit 1
fi

INSTANCE_ID="${LAMBDA_INSTANCE_ID:-}"
if [ -z "$INSTANCE_ID" ] && [ -f .lambda_instance_id ]; then
  INSTANCE_ID="$(tr -d '[:space:]' < .lambda_instance_id)"
fi
if [ -z "$INSTANCE_ID" ]; then
  echo "[teardown] No instance id (.lambda_instance_id or LAMBDA_INSTANCE_ID)" >&2
  exit 1
fi

API_BASE="${LAMBDA_API_BASE:-https://cloud.lambdalabs.com/api/v1}"
echo "[teardown] terminating instance ${INSTANCE_ID}..."

if command -v jq >/dev/null 2>&1; then
  payload="$(jq -n --arg id "$INSTANCE_ID" '{instance_ids: [$id]}')"
else
  payload="{\"instance_ids\":[\"${INSTANCE_ID}\"]}"
fi

curl -fsS -u "${LAMBDA_API_KEY}:" -X POST \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${API_BASE}/instance-operations/terminate"

echo "[teardown] terminate request sent for ${INSTANCE_ID}"
