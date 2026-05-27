#!/usr/bin/env bash
# Terminate a Lambda instance by ID (or .lambda_instance_id from provision.sh).
set -euo pipefail

API_BASE="${LAMBDA_API_BASE:-https://cloud.lambdalabs.com/api/v1}"

load_env() {
  local env_file="${ENV_FILE:-}"
  if [ -z "$env_file" ]; then
    if [ -f ".env" ]; then env_file=".env"
    elif [ -f "infra/gpu-cloud/.env" ]; then env_file="infra/gpu-cloud/.env"
    fi
  fi
  if [ -n "$env_file" ] && [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

load_env

INSTANCE_ID="${1:-}"
if [ -z "$INSTANCE_ID" ] && [ -f ".lambda_instance_id" ]; then
  INSTANCE_ID="$(cat .lambda_instance_id)"
fi
if [ -z "$INSTANCE_ID" ]; then
  echo "Usage: $0 <instance-id>" >&2
  exit 1
fi
if [ -z "${LAMBDA_API_KEY:-}" ]; then
  echo "LAMBDA_API_KEY not set" >&2
  exit 1
fi

read -r -p "Terminate instance ${INSTANCE_ID}? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

payload="$(jq -n --arg id "$INSTANCE_ID" '{instance_ids: [$id]}')"
curl -fsS -u "${LAMBDA_API_KEY}:" -X POST \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${API_BASE}/instance-operations/terminate" | jq .

echo "Terminated ${INSTANCE_ID}"
