#!/usr/bin/env bash
# List Lambda Cloud instance types and regions with capacity.
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

if [ -z "${LAMBDA_API_KEY:-}" ]; then
  load_env
fi
if [ -z "${LAMBDA_API_KEY:-}" ]; then
  echo "Set LAMBDA_API_KEY or add to .env" >&2
  exit 1
fi

curl -fsS -u "${LAMBDA_API_KEY}:" "${API_BASE}/instance-types" | jq -r '
  .data
  | to_entries[]
  | {
      type: .key,
      description: (.value.instance_type.description // ""),
      price: (.value.instance_type.price_cents_per_hour // null),
      regions: ([.value.regions_with_capacity_available[]?.name] | join(", "))
    }
  | "\(.type)\t\(.description)\t$\((.price/100)|tostring)/hr\t[\(.regions)]"
'
