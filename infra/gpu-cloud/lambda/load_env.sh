#!/usr/bin/env bash
# Load API keys from infra/gpu-cloud/.env for Terraform + bash scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/infra/gpu-cloud/.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Provider expects LAMBDALABS_API_KEY; repo uses LAMBDA_API_KEY.
export LAMBDALABS_API_KEY="${LAMBDALABS_API_KEY:-${LAMBDA_API_KEY:-}}"

if [ -z "${LAMBDALABS_API_KEY:-}" ]; then
  echo "Set LAMBDA_API_KEY in $ENV_FILE" >&2
  exit 1
fi
