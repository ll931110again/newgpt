#!/usr/bin/env bash
# Provision a Lambda Cloud GPU instance via the official REST API.
# Docs: https://docs-api.lambda.ai/api/cloud
#
# Usage:
#   export LAMBDA_API_KEY=...
#   ./infra/gpu-cloud/lambda/provision.sh
#
# Options (env vars):
#   LAMBDA_INSTANCE_TYPE   default: auto (pick 1x A100 80GB if available)
#   LAMBDA_REGION          default: auto (first region with capacity)
#   LAMBDA_SSH_KEY_NAME    default: auto (first registered key)
#   LAMBDA_INSTANCE_NAME   default: pretrainer-pretrain
#   LAMBDA_FILESYSTEM      optional filesystem name to attach

set -euo pipefail

API_BASE="${LAMBDA_API_BASE:-https://cloud.lambdalabs.com/api/v1}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

load_env() {
  local env_file="${ENV_FILE:-}"
  if [ -z "$env_file" ]; then
    # Prefer repo-root .env, then infra/gpu-cloud/.env
    if [ -f ".env" ]; then
      env_file=".env"
    elif [ -f "infra/gpu-cloud/.env" ]; then
      env_file="infra/gpu-cloud/.env"
    fi
  fi
  if [ -n "$env_file" ] && [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

api() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  if [ -z "${LAMBDA_API_KEY:-}" ]; then
    echo "LAMBDA_API_KEY is not set. Export it or add it to .env (see infra/gpu-cloud/env.example)." >&2
    exit 1
  fi
  if [ -n "$data" ]; then
    curl -fsS -u "${LAMBDA_API_KEY}:" -X "$method" \
      -H "Content-Type: application/json" \
      -d "$data" \
      "${API_BASE}${path}"
  else
    curl -fsS -u "${LAMBDA_API_KEY}:" -X "$method" "${API_BASE}${path}"
  fi
}

pick_ssh_key() {
  if [ -n "${LAMBDA_SSH_KEY_NAME:-}" ]; then
    echo "$LAMBDA_SSH_KEY_NAME"
    return
  fi
  local keys
  keys="$(api GET "/ssh-keys" | jq -r '.data[].name' | head -n 1)"
  if [ -z "$keys" ] || [ "$keys" = "null" ]; then
    echo "No SSH keys found in Lambda account. Add one at https://cloud.lambda.ai/ssh-keys" >&2
    exit 1
  fi
  echo "$keys"
}

# Resolve instance type name for 1x A100 80GB (or use LAMBDA_INSTANCE_TYPE).
pick_instance_type() {
  if [ -n "${LAMBDA_INSTANCE_TYPE:-}" ] && [ "$LAMBDA_INSTANCE_TYPE" != "auto" ]; then
    echo "$LAMBDA_INSTANCE_TYPE"
    return
  fi

  local types_json
  types_json="$(api GET "/instance-types")"

  # Prefer 1x A100 80GB SKU if Lambda offers it AND has capacity somewhere.
  local preferred
  preferred="$(echo "$types_json" | jq -r '
    .data
    | to_entries[]
    | select(
        (.key | test("^gpu_1x_a100.*80"; "i"))
        and ((.value.regions_with_capacity_available | length) > 0)
      )
    | .key
  ' | head -n 1)"

  if [ -n "$preferred" ] && [ "$preferred" != "null" ]; then
    echo "$preferred"
    return
  fi

  # Lambda often only lists 8x A100 80GB (gpu_8x_a100_80gb_sxm4), not 1x 80GB.
  local has_1x_80gb_sku
  has_1x_80gb_sku="$(echo "$types_json" | jq -r '.data | keys[] | select(test("^gpu_1x_a100.*80"; "i"))' | head -n 1)"
  if [ -n "$has_1x_80gb_sku" ]; then
    echo "[warn] Found $has_1x_80gb_sku but no regional capacity right now." >&2
    echo "[warn] Use ./poll_provision.sh or set LAMBDA_INSTANCE_TYPE to another SKU." >&2
  else
    echo "[warn] Lambda API has no 1x A100 80GB SKU (only 8x A100 80GB in catalog)." >&2
  fi

  # Optional fallback (also 80GB VRAM): 1x H100
  if [ "${LAMBDA_FALLBACK_80GB:-}" = "h100" ] || [ "${LAMBDA_FALLBACK_80GB:-}" = "1" ]; then
    local h100
    h100="$(echo "$types_json" | jq -r '
      .data
      | to_entries[]
      | select(.key | test("^gpu_1x_h100"; "i"))
      | select((.value.regions_with_capacity_available | length) > 0)
      | .key
    ' | head -n 1)"
    if [ -n "$h100" ]; then
      echo "[warn] Falling back to $h100 (80GB class GPU with capacity)." >&2
      echo "$h100"
      return
    fi
  fi

  # A100 40GB single-GPU with capacity (common fallback for pretrain_1-3b)
  local a100_40
  a100_40="$(echo "$types_json" | jq -r '
    .data
    | to_entries[]
    | select(.key | test("^gpu_1x_a100"; "i"))
    | select((.value.regions_with_capacity_available | length) > 0)
    | .key
  ' | head -n 1)"
  if [ -n "$a100_40" ]; then
    echo "[warn] No 1x A100 80GB capacity; using available A100 SKU: $a100_40 (likely 40GB)." >&2
    echo "[warn] For 80GB VRAM now: LAMBDA_FALLBACK_80GB=h100 ./provision.sh" >&2
    echo "$a100_40"
    return
  fi

  echo "No 1x A100 (80GB or 40GB) capacity. Run ./list_types.sh or ./poll_provision.sh <type>." >&2
  exit 1
}

pick_region() {
  local instance_type="$1"
  if [ -n "${LAMBDA_REGION:-}" ] && [ "$LAMBDA_REGION" != "auto" ]; then
    echo "$LAMBDA_REGION"
    return
  fi

  local region
  region="$(api GET "/instance-types" | jq -r --arg t "$instance_type" '
    .data[$t].regions_with_capacity_available[0].name // empty
  ')"
  if [ -z "$region" ]; then
    echo "No region with capacity for $instance_type. Try another region or poll later." >&2
    echo "Available regions (may be empty):" >&2
    api GET "/instance-types" | jq -r --arg t "$instance_type" '.data[$t].regions_with_capacity_available' >&2
    exit 1
  fi
  echo "$region"
}

launch_instance() {
  local region="$1"
  local instance_type="$2"
  local ssh_key="$3"
  local name="${LAMBDA_INSTANCE_NAME:-pretrainer-pretrain}"

  local payload
  payload="$(jq -n \
    --arg region "$region" \
    --arg it "$instance_type" \
    --arg key "$ssh_key" \
    --arg name "$name" \
    --arg fs "${LAMBDA_FILESYSTEM:-}" \
    '{
      region_name: $region,
      instance_type_name: $it,
      ssh_key_names: [$key],
      name: $name
    }
    + (if $fs != "" then {file_system_names: [$fs]} else {} end)'
  )"

  api POST "/instance-operations/launch" "$payload"
}

wait_for_active() {
  local instance_id="$1"
  local timeout="${LAMBDA_WAIT_TIMEOUT_SEC:-1800}"
  local start
  start="$(date +%s)"

  echo "[lambda] Waiting for instance $instance_id to become active (timeout ${timeout}s)..." >&2
  while true; do
    local now elapsed status ip
    now="$(date +%s)"
    elapsed=$((now - start))
    if [ "$elapsed" -gt "$timeout" ]; then
      echo "Timed out waiting for instance to become active." >&2
      exit 1
    fi

    local info
    info="$(api GET "/instances/${instance_id}")"
    status="$(echo "$info" | jq -r '.data.status')"
    ip="$(echo "$info" | jq -r '.data.ip // empty')"

    echo "[lambda] status=$status ip=${ip:-pending}" >&2
    if [ "$status" = "active" ] && [ -n "$ip" ]; then
      echo "$ip" >&1
      return
    fi
    if [ "$status" = "unhealthy" ] || [ "$status" = "terminated" ]; then
      echo "Instance entered bad state: $status" >&2
      echo "$info" | jq . >&2
      exit 1
    fi
    sleep 15
  done
}

main() {
  need_cmd curl
  need_cmd jq

  # Run from repo root if possible
  if [ -f "infra/gpu-cloud/env.example" ]; then
    :
  elif [ -f "../../env.example" ]; then
    cd "$(dirname "$0")/../.."
  fi

  load_env

  local ssh_key instance_type region
  ssh_key="$(pick_ssh_key)"
  instance_type="$(pick_instance_type)"
  region="$(pick_region "$instance_type")"

  echo "[lambda] ssh_key=$ssh_key"
  echo "[lambda] instance_type=$instance_type"
  echo "[lambda] region=$region"

  local launch_resp instance_id
  launch_resp="$(launch_instance "$region" "$instance_type" "$ssh_key")"
  instance_id="$(echo "$launch_resp" | jq -r '.data.instance_ids[0]')"
  if [ -z "$instance_id" ] || [ "$instance_id" = "null" ]; then
    echo "Launch failed:" >&2
    echo "$launch_resp" | jq . >&2
    exit 1
  fi

  echo "[lambda] launched instance_id=$instance_id"
  echo "$instance_id" > "${LAMBDA_INSTANCE_ID_FILE:-.lambda_instance_id}"

  local ip
  ip="$(wait_for_active "$instance_id")"

  echo ""
  echo "=== Instance ready ==="
  echo "instance_id: $instance_id"
  echo "ip:          $ip"
  echo ""
  echo "SSH:"
  echo "  ssh ubuntu@${ip}"
  echo ""
  echo "Next (on the instance): clone repo, copy .env, then run pretraining bootstrap."
  echo "  See docs/21_lambda_provision.md"
}

main "$@"
