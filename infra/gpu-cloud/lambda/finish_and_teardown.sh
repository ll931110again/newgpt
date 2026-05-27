#!/usr/bin/env bash
# Wait for pretraining to finish, download the final model, then terminate the Lambda instance.
#
# Usage:
#   ./infra/gpu-cloud/lambda/finish_and_teardown.sh
#   LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_rsa ./infra/gpu-cloud/lambda/finish_and_teardown.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# shellcheck source=load_env.sh
source "$SCRIPT_DIR/load_env.sh"

IP="${LAMBDA_INSTANCE_IP:-}"
if [ -z "$IP" ] && [ -f "$REPO_ROOT/.lambda_instance_ip" ]; then
  IP="$(cat "$REPO_ROOT/.lambda_instance_ip")"
fi
if [ -z "$IP" ]; then
  # Known instance from bash provision (May 2026 run)
  IP="150.136.112.30"
fi

INSTANCE_ID="${LAMBDA_INSTANCE_ID:-}"
if [ -z "$INSTANCE_ID" ] && [ -f "$REPO_ROOT/.lambda_instance_id" ]; then
  INSTANCE_ID="$(cat "$REPO_ROOT/.lambda_instance_id")"
fi

SSH_USER="${LAMBDA_SSH_USER:-ubuntu}"
SSH_KEY="${LAMBDA_SSH_PRIVATE_KEY:-}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
if [ -n "$SSH_KEY" ]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

REMOTE_DIR="${LAMBDA_REMOTE_DIR:-/home/ubuntu/pretrainer}"
RUN_NAME="${RUN_NAME:-pretrain_1-3b}"
REMOTE_FINAL="${REMOTE_DIR}/runs/${RUN_NAME}/final"
POLL_SECS="${POLL_SECS:-60}"

echo "[finish] monitoring ${SSH_USER}@${IP} for ${REMOTE_FINAL}"
echo "[finish] poll interval: ${POLL_SECS}s"

while true; do
  if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" "test -f '${REMOTE_FINAL}/config.json'"; then
    echo "[finish] final model ready"
    break
  fi

  progress="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" \
    "sudo docker logs \$(sudo docker ps -q | head -1) 2>&1 | grep -Eo '[0-9]+/5000' | tail -1" 2>/dev/null || true)"
  if [ -n "$progress" ]; then
    echo "[finish] training progress: ${progress} steps"
  else
    running="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" \
      "sudo docker ps -q | head -1" 2>/dev/null || true)"
    if [ -z "$running" ]; then
      echo "[finish] no running container; checking for final model or latest checkpoint..."
      if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" "test -d '${REMOTE_DIR}/runs/${RUN_NAME}'"; then
        ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" "ls -la '${REMOTE_DIR}/runs/${RUN_NAME}/'" || true
      fi
      # Fall back to latest checkpoint if training exited without writing final/
      latest="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" \
        "ls -d '${REMOTE_DIR}/runs/${RUN_NAME}/checkpoint-'* 2>/dev/null | sort -V | tail -1" || true)"
      if [ -n "$latest" ] && [ -z "${ARTIFACT:-}" ]; then
        echo "[finish] using latest checkpoint: $latest"
        export ARTIFACT="$(basename "$latest")"
        break
      fi
    fi
  fi

  sleep "$POLL_SECS"
done

echo "[finish] downloading model..."
export LAMBDA_INSTANCE_IP="$IP"
export LAMBDA_SSH_PRIVATE_KEY="$SSH_KEY"
export RUN_NAME
"$SCRIPT_DIR/download_model.sh"

if [ -z "$INSTANCE_ID" ]; then
  echo "[finish] no instance id found; skip terminate (set .lambda_instance_id or LAMBDA_INSTANCE_ID)"
  exit 0
fi

echo "[finish] terminating instance ${INSTANCE_ID}..."
payload="$(jq -n --arg id "$INSTANCE_ID" '{instance_ids: [$id]}')"
curl -fsS -u "${LAMBDA_API_KEY}:" -X POST \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${LAMBDA_API_BASE:-https://cloud.lambdalabs.com/api/v1}/instance-operations/terminate" | jq .

rm -f "$REPO_ROOT/.lambda_instance_ip" "$REPO_ROOT/.lambda_instance_id"
echo "[finish] done — model downloaded, instance terminated."
