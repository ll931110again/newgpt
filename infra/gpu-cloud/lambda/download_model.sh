#!/usr/bin/env bash
# Download trained model artifacts from a Lambda GPU instance to this machine.
#
# Usage:
#   ./infra/gpu-cloud/lambda/download_model.sh
#   LAMBDA_INSTANCE_IP=<ip> LOCAL_DIR=checkpoints/pretrain_1-3b ./infra/gpu-cloud/lambda/download_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

IP="${LAMBDA_INSTANCE_IP:-${1:-}}"
if [ -z "$IP" ] && [ -f "$REPO_ROOT/.lambda_instance_ip" ]; then
  IP="$(cat "$REPO_ROOT/.lambda_instance_ip")"
fi
if [ -z "$IP" ] && [ -d "$REPO_ROOT/infra/terraform/lambda/.terraform" ]; then
  IP="$(cd "$REPO_ROOT/infra/terraform/lambda" && terraform output -raw instance_ip 2>/dev/null || true)"
fi
if [ -z "$IP" ]; then
  echo "Set LAMBDA_INSTANCE_IP or run provision first." >&2
  exit 1
fi

SSH_USER="${LAMBDA_SSH_USER:-ubuntu}"
SSH_KEY="${LAMBDA_SSH_PRIVATE_KEY:-}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
if [ -n "$SSH_KEY" ]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

REMOTE_DIR="${LAMBDA_REMOTE_DIR:-/home/ubuntu/pretrainer}"
RUN_NAME="${RUN_NAME:-pretrain_1-3b}"
REMOTE_RUN="${REMOTE_DIR}/runs/${RUN_NAME}"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/checkpoints/${RUN_NAME}}"

ARTIFACT="${ARTIFACT:-final}"
REMOTE_PATH="${REMOTE_RUN}/${ARTIFACT}"

if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" "test -d '${REMOTE_PATH}'"; then
  echo "Remote path not found: ${SSH_USER}@${IP}:${REMOTE_PATH}" >&2
  echo "Training may still be running. Checkpoints live under ${REMOTE_RUN}/" >&2
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" "ls -la '${REMOTE_RUN}/' 2>/dev/null || true" >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"
echo "[download] ${SSH_USER}@${IP}:${REMOTE_PATH}/ -> ${LOCAL_DIR}/${ARTIFACT}/"

rsync -az --progress \
  -e "ssh ${SSH_OPTS[*]}" \
  "${SSH_USER}@${IP}:${REMOTE_PATH}/" \
  "${LOCAL_DIR}/${ARTIFACT}/"

echo "[download] done: ${LOCAL_DIR}/${ARTIFACT}/"
du -sh "${LOCAL_DIR}/${ARTIFACT}/"
