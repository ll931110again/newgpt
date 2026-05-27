#!/usr/bin/env bash
# Copy this repo to a Lambda instance and start pretraining bootstrap.
# Run after provision_tf.sh (or provision.sh).
#
# Usage:
#   ./infra/gpu-cloud/lambda/bootstrap_remote.sh
#   # or: LAMBDA_INSTANCE_IP=<ip> ./infra/gpu-cloud/lambda/bootstrap_remote.sh
set -euo pipefail

IP="${LAMBDA_INSTANCE_IP:-${1:-}}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

if [ -z "$IP" ] && [ -f "$REPO_ROOT/.lambda_instance_ip" ]; then
  IP="$(cat "$REPO_ROOT/.lambda_instance_ip")"
fi

if [ -z "$IP" ] && [ -d "$REPO_ROOT/infra/terraform/lambda/.terraform" ]; then
  IP="$(cd "$REPO_ROOT/infra/terraform/lambda" && terraform output -raw instance_ip 2>/dev/null || true)"
fi

if [ -z "$IP" ]; then
  echo "Usage: LAMBDA_INSTANCE_IP=<ip> $0" >&2
  echo "  Or run ./infra/gpu-cloud/lambda/provision_tf.sh first." >&2
  exit 1
fi

SSH_USER="${LAMBDA_SSH_USER:-ubuntu}"
SSH_KEY="${LAMBDA_SSH_PRIVATE_KEY:-}"
SSH_OPTS_STR="-o StrictHostKeyChecking=accept-new"
if [ -n "$SSH_KEY" ]; then
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -i "$SSH_KEY")
  SSH_OPTS_STR="-o StrictHostKeyChecking=accept-new -i ${SSH_KEY}"
else
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
fi

REMOTE_DIR="${LAMBDA_REMOTE_DIR:-/home/ubuntu/pretrainer}"

echo "[bootstrap] syncing repo to ${SSH_USER}@${IP}:${REMOTE_DIR}"
RSYNC_EXCLUDES=( \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'runs' --exclude 'wandb' \
  --exclude '.cache' --exclude 'outputs' \
  --exclude 'id_rsa' --exclude 'id_rsa.pub' --exclude '*.pem' \
)
if [ "${INCLUDE_CHECKPOINTS:-0}" != "1" ]; then
  RSYNC_EXCLUDES+=( --exclude 'checkpoints' )
fi

rsync -az --delete \
  -e "ssh ${SSH_OPTS_STR}" \
  "${RSYNC_EXCLUDES[@]}" \
  "$REPO_ROOT/" "${SSH_USER}@${IP}:${REMOTE_DIR}/"

echo "[bootstrap] installing docker (if needed) and starting pretrain job"
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${IP}" bash -s <<EOF
set -euo pipefail
cd "${REMOTE_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y docker.io git
  sudo usermod -aG docker "\$USER"
fi

DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

if [ -f infra/gpu-cloud/.env ]; then
  cp infra/gpu-cloud/.env .env
elif [ ! -f .env ]; then
  echo "Warning: no .env on remote; create one from infra/gpu-cloud/env.example"
fi

echo "[bootstrap] building docker image"
\$DOCKER build -t pretrainer:latest .

if [ "\${BOOTSTRAP_NO_RUN:-0}" = "1" ]; then
  echo "[bootstrap] BOOTSTRAP_NO_RUN=1; skipping job launch"
  exit 0
fi

chmod +x infra/gpu-cloud/lambda/remote_terminate_instance.sh \
  infra/gpu-cloud/lambda/remote_watch_and_teardown.sh \
  infra/gpu-cloud/lambda/run_pretrain_job_with_teardown.sh 2>/dev/null || true

if [ "\${AUTO_TERMINATE:-1}" = "1" ] && [ ! -f /tmp/pretrainer_teardown.pid ]; then
  echo "[bootstrap] starting auto-teardown watcher (survives laptop disconnect)"
  nohup ./infra/gpu-cloud/lambda/remote_watch_and_teardown.sh >> /tmp/pretrainer_teardown.log 2>&1 &
  echo \$! > /tmp/pretrainer_teardown.pid
fi

if [ "\${RUN_FULL_PIPELINE:-1}" = "1" ]; then
  chmod +x infra/gpu-cloud/lambda/run_pretrain_pipeline.sh
  ./infra/gpu-cloud/lambda/run_pretrain_pipeline.sh
else
  echo "[bootstrap] running pretrain job only (RUN_FULL_PIPELINE=0)"
  \$DOCKER run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --env-file .env \
    -v "\${PWD}:/workspace" \
    -w /workspace \
    pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_pretrain.yaml
fi
EOF

echo "[bootstrap] done"
