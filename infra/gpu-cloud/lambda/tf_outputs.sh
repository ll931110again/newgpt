#!/usr/bin/env bash
# Print Terraform outputs for the Lambda pretrain instance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TF_DIR="$REPO_ROOT/infra/terraform/lambda"

if [ ! -d "$TF_DIR/.terraform" ]; then
  echo "Run ./infra/gpu-cloud/lambda/provision_tf.sh first" >&2
  exit 1
fi

cd "$TF_DIR"
terraform output "$@"
