#!/usr/bin/env bash
# Destroy Lambda GPU instance via Terraform.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$REPO_ROOT/infra/terraform/lambda"

# shellcheck source=load_env.sh
source "$SCRIPT_DIR/load_env.sh"

cd "$TF_DIR"
terraform init -input=false
terraform destroy -input=false "$@"

rm -f "$REPO_ROOT/.lambda_instance_ip" "$REPO_ROOT/.lambda_instance_id"
echo "Destroyed Lambda instance and cleared local output files."
