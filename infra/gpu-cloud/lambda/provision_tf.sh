#!/usr/bin/env bash
# Provision Lambda GPU via Terraform (preferred over provision.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$REPO_ROOT/infra/terraform/lambda"

# shellcheck source=load_env.sh
source "$SCRIPT_DIR/load_env.sh"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing: $1" >&2
    exit 1
  }
}

need terraform

cd "$TF_DIR"

if [ ! -f terraform.tfvars ]; then
  echo "Create $TF_DIR/terraform.tfvars from terraform.tfvars.example" >&2
  exit 1
fi

terraform init -input=false
terraform apply -input=false "$@"

IP="$(terraform output -raw instance_ip)"
ID="$(terraform output -raw instance_id)"

echo "$IP" > "$REPO_ROOT/.lambda_instance_ip"
echo "$ID" > "$REPO_ROOT/.lambda_instance_id"

echo ""
echo "=== Terraform apply complete ==="
echo "instance_id: $ID"
echo "instance_ip: $IP"
echo ""
terraform output bootstrap_command
