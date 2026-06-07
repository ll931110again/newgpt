#!/usr/bin/env bash
# Multi-provider parallel fleet training (Kaggle + Paperspace).
#
# Usage:
#   ./scripts/fleet_train.sh start    Launch all enabled providers in parallel
#   ./scripts/fleet_train.sh status   Show fleet state + provider status
#   ./scripts/fleet_train.sh sync     Pull worker checkpoints to local store
#   ./scripts/fleet_train.sh merge    FedAvg merge into canonical checkpoint
#   ./scripts/fleet_train.sh resume   Relaunch dead workers
#   ./scripts/fleet_train.sh watch    Loop sync + merge (orchestrator daemon)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC="${FLEET_SPEC:-$ROOT/infra/fleet/fleet.yaml}"

if [[ ! -f "$SPEC" ]]; then
  SPEC="$ROOT/infra/fleet/fleet.example.yaml"
  echo "[fleet] Using example spec: $SPEC (copy to infra/fleet/fleet.yaml to customize)" >&2
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

cmd="${1:-status}"
shift || true

cd "$ROOT"
exec uv run python3 -m src.fleet.orchestrator --spec "$SPEC" "$cmd" "$@"
