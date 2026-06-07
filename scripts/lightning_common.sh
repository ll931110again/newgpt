#!/usr/bin/env bash
# Shared helpers for Lightning AI fleet workers.
set -euo pipefail

lightning_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

lightning_load_env() {
  ROOT="$(lightning_root)"
  WORKER_ID="${LIGHTNING_WORKER_ID:-lightning-1}"
  STATE_DIR="$ROOT/runs/lightning"
  STATE_FILE="$STATE_DIR/${WORKER_ID}.json"
  OUT="$ROOT/checkpoints/lightning_${WORKER_ID}"

  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi

  if [[ -z "${LIGHTNING_USER_ID:-}" || -z "${LIGHTNING_API_KEY:-}" ]]; then
    echo "Set LIGHTNING_USER_ID and LIGHTNING_API_KEY in .env" >&2
    exit 1
  fi

  export LIGHTNING_TEAMSPACE="${LIGHTNING_TEAMSPACE:-}"
  export LIGHTNING_USERNAME="${LIGHTNING_USERNAME:-}"
  export LIGHTNING_ORG="${LIGHTNING_ORG:-}"
  export LIGHTNING_STUDIO_NAME="${LIGHTNING_STUDIO_NAME:-pretrainer-${WORKER_ID}}"
  export LIGHTNING_MACHINE="${LIGHTNING_MACHINE:-T4}"
  export LIGHTNING_WORKSPACE_GIT="${LIGHTNING_WORKSPACE_GIT:-https://github.com/ll931110again/newgpt.git}"
  export LIGHTNING_STATE_FILE="$STATE_FILE"

  if ! uv run python -c "import lightning_sdk" 2>/dev/null; then
    uv pip install lightning-sdk --directory "$ROOT"
  fi
}

lightning_push_and_run() {
  uv run python "$ROOT/infra/lightning/fleet_launch.py"
}

lightning_run_phase() {
  if [[ ! -f "$STATE_FILE" ]]; then
    echo "unknown"
    return 0
  fi
  python3 -c "import json; print(json.load(open('$STATE_FILE')).get('status','running'))" 2>/dev/null || echo "running"
}

lightning_download_outputs() {
  if [[ ! -f "$STATE_FILE" ]]; then
    echo "[lightning] No state file" >&2
    return 1
  fi
  mkdir -p "$OUT"
  uv run python - "$STATE_FILE" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from lightning_sdk import Studio

state_path, out = Path(sys.argv[1]), Path(sys.argv[2])
state = json.loads(state_path.read_text())
studio_name = state["studio_name"]
teamspace = state.get("teamspace")
username = state.get("user")
org = state.get("org")
studio_kwargs = {"create_ok": False}
if teamspace:
    studio_kwargs["teamspace"] = teamspace
if org:
    studio_kwargs["org"] = org
elif username:
    studio_kwargs["user"] = username
studio = Studio(studio_name, **studio_kwargs)
base = state.get("remote_output", "/teamspace/studios/this_studio/pretrainer/runs/lightning-output")
out.mkdir(parents=True, exist_ok=True)
for remote in (f"{base}/final", "/teamspace/studios/this_studio/pretrainer/artifacts/final"):
    try:
        studio.download_folder(remote, str(out / "final"))
        print(f"[lightning] downloaded {remote}")
        sys.exit(0)
    except Exception as exc:
        print(f"[lightning] skip {remote}: {exc}")
sys.exit(1)
PY
}

lightning_show_status() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
    echo ""
  fi
  echo "=== Lightning run status ==="
  echo "worker_id:  $WORKER_ID"
  echo "phase:      $(lightning_run_phase)"
}
