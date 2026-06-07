#!/usr/bin/env bash
# Shared helpers for Paperspace Gradient fleet workers (remote notebooks).
set -euo pipefail

paperspace_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

paperspace_load_env() {
  ROOT="$(paperspace_root)"
  STATE_DIR="$ROOT/runs/paperspace"
  WORKER_ID="${PAPERSPACE_WORKER_ID:-paperspace-1}"
  STATE_FILE="$STATE_DIR/${WORKER_ID}.json"
  OUT="$ROOT/checkpoints/paperspace_${WORKER_ID}"

  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi

  if [[ -z "${PAPERSPACE_API_KEY:-}" ]]; then
    echo "Set PAPERSPACE_API_KEY in .env" >&2
    exit 1
  fi

  export PAPERSPACE_TEAM_ID="${PAPERSPACE_TEAM_ID:-}"
  export PAPERSPACE_PROJECT_ID="${PAPERSPACE_PROJECT_ID:-}"
  export PAPERSPACE_WORKSPACE_GIT="${PAPERSPACE_WORKSPACE_GIT:-https://github.com/ll931110again/newgpt.git}"
  export PAPERSPACE_MACHINE_TYPE="${PAPERSPACE_MACHINE_TYPE:-Free-GPU}"
  export PAPERSPACE_CONTAINER="${PAPERSPACE_CONTAINER:-paperspace/nb-pytorch:22.02-py3}"
  export PAPERSPACE_API_BASE="${PAPERSPACE_API_BASE:-https://api.paperspace.io}"
}

paperspace_api() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  if [[ -n "$data" ]]; then
    curl -fsS -X "$method" "${PAPERSPACE_API_BASE}${path}" \
      -H "x-api-key: ${PAPERSPACE_API_KEY}" \
      -H "Content-Type: application/json" \
      --data-raw "$data"
  else
    curl -fsS -X "$method" "${PAPERSPACE_API_BASE}${path}" \
      -H "x-api-key: ${PAPERSPACE_API_KEY}" \
      -H "Content-Type: application/json"
  fi
}

paperspace_resolve_project_id() {
  if [[ -n "${PAPERSPACE_PROJECT_ID:-}" ]]; then
    echo "$PAPERSPACE_PROJECT_ID"
    return 0
  fi
  echo "PAPERSPACE_PROJECT_ID is required for remote notebooks." >&2
  echo "  Team ID (from console URL): ${PAPERSPACE_TEAM_ID:-unknown}" >&2
  echo "  Find project ID: open https://console.paperspace.com/${PAPERSPACE_TEAM_ID:-TEAM}/notebooks" >&2
  echo "  → pick/create a project → copy its id (prj… or short id like psukfyemho7)" >&2
  echo "  Set PAPERSPACE_PROJECT_ID in .env (this is NOT the team slug t2y5yctyb8)." >&2
  exit 1
}

paperspace_write_state() {
  local status="$1"
  shift
  mkdir -p "$STATE_DIR"
  python3 - "$STATE_FILE" "$status" "$WORKER_ID" "$@" <<'PY'
import json, os, sys
from datetime import datetime, timezone

path, status, worker_id = sys.argv[1], sys.argv[2], sys.argv[3]
extra = {}
for arg in sys.argv[4:]:
    if "=" in arg:
        k, v = arg.split("=", 1)
        extra[k] = v

data = {}
if os.path.isfile(path):
    try:
        data = json.load(open(path))
    except Exception:
        pass

data.update({
    "worker_id": worker_id,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    **extra,
})
json.dump(data, open(path, "w"), indent=2)
print(f"[paperspace] State -> {path} ({status})")
PY
}

paperspace_push_and_run() {
  local project_id
  project_id="$(paperspace_resolve_project_id)"
  if [[ -z "$project_id" ]]; then
    echo "Set PAPERSPACE_PROJECT_ID in .env" >&2
    exit 1
  fi

  local name cmd payload resp notebook_id
  cmd="bash infra/paperspace/fleet_startup.sh"
  name="pretrainer-${WORKER_ID}-$(date -u +%Y%m%d-%H%M%S)"

  payload="$(python3 - "$project_id" "$name" "$cmd" <<'PY'
import json, os, sys
project_id, name, cmd = sys.argv[1:4]
keys = [
    "FLEET_ENABLED", "FLEET_WORKER_ID", "FLEET_RUN_NAME", "FLEET_RANK",
    "FLEET_WORLD_SIZE", "FLEET_SHARD_IDS", "FLEET_SYNC_EVERY_STEPS",
    "FLEET_SYNC_MODE", "FLEET_TRAIN_CONFIG", "FLEET_CANONICAL_INIT",
]
env = {k: os.environ[k] for k in keys if os.environ.get(k)}
env["FLEET_OUTPUT_DIR"] = "/notebooks/fleet-output"
print(json.dumps({
    "machineType": os.environ["PAPERSPACE_MACHINE_TYPE"],
    "container": os.environ["PAPERSPACE_CONTAINER"],
    "projectId": project_id,
    "name": name,
    "workspace": os.environ["PAPERSPACE_WORKSPACE_GIT"],
    "command": cmd,
    "environment": env,
    "shutdownTimeout": 6,
}))
PY
)"

  echo "[paperspace] Creating remote notebook (project=$project_id machine=$PAPERSPACE_MACHINE_TYPE)..."
  resp="$(paperspace_api POST "/notebooks/v2/createNotebook" "$payload")"
  notebook_id="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('id') or d.get('notebookId') or '')" "$resp")"
  if [[ -z "$notebook_id" ]]; then
    echo "[paperspace] Unexpected API response: $resp" >&2
    exit 1
  fi

  paperspace_write_state "running" "notebook_id=$notebook_id" "project_id=$project_id"
  echo "[paperspace] Remote notebook running: id=$notebook_id"
}

paperspace_notebook_status() {
  local notebook_id="$1"
  paperspace_api POST "/notebooks/getNotebook" "{\"notebookId\":\"$notebook_id\"}" 2>/dev/null || echo "{}"
}

paperspace_run_phase() {
  local notebook_id="${1:-}"
  if [[ -z "$notebook_id" && -f "$STATE_FILE" ]]; then
    notebook_id="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('notebook_id',''))" 2>/dev/null || true)"
  fi
  if [[ -z "$notebook_id" ]]; then
    echo "unknown"
    return 0
  fi
  local info phase
  info="$(paperspace_notebook_status "$notebook_id")"
  phase="$(python3 -c "import json,sys; d=json.loads(sys.argv[1] or '{}'); print((d.get('state') or d.get('status') or 'running').lower())" "$info")"
  case "$phase" in
    *stop*|*done*|*complete*) echo "completed" ;;
    *fail*|*error*) echo "failed" ;;
    *) echo "running" ;;
  esac
}

paperspace_download_outputs() {
  local notebook_id
  notebook_id="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('notebook_id',''))" 2>/dev/null || true)"
  if [[ -z "$notebook_id" ]]; then
    echo "[paperspace] No notebook_id in state" >&2
    return 1
  fi

  mkdir -p "$OUT"
  echo "[paperspace] Fetching artifacts for notebook $notebook_id ..."
  local resp
  resp="$(paperspace_api GET "/notebooks/artifactsList?notebookId=${notebook_id}" 2>/dev/null || echo '[]')"
  echo "$resp" > "$OUT/artifacts.json"

  python3 - "$resp" "$OUT" <<'PY'
import json, sys, urllib.request
from pathlib import Path

data = json.loads(sys.argv[1] or "[]")
out = Path(sys.argv[2])
items = data if isinstance(data, list) else data.get("data", data.get("files", []))
for item in items:
    url = item.get("url") or item.get("link")
    name = item.get("name") or item.get("file") or "artifact"
    if not url:
        continue
    dest = out / Path(name).name
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[paperspace] downloaded {dest}")
    except Exception as exc:
        print(f"[paperspace] skip {name}: {exc}")
PY

  if [[ -d "$OUT/final" ]]; then
    paperspace_write_state "downloaded" "local_checkpoint=$OUT/final"
    echo "[paperspace] Artifacts downloaded to $OUT"
    return 0
  fi

  echo "[paperspace] No final artifact yet; see $OUT/artifacts.json" >&2
  return 1
}

paperspace_show_status() {
  local phase="unknown"
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
    echo ""
    phase="$(paperspace_run_phase)"
  fi
  echo "=== Paperspace run status ==="
  echo "worker_id:  $WORKER_ID"
  echo "phase:      $phase"
}
