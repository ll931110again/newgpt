#!/usr/bin/env bash
# Shared helpers for Kaggle training scripts (start / resume / status / download).
set -euo pipefail

kaggle_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

kaggle_load_env() {
  ROOT="$(kaggle_root)"
  STAGE="$ROOT/infra/kaggle"
  OUT="$ROOT/checkpoints/kaggle_pretrain"
  STATE_DIR="$ROOT/runs/kaggle"
  STATE_FILE="$STATE_DIR/latest.json"
  KERNEL_ID="${KAGGLE_KERNEL_ID:-linhvuongnguyen/pretrainer-pretrain}"

  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi

  if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "$HOME/.kaggle/access_token" ]]; then
    export KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "$HOME/.kaggle/access_token")"
  fi

  if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "Set KAGGLE_API_TOKEN in .env or ~/.kaggle/access_token" >&2
    exit 1
  fi

  KAGGLE="$ROOT/.venv/bin/kaggle"
  if [[ ! -x "$KAGGLE" ]]; then
    uv pip install 'kaggle>=1.8.0' --directory "$ROOT"
    KAGGLE="$ROOT/.venv/bin/kaggle"
  fi

  export KAGGLE_DATASET="${KAGGLE_DATASET:-wikitext103}"
  export KAGGLE_TRAIN_HOURS="${KAGGLE_TRAIN_HOURS:-9}"
  export KAGGLE_KERNEL_TIMEOUT="${KAGGLE_KERNEL_TIMEOUT:-32400}"
  export KAGGLE_MAX_WAIT="${KAGGLE_MAX_WAIT:-36000}"
  export KAGGLE_POLL_INTERVAL="${KAGGLE_POLL_INTERVAL:-60}"
}

kaggle_parse_logs() {
  python3 -c "import sys,json; d=sys.stdin.read();
try:
  print(''.join(e.get('data','') for e in json.loads(d)))
except Exception:
  print(d)"
}

kaggle_fetch_logs() {
  "$KAGGLE" kernels logs "$KERNEL_ID" 2>&1 | kaggle_parse_logs
}

kaggle_run_phase() {
  # running | completed | failed | unknown
  local logs="${1:-}"
  if [[ -z "$logs" ]]; then
    logs="$(kaggle_fetch_logs 2>/dev/null || true)"
  fi
  if echo "$logs" | grep -q '\[kaggle\] Done\.'; then
    echo "completed"
  elif echo "$logs" | grep -qE 'CalledProcessError|FileNotFoundError|ModuleNotFoundError'; then
    if echo "$logs" | grep -q '\[kaggle\] Done\.'; then
      echo "completed"
    else
      echo "failed"
    fi
  elif echo "$logs" | grep -qE '\[kaggle\] Running:|\[kaggle\] GPU:|tokenize|Downloading corpus|Train steps:'; then
    echo "running"
  elif [[ -n "$logs" && "$logs" != "[]" ]]; then
    echo "running"
  else
    echo "unknown"
  fi
}

kaggle_write_state() {
  local status="$1"
  shift
  mkdir -p "$STATE_DIR"
  python3 - "$STATE_FILE" "$status" "$KERNEL_ID" "$@" <<'PY'
import json, os, sys
from datetime import datetime, timezone

path, status, kernel_id = sys.argv[1], sys.argv[2], sys.argv[3]
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
    "kernel_id": kernel_id,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "dataset": os.environ.get("KAGGLE_DATASET", "wikitext103"),
    "train_hours": os.environ.get("KAGGLE_TRAIN_HOURS", "9"),
    **extra,
})
if status == "pushed" and "started_at" not in data:
    data["started_at"] = data["updated_at"]

json.dump(data, open(path, "w"), indent=2)
print(f"[kaggle] State -> {path} ({status})")
PY
}

kaggle_read_state() {
  if [[ ! -f "$STATE_FILE" ]]; then
    echo ""
    return 0
  fi
  cat "$STATE_FILE"
}

kaggle_push_and_run() {
  "$ROOT/scripts/prepare_kaggle_dataset.sh"

  echo "[kaggle] Uploading/updating code dataset..."
  if ! "$KAGGLE" datasets version -p "$ROOT/infra/kaggle/dataset" --dir-mode tar \
      -m "pretrainer bundle $(date -u +%Y%m%d-%H%M%S)" 2>&1; then
    echo "[kaggle] Creating dataset for first time..."
    "$KAGGLE" datasets create -p "$ROOT/infra/kaggle/dataset" --dir-mode tar 2>&1
  fi

  "$ROOT/scripts/prepare_kaggle_kernel.sh"

  local accelerators=("NvidiaTeslaT4" "NvidiaTeslaP100" "NvidiaL4")
  local pushed=0 acc=""
  for acc in "${accelerators[@]}"; do
    echo "[kaggle] Pushing kernel (accelerator=$acc, dataset=$KAGGLE_DATASET, timeout=${KAGGLE_KERNEL_TIMEOUT}s)..."
    if "$KAGGLE" kernels push -p "$STAGE" --accelerator "$acc" -t "$KAGGLE_KERNEL_TIMEOUT"; then
      pushed=1
      break
    fi
    echo "[kaggle] Accelerator $acc unavailable, trying next..."
  done

  if [[ "$pushed" -ne 1 ]]; then
    echo "[kaggle] Falling back to default GPU..."
    "$KAGGLE" kernels push -p "$STAGE" -t "$KAGGLE_KERNEL_TIMEOUT"
    acc="default"
  fi

  kaggle_write_state "pushed" "accelerator=${acc:-unknown}" \
    "kernel_url=https://www.kaggle.com/code/${KERNEL_ID//\//\/}"
  echo "[kaggle] Kernel is running on Kaggle servers (safe to suspend this machine)."
  echo "[kaggle] Resume later: ./scripts/kaggle_train.sh resume"
}

kaggle_wait_for_completion() {
  local max_wait="$KAGGLE_MAX_WAIT"
  local interval="$KAGGLE_POLL_INTERVAL"
  local elapsed=0
  local phase="running"

  echo "[kaggle] Waiting for remote kernel (poll every ${interval}s, max ${max_wait}s)..."
  while [[ "$elapsed" -lt "$max_wait" ]]; do
    local logs
    logs="$(kaggle_fetch_logs 2>/dev/null || true)"
    phase="$(kaggle_run_phase "$logs")"
    kaggle_write_state "$phase"

    echo "[kaggle] $(date -u +%H:%M:%S) elapsed=${elapsed}s phase=$phase"
    echo "$logs" | tail -8

    if [[ "$phase" == "completed" ]]; then
      echo "[kaggle] Remote training finished."
      return 0
    fi
    if [[ "$phase" == "failed" ]]; then
      echo "[kaggle] Remote kernel failed:" >&2
      echo "$logs" | tail -80
      return 1
    fi

    sleep "$interval"
    elapsed=$((elapsed + interval))
  done

  echo "[kaggle] Poll timed out after ${max_wait}s (kernel may still be running on Kaggle)." >&2
  echo "[kaggle] Run ./scripts/kaggle_train.sh resume when back online." >&2
  kaggle_write_state "running"
  return 2
}

kaggle_download_outputs() {
  mkdir -p "$OUT"
  echo "[kaggle] Downloading outputs to $OUT ..."
  "$KAGGLE" kernels output "$KERNEL_ID" -p "$OUT"

  if [[ -d "$OUT/model-final" ]]; then
    mkdir -p "$ROOT/checkpoints"
    rm -rf "$ROOT/checkpoints/pretrain_kaggle"
    cp -R "$OUT/model-final" "$ROOT/checkpoints/pretrain_kaggle"
    kaggle_write_state "downloaded" "local_checkpoint=$ROOT/checkpoints/pretrain_kaggle"
    echo "[kaggle] Model copied to $ROOT/checkpoints/pretrain_kaggle"
    return 0
  fi

  echo "[kaggle] model-final not found in outputs yet; check $OUT" >&2
  ls -la "$OUT" 2>/dev/null || true
  return 1
}

kaggle_show_status() {
  local logs phase
  logs="$(kaggle_fetch_logs 2>/dev/null || true)"
  phase="$(kaggle_run_phase "$logs")"
  echo "=== Kaggle run status ==="
  echo "kernel_id:  $KERNEL_ID"
  echo "phase:      $phase"
  echo "dataset:    $KAGGLE_DATASET"
  if [[ -f "$STATE_FILE" ]]; then
    echo "state_file: $STATE_FILE"
    cat "$STATE_FILE"
    echo ""
  fi
  echo "--- recent logs ---"
  echo "$logs" | tail -20
}
