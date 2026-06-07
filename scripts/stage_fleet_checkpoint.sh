#!/usr/bin/env bash
# Stage the latest local checkpoint for fleet + Kaggle bundle upload.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="${FLEET_RUN_NAME:-pretrain_fleet_v1}"

latest="$(uv run python3 - <<'PY'
from pathlib import Path

root = Path("checkpoints")
best = None
best_key = (-1, 0.0)
for path in root.rglob("*"):
    if not path.is_dir():
        continue
    if path.name.startswith("checkpoint-"):
        try:
            step = int(path.name.split("-", 1)[1])
        except ValueError:
            continue
    elif path.name == "final":
        step = -1
    else:
        continue
    if not (path / "model.safetensors").is_file() and not (path / "pytorch_model.bin").is_file():
        continue
    key = (step, path.stat().st_mtime)
    if key > best_key:
        best_key = key
        best = path
if best is None:
    raise SystemExit("No checkpoint with model weights found under checkpoints/")
print(best)
PY
)"

echo "[stage] Latest checkpoint: $latest"

CANONICAL_REL="checkpoints/canonical"
BUNDLE_CANON="$ROOT/infra/kaggle/dataset/$CANONICAL_REL"
FLEET_CANON="$ROOT/runs/fleet/$RUN_NAME/canonical/checkpoint-latest"

rm -rf "$BUNDLE_CANON" "$FLEET_CANON"
mkdir -p "$BUNDLE_CANON" "$FLEET_CANON"

for f in model.safetensors pytorch_model.bin config.json generation_config.json tokenizer.json tokenizer_config.json; do
  if [[ -f "$latest/$f" ]]; then
    cp "$latest/$f" "$BUNDLE_CANON/"
    cp "$latest/$f" "$FLEET_CANON/"
  fi
done

step="$(basename "$latest" | sed 's/checkpoint-//')"
if [[ "$(basename "$latest")" == final ]]; then
  step="final"
fi

cat > "$ROOT/runs/fleet/$RUN_NAME/canonical/latest.json" <<EOF
{
  "source": "$latest",
  "step": "$step",
  "bundle_path": "$CANONICAL_REL"
}
EOF

echo "[stage] Bundled -> $BUNDLE_CANON"
echo "[stage] Fleet canonical -> $FLEET_CANON"
echo "$CANONICAL_REL"
