#!/usr/bin/env python3
"""Launch fleet pretraining on a remote Lightning AI Studio."""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path

from lightning_sdk import Machine, Studio


def _machine(name: str) -> Machine:
    key = name.strip().upper().replace("-", "_")
    if not hasattr(Machine, key):
        raise SystemExit(f"Unknown LIGHTNING_MACHINE={name!r}")
    return getattr(Machine, key)


def _write_state(state_file: Path, payload: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(payload, indent=2))
    print(f"[lightning] State -> {state_file}")


def _fleet_export_cmd() -> str | None:
    keys = [
        "FLEET_ENABLED",
        "FLEET_WORKER_ID",
        "FLEET_RUN_NAME",
        "FLEET_RANK",
        "FLEET_WORLD_SIZE",
        "FLEET_SHARD_IDS",
        "FLEET_SYNC_EVERY_STEPS",
        "FLEET_SYNC_MODE",
        "FLEET_TRAIN_CONFIG",
        "FLEET_CANONICAL_INIT",
        "FLEET_OUTPUT_DIR",
        "KAGGLE_API_TOKEN",
        "KAGGLE_CHECKPOINT_DATASET",
    ]
    exports = []
    for key in keys:
        val = os.environ.get(key)
        if val is not None:
            exports.append(f"export {key}={shlex.quote(val)}")
    return " ".join(exports) if exports else None


def main() -> None:
    user_id = os.environ.get("LIGHTNING_USER_ID")
    api_key = os.environ.get("LIGHTNING_API_KEY")
    if not user_id or not api_key:
        raise SystemExit("Set LIGHTNING_USER_ID and LIGHTNING_API_KEY in .env")

    worker_id = os.environ.get("LIGHTNING_WORKER_ID", "lightning-1")
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", f"pretrainer-{worker_id}")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "").strip()
    username = os.environ.get("LIGHTNING_USERNAME", "").strip()
    org = os.environ.get("LIGHTNING_ORG", "").strip()
    machine = _machine(os.environ.get("LIGHTNING_MACHINE", "T4"))
    git_url = os.environ.get(
        "LIGHTNING_WORKSPACE_GIT", "https://github.com/ll931110again/newgpt.git"
    )
    state_file = Path(
        os.environ.get(
            "LIGHTNING_STATE_FILE",
            f"runs/lightning/{worker_id}.json",
        )
    )

    print(f"[lightning] Opening Studio {studio_name!r} (machine={machine})...")
    studio_kwargs: dict = {"create_ok": True}
    if teamspace:
        studio_kwargs["teamspace"] = teamspace
    if org:
        studio_kwargs["org"] = org
    elif username:
        studio_kwargs["user"] = username
    studio = Studio(studio_name, **studio_kwargs)

    studio.start(machine)
    print(f"[lightning] Studio status: {studio.status}")

    repo_dir = "/teamspace/studios/this_studio/pretrainer"
    remote_root = "pretrainer"
    local_root = Path(__file__).resolve().parents[2]
    fleet_export = _fleet_export_cmd()
    commands = [
        "cd /teamspace/studios/this_studio",
        f'if [ ! -d pretrainer/.git ]; then git clone {shlex.quote(git_url)} pretrainer; else cd pretrainer && git pull --ff-only || true; fi',
        f"mkdir -p {remote_root}/infra/lightning {remote_root}/src {remote_root}/configs {remote_root}/data/manifests {remote_root}/data/shards",
        f"cd {repo_dir}",
    ]
    if fleet_export:
        commands.append(fleet_export)
    commands.extend(
        [
            "chmod +x infra/lightning/fleet_startup.sh",
            "nohup bash infra/lightning/fleet_startup.sh > fleet_train.log 2>&1 &",
            "echo started_pid=$!",
        ]
    )

    print("[lightning] Syncing local fleet code to Studio...")
    studio.run(f"mkdir -p {remote_root}/infra/lightning {remote_root}/src {remote_root}/configs")
    upload_pairs = [
        (local_root / "infra" / "lightning" / "fleet_startup.sh", f"{remote_root}/infra/lightning/fleet_startup.sh"),
        (local_root / "src", f"{remote_root}/src"),
        (local_root / "configs", f"{remote_root}/configs"),
        (local_root / "data" / "manifests", f"{remote_root}/data/manifests"),
        (local_root / "data" / "shards", f"{remote_root}/data/shards"),
    ]
    for local_path, remote_path in upload_pairs:
        if not local_path.exists():
            continue
        if local_path.is_file():
            studio.upload_file(str(local_path), remote_path, progress_bar=False)
        elif local_path.is_dir():
            studio.upload_folder(str(local_path), remote_path, progress_bar=False)

    print("[lightning] Starting remote training job...")
    out = studio.run(*commands)
    print(out)

    _write_state(
        state_file,
        {
            "worker_id": worker_id,
            "studio_name": studio_name,
            "teamspace": teamspace or None,
            "user": username or None,
            "org": org or None,
            "machine": os.environ.get("LIGHTNING_MACHINE", "T4"),
            "status": "running",
            "remote_output": f"{repo_dir}/runs/lightning-output",
            "started_at": time.time(),
        },
    )
    print(f"[lightning] Remote studio running: {studio_name}")


if __name__ == "__main__":
    main()
