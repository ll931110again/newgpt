from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.fleet.providers.base import Provider, ProviderStatus


class LightningProvider(Provider):
    provider_type = "lightning"

    def __init__(self, worker_id: str, config: Dict[str, Any], repo_root: Path):
        super().__init__(worker_id, config, repo_root)
        self._env_file = self.repo_root / "infra" / "lightning" / "fleet_worker.json"
        self._out = self.repo_root / "checkpoints" / f"lightning_{worker_id}"
        self._state = self.repo_root / "runs" / "lightning" / f"{worker_id}.json"

    def _write_fleet_env(self, env: Dict[str, str]) -> None:
        self._env_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"worker_id": self.worker_id, **env}
        self._env_file.write_text(json.dumps(payload, indent=2))

    def launch(self, env: Dict[str, str]) -> None:
        if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
            raise RuntimeError("LIGHTNING_USER_ID and LIGHTNING_API_KEY are required")

        self._write_fleet_env(env)
        merged = os.environ.copy()
        merged.update(env)
        merged["LIGHTNING_WORKER_ID"] = self.worker_id
        merged["FLEET_SYNC_MODE"] = env.get("FLEET_SYNC_MODE", "continuous")
        merged["FLEET_OUTPUT_DIR"] = "/teamspace/studios/this_studio/pretrainer/runs/lightning-output"
        if self.config.get("studio_name"):
            merged["LIGHTNING_STUDIO_NAME"] = str(self.config["studio_name"])
        if self.config.get("teamspace"):
            merged["LIGHTNING_TEAMSPACE"] = str(self.config["teamspace"])
        if self.config.get("user"):
            merged["LIGHTNING_USERNAME"] = str(self.config["user"])
        if self.config.get("org"):
            merged["LIGHTNING_ORG"] = str(self.config["org"])
        if self.config.get("machine"):
            merged["LIGHTNING_MACHINE"] = str(self.config["machine"])
        if self.config.get("workspace_git"):
            merged["LIGHTNING_WORKSPACE_GIT"] = str(self.config["workspace_git"])
        if os.environ.get("KAGGLE_API_TOKEN"):
            merged["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_API_TOKEN"]

        script = self.repo_root / "scripts" / "lightning_train.sh"
        subprocess.Popen(
            [str(script), "start"],
            cwd=str(self.repo_root),
            env=merged,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def status(self) -> ProviderStatus:
        script = self.repo_root / "scripts" / "lightning_train.sh"
        merged = os.environ.copy()
        merged["LIGHTNING_WORKER_ID"] = self.worker_id
        try:
            out = subprocess.check_output(
                [str(script), "status"],
                cwd=str(self.repo_root),
                env=merged,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            return ProviderStatus(self.worker_id, "failed", exc.output or str(exc))
        phase = "unknown"
        for line in out.splitlines():
            if line.startswith("phase:"):
                phase = line.split(":", 1)[1].strip()
                break
        return ProviderStatus(self.worker_id, phase, out[-200:] if out else "")

    def fetch_latest_checkpoint(self, dest: Path) -> Optional[Path]:
        script = self.repo_root / "scripts" / "lightning_train.sh"
        merged = os.environ.copy()
        merged["LIGHTNING_WORKER_ID"] = self.worker_id
        subprocess.run(
            [str(script), "download"],
            cwd=str(self.repo_root),
            env=merged,
            check=False,
        )
        src = self._out
        if not src.is_dir():
            return None
        ckpts = sorted(
            (p for p in src.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if ckpts:
            latest = ckpts[-1]
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-R", str(latest), str(dest / latest.name)], check=True)
            return dest / latest.name
        final = src / "final"
        if final.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-R", str(final), str(dest / "final")], check=True)
            return dest / "final"
        return None
