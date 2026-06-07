from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.fleet.providers.base import Provider, ProviderStatus


class KaggleProvider(Provider):
    provider_type = "kaggle"

    def __init__(self, worker_id: str, config: Dict[str, Any], repo_root: Path):
        super().__init__(worker_id, config, repo_root)
        self._env_file = self.repo_root / "infra" / "kaggle" / "dataset" / "fleet_worker.json"
        self._out = self.repo_root / "checkpoints" / "kaggle_pretrain"

    def _write_fleet_env(self, env: Dict[str, str]) -> None:
        self._env_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"worker_id": self.worker_id, **env}
        self._env_file.write_text(json.dumps(payload, indent=2))

    def launch(self, env: Dict[str, str]) -> None:
        self._write_fleet_env(env)
        merged = os.environ.copy()
        merged.update(env)
        merged["FLEET_SYNC_MODE"] = env.get("FLEET_SYNC_MODE", "session")
        script = self.repo_root / "scripts" / "kaggle_train.sh"
        subprocess.Popen(
            [str(script), "start"],
            cwd=str(self.repo_root),
            env=merged,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def status(self) -> ProviderStatus:
        script = self.repo_root / "scripts" / "kaggle_train.sh"
        try:
            out = subprocess.check_output(
                [str(script), "status"],
                cwd=str(self.repo_root),
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
        script = self.repo_root / "scripts" / "kaggle_train.sh"
        subprocess.run([str(script), "download"], cwd=str(self.repo_root), check=False)
        candidates = [
            self._out / "checkpoints" / "pretrain_kaggle",
            self.repo_root / "checkpoints" / "pretrain_kaggle",
        ]
        for base in candidates:
            if not base.is_dir():
                continue
            ckpts = sorted(
                (p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if ckpts:
                latest = ckpts[-1]
                dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(["cp", "-R", str(latest), str(dest / latest.name)], check=True)
                return dest / latest.name
            final = base / "final"
            if final.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(["cp", "-R", str(final), str(dest / "final")], check=True)
                return dest / "final"
        return None
