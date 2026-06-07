from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.fleet.providers.base import Provider, ProviderStatus


class PaperspaceProvider(Provider):
    """Launch a remote Paperspace Gradient notebook (same pattern as Kaggle)."""

    provider_type = "paperspace"

    def __init__(self, worker_id: str, config: Dict[str, Any], repo_root: Path):
        super().__init__(worker_id, config, repo_root)
        self.local_mode = bool(config.get("local", False))
        self._env_file = self.repo_root / "infra" / "paperspace" / "fleet_worker.json"
        self._out = self.repo_root / "checkpoints" / f"paperspace_{worker_id}"
        self._state = self.repo_root / "runs" / "paperspace" / f"{worker_id}.json"

    def _write_fleet_env(self, env: Dict[str, str]) -> None:
        self._env_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"worker_id": self.worker_id, **env}
        self._env_file.write_text(json.dumps(payload, indent=2))

    def launch(self, env: Dict[str, str]) -> None:
        if self.local_mode:
            self._launch_local(env)
            return

        if not os.environ.get("PAPERSPACE_API_KEY"):
            raise RuntimeError("PAPERSPACE_API_KEY is required for remote Paperspace workers")

        self._write_fleet_env(env)
        merged = os.environ.copy()
        merged.update(env)
        merged["PAPERSPACE_WORKER_ID"] = self.worker_id
        merged["FLEET_SYNC_MODE"] = env.get("FLEET_SYNC_MODE", "continuous")
        if os.environ.get("KAGGLE_API_TOKEN"):
            merged["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_API_TOKEN"]
        if self.config.get("project_id"):
            merged["PAPERSPACE_PROJECT_ID"] = str(self.config["project_id"])
        if self.config.get("machine_type"):
            merged["PAPERSPACE_MACHINE_TYPE"] = str(self.config["machine_type"])
        if self.config.get("workspace_git"):
            merged["PAPERSPACE_WORKSPACE_GIT"] = str(self.config["workspace_git"])

        script = self.repo_root / "scripts" / "paperspace_train.sh"
        subprocess.Popen(
            [str(script), "start"],
            cwd=str(self.repo_root),
            env=merged,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _launch_local(self, env: Dict[str, str]) -> None:
        """Optional dev-only path; not used in fleet.yaml by default."""
        out = self.repo_root / "runs" / "paperspace" / self.worker_id
        out.mkdir(parents=True, exist_ok=True)
        merged = os.environ.copy()
        merged.update(env)
        merged["FLEET_SYNC_MODE"] = env.get("FLEET_SYNC_MODE", "continuous")
        merged["FLEET_STORE_ROOT"] = str(self.repo_root / "runs" / "fleet")
        train_config = env.get("FLEET_TRAIN_CONFIG", "configs/pretrain_continue_from_current.yaml")
        subprocess.Popen(
            [
                "uv",
                "run",
                "python3",
                "-m",
                "src.train.pretrain",
                "--config",
                train_config,
                "--output_dir",
                str(out),
            ],
            cwd=str(self.repo_root),
            env=merged,
        )

    def status(self) -> ProviderStatus:
        if self.local_mode:
            out = self.repo_root / "runs" / "paperspace" / self.worker_id
            hb = out / "fleet" / "heartbeat.json"
            if hb.is_file():
                data = json.loads(hb.read_text())
                return ProviderStatus(self.worker_id, "running", f"step={data.get('step')}")
            return ProviderStatus(self.worker_id, "pending")

        script = self.repo_root / "scripts" / "paperspace_train.sh"
        merged = os.environ.copy()
        merged["PAPERSPACE_WORKER_ID"] = self.worker_id
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
        if not self.local_mode:
            script = self.repo_root / "scripts" / "paperspace_train.sh"
            merged = os.environ.copy()
            merged["PAPERSPACE_WORKER_ID"] = self.worker_id
            subprocess.run(
                [str(script), "download"],
                cwd=str(self.repo_root),
                env=merged,
                check=False,
            )

        src = self._out
        if not src.is_dir():
            src = self.repo_root / "runs" / "paperspace" / self.worker_id
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
