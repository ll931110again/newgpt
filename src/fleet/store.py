from __future__ import annotations

import json
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.fleet.state import FleetState, WorkerRecord


class FleetStore(ABC):
    @abstractmethod
    def root(self) -> Path: ...

    @abstractmethod
    def load_state(self) -> FleetState: ...

    @abstractmethod
    def save_state(self, state: FleetState) -> None: ...

    @abstractmethod
    def worker_checkpoint_dir(self, worker_id: str) -> Path: ...

    @abstractmethod
    def canonical_dir(self) -> Path: ...

    @abstractmethod
    def publish_worker_checkpoint(self, worker_id: str, local_ckpt: Path, step: int) -> Path: ...

    @abstractmethod
    def write_reload_signal(self, sync_generation: int) -> None: ...

    @abstractmethod
    def reload_ready(self, sync_generation: int) -> bool: ...

    @abstractmethod
    def heartbeat_path(self, worker_id: str) -> Path: ...

    def write_heartbeat(self, worker_id: str, step: int, checkpoint: Optional[str] = None) -> None:
        payload = {
            "worker_id": worker_id,
            "step": step,
            "checkpoint": checkpoint,
            "ts": time.time(),
        }
        path = self.heartbeat_path(worker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def read_heartbeat(self, worker_id: str) -> Optional[Dict[str, Any]]:
        path = self.heartbeat_path(worker_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def list_worker_checkpoints(self, worker_id: str) -> List[Path]:
        base = self.worker_checkpoint_dir(worker_id)
        if not base.is_dir():
            return []
        return sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint-"))


class LocalFleetStore(FleetStore):
    def __init__(self, run_name: str, base_dir: Path | None = None):
        self.run_name = run_name
        self._root = (base_dir or Path("runs/fleet")) / run_name

    def root(self) -> Path:
        return self._root

    def state_path(self) -> Path:
        return self._root / "state.json"

    def load_state(self) -> FleetState:
        return FleetState.load(self.state_path())

    def save_state(self, state: FleetState) -> None:
        state.save(self.state_path())

    def worker_checkpoint_dir(self, worker_id: str) -> Path:
        return self._root / "workers" / worker_id

    def canonical_dir(self) -> Path:
        return self._root / "canonical"

    def publish_worker_checkpoint(self, worker_id: str, local_ckpt: Path, step: int) -> Path:
        local_ckpt = Path(local_ckpt)
        dest = self.worker_checkpoint_dir(worker_id) / local_ckpt.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local_ckpt, dest)
        meta = {"step": step, "path": str(dest), "ts": time.time()}
        (dest.parent / f"{local_ckpt.name}.meta.json").write_text(json.dumps(meta, indent=2))
        return dest

    def reload_signal_path(self) -> Path:
        return self._root / "signals" / "reload.json"

    def write_reload_signal(self, sync_generation: int) -> None:
        path = self.reload_signal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        canonical = self.canonical_dir()
        payload = {
            "sync_generation": sync_generation,
            "canonical_path": str(canonical),
            "ts": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2))

    def reload_ready(self, sync_generation: int) -> bool:
        path = self.reload_signal_path()
        if not path.is_file():
            return False
        data = json.loads(path.read_text())
        return int(data.get("sync_generation", -1)) >= sync_generation

    def heartbeat_path(self, worker_id: str) -> Path:
        return self._root / "heartbeats" / f"{worker_id}.json"

    def init_run(
        self,
        train_config: str,
        sync_every_steps: int,
        sync_timeout_minutes: int,
        workers: List[WorkerRecord],
        total_shards: int,
    ) -> FleetState:
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "workers").mkdir(exist_ok=True)
        (self._root / "heartbeats").mkdir(exist_ok=True)
        (self._root / "signals").mkdir(exist_ok=True)
        self.canonical_dir().mkdir(exist_ok=True)
        state = FleetState(
            run_name=self.run_name,
            train_config=train_config,
            sync_every_steps=sync_every_steps,
            sync_timeout_minutes=sync_timeout_minutes,
            total_shards=total_shards,
            workers=workers,
        )
        self.save_state(state)
        return state


def fleet_store_from_config(cfg: Dict[str, Any]) -> FleetStore:
    backend = str(cfg.get("store_backend", "local")).lower()
    run_name = str(cfg["run_name"])
    if backend == "local":
        base = cfg.get("store_base_dir")
        base_dir = Path(base) if base else None
        return LocalFleetStore(run_name=run_name, base_dir=base_dir)
    if backend == "s3":
        raise NotImplementedError("S3FleetStore is planned for phase 2")
    raise ValueError(f"Unknown fleet store backend: {backend}")
