from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class WorkerRecord:
    worker_id: str
    provider_type: str
    rank: int
    shard_ids: List[int] = field(default_factory=list)
    status: str = "pending"  # pending | running | dead | completed
    last_step: int = 0
    last_heartbeat_ts: float = 0.0
    last_checkpoint: Optional[str] = None
    pending_reload: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerRecord":
        return cls(
            worker_id=str(data["worker_id"]),
            provider_type=str(data["provider_type"]),
            rank=int(data["rank"]),
            shard_ids=[int(x) for x in data.get("shard_ids", [])],
            status=str(data.get("status", "pending")),
            last_step=int(data.get("last_step", 0)),
            last_heartbeat_ts=float(data.get("last_heartbeat_ts", 0.0)),
            last_checkpoint=data.get("last_checkpoint"),
            pending_reload=bool(data.get("pending_reload", False)),
        )


@dataclass
class FleetState:
    run_name: str
    train_config: str
    sync_every_steps: int
    sync_timeout_minutes: int
    sync_generation: int = 0
    canonical_step: int = 0
    canonical_path: Optional[str] = None
    total_shards: int = 0
    workers: List[WorkerRecord] = field(default_factory=list)
    updated_ts: float = field(default_factory=time.time)

    @property
    def world_size(self) -> int:
        return len(self.workers)

    def worker_by_id(self, worker_id: str) -> Optional[WorkerRecord]:
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        return None

    def alive_workers(self) -> List[WorkerRecord]:
        return [w for w in self.workers if w.status in {"pending", "running"}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "train_config": self.train_config,
            "sync_every_steps": self.sync_every_steps,
            "sync_timeout_minutes": self.sync_timeout_minutes,
            "sync_generation": self.sync_generation,
            "canonical_step": self.canonical_step,
            "canonical_path": self.canonical_path,
            "total_shards": self.total_shards,
            "workers": [asdict(w) for w in self.workers],
            "updated_ts": self.updated_ts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FleetState":
        workers = [WorkerRecord.from_dict(w) for w in data.get("workers", [])]
        return cls(
            run_name=str(data["run_name"]),
            train_config=str(data["train_config"]),
            sync_every_steps=int(data["sync_every_steps"]),
            sync_timeout_minutes=int(data.get("sync_timeout_minutes", 30)),
            sync_generation=int(data.get("sync_generation", 0)),
            canonical_step=int(data.get("canonical_step", 0)),
            canonical_path=data.get("canonical_path"),
            total_shards=int(data.get("total_shards", 0)),
            workers=workers,
            updated_ts=float(data.get("updated_ts", time.time())),
        )

    def save(self, path: Path) -> None:
        self.updated_ts = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "FleetState":
        if not path.exists():
            raise FileNotFoundError(path)
        return cls.from_dict(json.loads(path.read_text()))
