from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from src.fleet.shards import assign_shards, parse_shard_ids


@dataclass(frozen=True)
class FleetEnv:
    enabled: bool
    worker_id: str
    rank: int
    world_size: int
    run_name: str
    sync_every_steps: int
    sync_mode: str  # session | continuous
    shard_ids: Optional[List[int]]
    canonical_init: Optional[str]
    store_root: Optional[str]

    @classmethod
    def from_os(cls) -> "FleetEnv":
        enabled = os.environ.get("FLEET_ENABLED", "0") == "1"
        rank = int(os.environ.get("FLEET_RANK", "0"))
        world_size = int(os.environ.get("FLEET_WORLD_SIZE", "1"))
        explicit = parse_shard_ids(os.environ.get("FLEET_SHARD_IDS"))
        shard_ids = explicit if explicit is not None else None
        if enabled and shard_ids is None and world_size > 1:
            # Shard ids assigned at orchestrator init time via FLEET_SHARD_IDS.
            pass
        return cls(
            enabled=enabled,
            worker_id=os.environ.get("FLEET_WORKER_ID", "worker-0"),
            rank=rank,
            world_size=world_size,
            run_name=os.environ.get("FLEET_RUN_NAME", "pretrain_fleet"),
            sync_every_steps=int(os.environ.get("FLEET_SYNC_EVERY_STEPS", "500")),
            sync_mode=os.environ.get("FLEET_SYNC_MODE", "session"),
            shard_ids=shard_ids,
            canonical_init=os.environ.get("FLEET_CANONICAL_INIT"),
            store_root=os.environ.get("FLEET_STORE_ROOT"),
        )

    def shard_ids_for(self, total_shards: int) -> List[int]:
        if self.shard_ids is not None:
            return list(self.shard_ids)
        return assign_shards(total_shards, self.rank, max(1, self.world_size))
