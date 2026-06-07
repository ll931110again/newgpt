from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set


def assign_shards(total_shards: int, rank: int, world_size: int) -> List[int]:
    """Assign shard indices to a worker by round-robin partition."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    return [i for i in range(total_shards) if i % world_size == rank]


def parse_shard_ids(raw: str | None) -> List[int] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def filter_shards_by_ids(shards: Sequence[Dict], shard_ids: Iterable[int]) -> List[Dict]:
    wanted: Set[int] = set(shard_ids)
    out: List[Dict] = []
    for idx, shard in enumerate(shards):
        if idx in wanted:
            out.append(shard)
    return out


def reassign_shards_after_failure(
    total_shards: int,
    workers: Sequence[Dict[str, object]],
) -> Dict[str, List[int]]:
    """Reassign all shards from alive workers; dead workers get empty lists."""
    alive = [w for w in workers if w.get("status") in {"pending", "running"}]
    if not alive:
        return {str(w["worker_id"]): [] for w in workers}

    world_size = len(alive)
    assignments: Dict[str, List[int]] = {}
    for new_rank, worker in enumerate(sorted(alive, key=lambda w: int(w["rank"]))):  # type: ignore[arg-type]
        wid = str(worker["worker_id"])
        assignments[wid] = assign_shards(total_shards, new_rank, world_size)

    for worker in workers:
        wid = str(worker["worker_id"])
        if wid not in assignments:
            assignments[wid] = []
    return assignments
