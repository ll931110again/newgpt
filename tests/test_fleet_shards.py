from __future__ import annotations

from src.fleet.shards import assign_shards, reassign_shards_after_failure


def test_assign_shards_round_robin():
    assert assign_shards(6, rank=0, world_size=3) == [0, 3]
    assert assign_shards(6, rank=1, world_size=3) == [1, 4]
    assert assign_shards(6, rank=2, world_size=3) == [2, 5]


def test_reassign_shards_after_failure():
    workers = [
        {"worker_id": "kaggle-0", "rank": 0, "status": "dead"},
        {"worker_id": "paperspace-1", "rank": 1, "status": "running"},
    ]
    out = reassign_shards_after_failure(4, workers)
    assert out["kaggle-0"] == []
    assert out["paperspace-1"] == [0, 1, 2, 3]
