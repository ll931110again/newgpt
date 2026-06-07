from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.fleet.shards import filter_shards_by_ids, parse_shard_ids


@dataclass(frozen=True)
class Manifest:
    dataset_version: str
    format: str
    tokenizer: str
    shards: List[Dict]
    total_tokens: int

    @staticmethod
    def load(path: str) -> "Manifest":
        data = json.loads(Path(path).read_text())
        return Manifest(
            dataset_version=data["dataset_version"],
            format=data["format"],
            tokenizer=data.get("tokenizer", ""),
            shards=list(data["shards"]),
            total_tokens=int(data.get("total_tokens", 0)),
        )


def pack_tokens_into_lm_batches(
    token_iter: Iterator[int], seq_len: int
) -> Iterator[Dict[str, torch.Tensor]]:
    """Pack a token stream into non-overlapping (input_ids, labels) examples."""
    need = int(seq_len) + 1
    carry: List[int] = []
    for tok in token_iter:
        carry.append(int(tok))
        while len(carry) >= need:
            chunk = np.asarray(carry[:need], dtype=np.int64)
            carry = carry[need:]
            yield {
                "input_ids": torch.from_numpy(chunk[:-1]).long(),
                "labels": torch.from_numpy(chunk[1:]).long(),
            }


class PackedTokensIterableDataset(IterableDataset):
    def __init__(
        self,
        manifest_path: str,
        seq_len: int,
        shard_ids: Optional[List[int]] = None,
        fleet_rank: Optional[int] = None,
        fleet_world_size: Optional[int] = None,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.seq_len = int(seq_len)
        self.manifest = Manifest.load(manifest_path)
        if self.manifest.format != "npy_int32_tokens":
            raise ValueError(f"Unsupported format: {self.manifest.format}")
        self._shard_ids = self._resolve_shard_ids(shard_ids, fleet_rank, fleet_world_size)

    def _resolve_shard_ids(
        self,
        shard_ids: Optional[List[int]],
        fleet_rank: Optional[int],
        fleet_world_size: Optional[int],
    ) -> Optional[List[int]]:
        if shard_ids is not None:
            return list(shard_ids)
        env_ids = parse_shard_ids(os.environ.get("FLEET_SHARD_IDS"))
        if env_ids is not None:
            return env_ids
        rank = fleet_rank
        world_size = fleet_world_size
        if rank is None:
            rank = int(os.environ["FLEET_RANK"]) if os.environ.get("FLEET_RANK") else None
        if world_size is None:
            world_size = (
                int(os.environ["FLEET_WORLD_SIZE"]) if os.environ.get("FLEET_WORLD_SIZE") else None
            )
        if rank is not None and world_size is not None and world_size > 1:
            from src.fleet.shards import assign_shards

            return assign_shards(len(self.manifest.shards), rank, world_size)
        return None

    def _shards_for_worker(self) -> List[Dict]:
        shards = list(self.manifest.shards)
        if self._shard_ids is not None:
            shards = filter_shards_by_ids(shards, self._shard_ids)
        info = get_worker_info()
        if info is None or info.num_workers <= 1:
            return shards
        return [s for i, s in enumerate(shards) if i % info.num_workers == info.id]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        carry: List[int] = []
        need = self.seq_len + 1
        for shard in self._shards_for_worker():
            arr = np.load(shard["path"], mmap_mode="r")
            flat = np.asarray(arr, dtype=np.int64).reshape(-1)
            if carry:
                flat = np.concatenate([np.asarray(carry, dtype=np.int64), flat])
                carry = []

            n = (len(flat) // need) * need
            for start in range(0, n, need):
                chunk = flat[start : start + need]
                yield {
                    "input_ids": torch.from_numpy(chunk[:-1]).long(),
                    "labels": torch.from_numpy(chunk[1:]).long(),
                }
            if n < len(flat):
                carry = flat[n:].tolist()
            else:
                carry = []

        if carry:
            yield from pack_tokens_into_lm_batches(iter(carry), self.seq_len)
