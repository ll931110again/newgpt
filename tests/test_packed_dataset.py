"""Tests for packed token dataset packing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.packed_dataset import (
    PackedTokensIterableDataset,
    pack_tokens_into_lm_batches,
)


def _pack_legacy(token_iter, seq_len: int):
    need = seq_len + 1
    buf: list[int] = []
    for t in token_iter:
        buf.append(int(t))
        if len(buf) >= need:
            chunk = buf[:need]
            buf = buf[need:]
            yield {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }


def test_pack_tokens_matches_legacy_implementation():
    tokens = list(range(100))
    seq_len = 16
    new_batches = list(pack_tokens_into_lm_batches(iter(tokens), seq_len))
    old_batches = list(_pack_legacy(iter(tokens), seq_len))
    assert len(new_batches) == len(old_batches)
    for a, b in zip(new_batches, old_batches):
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["labels"], b["labels"])


def test_pack_tokens_label_shift():
    batches = list(pack_tokens_into_lm_batches(iter([10, 11, 12, 13, 14, 15]), seq_len=2))
    assert len(batches) == 2
    assert batches[0]["input_ids"].tolist() == [10, 11]
    assert batches[0]["labels"].tolist() == [11, 12]


def test_packed_dataset_from_npy_shard(tmp_path: Path):
    seq_len = 8
    tokens = np.arange(40, dtype=np.int32)
    shard_path = tmp_path / "shard0.npy"
    np.save(shard_path, tokens)

    manifest = {
        "dataset_version": "vtest",
        "format": "npy_int32_tokens",
        "tokenizer": "gpt2",
        "total_tokens": int(tokens.size),
        "shards": [{"path": str(shard_path), "num_tokens": int(tokens.size)}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    ds = PackedTokensIterableDataset(str(manifest_path), seq_len=seq_len)
    examples = list(ds)
    expected = list(_pack_legacy(iter(tokens.tolist()), seq_len))
    assert len(examples) == len(expected)
    for got, want in zip(examples, expected):
        assert torch.equal(got["input_ids"], want["input_ids"])
        assert torch.equal(got["labels"], want["labels"])


def test_packed_dataset_fleet_shard_filter(tmp_path: Path):
    seq_len = 4
    shards = []
    for i in range(3):
        tokens = np.arange((i + 1) * 10, (i + 2) * 10, dtype=np.int32)
        p = tmp_path / f"s{i}.npy"
        np.save(p, tokens)
        shards.append({"path": str(p), "num_tokens": int(tokens.size)})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "vtest",
                "format": "npy_int32_tokens",
                "tokenizer": "gpt2",
                "total_tokens": sum(s["num_tokens"] for s in shards),
                "shards": shards,
            }
        )
    )
    ds_all = PackedTokensIterableDataset(str(manifest_path), seq_len=seq_len)
    ds_fleet = PackedTokensIterableDataset(
        str(manifest_path), seq_len=seq_len, shard_ids=[0, 2]
    )
    assert len(list(ds_all)) > len(list(ds_fleet))
    ds_rank = PackedTokensIterableDataset(
        str(manifest_path), seq_len=seq_len, fleet_rank=1, fleet_world_size=2
    )
    assert len(list(ds_rank)) < len(list(ds_all))


def test_shards_for_worker_partitions_without_overlap():
    from torch.utils.data import get_worker_info

    ds = PackedTokensIterableDataset.__new__(PackedTokensIterableDataset)
    ds.manifest = type("M", (), {"shards": [{"path": f"s{i}"} for i in range(6)]})()
    ds._shard_ids = None

    class FakeInfo:
        id = 1
        num_workers = 3

    import src.data.packed_dataset as pd

    original = pd.get_worker_info
    pd.get_worker_info = lambda: FakeInfo()
    try:
        got = ds._shards_for_worker()
    finally:
        pd.get_worker_info = original

    assert got == [{"path": "s1"}, {"path": "s4"}]


def test_packed_dataset_carries_remainder_across_shard_boundary(tmp_path: Path):
    seq_len = 4
    s0 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    s1 = np.array([6, 7, 8, 9, 10, 11], dtype=np.int32)
    p0 = tmp_path / "s0.npy"
    p1 = tmp_path / "s1.npy"
    np.save(p0, s0)
    np.save(p1, s1)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "vtest",
                "format": "npy_int32_tokens",
                "tokenizer": "gpt2",
                "total_tokens": int(s0.size + s1.size),
                "shards": [
                    {"path": str(p0), "num_tokens": int(s0.size)},
                    {"path": str(p1), "num_tokens": int(s1.size)},
                ],
            }
        )
    )
    all_tokens = list(s0.tolist()) + list(s1.tolist())
    expected = list(_pack_legacy(iter(all_tokens), seq_len))
    got = list(PackedTokensIterableDataset(str(manifest_path), seq_len=seq_len))
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert torch.equal(a["input_ids"], b["input_ids"])
