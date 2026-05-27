"""Tests for throughput metric helpers."""

from __future__ import annotations

import pytest

from src.train.metrics import effective_batch_tokens, tokens_per_second, tokens_per_step


def test_tokens_per_step():
    assert tokens_per_step(12, 2048, world_size=1) == 12 * 2048


def test_tokens_per_second():
    tps = tokens_per_second(4, 1024, step_time_seconds=0.5, world_size=2)
    assert tps == (4 * 1024 * 2) / 0.5


def test_effective_batch_tokens():
    assert effective_batch_tokens(8, 512, gradient_accumulation_steps=4) == 8 * 512 * 4


def test_tokens_per_second_rejects_non_positive_time():
    with pytest.raises(ValueError):
        tokens_per_second(1, 128, step_time_seconds=0.0)
