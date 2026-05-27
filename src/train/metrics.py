"""Training throughput metrics (6.S894-style measurement helpers)."""

from __future__ import annotations


def tokens_per_step(per_device_batch_size: int, seq_len: int, world_size: int = 1) -> int:
    """Tokens processed per optimizer step (all ranks)."""
    return int(per_device_batch_size) * int(seq_len) * int(world_size)


def tokens_per_second(
    per_device_batch_size: int,
    seq_len: int,
    step_time_seconds: float,
    world_size: int = 1,
) -> float:
    if step_time_seconds <= 0:
        raise ValueError("step_time_seconds must be positive")
    return tokens_per_step(per_device_batch_size, seq_len, world_size) / step_time_seconds


def effective_batch_tokens(
    per_device_batch_size: int,
    seq_len: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """Tokens per optimizer update including gradient accumulation."""
    return (
        tokens_per_step(per_device_batch_size, seq_len, world_size)
        * int(gradient_accumulation_steps)
    )
