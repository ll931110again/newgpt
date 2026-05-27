"""Tests for training performance helpers."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.train.performance import (
    model_load_kwargs,
    resolve_attn_implementation,
    training_args_performance_kwargs,
)


def test_resolve_attn_auto_falls_back_to_sdpa_without_flash_attn(monkeypatch):
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    cfg = {"train": {"attn_implementation": "auto"}}
    assert resolve_attn_implementation(cfg) == "sdpa"


def test_resolve_attn_auto_uses_flash_when_available(monkeypatch):
    fake = ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", fake)
    cfg = {"train": {"attn_implementation": "auto"}}
    assert resolve_attn_implementation(cfg) == "flash_attention_2"


def test_resolve_attn_flash_fallback(monkeypatch):
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    cfg = {"train": {"attn_implementation": "flash_attention_2"}}
    assert resolve_attn_implementation(cfg) == "sdpa"


def test_model_load_kwargs_includes_attn():
    cfg = {"train": {"attn_implementation": "sdpa"}}
    assert model_load_kwargs(cfg) == {"attn_implementation": "sdpa"}


def test_training_args_performance_kwargs_workers():
    cfg = {
        "train": {
            "optim": "adamw_torch_fused",
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 3,
        }
    }
    kw = training_args_performance_kwargs(cfg)
    assert kw["optim"] == "adamw_torch_fused"
    assert kw["dataloader_num_workers"] == 4
    assert kw["dataloader_pin_memory"] is True
    assert kw["dataloader_prefetch_factor"] == 3


def test_training_args_pin_memory_off_without_workers():
    cfg = {"train": {"dataloader_num_workers": 0, "dataloader_pin_memory": True}}
    kw = training_args_performance_kwargs(cfg)
    assert kw["dataloader_num_workers"] == 0
    assert kw["dataloader_pin_memory"] is False
    assert "dataloader_prefetch_factor" not in kw


def test_resolve_attn_invalid():
    with pytest.raises(ValueError, match="Unknown"):
        resolve_attn_implementation({"train": {"attn_implementation": "not_real"}})
