from __future__ import annotations

import torch

from src.fleet.merge import fedavg_state_dicts


def test_fedavg_averages_two_state_dicts():
    sd0 = {"w": torch.tensor([0.0, 2.0])}
    sd1 = {"w": torch.tensor([2.0, 4.0])}
    merged = fedavg_state_dicts([sd0, sd1])
    assert torch.allclose(merged["w"], torch.tensor([1.0, 3.0]))


def test_fedavg_single_state_dict_passthrough():
    sd = {"w": torch.tensor([1.0, 2.0])}
    merged = fedavg_state_dicts([sd])
    assert torch.allclose(merged["w"], sd["w"])
