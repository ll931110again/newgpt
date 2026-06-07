from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_yaml


def load_fleet_config(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    required = ("run_name", "train_config", "sync_every_steps", "providers")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Fleet config missing required key: {key}")
    return cfg
