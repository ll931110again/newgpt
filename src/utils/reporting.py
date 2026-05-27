"""Select experiment trackers (W&B, mlop) for Hugging Face Trainer."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.utils.wandb_setup import configure_wandb


def build_report_to(cfg: Dict[str, Any]) -> List[str]:
    integrations: List[str] = []
    if configure_wandb(cfg):
        integrations.append("wandb")
    if _configure_mlop(cfg):
        integrations.append("mlop")
    return integrations


def _configure_mlop(cfg: Dict[str, Any]) -> bool:
    try:
        from src.utils.mlop_setup import configure_mlop

        return configure_mlop(cfg)
    except ImportError:
        return False


def finish_reporting(cfg: Dict[str, Any]) -> None:
    logging_cfg = cfg.get("logging", {}) or {}
    if not (logging_cfg.get("mlop_project") or os.environ.get("MLOP_PROJECT")):
        return
    if not os.environ.get("MLOP_API_KEY"):
        return
    try:
        from src.utils.mlop_setup import finish_mlop

        finish_mlop()
    except ImportError:
        pass
