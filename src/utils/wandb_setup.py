"""Configure Weights & Biases from YAML config + environment variables."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.utils.config import deep_get


def configure_wandb(cfg: Dict[str, Any]) -> bool:
    """Return True if W&B reporting should be enabled."""
    logging_cfg = cfg.get("logging", {}) or {}
    project: Optional[str] = logging_cfg.get("wandb_project") or os.environ.get("WANDB_PROJECT")
    if not project:
        return False

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("[wandb] WANDB_API_KEY not set; skipping W&B logging")
        return False

    os.environ.setdefault("WANDB_PROJECT", str(project))
    entity = logging_cfg.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
    if entity:
        os.environ.setdefault("WANDB_ENTITY", str(entity))

    run_name = cfg.get("run_name") or logging_cfg.get("wandb_run_name") or "pretrain"
    os.environ.setdefault("WANDB_RUN_NAME", str(run_name))

    # Offline mode for air-gapped debugging
    if os.environ.get("WANDB_MODE") == "offline":
        print("[wandb] offline mode")

    print(f"[wandb] enabled project={os.environ['WANDB_PROJECT']} run={run_name}")
    return True
