from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.fleet.providers.base import Provider
from src.fleet.providers.kaggle import KaggleProvider
from src.fleet.providers.lightning import LightningProvider
from src.fleet.providers.paperspace import PaperspaceProvider


def build_provider(worker_id: str, provider_type: str, config: Dict[str, Any], repo_root: Path) -> Provider:
    if provider_type == "kaggle":
        return KaggleProvider(worker_id, config, repo_root)
    if provider_type == "lightning":
        return LightningProvider(worker_id, config, repo_root)
    if provider_type == "paperspace":
        return PaperspaceProvider(worker_id, config, repo_root)
    raise ValueError(
        f"Unknown provider type: {provider_type} (supported: kaggle, lightning, paperspace)"
    )
