from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProviderStatus:
    worker_id: str
    phase: str  # pending | running | completed | failed | dead | unknown
    detail: str = ""


class Provider(ABC):
    provider_type: str

    def __init__(self, worker_id: str, config: Dict[str, Any], repo_root: Path):
        self.worker_id = worker_id
        self.config = config
        self.repo_root = repo_root

    @abstractmethod
    def launch(self, env: Dict[str, str]) -> None: ...

    @abstractmethod
    def status(self) -> ProviderStatus: ...

    @abstractmethod
    def fetch_latest_checkpoint(self, dest: Path) -> Optional[Path]: ...

    def stop(self) -> None:
        """Best-effort stop; optional for batch providers."""
