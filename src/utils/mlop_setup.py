"""Configure mlop.ai experiment tracking (https://docs.mlop.ai)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.utils.config import deep_get


def ensure_mlop_transformers_integration() -> None:
    """Register MLOPCallback with Hugging Face before TrainingArguments."""
    from mlop.compat.transformers import MLOPCallback  # noqa: F401


def configure_mlop(cfg: Dict[str, Any]) -> bool:
    """Pre-init an mlop run when MLOP_API_KEY is set. Returns True if enabled."""
    logging_cfg = cfg.get("logging", {}) or {}
    project: Optional[str] = logging_cfg.get("mlop_project") or os.environ.get("MLOP_PROJECT")
    if not project:
        return False

    api_key = os.environ.get("MLOP_API_KEY")
    if not api_key:
        print("[mlop] MLOP_API_KEY not set; skipping mlop logging")
        return False

    try:
        import mlop
        from mlop.sets import Settings
    except ImportError:
        print("[mlop] mlop package not installed; pip install 'mlop[full]'")
        return False

    ensure_mlop_transformers_integration()

    run_name = str(cfg.get("run_name", "pretrain"))
    os.environ.setdefault("MLOP_PROJECT", str(project))

    settings = Settings()
    settings._auth = api_key
    settings.project = str(project)

    if not mlop.ops:
        mlop.init(
            project=str(project),
            name=run_name,
            config=cfg,
            settings=settings,
        )
    else:
        mlop.ops[-1].settings._auth = api_key

    print(f"[mlop] enabled project={project} run={run_name}")
    return True


def finish_mlop() -> None:
    try:
        import mlop
    except ImportError:
        return
    if mlop.ops:
        mlop.finish()
