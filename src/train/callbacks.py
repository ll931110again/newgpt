"""Hugging Face Trainer callbacks for artifact sync."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from transformers import TrainerCallback

from src.utils.s3_sync import s3_configured, upload_run_artifact


class S3CheckpointCallback(TrainerCallback):
    """Upload each saved checkpoint (and final model) to S3-compatible storage."""

    def __init__(self, run_name: str, enabled: bool = True):
        self.run_name = run_name
        self.enabled = enabled and s3_configured()

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt.is_dir():
            try:
                upload_run_artifact(ckpt, self.run_name)
            except Exception as exc:
                print(f"[s3] checkpoint upload failed: {exc}")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        final = Path(args.output_dir) / "final"
        if final.is_dir():
            try:
                upload_run_artifact(final, self.run_name)
            except Exception as exc:
                print(f"[s3] final model upload failed: {exc}")


def build_s3_callback(cfg: Dict[str, Any]) -> S3CheckpointCallback:
    artifacts = cfg.get("artifacts", {}) or {}
    enabled = artifacts.get("s3_upload", True)
    if enabled == "auto":
        enabled = s3_configured()
    run_name = str(cfg.get("run_name", "pretrain"))
    return S3CheckpointCallback(run_name=run_name, enabled=bool(enabled))
