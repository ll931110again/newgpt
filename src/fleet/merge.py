from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


def _load_state_dict(checkpoint_dir: Path) -> dict:
    """Load model weights from a HF Trainer checkpoint directory."""
    ckpt = Path(checkpoint_dir)
    model_bin = ckpt / "pytorch_model.bin"
    model_safetensors = ckpt / "model.safetensors"
    if model_safetensors.is_file():
        from safetensors.torch import load_file

        return load_file(str(model_safetensors))
    if model_bin.is_file():
        state = torch.load(model_bin, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            return state["state_dict"]
        return state
    raise FileNotFoundError(f"No model weights in {ckpt}")


def fedavg_state_dicts(state_dicts: Sequence[dict]) -> dict:
    if not state_dicts:
        raise ValueError("Need at least one state dict to merge")
    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    keys = state_dicts[0].keys()
    for sd in state_dicts[1:]:
        if sd.keys() != keys:
            raise ValueError("Checkpoint state dict keys do not match")

    merged: dict = {}
    n = len(state_dicts)
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        first = tensors[0]
        if not torch.is_floating_point(first):
            merged[key] = first.clone()
            continue
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        merged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
    return merged


def _save_merged_checkpoint(
    merged_state: dict,
    template_dir: Path,
    out_dir: Path,
) -> None:
    """Copy HF checkpoint metadata and write averaged weights."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(template_dir, out_dir)

    safetensors_path = out_dir / "model.safetensors"
    bin_path = out_dir / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import save_file

        save_file(merged_state, str(safetensors_path))
        bin_path.unlink(missing_ok=True)
    elif bin_path.is_file() or (template_dir / "pytorch_model.bin").is_file():
        torch.save(merged_state, bin_path)
        safetensors_path.unlink(missing_ok=True)
    else:
        torch.save(merged_state, out_dir / "pytorch_model.bin")


def fedavg_checkpoints(checkpoint_dirs: Iterable[Path], out_dir: Path) -> Path:
    """Average weights from multiple HF checkpoints into out_dir."""
    dirs: List[Path] = [Path(p) for p in checkpoint_dirs]
    if not dirs:
        raise ValueError("Need at least one checkpoint directory")

    state_dicts = [_load_state_dict(d) for d in dirs]
    merged = fedavg_state_dicts(state_dicts)
    _save_merged_checkpoint(merged, dirs[0], out_dir)
    return out_dir
