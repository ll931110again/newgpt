"""Training performance helpers (attention backend, dataloader, optimizer)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.config import deep_get


def resolve_attn_implementation(cfg: Dict[str, Any]) -> Optional[str]:
    """Pick an attention implementation, with safe fallbacks when optional deps are missing."""
    requested = str(deep_get(cfg, "train", "attn_implementation", default="auto") or "auto").lower()

    if requested in ("", "none", "null"):
        return None

    if requested == "auto":
        if _flash_attention_available():
            return "flash_attention_2"
        return "sdpa"

    if requested == "flash_attention_2":
        if _flash_attention_available():
            return "flash_attention_2"
        print("[perf] flash_attn not installed; falling back to sdpa")
        return "sdpa"

    if requested in ("sdpa", "eager", "flash_attention_2"):
        return requested

    raise ValueError(
        f"Unknown train.attn_implementation={requested!r} "
        "(use auto, flash_attention_2, sdpa, or eager)"
    )


def _flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def model_load_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extra kwargs for AutoModelForCausalLM.from_pretrained / from_config."""
    kwargs: Dict[str, Any] = {}
    attn = resolve_attn_implementation(cfg)
    if attn:
        kwargs["attn_implementation"] = attn
    return kwargs


def training_args_performance_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extra kwargs for Hugging Face TrainingArguments."""
    tcfg = cfg.get("train", {}) or {}
    num_workers = int(tcfg.get("dataloader_num_workers", 0))
    kwargs: Dict[str, Any] = {
        "optim": str(tcfg.get("optim", "adamw_torch_fused")),
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": bool(tcfg.get("dataloader_pin_memory", True)) and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = int(tcfg.get("dataloader_prefetch_factor", 2))
    return kwargs


def maybe_compile_model(model: Any, cfg: Dict[str, Any]) -> Any:
    if not bool(deep_get(cfg, "train", "torch_compile", default=False)):
        return model
    import torch

    print("[perf] torch.compile enabled")
    return torch.compile(model)


def apply_cuda_runtime_flags(cfg: Dict[str, Any]) -> None:
    """Enable Tensor Core–friendly matmul settings on CUDA (6.S894 Lab 6 / 10)."""
    import torch

    if not torch.cuda.is_available():
        return
    tcfg = cfg.get("train", {}) or {}
    if bool(tcfg.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    attn = resolve_attn_implementation(cfg)
    print(f"[perf] cuda flags applied (tf32={bool(tcfg.get('tf32', True))}, attn={attn})")
