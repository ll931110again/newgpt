#!/usr/bin/env python3
"""Micro-benchmark training step throughput (6.S894 measure-then-optimize workflow)."""

from __future__ import annotations

import argparse
import time

from src.train.metrics import effective_batch_tokens, tokens_per_second
from src.train.performance import apply_cuda_runtime_flags, model_load_kwargs
from src.utils.config import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    from src.data.packed_dataset import PackedTokensIterableDataset

    cfg = load_yaml(args.config)
    apply_cuda_runtime_flags(cfg)
    tcfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    seq_len = int(data_cfg.get("seq_len", 2048))
    batch_size = int(tcfg.get("per_device_train_batch_size", 1))
    manifest = data_cfg.get("dataset_manifest")
    if not manifest:
        raise SystemExit("data.dataset_manifest required for benchmark")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[benchmark] CUDA not available; results are not representative of GPU training")

    init_from = (cfg.get("model") or {}).get("init_from")
    load_kw = model_load_kwargs(cfg)
    if init_from:
        model = AutoModelForCausalLM.from_pretrained(
            init_from,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            **load_kw,
        )
    else:
        raise SystemExit("benchmark requires model.init_from for a quick load")

    model.to(device)
    model.train()
    if bool(tcfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    ds = PackedTokensIterableDataset(manifest, seq_len=seq_len)
    num_workers = int(tcfg.get("dataloader_num_workers", 0))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(tcfg.get("dataloader_pin_memory", False)) and num_workers > 0,
        prefetch_factor=int(tcfg.get("dataloader_prefetch_factor", 2)) if num_workers > 0 else None,
    )
    it = iter(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")

    def step_once() -> float:
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        step_once()

    times: list[float] = []
    for _ in range(args.steps):
        times.append(step_once())

    med = sorted(times)[len(times) // 2]
    tps = tokens_per_second(batch_size, seq_len, med)
    eff = effective_batch_tokens(
        batch_size,
        seq_len,
        int(tcfg.get("gradient_accumulation_steps", 1)),
    )
    vram = ""
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        vram = f" peak_vram_gb={peak:.2f}"

    print(
        f"[benchmark] median_step_s={med:.3f} tokens_per_sec={tps:,.0f} "
        f"batch={batch_size} seq_len={seq_len} eff_tokens_per_step={eff}{vram}"
    )


if __name__ == "__main__":
    main()
