from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoModelForCausalLM,
)

from src.fleet.env import FleetEnv
from src.utils.config import deep_get, load_yaml
from src.utils.reporting import build_report_to, finish_reporting
from src.data.packed_dataset import PackedTokensIterableDataset
from src.train.callbacks import build_s3_callback
from src.train.fleet_callback import build_fleet_callback
from src.train.performance import (
    apply_cuda_runtime_flags,
    maybe_compile_model,
    model_load_kwargs,
    training_args_performance_kwargs,
)


def _apply_fleet_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    fleet = FleetEnv.from_os()
    if not fleet.enabled:
        return cfg
    cfg = dict(cfg)
    model = dict(cfg.get("model") or {})
    if fleet.canonical_init:
        model["init_from"] = fleet.canonical_init
        print(f"[fleet] init_from canonical: {fleet.canonical_init}")
    cfg["model"] = model
    return cfg


def _build_model(cfg: Dict[str, Any]):
    init_from = deep_get(cfg, "model", "init_from")
    tokenizer_name = deep_get(cfg, "model", "tokenizer_name_or_path")
    if not tokenizer_name:
        raise SystemExit("model.tokenizer_name_or_path is required")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = model_load_kwargs(cfg)
    attn = load_kw.get("attn_implementation")
    if attn:
        print(f"[pretrain] attn_implementation={attn}")

    if init_from:
        model = AutoModelForCausalLM.from_pretrained(
            init_from,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            **load_kw,
        )
    else:
        base = deep_get(cfg, "model", "config_name_or_path")
        if base:
            mcfg = AutoConfig.from_pretrained(base)
        else:
            mcfg = AutoConfig.from_pretrained("gpt2")
            mcfg.n_positions = int(deep_get(cfg, "data", "seq_len", default=2048))
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_config(mcfg, torch_dtype=dtype, **load_kw)

    model = maybe_compile_model(model, cfg)
    return model, tokenizer


def _load_pretokenized_dataset(cfg: Dict[str, Any]):
    manifest = deep_get(cfg, "data", "dataset_manifest")
    if not manifest:
        raise SystemExit("data.dataset_manifest is required")
    fmt = deep_get(cfg, "data", "format", default="packed_npy")
    seq_len = int(deep_get(cfg, "data", "seq_len", default=2048))
    fleet = FleetEnv.from_os()
    if fmt == "packed_npy":
        return PackedTokensIterableDataset(
            manifest,
            seq_len=seq_len,
            shard_ids=fleet.shard_ids if fleet.enabled else None,
            fleet_rank=fleet.rank if fleet.enabled else None,
            fleet_world_size=fleet.world_size if fleet.enabled else None,
        )
    if fmt == "json_text":
        from datasets import load_dataset
        return load_dataset("json", data_files=manifest, split="train")
    raise SystemExit(f"Unknown data.format: {fmt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume_from", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = _apply_fleet_overrides(cfg)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    apply_cuda_runtime_flags(cfg)
    model, tokenizer = _build_model(cfg)
    ds = _load_pretokenized_dataset(cfg)

    seq_len = int(deep_get(cfg, "data", "seq_len", default=2048))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len)

    collator = None
    if hasattr(ds, "column_names") and "input_ids" not in getattr(ds, "column_names"):
        ds = ds.map(tok, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ds_conf = deep_get(cfg, "distributed", "deepspeed", default={}) or {}
    ds_enabled = bool(ds_conf.get("enabled", False))
    deepspeed_config = None
    if ds_enabled:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            print("[pretrain] deepspeed not installed; continuing without ZeRO")
            ds_enabled = False
    if ds_enabled:
        zero_stage = int(ds_conf.get("zero_stage", 3))
        deepspeed_config = {
            "train_micro_batch_size_per_gpu": int(ds_conf.get("micro_batch_size_per_gpu", 1)),
            "gradient_accumulation_steps": int(ds_conf.get("gradient_accumulation_steps", 1)),
            "zero_optimization": {"stage": zero_stage},
            "bf16": {"enabled": True},
        }

    tcfg = cfg.get("train", {})
    logging_cfg = cfg.get("logging", {})
    report_to = build_report_to(cfg)
    eval_every = logging_cfg.get("eval_every_steps")
    eval_strategy = "steps" if eval_every not in (None, "null", 0) else "no"
    perf_kw = training_args_performance_kwargs(cfg)
    if perf_kw.get("dataloader_num_workers", 0) > 0:
        print(f"[pretrain] dataloader_num_workers={perf_kw['dataloader_num_workers']}")

    training_args = TrainingArguments(
        output_dir=str(out),
        max_steps=int(tcfg.get("num_train_steps", 1000)),
        learning_rate=float(tcfg.get("learning_rate", 3e-4)),
        warmup_steps=int(tcfg.get("warmup_steps", 50)),
        weight_decay=float(tcfg.get("weight_decay", 0.1)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        bf16=bool(tcfg.get("bf16", True)),
        tf32=bool(tcfg.get("tf32", True)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        logging_steps=int(logging_cfg.get("log_every_steps", 10)),
        save_steps=int(logging_cfg.get("save_every_steps", 250)),
        eval_steps=int(eval_every) if eval_strategy == "steps" else None,
        eval_strategy=eval_strategy,
        save_total_limit=3,
        report_to=report_to,
        run_name=str(cfg.get("run_name", "pretrain")),
        deepspeed=deepspeed_config,
        **perf_kw,
    )

    callbacks = [build_s3_callback(cfg)]
    fleet_cb = build_fleet_callback(cfg, out)
    if fleet_cb is not None:
        callbacks.append(fleet_cb)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    finish_reporting(cfg)


if __name__ == "__main__":
    main()
