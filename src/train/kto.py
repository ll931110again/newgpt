from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import KTOTrainer

from src.utils.config import deep_get, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume_from", default=None)
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = deep_get(cfg, "base_model", "model_name_or_path")
    tok_path = deep_get(cfg, "base_model", "tokenizer_name_or_path") or model_path
    if not model_path:
        raise SystemExit("base_model.model_name_or_path is required for KTO")

    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = deep_get(cfg, "data", "train_path")
    eval_path = deep_get(cfg, "data", "eval_path")
    seq_len = int(deep_get(cfg, "data", "seq_len", default=2048))
    if not train_path:
        raise SystemExit("data.train_path is required")

    ds_train = load_dataset("json", data_files=train_path, split="train")
    ds_eval = load_dataset("json", data_files=eval_path, split="train") if eval_path else None

    tcfg = cfg.get("train", {})
    logging_cfg = cfg.get("logging", {})
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=float(tcfg.get("num_train_epochs", 1)),
        learning_rate=float(tcfg.get("learning_rate", 5e-6)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        bf16=bool(tcfg.get("bf16", True)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        logging_steps=int(logging_cfg.get("log_every_steps", 10)),
        save_steps=int(logging_cfg.get("save_every_steps", 250)),
        save_total_limit=3,
        report_to=["wandb"] if logging_cfg.get("wandb_project") else [],
        run_name=str(cfg.get("run_name", "kto")),
    )

    trainer = KTOTrainer(
        model=model_path,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        max_length=seq_len,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))


if __name__ == "__main__":
    main()

