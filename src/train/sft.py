from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from src.utils.config import deep_get, load_yaml


def _format_example(ex: Dict[str, Any]) -> str:
    # Preferred: precomputed `text` field (e.g. from src.data.download_sft_alpaca)
    if ex.get("text"):
        return str(ex["text"])

    # Chat format: {messages:[{role,content}...]}
    if isinstance(ex.get("messages"), list):
        parts = []
        for m in ex["messages"]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not role or not content:
                continue
            parts.append(f"{role.title()}: {content}")
        return "\n".join(parts)

    # Prompt/response format
    if ex.get("prompt") is not None and ex.get("response") is not None:
        return f"{ex['prompt']}{ex['response']}"

    # Alpaca-style format
    if ex.get("instruction") is not None and ex.get("output") is not None:
        instruction = str(ex.get("instruction") or "").strip()
        inp = str(ex.get("input") or "").strip()
        output = str(ex.get("output") or "").strip()
        if inp:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt + output

    raise ValueError("Unsupported SFT row format; expected text/messages/prompt+response/instruction+output")


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
        raise SystemExit("base_model.model_name_or_path is required for SFT")

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
    peft_cfg = cfg.get("peft", {}) or {}
    report_to = []
    if logging_cfg.get("wandb_project"):
        report_to.append("wandb")

    training_args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=float(tcfg.get("num_train_epochs", 1)),
        learning_rate=float(tcfg.get("learning_rate", 2e-5)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        bf16=bool(tcfg.get("bf16", True)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        logging_steps=int(logging_cfg.get("log_every_steps", 10)),
        save_steps=int(logging_cfg.get("save_every_steps", 250)),
        save_total_limit=3,
        report_to=report_to or "none",
        run_name=str(cfg.get("run_name", "sft")),
        dataset_text_field="text",
        max_length=seq_len,
        packing=True,
    )

    peft_config = None
    if bool(peft_cfg.get("enabled", False)):
        try:
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=int(peft_cfg.get("lora_r", 16)),
                lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
                lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(f"[sft] peft enabled (lora_r={peft_config.r})")
        except Exception as exc:
            print(f"[sft] peft requested but unavailable ({type(exc).__name__}: {exc}); continuing without peft")

    trainer = SFTTrainer(
        model=model_path,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        processing_class=tokenizer,
        formatting_func=_format_example,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))


if __name__ == "__main__":
    main()

