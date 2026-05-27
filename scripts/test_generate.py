#!/usr/bin/env python3
"""Quick local smoke test for a saved causal LM checkpoint."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default="checkpoints/pretrain_1-3b/final",
        help="Path to saved model directory",
    )
    ap.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text. Pass multiple times to generate from multiple prompts.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="Base seed; each sample uses seed+i")
    args = ap.parse_args()

    prompts = args.prompt or [
        "In a surprising finding, scientists discovered that",
        "Once upon a time, in a distant kingdom,",
        "Write a short poem about the ocean:",
        "Q: What is the capital of France?\nA:",
        "User: Explain gradient descent in one paragraph.\nAssistant:",
        "Write a Python function that computes factorial(n):",
        "Summarize the following text in one sentence:\n\nArtificial intelligence is transforming many industries by automating tasks and enabling new products.\n\nSummary:",
        "Dialogue:\nAlice: Do you want coffee or tea?\nBob:",
    ]

    device, dtype = _device_and_dtype()
    print(f"[test_generate] device={device} dtype={dtype}", flush=True)
    print(f"[test_generate] loading {args.model_dir} ...", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype)
    model.eval()
    model = model.to(device)
    print("[test_generate] model loaded; generating ...", flush=True)

    for p_i, prompt in enumerate(prompts, start=1):
        print(f"\n=== prompt {p_i}/{len(prompts)} ===", flush=True)
        print(prompt, flush=True)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        for i in range(int(args.num_samples)):
            seed = int(args.seed) + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"\n--- sample {i+1}/{args.num_samples} (seed={seed}) ---", flush=True)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=tok.eos_token_id,
                )
            print(tok.decode(out[0], skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    main()
