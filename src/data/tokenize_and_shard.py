from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def _load_tokenizer(name_or_path: str) -> Tokenizer:
    # Fast path: use tokenizers directly if a tokenizer.json exists.
    p = Path(name_or_path)
    if p.is_file() and p.name.endswith(".json"):
        return Tokenizer.from_file(str(p))
    # Fallback: load via HF and extract underlying tokenizer.json if available.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if getattr(tok, "backend_tokenizer", None) is None:
        raise SystemExit("Expected a fast tokenizer backend (tokenizers).")
    return tok.backend_tokenizer


def _iter_text(ds) -> Iterable[str]:
    if "text" not in ds.column_names:
        raise SystemExit("Dataset must contain a 'text' column (jsonl with {\"text\": ...})")
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path/glob or HF datasets spec for json/jsonl")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer name/path (HF) or tokenizer.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_version", required=True)
    ap.add_argument("--shard_tokens", type=int, default=50_000_000, help="Approx tokens per shard")
    ap.add_argument("--eos_token_id", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    manifests_dir = out_dir / "manifests"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    tok = _load_tokenizer(args.tokenizer)
    eos_id = args.eos_token_id
    if eos_id is None:
        # Try common convention
        eos_id = tok.token_to_id("</s>") or tok.token_to_id("")  # type: ignore[arg-type]

    ds = load_dataset("json", data_files=args.input, split="train")

    shard_tokens = int(args.shard_tokens)
    cur: List[int] = []
    shard_idx = 0
    total_tokens = 0

    def flush() -> Optional[Dict]:
        nonlocal shard_idx, total_tokens, cur
        if not cur:
            return None
        arr = np.asarray(cur, dtype=np.int32)
        shard_path = shards_dir / f"shard_{shard_idx:05d}.npy"
        np.save(shard_path, arr)
        meta = {"path": str(shard_path), "tokens": int(arr.shape[0])}
        total_tokens += int(arr.shape[0])
        shard_idx += 1
        cur = []
        return meta

    shard_metas: List[Dict] = []
    for text in tqdm(_iter_text(ds), desc="tokenize"):
        enc = tok.encode(text)
        cur.extend(enc.ids)
        if eos_id is not None:
            cur.append(int(eos_id))
        if len(cur) >= shard_tokens:
            m = flush()
            if m:
                shard_metas.append(m)

    m = flush()
    if m:
        shard_metas.append(m)

    manifest = {
        "dataset_version": args.dataset_version,
        "format": "npy_int32_tokens",
        "tokenizer": args.tokenizer,
        "shards": shard_metas,
        "total_tokens": total_tokens,
    }
    manifest_path = manifests_dir / f"{args.dataset_version}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(str(manifest_path))


if __name__ == "__main__":
    main()

