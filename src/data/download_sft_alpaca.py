from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset


def _alpaca_to_text(ex: Dict[str, str]) -> str:
    instruction = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    output = (ex.get("output") or "").strip()

    if inp:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt + output


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-eval", type=int, default=1000)
    args = ap.parse_args()

    random.seed(args.seed)
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    ds = ds.shuffle(seed=args.seed)

    n_train = min(int(args.max_train), max(len(ds) - 1, 1))
    n_eval = min(int(args.max_eval), max(len(ds) - n_train, 0))

    train = ds.select(range(0, n_train))
    eval_ = ds.select(range(n_train, n_train + n_eval)) if n_eval > 0 else None

    def rows(dset) -> List[Dict]:
        out: List[Dict] = []
        for ex in dset:
            text = _alpaca_to_text(ex)
            out.append(
                {
                    "instruction": ex.get("instruction"),
                    "input": ex.get("input"),
                    "output": ex.get("output"),
                    "text": text,
                }
            )
        return out

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"

    _write_jsonl(train_path, rows(train))
    if eval_ is not None:
        _write_jsonl(eval_path, rows(eval_))

    print(f"[sft] wrote: {train_path} ({len(train)} rows)")
    if eval_ is not None:
        print(f"[sft] wrote: {eval_path} ({len(eval_)} rows)")


if __name__ == "__main__":
    main()

