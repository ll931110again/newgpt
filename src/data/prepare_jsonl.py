"""Download a Hugging Face text dataset and write jsonl with a `text` column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="wikitext",
        help="HF datasets name (default: wikitext)",
    )
    ap.add_argument(
        "--config",
        default="wikitext-103-raw-v1",
        help="HF dataset config/subset",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--out", required=True, help="Output jsonl path")
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Cap rows (0 = all). Useful for smoke runs.",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    # Use trust_remote_code for older dataset scripts; clear HF cache if load fails.
    try:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    except Exception:
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=args.split,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get(args.text_column) or "").strip()
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
            if args.max_rows and n >= args.max_rows:
                break

    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
