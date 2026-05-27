from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from src.utils.config import deep_get, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_path", default=None)
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_yaml(args.config)
    model_path = args.model_path or deep_get(cfg, "model", "model_name_or_path")
    if not model_path:
        raise SystemExit("Provide --model_path or set model.model_name_or_path in config")

    host = str(deep_get(cfg, "serve", "host", default="0.0.0.0"))
    port = int(deep_get(cfg, "serve", "port", default=8000))
    tp = int(deep_get(cfg, "serve", "tensor_parallel_size", default=1))
    dtype = str(deep_get(cfg, "serve", "dtype", default="bfloat16"))
    max_len = int(deep_get(cfg, "serve", "max_model_len", default=4096))

    # vLLM exposes an OpenAI-compatible API via `vllm.entrypoints.openai.api_server`.
    cmd = (
        "python3 -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--host {host} --port {port} "
        f"--tensor-parallel-size {tp} "
        f"--dtype {dtype} "
        f"--max-model-len {max_len}"
    )
    os.execvp("bash", ["bash", "-lc", cmd])


if __name__ == "__main__":
    main()

