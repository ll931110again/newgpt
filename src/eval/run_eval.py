from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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

    tasks: List[str] = list(deep_get(cfg, "eval", "tasks", default=[]))
    batch_size = int(deep_get(cfg, "eval", "batch_size", default=1))
    limit = deep_get(cfg, "eval", "limit", default=None)
    out_dir = Path(deep_get(cfg, "logging", "output_dir", default="runs/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # lm-eval entrypoint is installed as `lm_eval`
    # We invoke it as a subprocess to keep integration simple.
    import subprocess

    cmd = [
        "python3",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path}",
        "--tasks",
        ",".join(tasks) if tasks else "arc_easy",
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(out_dir / "lm_eval_results.json"),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    subprocess.check_call(cmd)

    # Convenience summary
    res_path = out_dir / "lm_eval_results.json"
    if res_path.exists():
        data = json.loads(res_path.read_text())
        (out_dir / "summary.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

