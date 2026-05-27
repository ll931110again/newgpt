import argparse
import os
from pathlib import Path

import yaml


def _require_exists(path: str, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what} not found: {path}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to infra/gpu-cloud/job_*.yaml")
    args = ap.parse_args()

    spec_path = _require_exists(args.spec, "job spec")
    spec = yaml.safe_load(spec_path.read_text())
    kind = spec.get("kind")
    cfg = spec.get("config")

    if kind in {"pretrain", "sft", "dpo"}:
        _require_exists(cfg, "config")
        output_dir = spec.get("output_dir", "runs/unnamed")
        resume_from = spec.get("resume_from")
        module = {
            "pretrain": "src.train.pretrain",
            "sft": "src.train.sft",
            "dpo": "src.train.dpo",
        }[kind]
        cmd = f"python3 -m {module} --config {cfg} --output_dir {output_dir}"
        if resume_from:
            cmd += f" --resume_from {resume_from}"
        os.execvp("bash", ["bash", "-lc", cmd])

    if kind == "eval":
        _require_exists(cfg, "config")
        model_path = spec.get("model_path")
        cmd = f"python3 -m src.eval.run_eval --config {cfg}"
        if model_path:
            cmd += f" --model_path {model_path}"
        os.execvp("bash", ["bash", "-lc", cmd])

    if kind == "serve":
        _require_exists(cfg, "config")
        model_path = spec.get("model_path")
        cmd = f"python3 -m src.serve.launch_vllm --config {cfg}"
        if model_path:
            cmd += f" --model_path {model_path}"
        os.execvp("bash", ["bash", "-lc", cmd])

    raise SystemExit(f"Unknown job kind: {kind}")


if __name__ == "__main__":
    main()

