#!/usr/bin/env python3
"""Kaggle GPU kernel entrypoint: probe hardware, tune config, run pretraining."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def _find_bundle_root() -> Path:
    """Locate staged code bundle (Kaggle dataset mount or local staging)."""
    candidates = [
        Path("/kaggle/input/pretrainer-bundle"),
        Path("/kaggle/input/linhvuongnguyen/pretrainer-bundle"),
        ROOT,
    ]
    input_dir = Path("/kaggle/input")
    if input_dir.is_dir():
        for child in sorted(input_dir.iterdir()):
            if (child / "src" / "train" / "pretrain.py").exists():
                return child
    for p in candidates:
        if (p / "src" / "train" / "pretrain.py").exists():
            return p
    raise FileNotFoundError(
        "Could not find pretrainer bundle (expected src/train/pretrain.py). "
        "Attach dataset linhvuongnguyen/pretrainer-bundle to the kernel."
    )


BUNDLE = _find_bundle_root()
sys.path.insert(0, str(BUNDLE))
os.chdir(BUNDLE)


def _kaggle_working() -> Path:
    """Writable output dir on Kaggle; local fallback for smoke tests."""
    for candidate in (
        os.environ.get("KAGGLE_WORKING_DIR"),
        "/kaggle/working",
    ):
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
    return ROOT / "kaggle_output"


WORKING = _kaggle_working()
WORKING.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = WORKING / "checkpoints" / "pretrain_kaggle"
FINAL_DIR = WORKING / "model-final"
DATA_ROOT = WORKING / "data"

# Largest practical presets for Kaggle (~20GB disk, 9–12h GPU session).
# wikitext103: ~103M tokens — best default; ~1 epoch fits a 9h T4 session.
# c4_250k:     ~150–300M tokens — slightly larger, still finishes prep quickly.
# c4_1m:       ~500M–1B tokens — max corpus; use only on a fresh run with time
#              budget for streaming prep + accept a partial epoch in one session.
DATASET_PRESETS: dict[str, dict] = {
    "wikitext103": {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "streaming": False,
        "max_rows": 0,
    },
    "c4_250k": {
        "dataset": "allenai/c4",
        "config": "en.noclean",
        "split": "train",
        "streaming": True,
        "max_rows": 250_000,
    },
    "c4_1m": {
        "dataset": "allenai/c4",
        "config": "en.noclean",
        "split": "train",
        "streaming": True,
        "max_rows": 1_000_000,
    },
}


def _bundled_fleet_manifest() -> Path | None:
    """Return bundled v1 manifest when fleet training uses pre-sharded data."""
    if os.environ.get("FLEET_ENABLED", "0") != "1":
        return None
    for candidate in (
        BUNDLE / "data" / "manifests" / "v1.json",
        ROOT / "data" / "manifests" / "v1.json",
    ):
        if candidate.is_file():
            manifest = json.loads(candidate.read_text())
            if len(manifest.get("shards", [])) >= 2:
                print(f"[kaggle] Using bundled fleet manifest ({len(manifest['shards'])} shards): {candidate}")
                return candidate
    return None


def _prepare_data() -> tuple[Path, int]:
    """Download HF corpus, tokenize to /kaggle/working/data, return manifest + token count."""
    bundled = _bundled_fleet_manifest()
    if bundled is not None:
        manifest = json.loads(bundled.read_text())
        return bundled, int(manifest["total_tokens"])

    choice = os.environ.get("KAGGLE_DATASET", "wikitext103").lower()
    spec = DATASET_PRESETS.get(choice, DATASET_PRESETS["wikitext103"])
    print(f"[kaggle] Dataset preset: {choice} ({spec['dataset']})")

    raw_dir = DATA_ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl = raw_dir / f"{choice}.jsonl"

    prep_cmd = [
        sys.executable,
        "-m",
        "src.data.prepare_jsonl",
        "--dataset",
        spec["dataset"],
        "--config",
        spec["config"],
        "--split",
        spec["split"],
        "--out",
        str(jsonl),
    ]
    if spec.get("streaming"):
        prep_cmd.append("--streaming")
    if spec.get("max_rows"):
        prep_cmd.extend(["--max_rows", str(spec["max_rows"])])
    print("[kaggle] Downloading corpus:", " ".join(prep_cmd))
    subprocess.check_call(prep_cmd)

    shard_cmd = [
        sys.executable,
        "-m",
        "src.data.tokenize_and_shard",
        "--input",
        str(jsonl),
        "--tokenizer",
        "gpt2",
        "--out_dir",
        str(DATA_ROOT),
        "--dataset_version",
        "kaggle_v1",
        "--shard_tokens",
        "5000000",
    ]
    print("[kaggle] Tokenizing:", " ".join(shard_cmd))
    subprocess.check_call(shard_cmd)

    manifest_path = DATA_ROOT / "manifests" / "kaggle_v1.json"
    total_tokens = int(json.loads(manifest_path.read_text())["total_tokens"])
    print(f"[kaggle] Corpus ready: {total_tokens:,} tokens -> {manifest_path}")
    return manifest_path, total_tokens


def _train_steps(total_tokens: int, profile: dict) -> int:
    """Steps for one epoch or GPU time budget, whichever is smaller."""
    hours = float(os.environ.get("KAGGLE_TRAIN_HOURS", "9"))
    sec_per_step = float(os.environ.get("KAGGLE_SEC_PER_STEP", "1.5"))
    tokens_per_step = (
        profile["batch_size"] * profile["seq_len"] * profile["grad_accum"]
    )
    steps_epoch = max(1, total_tokens // max(1, tokens_per_step))
    steps_budget = max(1, int((hours * 3600) / sec_per_step))
    steps = min(steps_epoch, steps_budget)
    print(
        f"[kaggle] Train steps: {steps} "
        f"(epoch={steps_epoch}, budget@{hours}h≈{steps_budget}, "
        f"{tokens_per_step} tok/step)"
    )
    return steps


def _pip_install() -> None:
    req = BUNDLE / "requirements-kaggle.txt"
    if not req.exists():
        req = ROOT / "requirements-kaggle.txt"
    if req.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            stdout=subprocess.DEVNULL,
        )


def _gpu_profile() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {
            "name": "cpu",
            "total_gb": 0.0,
            "model_config": "gpt2",
            "seq_len": 512,
            "batch_size": 1,
            "grad_accum": 1,
            "checkpointing": False,
        }

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    name = props.name
    print(f"[kaggle] GPU: {name}, {total_gb:.1f} GB")

    # Tiered presets — maximize throughput without OOM on common Kaggle SKUs.
    if total_gb >= 38:
        return {
            "name": name,
            "total_gb": total_gb,
            "model_config": "configs/model/gpt2_1.3b.json",
            "seq_len": 2048,
            "batch_size": 8,
            "grad_accum": 1,
            "checkpointing": False,
        }
    if total_gb >= 22:
        return {
            "name": name,
            "total_gb": total_gb,
            "model_config": "gpt2-large",
            "seq_len": 1024,
            "batch_size": 4,
            "grad_accum": 2,
            "checkpointing": False,
        }
    if total_gb >= 14:
        return {
            "name": name,
            "total_gb": total_gb,
            "model_config": "gpt2-medium",
            "seq_len": 1024,
            "batch_size": 6,
            "grad_accum": 1,
            "checkpointing": False,
        }
    return {
        "name": name,
        "total_gb": total_gb,
        "model_config": "gpt2",
        "seq_len": 512,
        "batch_size": 8,
        "grad_accum": 1,
        "checkpointing": False,
    }


def _maybe_reduce_batch(profile: dict) -> dict:
    """Binary-search down batch size if a dry-run forward OOMs."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        return profile

    cfg_path = profile["model_config"]
    if cfg_path.endswith(".json"):
        mcfg = AutoConfig.from_pretrained(str(BUNDLE / cfg_path))
    else:
        mcfg = AutoConfig.from_pretrained(cfg_path)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    seq = profile["seq_len"]
    batch = profile["batch_size"]

    while batch >= 1:
        try:
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_config(mcfg, torch_dtype=torch.bfloat16)
            model.cuda()
            ids = torch.randint(0, 50257, (batch, seq), device="cuda")
            model(input_ids=ids, labels=ids).loss.backward()
            del model, ids
            torch.cuda.empty_cache()
            profile["batch_size"] = batch
            print(f"[kaggle] Verified batch_size={batch} seq_len={seq}")
            return profile
        except torch.cuda.OutOfMemoryError:
            print(f"[kaggle] OOM at batch_size={batch}, reducing...")
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()

    profile["batch_size"] = 1
    profile["checkpointing"] = True
    profile["grad_accum"] = max(profile["grad_accum"], 2)
    return profile


def _write_runtime_config(profile: dict, manifest_path: Path, total_tokens: int) -> Path:
    import torch

    fleet = os.environ.get("FLEET_ENABLED", "0") == "1"
    fleet_cfg = os.environ.get("FLEET_TRAIN_CONFIG", "").strip()
    if fleet and fleet_cfg:
        cfg_path = BUNDLE / fleet_cfg
        if not cfg_path.is_file():
            cfg_path = ROOT / fleet_cfg
        if cfg_path.is_file():
            base = yaml.safe_load(cfg_path.read_text())
            print(f"[kaggle] Fleet runtime base config: {cfg_path}")
        else:
            base = yaml.safe_load((BUNDLE / "configs" / "pretrain_kaggle.yaml").read_text())
    else:
        base = yaml.safe_load((BUNDLE / "configs" / "pretrain_kaggle.yaml").read_text())
    canonical = _resolve_canonical_init()
    if canonical:
        base["model"]["init_from"] = canonical
        base["model"]["config_name_or_path"] = None
    else:
        base["model"]["config_name_or_path"] = profile["model_config"]
    base["data"]["dataset_manifest"] = str(manifest_path)
    base["data"]["format"] = "packed_npy"
    base["data"]["seq_len"] = profile["seq_len"]
    steps = _train_steps(total_tokens, profile)
    base["train"]["num_train_steps"] = steps
    base["train"]["warmup_steps"] = min(500, max(25, steps // 20))
    base["train"]["per_device_train_batch_size"] = profile["batch_size"]
    base["train"]["gradient_accumulation_steps"] = profile["grad_accum"]
    base["train"]["gradient_checkpointing"] = profile.get("checkpointing", False)
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        base["train"]["tf32"] = major >= 8  # Ampere+ only (T4/P100 need false)
    else:
        base["train"]["tf32"] = False

    runtime = WORKING / "pretrain_kaggle_runtime.yaml"
    runtime.write_text(yaml.dump(base, default_flow_style=False, sort_keys=False))
    (WORKING / "gpu_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[kaggle] Runtime config written to {runtime}")
    return runtime


def _resolve_canonical_init() -> str | None:
    """Resolve fleet canonical weights bundled with the dataset or from env."""
    explicit = os.environ.get("FLEET_CANONICAL_INIT", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            BUNDLE / "checkpoints" / "canonical",
            ROOT / "checkpoints" / "canonical",
            Path("/kaggle/input/pretrainer-checkpoint/checkpoints/canonical"),
        ]
    )
    input_dir = Path("/kaggle/input")
    if input_dir.is_dir():
        for child in sorted(input_dir.iterdir()):
            p = child / "checkpoints" / "canonical"
            candidates.append(p)
    for path in candidates:
        if path.is_dir() and (
            (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
        ):
            print(f"[kaggle] Using canonical init: {path}")
            return str(path)
    return explicit or None


def _load_fleet_env() -> None:
    """Apply fleet worker env from bundled fleet_worker.json (written by orchestrator)."""
    candidates = [
        BUNDLE / "fleet_worker.json",
        ROOT / "fleet_worker.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        worker_id = data.pop("worker_id", None)
        if worker_id is not None:
            os.environ["FLEET_WORKER_ID"] = str(worker_id)
        for key, value in data.items():
            os.environ[str(key)] = str(value)
        os.environ.setdefault("FLEET_ENABLED", "1")
        print(f"[kaggle] Fleet env loaded from {path}")
        return


def main() -> None:
    _pip_install()
    _load_fleet_env()
    profile = _gpu_profile()
    profile = _maybe_reduce_batch(profile)
    manifest_path, total_tokens = _prepare_data()
    runtime_cfg = _write_runtime_config(profile, manifest_path, total_tokens)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.train.pretrain",
        "--config",
        str(runtime_cfg),
        "--output_dir",
        str(OUTPUT_DIR),
    ]
    print("[kaggle] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    src_final = OUTPUT_DIR / "final"
    if src_final.exists():
        if FINAL_DIR.exists():
            shutil.rmtree(FINAL_DIR)
        shutil.copytree(src_final, FINAL_DIR)
        print(f"[kaggle] Model saved to {FINAL_DIR}")

    # Manifest for local download script.
    manifest = {
        "output_dir": str(OUTPUT_DIR),
        "final_dir": str(FINAL_DIR),
        "gpu_profile": profile,
        "dataset_manifest": str(manifest_path),
        "total_tokens": total_tokens,
        "kaggle_dataset": os.environ.get("KAGGLE_DATASET", "wikitext103"),
    }
    (WORKING / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[kaggle] Done.")


if __name__ == "__main__":
    main()
