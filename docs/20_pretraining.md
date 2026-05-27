# Pretraining

Pretraining runs on GPU VMs using Docker + DeepSpeed ZeRO-3.

## Prereqs

- A GPU VM (RunPod/Lambda/Paperspace-style)
- Docker + NVIDIA container runtime
- Dataset manifest + shards (see `docs/10_data_pipeline.md`)

## Run (single node)

Edit `infra/gpu-cloud/job_pretrain.yaml` to point at a config preset, then:

```bash
docker build -t pretrainer:latest .
docker run --rm --gpus all --ipc=host \
  --env-file .env \
  -v $PWD:/workspace -w /workspace \
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_pretrain.yaml
```

## Train a better model (start from a baseline)

The fastest way to get **more coherent generations** is to **continue pretraining from a strong baseline** instead of random init.

This repo includes a starting point config for continuing from `gpt2-xl`:

- Config: `configs/pretrain_from_gpt2-xl.yaml`
- Job spec: `infra/gpu-cloud/job_pretrain_from_gpt2xl.yaml`

Important: `gpt2-xl` has a default context length of **1024**, so you must pack your dataset with `seq_len=1024` for this run.

Run:

```bash
docker run --rm --gpus all --ipc=host \
  --env-file .env \
  -v $PWD:/workspace -w /workspace \
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_pretrain_from_gpt2xl.yaml
```

## Multi-node

For multi-node, use your provider’s mechanism to provision multiple VMs on the same private network and run `torchrun` with a shared rendezvous.

This repo’s `src/train/pretrain.py` is written to support `torchrun` and reads its distributed settings from environment.

## Performance (6.S894 accelerated computing)

See **`docs/23_accelerated_computing.md`** for the full mapping from [MIT 6.S894](https://accelerated-computing.academy/fall25/) (architecture + labs + measurement).

Pretrain configs enable:

- **`attn_implementation: auto`** — Flash Attention 2 when `flash-attn` is installed, else PyTorch **SDPA**
- **`optim: adamw_torch_fused`** — fused AdamW on GPU
- **`dataloader_num_workers`** + **`pin_memory`** — overlap host packing with GPU steps
- **`gradient_checkpointing: false`** when VRAM allows (set `true` if OOM)

Benchmark on the VM:

```bash
docker run --rm --gpus all --ipc=host --env-file .env \
  -v $PWD:/workspace -w /workspace pretrainer:latest \
  scripts/benchmark_train_step.py --config configs/pretrain_continue_from_current.yaml
```

Tests: `uv run pytest tests/`

## Scale presets

- `configs/pretrain_1-3b.yaml`
- `configs/pretrain_7-13b.yaml`
- `configs/pretrain_30-70b.yaml`
