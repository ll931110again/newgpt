# Accelerated computing playbook (6.S894 → pretrainer)

This doc maps [MIT 6.S894: Accelerated Computing (Fall 2025)](https://accelerated-computing.academy/fall25/) onto this repo. The course teaches **why** accelerators look the way they do and **how** to write fast code on them ([syllabus](https://accelerated-computing.academy/fall25/syllabus/)): SIMD, latency hiding, memory hierarchy, matrix units, and first-principles performance reasoning (spiritual follow-on to 6.106).

We do not assign CUDA homework here; we apply the same ideas to **LLM pretraining on NVIDIA GPUs** via PyTorch, Hugging Face Trainer, and tuned configs.

## Course pillars → pretrainer

| 6.S894 theme | What it means | Where in pretrainer |
|--------------|---------------|-------------------|
| **Why GPUs?** | Massive parallelism for independent work (batch × seq) | `per_device_train_batch_size`, packed sequences |
| **Memory hierarchy** | HBM is slow; SRAM/registers fast — tile & reuse | Flash Attention / SDPA, large micro-batch, `mmap` shards |
| **Latency hiding** | Overlap compute with memory & I/O | `dataloader_num_workers`, `pin_memory`, prefetch |
| **Matrix units (Tensor Cores)** | bf16/fp16 GEMMs accumulate in fp32 | `bf16`, `tf32`, `attn_implementation: auto` |
| **SIMD / warps** | Same instruction on many lanes | Handled inside cuBLAS / attention kernels |
| **Measure, then optimize** | Roofline-style: are we compute or memory bound? | `scripts/benchmark_train_step.py`, W&B throughput |
| **Scheduling** | Keep device fed, avoid host bubbles | `log_every_steps: 50`, fused optimizer, fewer syncs |

## Lab sequence → training checklist

Use this as a **pre-flight and tuning list** before long GPU runs ([labs index](https://accelerated-computing.academy/fall25/labs/)).

| Lab | Idea | Pretrainer action |
|-----|------|-------------------|
| 1–2 SIMD / massive parallelism | Expose parallelism | ↑ micro-batch until ~90–95% VRAM |
| 3 Wave / pipelines | Overlap stages | DataLoader workers + pinned memory |
| 4 Tiling & reuse | Reuse data in fast memory | Packed `npy` shards; Flash/SDPA attention |
| 5 Improved scheduling | Reduce idle time | Fused AdamW; less frequent `logging_steps` |
| 6 Tensor cores | Use TC-friendly dtypes | `bf16: true`, `tf32: true` |
| 7 Compression | Less traffic | (Optional) smaller dtypes; ZeRO for scale-out |
| 8–9 Advanced scheduling | Expert-level GPU control | `torch_compile` (config flag) if stable |
| 10 H100 matmul | Hardware-specific GEMM | Same TC path; pick `gpu_1x_*` SKU on Lambda |
| 11 TPU | Different accelerator | N/A on Lambda; same *principles* if you port |

## Config knobs (`train:` block)

See `configs/pretrain_continue_from_current.yaml` and `src/train/performance.py`.

```yaml
train:
  attn_implementation: auto      # flash_attention_2 → sdpa fallback
  optim: adamw_torch_fused
  bf16: true
  tf32: true
  per_device_train_batch_size: 12
  gradient_accumulation_steps: 1
  gradient_checkpointing: false    # true if OOM (trades compute for VRAM)
  dataloader_num_workers: 2
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 2
  torch_compile: false
```

## Measurement workflow (6.106-style)

1. **Baseline** — note `tokens/sec` and peak VRAM from benchmark script or W&B.
2. **Change one knob** — e.g. disable checkpointing *or* enable Flash, not both at once.
3. **Compare** — throughput ↑ and no OOM → keep; else revert.

On the GPU VM:

```bash
docker run --rm --gpus all --ipc=host --env-file .env \
  -v $PWD:/workspace -w /workspace pretrainer:latest \
  scripts/benchmark_train_step.py --config configs/pretrain_continue_from_current.yaml --steps 20
```

Locally (CPU smoke / import only):

```bash
uv run pytest tests/ -q
```

## First-principles sanity checks

Ask on every long run:

1. **Arithmetic intensity** — Are we doing enough matmul FLOPs per byte loaded? (Low batch / short seq → memory-bound.)
2. **Occupancy** — Is GPU util often high in `nvidia-smi`? (If not, host bottleneck or tiny batch.)
3. **Peak memory** — Are we leaving GB unused? (Increase micro-batch until ~95% VRAM.)
4. **Correctness** — Do packed labels shift by one? (`tests/test_packed_dataset.py`.)

## Multi-GPU (course: scale-out)

When a model does not fit on one GPU, use **data parallel + ZeRO** (`distributed.deepspeed` in configs, `Dockerfile.deepspeed`) — same “shard state across ranks” idea as splitting work in multi-block CUDA labs.

## References

- [6.S894 course home](https://accelerated-computing.academy/fall25/)
- [Syllabus](https://accelerated-computing.academy/fall25/syllabus/)
- [Labs](https://accelerated-computing.academy/fall25/labs/)
- Repo: `docs/20_pretraining.md`, `docs/22_monitoring.md`
