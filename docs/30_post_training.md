# Post-training (SFT + DPO/ORPO/KTO)

This stage turns a base model into an instruction-following and preference-aligned model.\n
## SFT\n
- Config: `configs/sft_1-3b.yaml` (and future scale presets)\n
- Job spec: `infra/gpu-cloud/job_sft.yaml`\n
\n
Run:\n
```bash
docker run --rm --gpus all --ipc=host \\\n
  --env-file .env \\\n
  -v $PWD:/workspace -w /workspace \\\n
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_sft.yaml\n
```\n

### SFT on your pretrained checkpoint (recommended)

After pretraining, point SFT at your saved model directory (the `final/` folder).

#### Get an SFT dataset (download + convert)

This repo includes a small helper that downloads the `yahma/alpaca-cleaned` dataset from Hugging Face and writes:

- `data/sft/train.jsonl`
- `data/sft/eval.jsonl`

Run (from repo root):

```bash
uv run python -m src.data.download_sft_alpaca --out-dir data/sft --max-train 20000 --max-eval 1000
```

This repo includes:

- Config: `configs/sft_from_pretrained.yaml`
- Job spec: `infra/gpu-cloud/job_sft_from_pretrained.yaml`

Run:

```bash
docker run --rm --gpus all --ipc=host \
  --env-file .env \
  -v $PWD:/workspace -w /workspace \
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_sft_from_pretrained.yaml
```
## Preference optimization\n
Default is **DPO** (direct preference optimization).\n
- Config: `configs/dpo_1-3b.yaml`\n
- Job spec: `infra/gpu-cloud/job_dpo.yaml`\n
