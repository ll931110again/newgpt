# pretrainer

End-to-end pipeline for training a decoder-only LLM on **GPU cloud**: **data → pretraining → post-training (SFT + DPO/ORPO/KTO) → evaluation → inference (vLLM)**.

## What you get
- **Reproducible configs** for 1–3B, 7–13B, and 30–70B scale presets in `configs/`.
- **Runnable entrypoints** in `src/` for pretraining, SFT, preference optimization, eval, and serving.
- **GPU-cloud templates** (RunPod/Lambda/Paperspace-style VMs) in `infra/gpu-cloud/`.
- **Docker-first** workflow for training and serving.

## Quickstart (local only builds; training runs on cloud GPUs)

### 1) Build the container
```bash
docker build -t pretrainer:latest .
```

### 2) Prepare object storage (recommended)
This repo assumes an **S3-compatible** bucket (AWS S3, Cloudflare R2, MinIO, etc.) for checkpoints and dataset shards.

Set these environment variables (see `infra/gpu-cloud/env.example`):
- `S3_ENDPOINT_URL` (optional for AWS S3)
- `S3_BUCKET`
- `S3_PREFIX` (e.g. `pretrainer`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

### 3) Data pipeline
See `docs/10_data_pipeline.md`.

### 4) Provision Lambda GPU (Terraform)

See `docs/21_lambda_provision.md`:

```bash
cp infra/terraform/lambda/terraform.tfvars.example infra/terraform/lambda/terraform.tfvars
./infra/gpu-cloud/lambda/provision_tf.sh
./infra/gpu-cloud/lambda/bootstrap_remote.sh
```

Legacy bash API scripts remain in `infra/gpu-cloud/lambda/provision.sh`.

### 5) Pretraining
See `docs/20_pretraining.md`.

### 6) Post-training (SFT + DPO/ORPO/KTO)
See `docs/30_post_training.md`.

### 7) Evaluation
See `docs/40_evaluation.md`.

### 8) Monitoring (W&B + S3 checkpoints)
See `docs/22_monitoring.md`.

### 9) Inference deployment (vLLM)
See `docs/50_inference_and_deploy.md`.

## Directory map
- `docs/` end-to-end documentation
- `configs/` scale presets and job configs (YAML)
- `src/` Python entrypoints and utilities
- `infra/gpu-cloud/` GPU-cloud VM templates and job specs
- `infra/terraform/` Terraform modules (Lambda provisioning)
- `docker/` container helpers

# newgpt
