# Monitoring: W&B + mlop + S3 checkpoints

Train on a GPU VM and monitor from your laptop without SSH.

## Weights & Biases

### Setup

1. Create a key at https://wandb.ai/settings
2. Add to `infra/gpu-cloud/.env` (copied to `.env` on the VM):

```bash
WANDB_API_KEY=your_key
WANDB_PROJECT=pretrainer
WANDB_ENTITY=your_team_or_username   # optional
```

3. In `configs/pretrain_1-3b.yaml`, `logging.wandb_project` is set to `pretrainer`.

W&B turns on automatically when **both** `wandb_project` (config or env) and `WANDB_API_KEY` are present.

### View runs

Open https://wandb.ai and select project **pretrainer**. You will see loss, learning rate, and throughput logged every `log_every_steps`.

## mlop.ai

[mlop](https://docs.mlop.ai) provides experiment tracking with GPU/system metrics and Hugging Face Transformers integration.

### Setup

1. Create an API key at [unified-capital → Settings → Developers](https://app.mlop.ai/o/unified-capital/settings/org/developers)
2. Add to `infra/gpu-cloud/.env`:

```bash
MLOP_API_KEY=your_key
MLOP_PROJECT=pretrainer
```

3. In `configs/pretrain_1-3b.yaml`, `logging.mlop_project` is set to `pretrainer`.

mlop turns on automatically when **both** `mlop_project` (config or `MLOP_PROJECT` env) and `MLOP_API_KEY` are present.

### View runs

Open https://app.mlop.ai and select project **pretrainer**. Runs include training loss, learning rate, and system metrics (GPU util, memory, etc.).

W&B and mlop can run together — both are enabled when their keys are set.

## S3 checkpoint uploads

Checkpoints are uploaded after each `save_steps` and when training finishes.

### Setup

Add to `.env`:

```bash
S3_BUCKET=your-bucket
S3_PREFIX=pretrainer
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
# Optional for R2/MinIO:
# S3_ENDPOINT_URL=https://...
```

### Object layout

```
s3://<bucket>/<prefix>/runs/<run_name>/checkpoint-500/...
s3://<bucket>/<prefix>/runs/<run_name>/checkpoint-1000/...
s3://<bucket>/<prefix>/runs/<run_name>/final/...
```

Example for this repo:

```
s3://my-bucket/pretrainer/runs/pretrain_1-3b/final/model.safetensors
```

### Disable uploads

In config:

```yaml
artifacts:
  s3_upload: false
```

## Apply to a running job

Changes take effect on the **next** training run. To pick up W&B, mlop, or S3 on the VM:

```bash
# 1) Fill infra/gpu-cloud/.env with WANDB_*, MLOP_*, and S3_* keys
# 2) Sync + restart pipeline
export LAMBDA_INSTANCE_IP=<ip>
export LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_rsa
./infra/gpu-cloud/lambda/bootstrap_remote.sh
```

To resume from a local checkpoint instead of restarting from scratch, set `resume_from` in `infra/gpu-cloud/job_pretrain.yaml`.

## Quick verify (before a long run)

```bash
docker run --rm --env-file .env pretrainer:latest python3 -c "
import os
from src.utils.s3_sync import s3_configured
from src.utils.wandb_setup import configure_wandb
from src.utils.mlop_setup import configure_mlop
cfg = {'run_name':'test','logging':{'wandb_project': os.getenv('WANDB_PROJECT','pretrainer'), 'mlop_project': os.getenv('MLOP_PROJECT','pretrainer')}}
print('s3:', s3_configured())
print('wandb:', configure_wandb(cfg))
print('mlop:', configure_mlop(cfg))
"
```
