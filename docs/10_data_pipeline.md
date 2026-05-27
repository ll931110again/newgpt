# Data pipeline

Build a **versioned dataset** and **token shards** for pretraining.

## Outputs

- `data/manifests/<dataset_version>.json`
- `data/shards/shard_*.npy` (packed int32 tokens)

## Quick path (on GPU VM)

From repo root after `docker build -t pretrainer:latest .`:

```bash
chmod +x infra/gpu-cloud/lambda/run_pretrain_pipeline.sh
./infra/gpu-cloud/lambda/run_pretrain_pipeline.sh
```

This will:

1. Download **WikiText-103** → `data/raw/wikitext.jsonl`
2. Tokenize with **gpt2** → `data/manifests/v1.json` + shards
3. Launch **pretrain** using `configs/pretrain_1-3b.yaml`

Optional env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASET_VERSION` | `v1` | Manifest name |
| `TOKENIZER` | `gpt2` | HF tokenizer |
| `MAX_ROWS` | `0` | Cap jsonl rows (`10000` for a quick smoke) |
| `PRETRAIN_CONFIG` | `configs/pretrain_1-3b.yaml` | Training config |

## Manual steps

### 1) Prepare jsonl

```bash
docker run --rm --gpus all -v $PWD:/workspace -w /workspace pretrainer:latest \
  src.data.prepare_jsonl \
  --dataset wikitext --config wikitext-103-raw-v1 --split train \
  --out data/raw/wikitext.jsonl
```

### 2) Tokenize + shard

```bash
docker run --rm --gpus all -v $PWD:/workspace -w /workspace pretrainer:latest \
  src.data.tokenize_and_shard \
  --input data/raw/wikitext.jsonl \
  --tokenizer gpt2 \
  --out_dir data \
  --dataset_version v1
```

### 3) Point config at manifest

In `configs/pretrain_1-3b.yaml`:

```yaml
data:
  dataset_manifest: data/manifests/v1.json
  format: packed_npy
```

### 4) Train

```bash
docker run --rm --gpus all --ipc=host --env-file .env \
  -v $PWD:/workspace -w /workspace \
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_pretrain.yaml
```

## Your own corpus

Replace step 1 with your jsonl (`{"text": "..."}` per line), then run tokenize + train.

For large corpora, upload shards to S3 and sync to the VM before training.
