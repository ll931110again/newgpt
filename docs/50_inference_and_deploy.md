# Inference and deployment (vLLM)

This repo serves models using **vLLM** on a GPU VM.\n
## Secrets\n
Do **not** commit any keys. Put secrets in `.env` (gitignored) or set them in your cloud provider UI.\n
\n
If you are using Lambda.ai, set:\n
```bash\n
export LAMBDA_API_KEY="YOUR_KEY"\n
```\n
## Run\n
Edit `infra/gpu-cloud/job_serve.yaml` to set `model_path`, then:\n
```bash
docker run --rm --gpus all --ipc=host \\\n
  --env-file .env \\\n
  -v $PWD:/workspace -w /workspace \\\n
  -p 8000:8000 \\\n
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_serve.yaml\n
```\n
## Smoke test\n
Once running, vLLM exposes an OpenAI-compatible API (configurable). You can test with curl against `/v1/completions` or `/v1/chat/completions`.\n
