# Evaluation + gating

Evaluation is run via `lm-eval` (lm-evaluation-harness).\n
## Run\n
```bash
docker run --rm --gpus all --ipc=host \\\n
  --env-file .env \\\n
  -v $PWD:/workspace -w /workspace \\\n
  pretrainer:latest src.infra.run_job --spec infra/gpu-cloud/job_eval.yaml\n
```\n
## Gating (recommended)\n
Define acceptance thresholds for the tasks you care about and only promote a model to serving if it passes.\n
\n
This repo writes:\n
- `runs/eval/lm_eval_results.json`\n
- `runs/eval/summary.json`\n
\n
Example gating policy (edit to your needs):\n
- `arc_easy` accuracy >= 0.55\n
- `hellaswag` accuracy >= 0.45\n
- no safety regression on your internal refusal tests\n
\n
You can implement gating as a simple script that loads `runs/eval/summary.json` and fails a CI step if thresholds are not met.\n
