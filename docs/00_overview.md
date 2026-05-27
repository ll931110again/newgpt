# Overview

This repository is a **GPU-cloud-first** pipeline for building a decoder-only LLM:

1. **Data**: ingest → clean/filter → dedup → tokenize → pack → shard
2. **Pretraining**: train a base model on packed token shards
3. **Post-training**: instruction tuning (**SFT**) then preference optimization (**DPO/ORPO/KTO**)
4. **Evaluation**: automated suite + promotion gating
5. **Inference**: export and serve with **vLLM**

The implementation is organized around:

- `configs/` for **scale presets** and job configs
- `src/` for Python entrypoints
- `infra/gpu-cloud/` for cloud VM templates and job specs

## Where things run

- **Your laptop**: author configs, build images, upload code; do not train.
- **GPU VM(s)**: execute training jobs using Docker.
- **Object storage**: dataset shards + checkpoints + exported models.

## Artifacts

- Dataset manifests: `data/manifests/<dataset_version>.json`
- Training runs: `runs/<run_id>/` (metadata) + checkpoints in object storage
- Model export: `s3://<bucket>/<prefix>/models/<name_or_run_id>/`

## Performance engineering

GPU tuning follows [MIT 6.S894 Accelerated Computing](https://accelerated-computing.academy/fall25/) principles — see **`docs/23_accelerated_computing.md`**.
