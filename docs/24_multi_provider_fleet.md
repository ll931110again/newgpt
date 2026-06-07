# Multi-Provider Fleet Training

Train one logical model **in parallel** across free GPU providers (Kaggle, Paperspace) with checkpoint-level **FedAvg** synchronization. If one provider dies, the others keep going; you lose at most one sync window of work from the failed worker.

## Why not NCCL / torchrun across clouds?

Standard DDP and DeepSpeed ZeRO need a low-latency private network between GPUs. Kaggle, Paperspace, and Colab run on isolated networks—you cannot `torchrun` across them. Instead, each worker trains on a **disjoint data shard** and periodically uploads checkpoints; the orchestrator **averages weights** (FedAvg) into a canonical model.

Future upgrade: **DiLoCo** (FedAvg + outer optimizer step) for better cross-worker convergence.

## Architecture

```
Laptop (orchestrator)          Kaggle worker 0        Paperspace worker 1
       |                              |                        |
       |---- launch parallel -------->|                        |
       |---- launch parallel ------------------------------->|
       |                              |                        |
 runs/fleet/<run>/                   shard 0,3,6...           shard 1,4,7...
       |<---- sync checkpoints -------|                        |
       |<---- sync checkpoints -------------------------------|
       |---- FedAvg merge ----------> canonical/checkpoint-*  |
       |---- resume dead workers ---> (optional)               |
```

## Quickstart

1. Copy the example fleet spec:

```bash
cp infra/fleet/fleet.example.yaml infra/fleet/fleet.yaml
```

2. Set API keys in `.env` (`KAGGLE_API_TOKEN`, `PAPERSPACE_API_KEY`).

3. Launch all enabled providers in parallel:

```bash
chmod +x scripts/fleet_train.sh
./scripts/fleet_train.sh start
```

4. Poll, sync, and merge:

```bash
./scripts/fleet_train.sh status
./scripts/fleet_train.sh sync
./scripts/fleet_train.sh merge
```

5. Relaunch failed workers after a provider outage:

```bash
./scripts/fleet_train.sh resume
```

6. Or run a background watch loop (sync + merge every 2 minutes):

```bash
./scripts/fleet_train.sh watch
```

## Fleet config

See [`infra/fleet/fleet.example.yaml`](../infra/fleet/fleet.example.yaml):

- `run_name` — logical run id; state under `runs/fleet/<run_name>/`
- `train_config` — base YAML passed to `src.train.pretrain`
- `sync_every_steps` — FedAvg interval (align with `logging.save_every_steps`)
- `providers` — list of `kaggle`, `lightning`, and/or `paperspace` workers (both remote)

**Lightning AI** uses [`lightning-sdk`](https://pypi.org/project/lightning-sdk/) via [`scripts/lightning_train.sh`](../scripts/lightning_train.sh). Set `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` in `.env` (Account → Keys → Programmatic Login). Optional: `LIGHTNING_TEAMSPACE`, `LIGHTNING_MACHINE` (default `T4`).

**Paperspace** launches a remote Gradient notebook via [`scripts/paperspace_train.sh`](../scripts/paperspace_train.sh).

Set in `.env`:
- `PAPERSPACE_API_KEY` — from [Team Settings → API Keys](https://console.paperspace.com/t2y5yctyb8/settings/profile)
- `PAPERSPACE_TEAM_ID` — team slug from the console URL (e.g. `t2y5yctyb8` in `https://console.paperspace.com/t2y5yctyb8/...`)
- `PAPERSPACE_PROJECT_ID` — **separate** Gradient project id for notebooks (open **Notebooks** in the console, create/select a project, copy its id — often `prj…` or a short id like `psukfyemho7`). This is **not** the team slug.

For dev-only testing without remote GPUs, add `local: true` under a paperspace provider entry.

## Worker environment

The orchestrator injects:

| Variable | Meaning |
|----------|---------|
| `FLEET_ENABLED` | `1` on fleet workers |
| `FLEET_WORKER_ID` | e.g. `kaggle-0` |
| `FLEET_RANK` / `FLEET_WORLD_SIZE` | Data shard partition |
| `FLEET_SHARD_IDS` | Explicit shard indices |
| `FLEET_SYNC_EVERY_STEPS` | Sync interval |
| `FLEET_SYNC_MODE` | `session` (Kaggle) or `continuous` (Paperspace) |
| `FLEET_CANONICAL_INIT` | Path to merged weights |

Kaggle reads these from `fleet_worker.json` bundled with the kernel dataset.

## Local store layout

```
runs/fleet/<run_name>/
  state.json
  canonical/checkpoint-sync-<N>/
  workers/<worker_id>/checkpoint-<step>/
  heartbeats/<worker_id>.json
  signals/reload.json
```

## Resilience

- **Heartbeat** on each checkpoint save
- **Stale detection** marks workers dead after ~2h without heartbeat
- **Shard reassignment** gives dead worker shards to survivors
- **Quorum merge** combines all available checkpoints (≥1)

## Phase 2: S3 backend

When S3 is configured, set `store_backend: s3` (planned) so workers upload directly and the orchestrator only merges—lower latency than laptop relay.

## Paid providers

Use `infra/gpu-cloud/` (Lambda, etc.) for single large-GPU jobs. Fleet mode targets **free-tier** providers only.
