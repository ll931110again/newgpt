# Lambda Cloud: provision 1×A100 for pretraining

Training runs on a **Lambda GPU VM**, not your laptop. Use **Terraform** (recommended) or the legacy **bash REST scripts** in `infra/gpu-cloud/lambda/`.

## Prerequisites

1. **Lambda API key** in `infra/gpu-cloud/.env`:
   ```bash
   LAMBDA_API_KEY=...
   WANDB_API_KEY=...   # optional but recommended for training
   ```
2. **SSH public key registered** in Lambda: https://cloud.lambda.ai/ssh-keys  
   Terraform and bash scripts use a named key (e.g. `pretrainer-laptop`).
3. **Local tools**: `terraform` (preferred) or `curl` + `jq` (legacy), plus `rsync`, `ssh`.

---

## Option A — Terraform (recommended)

Declarative provisioning lives in `infra/terraform/lambda/`. Wrapper scripts load `LAMBDA_API_KEY` from `.env` and map it to `LAMBDALABS_API_KEY` for the provider.

### 1) Configure instance type and SSH key

```bash
cp infra/terraform/lambda/terraform.tfvars.example infra/terraform/lambda/terraform.tfvars
# edit: region_name, instance_type_name, ssh_key_names
```

Common instance types (availability varies by account/region):

| Goal | `instance_type_name` |
|------|----------------------|
| A100 40GB (single GPU) | `gpu_1x_a100_sxm4` |
| H100 80GB (single GPU) | `gpu_1x_h100_sxm5` |

Lambda may **not** offer a 1× A100 80GB SKU; run `./infra/gpu-cloud/lambda/list_types.sh` to see your catalog.

### 2) Launch instance

```bash
chmod +x infra/gpu-cloud/lambda/*.sh
./infra/gpu-cloud/lambda/provision_tf.sh
```

This runs `terraform init` + `apply`, writes `.lambda_instance_ip` / `.lambda_instance_id`, and prints SSH + bootstrap commands.

Optional: pass Terraform flags, e.g. `./infra/gpu-cloud/lambda/provision_tf.sh -auto-approve`.

### 3) Bootstrap pretraining

IP is auto-detected from Terraform output or `.lambda_instance_ip`:

```bash
export LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_rsa   # if not your default key
./infra/gpu-cloud/lambda/bootstrap_remote.sh
```

### 4) Inspect Terraform outputs

```bash
./infra/gpu-cloud/lambda/tf_outputs.sh
# or: cd infra/terraform/lambda && terraform output instance_ip
```

### 5) Destroy when done (stop billing)

```bash
./infra/gpu-cloud/lambda/destroy_tf.sh
```

See also `infra/terraform/lambda/README.md`.

---

## Option B — Bash REST API (legacy fallback)

Use when Terraform is not installed. Scripts call `https://cloud.lambdalabs.com/api/v1` directly.

### 1) List available GPU types (optional)

```bash
chmod +x infra/gpu-cloud/lambda/*.sh
./infra/gpu-cloud/lambda/list_types.sh
```

Look for a **1× A100** line. If auto-detection picks the wrong SKU, set:

```bash
export LAMBDA_INSTANCE_TYPE=gpu_1x_a100_sxm4   # use name from list_types
```

### 2) Launch 1×A100 (auto region + SSH key)

```bash
./infra/gpu-cloud/lambda/provision.sh
```

**Important:** Lambda's API catalog may **not include a 1× A100 80GB** SKU (often only `gpu_8x_a100_80gb_sxm4`). Run `list_types.sh` to see what your account offers today.

- **Want 80GB VRAM on a single GPU now?** Use H100 fallback:
  ```bash
  LAMBDA_FALLBACK_80GB=h100 ./infra/gpu-cloud/lambda/provision.sh
  ```
- **Want A100 specifically (40GB SKU)?** Pin the type:
  ```bash
  LAMBDA_INSTANCE_TYPE=gpu_1x_a100_sxm4 ./infra/gpu-cloud/lambda/provision.sh
  ```
- **Wait for capacity:**
  ```bash
  ./infra/gpu-cloud/lambda/poll_provision.sh gpu_1x_a100_sxm4
  ```

This will:

- Pick **1× A100 80GB** when that SKU exists **and** has capacity (otherwise warn + fallback rules above)
- Pick the **first region with capacity**
- Launch the instance and wait until **active**
- Write the instance id to `.lambda_instance_id`
- Print the **SSH command**

Optional env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LAMBDA_INSTANCE_TYPE` | `auto` | Instance type name from API |
| `LAMBDA_REGION` | `auto` | Region with capacity |
| `LAMBDA_SSH_KEY_NAME` | first key | SSH key name in Lambda |
| `LAMBDA_INSTANCE_NAME` | `pretrainer-pretrain` | Instance display name |
| `LAMBDA_FILESYSTEM` | (none) | Attach a Lambda filesystem |

### 3) Bootstrap pretraining on the VM

After provision prints an IP:

```bash
export LAMBDA_INSTANCE_IP=<ip-from-provision>
# optional if your key isn't default:
# export LAMBDA_SSH_PRIVATE_KEY=~/.ssh/id_ed25519

./infra/gpu-cloud/lambda/bootstrap_remote.sh
```

This **rsyncs the repo** to the VM, builds Docker, and runs the pretrain pipeline.

---

## Before bootstrap: dataset + config

On the VM (or sync from laptop), you still need:

1. **Tokenized dataset** — see `docs/10_data_pipeline.md`
2. **`configs/pretrain_1-3b.yaml`** — set `data.dataset_manifest` to your manifest (e.g. `data/manifests/v1.json`)

For a quick sanity check, create `data/raw/train.jsonl` and run tokenization on the VM before the full pretrain job.

## SSH manually (debugging)

```bash
ssh ubuntu@<ip>
nvidia-smi
cd ~/pretrainer
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Auto-terminate after training (laptop can disconnect)

Training on the VM can **shut down the Lambda instance** when the job ends, without your laptop staying online.

**Requirements on the VM** (synced via `bootstrap_remote.sh`):

1. `.env` includes `LAMBDA_API_KEY` (same key as provision).
2. `.lambda_instance_id` in the repo root on the VM (written by `provision.sh` / `provision_tf.sh` and rsynced).

**How it works:** `remote_watch_and_teardown.sh` runs on the VM under `nohup`, polls until `src.train.pretrain` exits, waits briefly for S3 uploads, then calls the Lambda terminate API.

`bootstrap_remote.sh` starts this watcher automatically when `AUTO_TERMINATE=1` (default).

**Manual start** (e.g. if you launched training by hand):

```bash
# On the VM
cd ~/pretrainer
chmod +x infra/gpu-cloud/lambda/remote_*.sh
nohup ./infra/gpu-cloud/lambda/remote_watch_and_teardown.sh >> /tmp/pretrainer_teardown.log 2>&1 &
echo $! > /tmp/pretrainer_teardown.pid
```

**Disable** auto-terminate: `export AUTO_TERMINATE=0` before training or in `.env`.

**Logs:** `/tmp/pretrainer_teardown.log` on the VM (until the instance is destroyed).

`finish_and_teardown.sh` on your laptop is optional — it polls over SSH, downloads the model, then terminates. Use it when you want the artifact on your machine without S3.

## Terminate (bash legacy)

If you used `provision.sh` instead of Terraform:

```bash
./infra/gpu-cloud/lambda/terminate.sh
# or: ./infra/gpu-cloud/lambda/terminate.sh <instance-id>
```

## Troubleshooting

- **No capacity**: run `list_types.sh` and try another region, or re-run provision later.
- **Wrong GPU SKU**: set `instance_type_name` in `terraform.tfvars` or `LAMBDA_INSTANCE_TYPE` for bash.
- **SSH fails**: confirm the key name in Lambda matches the private key you use (`LAMBDA_SSH_PRIVATE_KEY`).
- **Docker permission denied** on VM: `sudo usermod -aG docker $USER && newgrp docker`.
- **Terraform provider errors**: ensure `LAMBDA_API_KEY` is set in `.env`; wrapper scripts export `LAMBDALABS_API_KEY`.

## API reference (bash scripts)

- Base URL: `https://cloud.lambdalabs.com/api/v1`
- Auth: HTTP basic, username = API key, password empty (`curl -u "$LAMBDA_API_KEY:"`)
- Launch: `POST /instance-operations/launch`
