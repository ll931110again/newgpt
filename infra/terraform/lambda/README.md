# Lambda GPU — Terraform

Declarative provisioning for Lambda Cloud GPU instances, replacing the bash
`provision.sh` / `terminate.sh` scripts for infrastructure lifecycle.

Uses the community provider [`elct9620/lambdalabs`](https://registry.terraform.io/providers/elct9620/lambdalabs/latest/docs) (not official Lambda).

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- Lambda API key in `infra/gpu-cloud/.env` as `LAMBDA_API_KEY`
- SSH key registered at https://cloud.lambda.ai/ssh-keys

## Quick start

From repo root:

```bash
cp infra/terraform/lambda/terraform.tfvars.example infra/terraform/lambda/terraform.tfvars
# edit terraform.tfvars (region, instance type, ssh key name)

./infra/gpu-cloud/lambda/provision_tf.sh
./infra/gpu-cloud/lambda/bootstrap_remote.sh   # reads IP from Terraform output
```

## Destroy (stop billing)

```bash
./infra/gpu-cloud/lambda/destroy_tf.sh
```

## What Terraform manages vs scripts

| Terraform | Bash scripts (unchanged) |
|-----------|-------------------------|
| GPU instance create/destroy | Docker build, data prep, training |
| Instance IP / ID outputs | `bootstrap_remote.sh`, job YAML |
| SSH key name (reference) | W&B, S3 uploads |

## API key env var

The provider reads `LAMBDALABS_API_KEY`. Wrapper scripts map from `LAMBDA_API_KEY`
in `infra/gpu-cloud/.env` automatically.

## Legacy bash provision

`infra/gpu-cloud/lambda/provision.sh` remains available as a fallback when
Terraform is not installed.
