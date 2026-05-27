"""Upload training artifacts to S3-compatible object storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def s3_configured() -> bool:
    return bool(os.environ.get("S3_BUCKET") and os.environ.get("AWS_ACCESS_KEY_ID"))


def _client():
    import boto3

    kwargs = {}
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    region = os.environ.get("AWS_DEFAULT_REGION")
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _base_prefix(extra: str = "") -> str:
    root = os.environ.get("S3_PREFIX", "pretrainer").strip("/")
    extra = extra.strip("/")
    return f"{root}/{extra}" if extra else root


def upload_file(local_path: Path, s3_key: str) -> None:
    bucket = os.environ["S3_BUCKET"]
    client = _client()
    client.upload_file(str(local_path), bucket, s3_key)
    print(f"[s3] uploaded {local_path} -> s3://{bucket}/{s3_key}")


def upload_directory(local_dir: Path, s3_key_prefix: str) -> None:
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(local_dir)

    bucket = os.environ["S3_BUCKET"]
    client = _client()
    prefix = _base_prefix(s3_key_prefix).strip("/") + "/"
    n = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}{rel}"
        client.upload_file(str(path), bucket, key)
        n += 1
    print(f"[s3] uploaded {n} files from {local_dir} -> s3://{bucket}/{prefix}")


def upload_run_artifact(local_path: Path, run_name: str) -> None:
    """Upload a checkpoint dir or final model dir under runs/<run_name>/."""
    local_path = Path(local_path)
    artifact_name = local_path.name
    s3_key_prefix = f"runs/{run_name}/{artifact_name}"
    upload_directory(local_path, s3_key_prefix)
