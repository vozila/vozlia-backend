# services/storage_service.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import boto3
from botocore.config import Config


@dataclass(frozen=True)
class StorageConfig:
    bucket: str
    region: str
    access_key_id: str
    secret_access_key: str
    endpoint_url: Optional[str] = None  # Set for Cloudflare R2, MinIO, etc.
    presign_ttl_seconds: int = 900


def load_storage_config() -> StorageConfig:
    bucket = (os.getenv("KB_S3_BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("KB_S3_BUCKET is not set")

    return StorageConfig(
        bucket=bucket,
        region=(os.getenv("KB_S3_REGION") or "us-east-1").strip(),
        access_key_id=(os.getenv("KB_S3_ACCESS_KEY_ID") or "").strip(),
        secret_access_key=(os.getenv("KB_S3_SECRET_ACCESS_KEY") or "").strip(),
        endpoint_url=(os.getenv("KB_S3_ENDPOINT_URL") or "").strip() or None,
        presign_ttl_seconds=int(os.getenv("KB_PRESIGN_TTL_SECONDS") or "900"),
    )


def _client(cfg: StorageConfig):
    # S3-compatible signing works for AWS S3 and R2.
    # Use v4 signatures; avoid retries causing long hangs in request paths.
    return boto3.client(
        "s3",
        region_name=cfg.region,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        endpoint_url=cfg.endpoint_url,
        config=Config(signature_version="s3v4", retries={"max_attempts": 2}),
    )


def presign_put_object(
    key: str,
    content_type: str,
    *,
    cfg: Optional[StorageConfig] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> str:
    cfg = cfg or load_storage_config()
    s3 = _client(cfg)

    params: Dict[str, Any] = {
        "Bucket": cfg.bucket,
        "Key": key,
        "ContentType": content_type or "application/octet-stream",
    }
    if extra_args:
        params.update(extra_args)

    return s3.generate_presigned_url(
        ClientMethod="put_object",
        Params=params,
        ExpiresIn=cfg.presign_ttl_seconds,
    )
