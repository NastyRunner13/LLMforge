"""S3 client for dataset, checkpoint, and model storage."""

import boto3
from botocore.config import Config as BotoConfig

from app.core.config import settings


def get_s3_client():
    """Create and return an S3 client configured for MinIO/S3."""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


def generate_presigned_upload_url(
    bucket: str, key: str, content_type: str = "application/octet-stream", expires_in: int = 900
) -> str:
    """Generate a presigned URL for direct client upload to S3."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires_in,
    )


def generate_presigned_download_url(bucket: str, key: str, expires_in: int = 900) -> str:
    """Generate a presigned URL for downloading from S3."""
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def ensure_buckets_exist():
    """Create required S3 buckets if they don't exist (for local dev with MinIO)."""
    client = get_s3_client()
    buckets_needed = [
        settings.S3_BUCKET_DATASETS,
        settings.S3_BUCKET_CHECKPOINTS,
        settings.S3_BUCKET_MODELS,
    ]
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    for bucket in buckets_needed:
        if bucket not in existing:
            client.create_bucket(Bucket=bucket)
            print(f"✅ Created S3 bucket: {bucket}")
