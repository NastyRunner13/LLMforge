"""Dataset management API routes — wired to real DB + S3."""

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import verify_internal_key
from app.core.storage import (
    generate_presigned_upload_url,
    get_s3_client,
)
from app.models.db_models import DatasetStatus
from app.services import crud

router = APIRouter()


class UploadRequest(BaseModel):
    """Request body to initiate a dataset upload."""

    project_id: str
    filename: str
    content_type: str
    file_size_bytes: int


class CleaningNodeConfig(BaseModel):
    """Configuration for a single cleaning pipeline node."""

    node_type: str  # dedup, language_filter, length_filter, pii_redact, regex_filter
    params: dict = {}


class CleanRequest(BaseModel):
    """Request body to launch a cleaning pipeline."""

    nodes: list[CleaningNodeConfig]


class FormatRequest(BaseModel):
    """Request body for instruction format mapping."""

    system_column: str | None = None
    user_column: str
    assistant_column: str
    tokenizer: str = "cl100k_base"


@router.post("/upload")
async def initiate_upload(
    req: UploadRequest,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get a presigned S3 URL for direct client upload. Registers dataset record."""
    # Derive file format from extension
    file_ext = req.filename.rsplit(".", 1)[-1].lower() if "." in req.filename else "unknown"

    # Create DB record
    dataset = crud.create_dataset(
        db,
        project_id=req.project_id,
        name=req.filename.rsplit(".", 1)[0] if "." in req.filename else req.filename,
        original_filename=req.filename,
        file_format=file_ext,
        file_size_bytes=req.file_size_bytes,
    )

    # S3 key: datasets/{project_id}/{dataset_id}/{filename}
    s3_key = f"datasets/{req.project_id}/{dataset.id}/{req.filename}"
    crud.update_dataset(db, dataset.id, file_path_s3=s3_key)

    # Generate presigned upload URL
    upload_url = generate_presigned_upload_url(
        bucket=settings.S3_BUCKET_DATASETS,
        key=s3_key,
        content_type=req.content_type,
        expires_in=900,
    )

    return {"upload_url": upload_url, "dataset_id": dataset.id}


@router.post("/{dataset_id}/confirm-upload")
async def confirm_upload(
    dataset_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Called after client completes S3 upload. Triggers processing."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    crud.update_dataset_status(db, dataset_id, DatasetStatus.PROCESSING)

    # Queue file conversion/parsing task
    from app.core.celery_app import celery_app

    celery_app.send_task(
        "data.convert_file_format",
        args=[dataset_id, dataset.file_format or "jsonl", "jsonl"],
    )

    return {"dataset_id": dataset_id, "status": "processing"}


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get dataset metadata and status."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "name": dataset.name,
        "original_filename": dataset.original_filename,
        "file_format": dataset.file_format,
        "file_size_bytes": dataset.file_size_bytes,
        "row_count": dataset.row_count,
        "token_count": dataset.token_count,
        "schema_json": dataset.schema_json,
        "status": dataset.status.value if dataset.status else None,
        "error_message": dataset.error_message,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
    }


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    page: int = 1,
    page_size: int = 50,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Return paginated rows from the dataset."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.file_path_s3:
        return {"rows": [], "total": 0, "page": page}

    try:
        client = get_s3_client()
        response = client.get_object(Bucket=settings.S3_BUCKET_DATASETS, Key=dataset.file_path_s3)
        raw_data = response["Body"].read().decode("utf-8")

        # Parse JSONL
        rows = []
        for line in raw_data.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        total = len(rows)
        start = (page - 1) * page_size
        end = start + page_size
        return {"rows": rows[start:end], "total": total, "page": page}
    except Exception:
        return {"rows": [], "total": 0, "page": page}


@router.post("/{dataset_id}/clean")
async def launch_cleaning(
    dataset_id: str,
    req: CleanRequest,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Launch a cleaning pipeline job."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    crud.update_dataset_status(db, dataset_id, DatasetStatus.CLEANING)

    pipeline_config = [{"node_type": n.node_type, "params": n.params} for n in req.nodes]
    crud.update_dataset(db, dataset_id, cleaning_config_json=pipeline_config)

    # Queue Celery task
    from app.core.celery_app import celery_app

    task = celery_app.send_task(
        "data.run_cleaning_pipeline",
        args=[dataset_id, pipeline_config],
    )

    return {"job_id": task.id, "status": "queued"}


@router.get("/{dataset_id}/clean/status")
async def cleaning_status(
    dataset_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get cleaning pipeline job status and progress."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "status": dataset.status.value if dataset.status else None,
        "error_message": dataset.error_message,
    }


@router.post("/{dataset_id}/format")
async def format_dataset(
    dataset_id: str,
    req: FormatRequest,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Apply instruction format mapping and return token count."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset or not dataset.file_path_s3:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        client = get_s3_client()
        response = client.get_object(Bucket=settings.S3_BUCKET_DATASETS, Key=dataset.file_path_s3)
        raw_data = response["Body"].read().decode("utf-8")

        rows = []
        for line in raw_data.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Format into instruction tuples
        formatted_rows = []
        for row in rows:
            formatted = {}
            if req.system_column and req.system_column in row:
                formatted["system"] = str(row[req.system_column])
            if req.user_column in row:
                formatted["user"] = str(row[req.user_column])
            if req.assistant_column in row:
                formatted["assistant"] = str(row[req.assistant_column])
            if formatted.get("user") and formatted.get("assistant"):
                formatted_rows.append(formatted)

        # Count tokens
        total_tokens = 0
        try:
            import tiktoken

            enc = tiktoken.get_encoding(req.tokenizer)
            for row in formatted_rows:
                for val in row.values():
                    total_tokens += len(enc.encode(val))
        except Exception:
            # Fallback: rough estimate (4 chars ≈ 1 token)
            for row in formatted_rows:
                for val in row.values():
                    total_tokens += len(val) // 4

        # Save formatted data back to S3
        formatted_key = dataset.file_path_s3.rsplit(".", 1)[0] + "_formatted.jsonl"
        formatted_content = "\n".join(json.dumps(r) for r in formatted_rows)
        client.put_object(
            Bucket=settings.S3_BUCKET_DATASETS,
            Key=formatted_key,
            Body=formatted_content.encode("utf-8"),
            ContentType="application/jsonl",
        )

        crud.update_dataset(
            db,
            dataset_id,
            row_count=len(formatted_rows),
            token_count=total_tokens,
            file_path_s3=formatted_key,
            status=DatasetStatus.READY,
        )

        return {"formatted_rows": len(formatted_rows), "total_tokens": total_tokens}
    except HTTPException:
        raise
    except Exception as e:
        crud.update_dataset_status(db, dataset_id, DatasetStatus.FAILED, error_message=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Delete dataset and S3 artifacts."""
    dataset = crud.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete from S3
    if dataset.file_path_s3:
        try:
            client = get_s3_client()
            client.delete_object(Bucket=settings.S3_BUCKET_DATASETS, Key=dataset.file_path_s3)
        except Exception:
            pass  # S3 cleanup is best-effort

    crud.delete_dataset(db, dataset_id)
    return {"deleted": True}
