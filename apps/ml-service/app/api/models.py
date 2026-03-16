"""Model registry and management API routes — wired to real DB + S3."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.core.security import verify_internal_key
from app.core.storage import generate_presigned_download_url
from app.services import crud

router = APIRouter()


class DeployRequest(BaseModel):
    """Request body to deploy a model to inference."""
    gpu_type: str = "A100_40GB"
    replicas: int = 1


@router.get("/")
async def list_models(
    project_id: str | None = None,
    page: int = 1,
    limit: int = 20,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """List registered models, optionally filtered by project."""
    models = crud.list_models(db, project_id=project_id, skip=(page - 1) * limit, limit=limit)
    return {
        "models": [
            {
                "id": m.id,
                "project_id": m.project_id,
                "run_id": m.run_id,
                "name": m.name,
                "base_model": m.base_model,
                "architecture": m.architecture,
                "param_count": m.param_count,
                "training_method": m.training_method,
                "quantization": m.quantization,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in models
        ],
    }


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get model details including architecture, training run, and checkpoint info."""
    model = crud.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": model.id,
        "project_id": model.project_id,
        "run_id": model.run_id,
        "checkpoint_id": model.checkpoint_id,
        "name": model.name,
        "base_model": model.base_model,
        "architecture": model.architecture,
        "param_count": model.param_count,
        "training_method": model.training_method,
        "quantization": model.quantization,
        "s3_path": model.s3_path,
        "created_at": model.created_at.isoformat() if model.created_at else None,
    }


@router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    req: DeployRequest,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Deploy model to a managed inference endpoint."""
    model = crud.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create endpoint record
    endpoint = crud.create_endpoint(
        db, model_id=model_id, gpu_type=req.gpu_type, replicas=req.replicas,
    )

    # Queue deployment task
    from app.core.celery_app import celery_app
    celery_app.send_task(
        "training.deploy_model",
        args=[model_id, req.gpu_type, req.replicas],
        queue="training",
    )

    return {"endpoint_id": endpoint.id, "status": "starting"}


@router.get("/{model_id}/download")
async def download_model(
    model_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get signed S3 URL to download model weights."""
    model = crud.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.s3_path:
        raise HTTPException(status_code=400, detail="Model has no stored weights")

    download_url = generate_presigned_download_url(
        bucket=settings.S3_BUCKET_MODELS,
        key=model.s3_path,
        expires_in=3600,
    )
    return {"download_url": download_url}


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Delete a model and its S3 artifacts."""
    model = crud.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Clean up S3
    if model.s3_path:
        try:
            from app.core.storage import get_s3_client
            client = get_s3_client()
            client.delete_object(Bucket=settings.S3_BUCKET_MODELS, Key=model.s3_path)
        except Exception:
            pass

    crud.delete_model(db, model_id)
    return {"deleted": True}
