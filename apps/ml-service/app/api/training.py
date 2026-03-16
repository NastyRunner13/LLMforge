"""Training orchestration API routes — wired to real DB + Celery."""

import contextlib
import json

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import verify_internal_key
from app.models.db_models import RunStatus
from app.services import crud

router = APIRouter()


class LoraConfig(BaseModel):
    """LoRA-specific configuration."""

    r: int = 16
    alpha: int = 32
    target_modules: list[str] = ["q_proj", "v_proj"]
    dropout: float = 0.05


class TrainingConfig(BaseModel):
    """Full training configuration payload."""

    project_id: str
    dataset_id: str
    base_model: str  # HuggingFace model ID
    method: str = "lora"  # sft, lora, qlora
    lora_config: LoraConfig | None = None

    # Hyperparameters
    learning_rate: float = 2e-4
    scheduler: str = "cosine"
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_steps: int = -1
    num_epochs: int = 3
    context_length: int = 2048
    mixed_precision: str = "bf16"

    # Compute
    gpu_type: str = "A100_40GB"
    gpu_count: int = 1

    # Checkpointing
    checkpoint_steps: int = 500
    max_checkpoints: int = 3
    resume_from_checkpoint: str | None = None

    # Metadata
    experiment_name: str = ""
    tags: list[str] = []


def _estimate_cost(config: TrainingConfig) -> float:
    """Rough credit estimate based on GPU type and training steps."""
    gpu_credits_per_hour = {
        "T4": 5,
        "L4": 8,
        "A10G": 10,
        "A100_40GB": 20,
        "A100_80GB": 35,
    }
    credits_hr = gpu_credits_per_hour.get(config.gpu_type, 20)
    estimated_hours = (
        (config.num_epochs * 2) if config.max_steps == -1 else (config.max_steps / 500)
    )
    return round(credits_hr * estimated_hours * config.gpu_count, 2)


@router.post("/runs")
async def launch_training(
    config: TrainingConfig,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Launch a training job. Validates config, checks credits, queues job."""
    # Verify dataset exists and is ready
    dataset = crud.get_dataset(db, config.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status and dataset.status.value != "ready":
        raise HTTPException(
            status_code=400, detail=f"Dataset is not ready (status: {dataset.status.value})"
        )

    estimated_cost = _estimate_cost(config)

    # Build config dicts for storage
    model_config = {
        "base_model": config.base_model,
        "method": config.method,
        "lora_config": config.lora_config.model_dump() if config.lora_config else None,
    }
    training_config = {
        "learning_rate": config.learning_rate,
        "scheduler": config.scheduler,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "max_steps": config.max_steps,
        "num_epochs": config.num_epochs,
        "context_length": config.context_length,
        "mixed_precision": config.mixed_precision,
        "checkpoint_steps": config.checkpoint_steps,
        "max_checkpoints": config.max_checkpoints,
        "resume_from_checkpoint": config.resume_from_checkpoint,
    }

    # Create DB record
    run = crud.create_training_run(
        db,
        project_id=config.project_id,
        dataset_id=config.dataset_id,
        model_config=model_config,
        training_config=training_config,
        experiment_name=config.experiment_name,
        tags=config.tags,
        gpu_type=config.gpu_type,
        gpu_count=config.gpu_count,
    )

    # Queue Celery task
    from app.core.celery_app import celery_app

    full_config = {**model_config, **training_config, "dataset_id": config.dataset_id}
    celery_app.send_task(
        "training.launch_training_job",
        args=[run.id, full_config],
        queue="training",
    )

    return {"run_id": run.id, "status": "queued", "estimated_cost": estimated_cost}


@router.get("/runs")
async def list_runs(
    project_id: str | None = None,
    status: str | None = None,
    page: int = 1,
    limit: int = 20,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """List training runs, optionally filtered by project or status."""
    runs = crud.list_training_runs(
        db,
        project_id=project_id,
        status=status,
        skip=(page - 1) * limit,
        limit=limit,
    )
    return {
        "runs": [
            {
                "id": r.id,
                "project_id": r.project_id,
                "dataset_id": r.dataset_id,
                "experiment_name": r.experiment_name,
                "status": r.status.value if r.status else None,
                "current_step": r.current_step,
                "total_steps": r.total_steps,
                "gpu_type": r.gpu_type,
                "cost_credits": r.cost_credits,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "model_config": r.model_config_json,
            }
            for r in runs
        ],
    }


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get full run detail: config, status, cost, checkpoints."""
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    checkpoints = crud.list_checkpoints(db, run_id)
    return {
        "id": run.id,
        "project_id": run.project_id,
        "dataset_id": run.dataset_id,
        "experiment_name": run.experiment_name,
        "tags": run.tags,
        "model_config": run.model_config_json,
        "training_config": run.training_config_json,
        "status": run.status.value if run.status else None,
        "error_message": run.error_message,
        "current_step": run.current_step,
        "total_steps": run.total_steps,
        "gpu_type": run.gpu_type,
        "gpu_count": run.gpu_count,
        "cost_credits": run.cost_credits,
        "gpu_seconds": run.gpu_seconds,
        "queued_at": run.queued_at.isoformat() if run.queued_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "checkpoints": [
            {
                "id": c.id,
                "step": c.step,
                "epoch": c.epoch,
                "val_loss": c.val_loss,
                "is_best": c.is_best,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in checkpoints
        ],
    }


@router.get("/runs/{run_id}/metrics")
async def get_metrics(
    run_id: str,
    from_step: int = 0,
    to_step: int | None = None,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Time-series metrics for a training run."""
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    metrics = crud.list_run_metrics(db, run_id, from_step=from_step, to_step=to_step)
    return {
        "metrics": [
            {
                "step": m.step,
                "loss": m.loss,
                "val_loss": m.val_loss,
                "learning_rate": m.learning_rate,
                "throughput": m.throughput,
                "gpu_utilization": m.gpu_utilization,
                "vram_usage_gb": m.vram_usage_gb,
                "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
            }
            for m in metrics
        ],
    }


@router.get("/runs/{run_id}/logs")
async def get_logs(
    run_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Paginated historical training logs."""
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Logs are derived from metrics for now
    metrics = crud.list_run_metrics(db, run_id)
    logs = [
        {
            "step": m.step,
            "message": f"Step {m.step} — loss: {m.loss:.4f}"
            + (f", val_loss: {m.val_loss:.4f}" if m.val_loss else ""),
            "timestamp": m.recorded_at.isoformat() if m.recorded_at else None,
        }
        for m in metrics
        if m.loss is not None
    ]
    return {"logs": logs}


@router.post("/runs/{run_id}/pause")
async def pause_run(
    run_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Pause a training run after current step."""
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status != RunStatus.TRAINING:
        raise HTTPException(
            status_code=400, detail=f"Cannot pause run in status: {run.status.value}"
        )

    crud.update_run_status(db, run_id, RunStatus.PAUSED)

    # Signal worker via Redis
    try:
        import redis

        from app.core.config import settings

        r = redis.from_url(settings.REDIS_URL)
        r.publish(f"training:{run_id}:control", "pause")
    except Exception:
        pass

    return {"status": "paused"}


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Resume a paused run from latest checkpoint."""
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status != RunStatus.PAUSED:
        raise HTTPException(
            status_code=400, detail=f"Cannot resume run in status: {run.status.value}"
        )

    # Find latest checkpoint
    best = crud.get_best_checkpoint(db, run_id)
    checkpoints = crud.list_checkpoints(db, run_id)
    resume_ckpt = best or (checkpoints[-1] if checkpoints else None)

    crud.update_run_status(db, run_id, RunStatus.QUEUED)

    # Re-queue with checkpoint
    from app.core.celery_app import celery_app

    config = {**(run.model_config_json or {}), **(run.training_config_json or {})}
    if resume_ckpt:
        config["resume_from_checkpoint"] = resume_ckpt.s3_path
    celery_app.send_task(
        "training.launch_training_job",
        args=[run.id, config],
        queue="training",
    )

    return {"status": "resuming", "checkpoint_step": resume_ckpt.step if resume_ckpt else None}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Cancel and clean up a training job."""
    run = crud.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    if run.status in (RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED):
        raise HTTPException(
            status_code=400, detail=f"Run already in terminal state: {run.status.value}"
        )

    crud.update_run_status(db, run_id, RunStatus.CANCELLED)

    # Signal worker via Redis
    try:
        import redis

        from app.core.config import settings

        r = redis.from_url(settings.REDIS_URL)
        r.publish(f"training:{run_id}:control", "cancel")
    except Exception:
        pass

    return {"status": "cancelled"}


@router.websocket("/ws/runs/{run_id}/metrics")
async def metrics_websocket(websocket: WebSocket, run_id: str):
    """WebSocket stream for real-time training metric updates via Redis pub/sub."""
    await websocket.accept()
    try:
        import redis.asyncio as aioredis

        from app.core.config import settings

        r = aioredis.from_url(settings.REDIS_URL)
        pubsub = r.pubsub()
        await pubsub.subscribe(f"training:{run_id}:metrics")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    try:
                        await websocket.send_json(json.loads(data))
                    except Exception:
                        await websocket.send_text(data)
        finally:
            await pubsub.unsubscribe(f"training:{run_id}:metrics")
            await r.aclose()
    except WebSocketDisconnect:
        pass
    except Exception:
        with contextlib.suppress(Exception):
            await websocket.close()
