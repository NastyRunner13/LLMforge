"""CRUD operations for all ML service database models.

Provides typed create/read/update/delete functions used
by the API route handlers and Celery workers.
"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.db_models import (
    Checkpoint,
    Dataset,
    DatasetStatus,
    Endpoint,
    EndpointStatus,
    Model,
    RunMetric,
    RunStatus,
    TrainingRun,
)


def _id() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ═══════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════


def create_dataset(
    db: Session,
    *,
    project_id: str,
    name: str,
    original_filename: str | None = None,
    file_format: str | None = None,
    file_size_bytes: int = 0,
) -> Dataset:
    ds = Dataset(
        id=_id(),
        project_id=project_id,
        name=name,
        original_filename=original_filename,
        file_format=file_format,
        file_size_bytes=file_size_bytes,
        status=DatasetStatus.UPLOADING,
        created_at=_now(),
        updated_at=_now(),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def get_dataset(db: Session, dataset_id: str) -> Dataset | None:
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def list_datasets(db: Session, project_id: str, skip: int = 0, limit: int = 50) -> list[Dataset]:
    return (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(desc(Dataset.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_dataset(db: Session, dataset_id: str, **kwargs) -> Dataset | None:
    ds = get_dataset(db, dataset_id)
    if not ds:
        return None
    for k, v in kwargs.items():
        setattr(ds, k, v)
    ds.updated_at = _now()
    db.commit()
    db.refresh(ds)
    return ds


def update_dataset_status(
    db: Session, dataset_id: str, status: DatasetStatus, error_message: str | None = None
) -> Dataset | None:
    return update_dataset(db, dataset_id, status=status, error_message=error_message)


def delete_dataset(db: Session, dataset_id: str) -> bool:
    ds = get_dataset(db, dataset_id)
    if not ds:
        return False
    db.delete(ds)
    db.commit()
    return True


# ═══════════════════════════════════════════════
# Training Runs
# ═══════════════════════════════════════════════


def create_training_run(
    db: Session,
    *,
    project_id: str,
    dataset_id: str,
    model_config: dict,
    training_config: dict,
    experiment_name: str = "",
    tags: list[str] | None = None,
    gpu_type: str = "A100_40GB",
    gpu_count: int = 1,
) -> TrainingRun:
    run = TrainingRun(
        id=_id(),
        project_id=project_id,
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        tags=tags or [],
        model_config_json=model_config,
        training_config_json=training_config,
        status=RunStatus.QUEUED,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        queued_at=_now(),
        created_at=_now(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_training_run(db: Session, run_id: str) -> TrainingRun | None:
    return db.query(TrainingRun).filter(TrainingRun.id == run_id).first()


def list_training_runs(
    db: Session,
    project_id: str | None = None,
    status: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[TrainingRun]:
    q = db.query(TrainingRun)
    if project_id:
        q = q.filter(TrainingRun.project_id == project_id)
    if status:
        q = q.filter(TrainingRun.status == status)
    return q.order_by(desc(TrainingRun.created_at)).offset(skip).limit(limit).all()


def update_run_status(
    db: Session,
    run_id: str,
    status: RunStatus,
    *,
    error_message: str | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    worker_id: str | None = None,
    cost_credits: float | None = None,
    gpu_seconds: int | None = None,
) -> TrainingRun | None:
    run = get_training_run(db, run_id)
    if not run:
        return None
    run.status = status
    if error_message is not None:
        run.error_message = error_message
    if current_step is not None:
        run.current_step = current_step
    if total_steps is not None:
        run.total_steps = total_steps
    if worker_id is not None:
        run.worker_id = worker_id
    if cost_credits is not None:
        run.cost_credits = cost_credits
    if gpu_seconds is not None:
        run.gpu_seconds = gpu_seconds

    # Auto-set timestamps
    if status == RunStatus.TRAINING and run.started_at is None:
        run.started_at = _now()
    if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
        run.finished_at = _now()

    db.commit()
    db.refresh(run)
    return run


# ═══════════════════════════════════════════════
# Checkpoints
# ═══════════════════════════════════════════════


def create_checkpoint(
    db: Session,
    *,
    run_id: str,
    step: int,
    s3_path: str,
    epoch: float | None = None,
    val_loss: float | None = None,
    is_best: bool = False,
) -> Checkpoint:
    ckpt = Checkpoint(
        id=_id(),
        run_id=run_id,
        step=step,
        epoch=epoch,
        s3_path=s3_path,
        val_loss=val_loss,
        is_best=is_best,
        created_at=_now(),
    )
    db.add(ckpt)
    db.commit()
    db.refresh(ckpt)
    return ckpt


def list_checkpoints(db: Session, run_id: str) -> list[Checkpoint]:
    return db.query(Checkpoint).filter(Checkpoint.run_id == run_id).order_by(Checkpoint.step).all()


def get_best_checkpoint(db: Session, run_id: str) -> Checkpoint | None:
    return (
        db.query(Checkpoint)
        .filter(Checkpoint.run_id == run_id, Checkpoint.is_best.is_(True))
        .first()
    )


# ═══════════════════════════════════════════════
# Run Metrics
# ═══════════════════════════════════════════════


def create_run_metric(
    db: Session,
    *,
    run_id: str,
    step: int,
    loss: float | None = None,
    val_loss: float | None = None,
    learning_rate: float | None = None,
    throughput: float | None = None,
    gpu_utilization: float | None = None,
    vram_usage_gb: float | None = None,
) -> RunMetric:
    metric = RunMetric(
        id=_id(),
        run_id=run_id,
        step=step,
        loss=loss,
        val_loss=val_loss,
        learning_rate=learning_rate,
        throughput=throughput,
        gpu_utilization=gpu_utilization,
        vram_usage_gb=vram_usage_gb,
        recorded_at=_now(),
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def list_run_metrics(
    db: Session,
    run_id: str,
    from_step: int = 0,
    to_step: int | None = None,
) -> list[RunMetric]:
    q = db.query(RunMetric).filter(RunMetric.run_id == run_id, RunMetric.step >= from_step)
    if to_step is not None:
        q = q.filter(RunMetric.step <= to_step)
    return q.order_by(RunMetric.step).all()


# ═══════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════


def create_model(
    db: Session,
    *,
    project_id: str,
    name: str,
    run_id: str | None = None,
    checkpoint_id: str | None = None,
    base_model: str | None = None,
    architecture: str | None = None,
    param_count: int | None = None,
    training_method: str | None = None,
    quantization: str | None = None,
    s3_path: str | None = None,
) -> Model:
    model = Model(
        id=_id(),
        project_id=project_id,
        name=name,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        base_model=base_model,
        architecture=architecture,
        param_count=param_count,
        training_method=training_method,
        quantization=quantization,
        s3_path=s3_path,
        created_at=_now(),
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def get_model(db: Session, model_id: str) -> Model | None:
    return db.query(Model).filter(Model.id == model_id).first()


def list_models(
    db: Session, project_id: str | None = None, skip: int = 0, limit: int = 50
) -> list[Model]:
    q = db.query(Model)
    if project_id:
        q = q.filter(Model.project_id == project_id)
    return q.order_by(desc(Model.created_at)).offset(skip).limit(limit).all()


def delete_model(db: Session, model_id: str) -> bool:
    model = get_model(db, model_id)
    if not model:
        return False
    db.delete(model)
    db.commit()
    return True


# ═══════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════


def create_endpoint(
    db: Session,
    *,
    model_id: str,
    gpu_type: str = "A100_40GB",
    replicas: int = 1,
) -> Endpoint:
    ep = Endpoint(
        id=_id(),
        model_id=model_id,
        status=EndpointStatus.STARTING,
        gpu_type=gpu_type,
        replicas=replicas,
        created_at=_now(),
    )
    db.add(ep)
    db.commit()
    db.refresh(ep)
    return ep


def get_endpoint(db: Session, endpoint_id: str) -> Endpoint | None:
    return db.query(Endpoint).filter(Endpoint.id == endpoint_id).first()


def list_endpoints(db: Session, skip: int = 0, limit: int = 50) -> list[Endpoint]:
    return db.query(Endpoint).order_by(desc(Endpoint.created_at)).offset(skip).limit(limit).all()


def update_endpoint_status(
    db: Session,
    endpoint_id: str,
    status: EndpointStatus,
    **kwargs,
) -> Endpoint | None:
    ep = get_endpoint(db, endpoint_id)
    if not ep:
        return None
    ep.status = status
    for k, v in kwargs.items():
        setattr(ep, k, v)
    if status == EndpointStatus.STOPPED:
        ep.stopped_at = _now()
    db.commit()
    db.refresh(ep)
    return ep


def delete_endpoint(db: Session, endpoint_id: str) -> bool:
    ep = get_endpoint(db, endpoint_id)
    if not ep:
        return False
    db.delete(ep)
    db.commit()
    return True
