"""SQLAlchemy models for the ML service.

These mirror the Prisma schema for tables the ML service needs to read/write directly.
The Next.js API (Prisma) is the primary owner of all tables, but the ML service
needs direct access to training-related tables for performance.
"""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


def _default_id():
    """Generate a UUID string for use as a primary key default."""
    return str(uuid.uuid4())


def _utcnow():
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


# === Enums ===


class DatasetStatus(enum.StrEnum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    CLEANING = "cleaning"
    READY = "ready"
    FAILED = "failed"


class RunStatus(enum.StrEnum):
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    DOWNLOADING = "downloading"
    TRAINING = "training"
    SAVING = "saving"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EndpointStatus(enum.StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


# === Models ===


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=_default_id)
    project_id = Column(String(36), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    original_filename = Column(String(500))
    file_path_s3 = Column(String(1000))
    file_format = Column(String(20))  # csv, jsonl, txt, pdf, docx
    file_size_bytes = Column(BigInteger, default=0)
    row_count = Column(Integer, default=0)
    token_count = Column(BigInteger, default=0)
    schema_json = Column(JSON)
    cleaning_config_json = Column(JSON)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.UPLOADING)
    error_message = Column(Text)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String(36), primary_key=True, default=_default_id)
    project_id = Column(String(36), nullable=False, index=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    experiment_name = Column(String(255), default="")
    tags = Column(JSON, default=list)

    # Configuration (stored as JSONB for flexibility)
    model_config_json = Column(JSON, nullable=False)
    training_config_json = Column(JSON, nullable=False)

    # Status
    status = Column(Enum(RunStatus), default=RunStatus.QUEUED)
    error_message = Column(Text)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer)

    # Compute
    gpu_type = Column(String(50))
    gpu_count = Column(Integer, default=1)
    worker_id = Column(String(100))  # Celery worker ID for tracking

    # Cost
    cost_credits = Column(Float, default=0.0)
    gpu_seconds = Column(Integer, default=0)

    # Timestamps
    queued_at = Column(DateTime, default=_utcnow)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    checkpoints = relationship("Checkpoint", back_populates="run", cascade="all, delete-orphan")
    metrics = relationship("RunMetric", back_populates="run", cascade="all, delete-orphan")


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(String(36), primary_key=True, default=_default_id)
    run_id = Column(String(36), ForeignKey("training_runs.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Float)
    s3_path = Column(String(1000), nullable=False)
    val_loss = Column(Float)
    is_best = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    run = relationship("TrainingRun", back_populates="checkpoints")


class RunMetric(Base):
    __tablename__ = "run_metrics"

    id = Column(String(36), primary_key=True, default=_default_id)
    run_id = Column(String(36), ForeignKey("training_runs.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    loss = Column(Float)
    val_loss = Column(Float)
    learning_rate = Column(Float)
    throughput = Column(Float)  # tokens/sec
    gpu_utilization = Column(Float)  # 0-100%
    vram_usage_gb = Column(Float)
    recorded_at = Column(DateTime, default=_utcnow)

    # Relationships
    run = relationship("TrainingRun", back_populates="metrics")


class Model(Base):
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=_default_id)
    project_id = Column(String(36), nullable=False, index=True)
    run_id = Column(String(36), ForeignKey("training_runs.id"))
    checkpoint_id = Column(String(36), ForeignKey("checkpoints.id"))
    name = Column(String(255), nullable=False)
    base_model = Column(String(255))
    architecture = Column(String(100))
    param_count = Column(BigInteger)
    training_method = Column(String(50))  # sft, lora, qlora
    quantization = Column(String(20))
    s3_path = Column(String(1000))
    created_at = Column(DateTime, default=_utcnow)


class Endpoint(Base):
    __tablename__ = "endpoints"

    id = Column(String(36), primary_key=True, default=_default_id)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    status = Column(Enum(EndpointStatus), default=EndpointStatus.STARTING)
    api_url = Column(String(500))
    gpu_type = Column(String(50))
    replicas = Column(Integer, default=1)
    container_id = Column(String(100))  # Docker/K8s container reference
    error_message = Column(Text)
    created_at = Column(DateTime, default=_utcnow)
    stopped_at = Column(DateTime)
