"""Initial schema — datasets, training_runs, checkpoints, run_metrics, models, endpoints.

Revision ID: 001_initial
Revises: None
Create Date: 2026-03-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- datasets ---
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(500)),
        sa.Column("file_path_s3", sa.String(1000)),
        sa.Column("file_format", sa.String(20)),
        sa.Column("file_size_bytes", sa.BigInteger, server_default="0"),
        sa.Column("row_count", sa.Integer, server_default="0"),
        sa.Column("token_count", sa.BigInteger, server_default="0"),
        sa.Column("schema_json", sa.JSON),
        sa.Column("cleaning_config_json", sa.JSON),
        sa.Column(
            "status",
            sa.Enum("uploading", "processing", "cleaning", "ready", "failed", name="datasetstatus"),
            server_default="uploading",
        ),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- training_runs ---
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False, index=True),
        sa.Column("dataset_id", sa.String(36), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("experiment_name", sa.String(255), server_default=""),
        sa.Column("tags", sa.JSON, server_default="[]"),
        sa.Column("model_config_json", sa.JSON, nullable=False),
        sa.Column("training_config_json", sa.JSON, nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "queued", "provisioning", "downloading", "training",
                "saving", "paused", "completed", "failed", "cancelled",
                name="runstatus",
            ),
            server_default="queued",
        ),
        sa.Column("error_message", sa.Text),
        sa.Column("current_step", sa.Integer, server_default="0"),
        sa.Column("total_steps", sa.Integer),
        sa.Column("gpu_type", sa.String(50)),
        sa.Column("gpu_count", sa.Integer, server_default="1"),
        sa.Column("worker_id", sa.String(100)),
        sa.Column("cost_credits", sa.Float, server_default="0"),
        sa.Column("gpu_seconds", sa.Integer, server_default="0"),
        sa.Column("queued_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_training_runs_status", "training_runs", ["status"])

    # --- checkpoints ---
    op.create_table(
        "checkpoints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("step", sa.Integer, nullable=False),
        sa.Column("epoch", sa.Float),
        sa.Column("s3_path", sa.String(1000), nullable=False),
        sa.Column("val_loss", sa.Float),
        sa.Column("is_best", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- run_metrics ---
    op.create_table(
        "run_metrics",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step", sa.Integer, nullable=False),
        sa.Column("loss", sa.Float),
        sa.Column("val_loss", sa.Float),
        sa.Column("learning_rate", sa.Float),
        sa.Column("throughput", sa.Float),
        sa.Column("gpu_utilization", sa.Float),
        sa.Column("vram_usage_gb", sa.Float),
        sa.Column("recorded_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_run_metrics_run_step", "run_metrics", ["run_id", "step"])

    # --- models ---
    op.create_table(
        "models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False, index=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("training_runs.id")),
        sa.Column("checkpoint_id", sa.String(36), sa.ForeignKey("checkpoints.id")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("base_model", sa.String(255)),
        sa.Column("architecture", sa.String(100)),
        sa.Column("param_count", sa.BigInteger),
        sa.Column("training_method", sa.String(50)),
        sa.Column("quantization", sa.String(20)),
        sa.Column("s3_path", sa.String(1000)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- endpoints ---
    op.create_table(
        "endpoints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.String(36), sa.ForeignKey("models.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "status",
            sa.Enum("starting", "running", "stopping", "stopped", "failed", name="endpointstatus"),
            server_default="starting",
        ),
        sa.Column("api_url", sa.String(500)),
        sa.Column("gpu_type", sa.String(50)),
        sa.Column("replicas", sa.Integer, server_default="1"),
        sa.Column("container_id", sa.String(100)),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("stopped_at", sa.DateTime),
    )


def downgrade() -> None:
    op.drop_table("endpoints")
    op.drop_table("models")
    op.drop_table("run_metrics")
    op.drop_table("checkpoints")
    op.drop_table("training_runs")
    op.drop_table("datasets")
    sa.Enum(name="datasetstatus").drop(op.get_bind())
    sa.Enum(name="runstatus").drop(op.get_bind())
    sa.Enum(name="endpointstatus").drop(op.get_bind())
