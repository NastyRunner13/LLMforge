"""Celery application configuration."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "llm_forge",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=86400,  # 24 hours
    task_time_limit=90000,       # 25 hours hard limit
    task_routes={
        "app.workers.data_tasks.*": {"queue": "data"},
        "app.workers.training_tasks.*": {"queue": "training"},
    },
)

# Auto-discover task modules
celery_app.autodiscover_tasks(["app.workers"])
