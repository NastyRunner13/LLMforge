"""Custom HuggingFace Trainer callbacks for LLM Forge.

- MetricsCallback: publishes step metrics to Redis pub/sub and DB
- CheckpointUploadCallback: uploads checkpoint directories to S3
"""

from __future__ import annotations

import json
import logging
import time

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):
    """Publishes training metrics to Redis pub/sub for real-time WebSocket streaming."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._redis = None
        self._start_time = time.time()

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis as _redis
                from app.core.config import settings
                self._redis = _redis.from_url(settings.REDIS_URL)
            except Exception as e:
                logger.warning("Could not connect to Redis: %s", e)
        return self._redis

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        if not logs:
            return

        metrics_payload = {
            "run_id": self.run_id,
            "step": state.global_step,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "epoch": logs.get("epoch"),
            "elapsed_seconds": time.time() - self._start_time,
        }

        # Publish to Redis
        r = self._get_redis()
        if r:
            try:
                channel = f"training:{self.run_id}:metrics"
                r.publish(channel, json.dumps(metrics_payload))
            except Exception as e:
                logger.warning("Failed to publish metrics to Redis: %s", e)

        # Save to DB
        try:
            from app.core.database import SessionLocal
            from app.services import crud
            db = SessionLocal()
            try:
                crud.create_run_metric(
                    db,
                    run_id=self.run_id,
                    step=state.global_step,
                    loss=logs.get("loss"),
                    learning_rate=logs.get("learning_rate"),
                )
                crud.update_run_status(
                    db, self.run_id, status=None,
                    current_step=state.global_step,
                    total_steps=state.max_steps if state.max_steps > 0 else None,
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to save metrics to DB: %s", e)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation."""
        if not metrics:
            return

        r = self._get_redis()
        if r:
            try:
                payload = {
                    "run_id": self.run_id,
                    "step": state.global_step,
                    "val_loss": metrics.get("eval_loss"),
                    "type": "evaluation",
                }
                r.publish(f"training:{self.run_id}:metrics", json.dumps(payload))
            except Exception:
                pass


class CheckpointUploadCallback(TrainerCallback):
    """Uploads checkpoint directories to S3 after each save."""

    def __init__(self, run_id: str):
        self.run_id = run_id

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when a checkpoint is saved."""
        import os
        from pathlib import Path

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return

        s3_prefix = f"checkpoints/{self.run_id}/step-{state.global_step}"

        try:
            from app.core.storage import get_s3_client
            from app.core.config import settings

            client = get_s3_client()
            ckpt_path = Path(checkpoint_dir)

            for file_path in ckpt_path.rglob("*"):
                if file_path.is_file():
                    relative = file_path.relative_to(ckpt_path)
                    s3_key = f"{s3_prefix}/{relative}"
                    client.upload_file(str(file_path), settings.S3_BUCKET_CHECKPOINTS, s3_key)

            logger.info("[%s] Checkpoint step %d uploaded to S3", self.run_id, state.global_step)

            # Register in DB
            from app.core.database import SessionLocal
            from app.services import crud
            db = SessionLocal()
            try:
                crud.create_checkpoint(
                    db,
                    run_id=self.run_id,
                    step=state.global_step,
                    s3_path=s3_prefix,
                    epoch=state.epoch,
                )
            finally:
                db.close()

        except Exception as e:
            logger.error("[%s] Failed to upload checkpoint: %s", self.run_id, e)
