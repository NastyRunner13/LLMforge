"""Training orchestration Celery tasks."""

import json
import logging
import os
import shutil
import tempfile

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="training.launch_training_job", bind=True, max_retries=3)
def launch_training_job(self, run_id: str, config: dict):
    """
    Main training job task.

    Steps:
    1. Update run status to "provisioning"
    2. Download dataset from S3
    3. Configure & run TrainingEngine (HuggingFace TRL + PEFT)
    4. Metrics pushed to Redis pub/sub via callbacks
    5. Checkpoints saved to S3 via callbacks
    6. Register final model on completion
    """
    from app.core.database import SessionLocal
    from app.core.config import settings
    from app.core.storage import get_s3_client
    from app.models.db_models import RunStatus
    from app.services import crud

    db = SessionLocal()
    try:
        # 1. Set status to provisioning
        crud.update_run_status(
            db, run_id, RunStatus.PROVISIONING,
            worker_id=self.request.hostname,
        )

        run = crud.get_training_run(db, run_id)
        if not run:
            logger.error("Run %s not found", run_id)
            return

        # 2. Download dataset from S3
        crud.update_run_status(db, run_id, RunStatus.DOWNLOADING)

        dataset = crud.get_dataset(db, run.dataset_id)
        if not dataset or not dataset.file_path_s3:
            crud.update_run_status(
                db, run_id, RunStatus.FAILED,
                error_message="Dataset not found or missing S3 path",
            )
            return

        client = get_s3_client()
        dataset_tmpdir = tempfile.mkdtemp(prefix="llmforge_data_")
        local_dataset_path = os.path.join(dataset_tmpdir, "dataset.jsonl")

        client.download_file(
            settings.S3_BUCKET_DATASETS,
            dataset.file_path_s3,
            local_dataset_path,
        )
        logger.info("[%s] Dataset downloaded to %s", run_id, local_dataset_path)

        # 3. Configure and run training
        crud.update_run_status(db, run_id, RunStatus.TRAINING)

        from app.training.engine import TrainingEngine

        engine = TrainingEngine(
            run_id=run_id,
            config=config,
            dataset_path=local_dataset_path,
        )
        engine.setup()
        output_dir = engine.train()

        # 4. Save final model
        crud.update_run_status(db, run_id, RunStatus.SAVING)

        final_model_dir = engine.save_model()

        # Upload final model to S3
        s3_model_prefix = f"models/{run.project_id}/{run_id}/final"
        from pathlib import Path
        model_path = Path(final_model_dir)
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(model_path)
                s3_key = f"{s3_model_prefix}/{relative}"
                client.upload_file(str(file_path), settings.S3_BUCKET_MODELS, s3_key)

        # 5. Register model in DB
        model_config = config.get("base_model", "unknown")
        crud.create_model(
            db,
            project_id=run.project_id,
            name=f"{run.experiment_name or 'model'}-{run_id[:8]}",
            run_id=run_id,
            base_model=config.get("base_model"),
            training_method=config.get("method", "lora"),
            s3_path=s3_model_prefix,
        )

        # 6. Mark complete
        crud.update_run_status(db, run_id, RunStatus.COMPLETED)
        logger.info("[%s] Training complete!", run_id)

        # Cleanup temp files
        try:
            shutil.rmtree(dataset_tmpdir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass

    except Exception as exc:
        logger.error("[%s] Training failed: %s", run_id, exc)
        try:
            crud.update_run_status(
                db, run_id, RunStatus.FAILED,
                error_message=str(exc),
            )
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))
    finally:
        db.close()


@celery_app.task(name="training.save_checkpoint")
def save_checkpoint(run_id: str, step: int, checkpoint_dir: str):
    """Save a training checkpoint to S3 and register in DB."""
    from pathlib import Path
    from app.core.database import SessionLocal
    from app.core.config import settings
    from app.core.storage import get_s3_client
    from app.services import crud

    db = SessionLocal()
    try:
        s3_prefix = f"checkpoints/{run_id}/step-{step}"
        client = get_s3_client()
        ckpt_path = Path(checkpoint_dir)

        for file_path in ckpt_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(ckpt_path)
                s3_key = f"{s3_prefix}/{relative}"
                client.upload_file(str(file_path), settings.S3_BUCKET_CHECKPOINTS, s3_key)

        crud.create_checkpoint(
            db, run_id=run_id, step=step, s3_path=s3_prefix,
        )
        logger.info("[%s] Checkpoint at step %d saved", run_id, step)
    except Exception as exc:
        logger.error("[%s] Checkpoint save failed: %s", run_id, exc)
    finally:
        db.close()


@celery_app.task(name="training.deploy_model")
def deploy_model(model_id: str, gpu_type: str, replicas: int):
    """Deploy a trained model to an inference endpoint.

    In production, this would:
    - Pull model weights from S3
    - Launch a vLLM or TGI container
    - Register the endpoint URL
    """
    from app.core.database import SessionLocal
    from app.models.db_models import EndpointStatus
    from app.services import crud

    db = SessionLocal()
    try:
        model = crud.get_model(db, model_id)
        if not model:
            logger.error("Model %s not found", model_id)
            return

        # Find the endpoint record
        endpoints = crud.list_endpoints(db)
        endpoint = next((ep for ep in endpoints if ep.model_id == model_id), None)

        if endpoint:
            # In production: launch vLLM container here
            # For now, mark as running with a placeholder URL
            crud.update_endpoint_status(
                db, endpoint.id, EndpointStatus.RUNNING,
                api_url=f"http://localhost:8080/v1",
                container_id=f"llmforge-inference-{model_id[:8]}",
            )
            logger.info("[%s] Model deployed (placeholder)", model_id)

    except Exception as exc:
        logger.error("[%s] Deployment failed: %s", model_id, exc)
        if endpoint:
            try:
                crud.update_endpoint_status(
                    db, endpoint.id, EndpointStatus.FAILED,
                    error_message=str(exc),
                )
            except Exception:
                pass
    finally:
        db.close()
