"""Training orchestration Celery tasks."""

import contextlib
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
    from app.core.config import settings
    from app.core.database import SessionLocal
    from app.core.storage import get_s3_client
    from app.models.db_models import RunStatus
    from app.services import crud

    db = SessionLocal()
    try:
        # 1. Set status to provisioning
        crud.update_run_status(
            db,
            run_id,
            RunStatus.PROVISIONING,
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
                db,
                run_id,
                RunStatus.FAILED,
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
        config.get("base_model", "unknown")
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
        with contextlib.suppress(Exception):
            crud.update_run_status(
                db,
                run_id,
                RunStatus.FAILED,
                error_message=str(exc),
            )
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries)) from exc
    finally:
        db.close()


@celery_app.task(name="training.save_checkpoint")
def save_checkpoint(run_id: str, step: int, checkpoint_dir: str):
    """Save a training checkpoint to S3 and register in DB."""
    from pathlib import Path

    from app.core.config import settings
    from app.core.database import SessionLocal
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
            db,
            run_id=run_id,
            step=step,
            s3_path=s3_prefix,
        )
        logger.info("[%s] Checkpoint at step %d saved", run_id, step)
    except Exception as exc:
        logger.error("[%s] Checkpoint save failed: %s", run_id, exc)
    finally:
        db.close()


@celery_app.task(name="training.deploy_model", bind=True, max_retries=2)
def deploy_model(self, model_id: str, gpu_type: str, replicas: int):
    """Deploy a trained model to an inference endpoint via vLLM Docker container.

    Steps:
    1. Download model weights from S3 to a local volume
    2. Launch a vLLM Docker container with GPU passthrough
    3. Wait for health check
    4. Register the container URL in the endpoint record
    """
    import tempfile
    import time

    from app.core.config import settings
    from app.core.database import SessionLocal
    from app.core.storage import get_s3_client
    from app.models.db_models import EndpointStatus
    from app.services import crud

    db = SessionLocal()
    endpoint = None
    try:
        model = crud.get_model(db, model_id)
        if not model:
            logger.error("Model %s not found", model_id)
            return

        # Find the endpoint record
        endpoints = crud.list_endpoints(db)
        endpoint = next((ep for ep in endpoints if ep.model_id == model_id), None)
        if not endpoint:
            logger.error("No endpoint record for model %s", model_id)
            return

        # Step 1: Download model from S3
        if model.s3_path:
            model_dir = os.path.join(tempfile.gettempdir(), "llmforge_models", model_id)
            os.makedirs(model_dir, exist_ok=True)

            s3 = get_s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=settings.S3_BUCKET_MODELS, Prefix=model.s3_path):
                for obj in page.get("Contents", []):
                    relative = obj["Key"].removeprefix(model.s3_path).lstrip("/")
                    local_path = os.path.join(model_dir, relative)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3.download_file(settings.S3_BUCKET_MODELS, obj["Key"], local_path)

            logger.info("[%s] Model downloaded to %s", model_id, model_dir)
        else:
            # Use the base model name directly (for HuggingFace Hub models)
            model_dir = model.base_model or model.name

        # Step 2: Launch vLLM container
        container_name = f"llmforge-vllm-{model_id[:8]}"
        vllm_port = 8080  # Internal port

        try:
            import docker

            docker_client = docker.from_env()

            # Remove existing container if any
            try:
                old = docker_client.containers.get(container_name)
                old.stop(timeout=10)
                old.remove()
            except docker.errors.NotFound:
                pass

            # Determine GPU configuration
            device_requests = []
            environment = {
                "MODEL_NAME": model_dir,
                "MAX_MODEL_LEN": "4096",
            }

            if gpu_type and gpu_type != "cpu":
                device_requests = [
                    docker.types.DeviceRequest(count=replicas, capabilities=[["gpu"]])
                ]

            container = docker_client.containers.run(
                image="vllm/vllm-openai:latest",
                name=container_name,
                command=[
                    "--model",
                    model_dir,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(vllm_port),
                    "--max-model-len",
                    "4096",
                    "--dtype",
                    "auto",
                ],
                ports={f"{vllm_port}/tcp": None},  # Random host port
                volumes={
                    model_dir: {"bind": "/model", "mode": "ro"},
                    os.path.join(tempfile.gettempdir(), "hf_cache"): {
                        "bind": "/root/.cache/huggingface",
                        "mode": "rw",
                    },
                },
                device_requests=device_requests if device_requests else None,
                environment=environment,
                detach=True,
                restart_policy={"Name": "unless-stopped"},
            )

            # Step 3: Wait for health check (up to 5 minutes)
            api_url = None
            for _attempt in range(30):
                time.sleep(10)
                container.reload()

                if container.status == "exited":
                    logs = container.logs(tail=50).decode()
                    raise RuntimeError(f"vLLM container exited unexpectedly:\n{logs}")

                # Get the mapped port
                ports = container.ports.get(f"{vllm_port}/tcp")
                if ports:
                    host_port = ports[0]["HostPort"]
                    api_url = f"http://localhost:{host_port}/v1"

                    # Check if vLLM is actually ready
                    try:
                        import httpx

                        resp = httpx.get(f"{api_url}/models", timeout=5.0)
                        if resp.status_code == 200:
                            logger.info("[%s] vLLM ready at %s", model_id, api_url)
                            break
                    except Exception:
                        continue

            if not api_url:
                raise RuntimeError("vLLM container started but no port binding found")

            # Step 4: Update endpoint record
            crud.update_endpoint_status(
                db,
                endpoint.id,
                EndpointStatus.RUNNING,
                api_url=api_url,
                container_id=container.id,
            )
            logger.info("[%s] Model deployed at %s", model_id, api_url)

        except ImportError:
            # Docker SDK not installed — use placeholder for dev
            logger.warning(
                "[%s] Docker SDK not available — marking endpoint as running with placeholder URL",
                model_id,
            )
            crud.update_endpoint_status(
                db,
                endpoint.id,
                EndpointStatus.RUNNING,
                api_url="http://localhost:8080/v1",
                container_id=f"placeholder-{model_id[:8]}",
            )

    except Exception as exc:
        logger.error("[%s] Deployment failed: %s", model_id, exc)
        if endpoint:
            with contextlib.suppress(Exception):
                crud.update_endpoint_status(
                    db,
                    endpoint.id,
                    EndpointStatus.FAILED,
                    error_message=str(exc),
                )
        raise self.retry(exc=exc, countdown=60) from exc
    finally:
        db.close()
