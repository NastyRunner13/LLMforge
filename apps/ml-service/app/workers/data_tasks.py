"""Data processing Celery tasks — cleaning pipeline workers."""

import json
import logging
import tempfile

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="data.run_cleaning_pipeline", bind=True, max_retries=3)
def run_cleaning_pipeline(self, dataset_id: str, pipeline_config: list[dict]):
    """
    Execute a cleaning pipeline on a dataset.

    Each node in pipeline_config is processed sequentially:
    - dedup: Remove duplicate rows (exact or hash-based)
    - language_filter: Filter to target language(s)
    - length_filter: Remove rows outside min/max character count
    - pii_redact: Mask PII using regex patterns
    - regex_filter: Apply regex pattern to include/exclude rows
    """
    from app.core.database import SessionLocal
    from app.core.config import settings
    from app.core.storage import get_s3_client
    from app.models.db_models import DatasetStatus
    from app.services import crud
    from app.services.cleaning import run_pipeline

    db = SessionLocal()
    try:
        dataset = crud.get_dataset(db, dataset_id)
        if not dataset or not dataset.file_path_s3:
            logger.error("Dataset %s not found or has no S3 path", dataset_id)
            return

        # Download dataset from S3
        client = get_s3_client()
        response = client.get_object(
            Bucket=settings.S3_BUCKET_DATASETS, Key=dataset.file_path_s3
        )
        raw_data = response["Body"].read().decode("utf-8")

        # Parse JSONL
        records = []
        for line in raw_data.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        original_count = len(records)
        logger.info("[%s] Cleaning pipeline: %d records, %d nodes", dataset_id, original_count, len(pipeline_config))

        # Run cleaning pipeline
        cleaned = run_pipeline(records, pipeline_config)
        logger.info("[%s] Cleaning done: %d -> %d records", dataset_id, original_count, len(cleaned))

        # Save cleaned data back to S3
        cleaned_key = dataset.file_path_s3.rsplit(".", 1)[0] + "_cleaned.jsonl"
        cleaned_content = "\n".join(json.dumps(r) for r in cleaned)
        client.put_object(
            Bucket=settings.S3_BUCKET_DATASETS,
            Key=cleaned_key,
            Body=cleaned_content.encode("utf-8"),
            ContentType="application/jsonl",
        )

        # Update database
        crud.update_dataset(
            db, dataset_id,
            file_path_s3=cleaned_key,
            row_count=len(cleaned),
            status=DatasetStatus.READY,
        )

        logger.info("[%s] Cleaning pipeline complete", dataset_id)

    except Exception as exc:
        logger.error("[%s] Cleaning pipeline failed: %s", dataset_id, exc)
        try:
            crud.update_dataset_status(db, dataset_id, DatasetStatus.FAILED, error_message=str(exc))
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
    finally:
        db.close()


@celery_app.task(name="data.convert_file_format")
def convert_file_format(dataset_id: str, source_format: str, target_format: str = "jsonl"):
    """Convert uploaded file (CSV, JSONL, TXT) to clean JSONL."""
    import csv
    import io

    from app.core.database import SessionLocal
    from app.core.config import settings
    from app.core.storage import get_s3_client
    from app.models.db_models import DatasetStatus
    from app.services import crud

    db = SessionLocal()
    try:
        dataset = crud.get_dataset(db, dataset_id)
        if not dataset or not dataset.file_path_s3:
            return

        crud.update_dataset_status(db, dataset_id, DatasetStatus.PROCESSING)

        client = get_s3_client()
        response = client.get_object(
            Bucket=settings.S3_BUCKET_DATASETS, Key=dataset.file_path_s3
        )
        raw_data = response["Body"].read().decode("utf-8")

        records = []

        if source_format == "csv":
            reader = csv.DictReader(io.StringIO(raw_data))
            for row in reader:
                records.append(dict(row))

        elif source_format in ("jsonl", "json"):
            for line in raw_data.strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, list):
                            records.extend(parsed)
                        else:
                            records.append(parsed)
                    except json.JSONDecodeError:
                        continue

        elif source_format == "txt":
            for line in raw_data.strip().split("\n"):
                line = line.strip()
                if line:
                    records.append({"text": line})

        else:
            # Unsupported format — store raw
            records.append({"text": raw_data})

        # Save as JSONL
        jsonl_key = dataset.file_path_s3.rsplit(".", 1)[0] + ".jsonl"
        content = "\n".join(json.dumps(r) for r in records)
        client.put_object(
            Bucket=settings.S3_BUCKET_DATASETS,
            Key=jsonl_key,
            Body=content.encode("utf-8"),
            ContentType="application/jsonl",
        )

        # Detect schema from first record
        schema = {}
        if records:
            schema = {k: type(v).__name__ for k, v in records[0].items()}

        crud.update_dataset(
            db, dataset_id,
            file_path_s3=jsonl_key,
            row_count=len(records),
            schema_json=schema,
            status=DatasetStatus.READY,
        )

        logger.info("[%s] Converted %s -> jsonl: %d records", dataset_id, source_format, len(records))

    except Exception as exc:
        logger.error("[%s] Format conversion failed: %s", dataset_id, exc)
        try:
            crud.update_dataset_status(db, dataset_id, DatasetStatus.FAILED, error_message=str(exc))
        except Exception:
            pass
    finally:
        db.close()


@celery_app.task(name="data.count_tokens")
def count_tokens(dataset_id: str, tokenizer_name: str = "cl100k_base"):
    """Count tokens in a formatted dataset using the specified tokenizer."""
    from app.core.database import SessionLocal
    from app.core.config import settings
    from app.core.storage import get_s3_client
    from app.services import crud

    db = SessionLocal()
    try:
        dataset = crud.get_dataset(db, dataset_id)
        if not dataset or not dataset.file_path_s3:
            return

        client = get_s3_client()
        response = client.get_object(
            Bucket=settings.S3_BUCKET_DATASETS, Key=dataset.file_path_s3
        )
        raw_data = response["Body"].read().decode("utf-8")

        total_tokens = 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer_name)
            for line in raw_data.strip().split("\n"):
                total_tokens += len(enc.encode(line))
        except ImportError:
            # Fallback: rough estimate
            total_tokens = len(raw_data) // 4

        crud.update_dataset(db, dataset_id, token_count=total_tokens)
        logger.info("[%s] Token count: %d", dataset_id, total_tokens)

    except Exception as exc:
        logger.error("[%s] Token counting failed: %s", dataset_id, exc)
    finally:
        db.close()
