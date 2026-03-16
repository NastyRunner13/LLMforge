"""Application configuration — loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation via Pydantic."""

    # App
    APP_NAME: str = "LLM Forge ML Service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/llmforge"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # S3 / MinIO
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_DATASETS: str = "llmforge-datasets"
    S3_BUCKET_CHECKPOINTS: str = "llmforge-checkpoints"
    S3_BUCKET_MODELS: str = "llmforge-models"
    S3_REGION: str = "us-east-1"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Auth (shared secret with Next.js for inter-service auth)
    INTERNAL_API_SECRET: str = "dev-secret-change-in-production"

    # HuggingFace
    HF_TOKEN: str = ""
    HF_CACHE_DIR: str = "/tmp/hf_cache"

    # Training Defaults
    MAX_CONCURRENT_JOBS: int = 10
    DEFAULT_CHECKPOINT_INTERVAL: int = 500
    MAX_TRAINING_HOURS: int = 24

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()
