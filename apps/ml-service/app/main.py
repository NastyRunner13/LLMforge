"""LLM Forge ML Service -- FastAPI Application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, health, inference, models, training
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    print(f"  LLM Forge ML Service starting on {settings.HOST}:{settings.PORT}")

    # Ensure S3 buckets exist (for local dev with MinIO)
    try:
        from app.core.storage import ensure_buckets_exist

        ensure_buckets_exist()
    except Exception as e:
        print(f"  S3 bucket setup skipped: {e}")

    # Verify database connectivity
    try:
        from app.core.database import engine

        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        print("  Database connection OK")
    except Exception as e:
        print(f"  Database connection warning: {e}")

    yield

    # Shutdown
    print("  LLM Forge ML Service shutting down")


app = FastAPI(
    title="LLM Forge ML Service",
    description="ML operations backend for LLM Forge platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
