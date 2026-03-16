"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Liveness probe — is the service running?"""
    return {"status": "healthy", "service": "llm-forge-ml"}


@router.get("/ready")
async def readiness_check():
    """Readiness probe — is the service ready to accept requests?"""
    # TODO: Check DB connection, Redis connection, S3 access
    return {"status": "ready"}
