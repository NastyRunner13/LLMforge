"""Security utilities — API key validation, internal auth."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name="X-Internal-Key", auto_error=False)


async def verify_internal_key(api_key: str = Security(api_key_header)):
    """Verify the internal API key for inter-service communication."""
    if not api_key or api_key != settings.INTERNAL_API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key",
        )
    return api_key
