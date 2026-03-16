"""Inference and endpoint management API routes — wired to real DB."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.rate_limit import inference_limiter

from app.core.database import get_db
from app.core.security import verify_internal_key
from app.models.db_models import EndpointStatus
from app.services import crud

router = APIRouter()


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # system, user, assistant
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False


@router.get("/endpoints")
async def list_endpoints(
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """List all active inference endpoints."""
    endpoints = crud.list_endpoints(db)
    return {
        "endpoints": [
            {
                "id": ep.id,
                "model_id": ep.model_id,
                "status": ep.status.value if ep.status else None,
                "api_url": ep.api_url,
                "gpu_type": ep.gpu_type,
                "replicas": ep.replicas,
                "error_message": ep.error_message,
                "created_at": ep.created_at.isoformat() if ep.created_at else None,
            }
            for ep in endpoints
        ],
    }


@router.get("/endpoints/{endpoint_id}")
async def get_endpoint(
    endpoint_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Get endpoint status, API URL, and latency stats."""
    ep = crud.get_endpoint(db, endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return {
        "id": ep.id,
        "model_id": ep.model_id,
        "status": ep.status.value if ep.status else None,
        "api_url": ep.api_url,
        "gpu_type": ep.gpu_type,
        "replicas": ep.replicas,
        "container_id": ep.container_id,
        "error_message": ep.error_message,
        "created_at": ep.created_at.isoformat() if ep.created_at else None,
        "stopped_at": ep.stopped_at.isoformat() if ep.stopped_at else None,
    }


@router.post("/endpoints/{endpoint_id}/stop")
async def stop_endpoint(
    endpoint_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Stop and tear down an inference endpoint."""
    ep = crud.get_endpoint(db, endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    if ep.status in (EndpointStatus.STOPPED, EndpointStatus.FAILED):
        raise HTTPException(status_code=400, detail=f"Endpoint already in state: {ep.status.value}")

    crud.update_endpoint_status(db, endpoint_id, EndpointStatus.STOPPED)
    return {"status": "stopped"}


@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint(
    endpoint_id: str,
    _key: str = Depends(verify_internal_key),
    db: Session = Depends(get_db),
):
    """Delete an endpoint permanently."""
    ep = crud.get_endpoint(db, endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")

    crud.delete_endpoint(db, endpoint_id)
    return {"deleted": True}


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible inference endpoint.
    Authenticated via endpoint-specific API key (not internal key).
    Rate limited to 30 req/min, 500 req/hour per client.
    """
    # Apply rate limiting
    inference_limiter.check(request)

    # Extract API key from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    api_key = auth_header.removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    # In production, this would:
    # 1. Look up the endpoint by model name
    # 2. Verify the API key hash
    # 3. Proxy the request to the actual vLLM/TGI instance
    # 4. Log the request for billing

    # For now, return a structured placeholder
    import time
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[LLM Forge] Inference endpoint for model '{req.model}' is not yet connected to a live vLLM instance. Deploy a model first.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
