"""Inference and endpoint management API routes — with real vLLM proxy."""

import hashlib

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.rate_limit import inference_limiter
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

    # Stop the vLLM container if running
    if ep.container_id:
        try:
            import docker

            docker_client = docker.from_env()
            container = docker_client.containers.get(ep.container_id)
            container.stop(timeout=30)
            container.remove()
        except Exception:
            pass  # Container may already be stopped

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

    # Clean up container if exists
    if ep.container_id:
        try:
            import docker

            docker_client = docker.from_env()
            container = docker_client.containers.get(ep.container_id)
            container.stop(timeout=10)
            container.remove()
        except Exception:
            pass

    crud.delete_endpoint(db, endpoint_id)
    return {"deleted": True}


def _find_endpoint_for_model(db: Session, model_name: str):
    """Look up a running endpoint by model name or ID."""
    # Try exact model ID match first
    endpoints = crud.list_endpoints(db)
    for ep in endpoints:
        if ep.status == EndpointStatus.RUNNING:
            model = crud.get_model(db, ep.model_id)
            if model and (
                model.name == model_name or model.id == model_name or model.base_model == model_name
            ):
                return ep, model
    return None, None


def _verify_api_key(db: Session, api_key: str, endpoint_id: str) -> bool:
    """Verify an API key against stored hash for endpoint authentication."""
    # Hash the provided key and compare against stored hashes
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Query endpoint API keys table
    try:
        from sqlalchemy import text

        result = db.execute(
            text(
                "SELECT id FROM endpoint_api_keys "
                "WHERE endpoint_id = :eid AND key_hash = :kh AND revoked_at IS NULL"
            ),
            {"eid": endpoint_id, "kh": key_hash},
        ).fetchone()
        return result is not None
    except Exception:
        # If table doesn't exist yet, allow all keys in dev mode
        from app.core.config import settings

        return settings.APP_ENV == "development"


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible inference endpoint.
    Authenticated via endpoint-specific API key (Bearer token).
    Rate limited to 30 req/min, 500 req/hour per client.

    Proxies requests to the actual vLLM/TGI instance running for the model.
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

    # Look up the endpoint for this model
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        endpoint, model = _find_endpoint_for_model(db, req.model)

        if not endpoint or not endpoint.api_url:
            raise HTTPException(
                status_code=404,
                detail=f"No running inference endpoint found for model '{req.model}'. "
                f"Deploy the model first via POST /api/models/{{model_id}}/deploy",
            )

        # Verify API key
        if not _verify_api_key(db, api_key, endpoint.id):
            raise HTTPException(status_code=403, detail="Invalid API key for this endpoint")

    finally:
        db.close()

    # Proxy the request to the actual vLLM/TGI instance
    vllm_url = f"{endpoint.api_url}/chat/completions"
    payload = {
        "model": model.base_model or req.model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if req.stream:
                # Streaming response — proxy SSE events
                async def stream_generator():
                    async with client.stream("POST", vllm_url, json=payload) as resp:
                        async for chunk in resp.aiter_text():
                            yield chunk

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Non-streaming — forward and return
                resp = await client.post(vllm_url, json=payload)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Inference backend error: {resp.text}",
                    )
                return resp.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Inference endpoint for '{req.model}' is unreachable. "
            f"The vLLM container may be starting up or has crashed.",
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Inference request timed out. "
            "Try reducing max_tokens or simplifying the prompt.",
        ) from None
