"""Redis-based sliding window rate limiter.

Provides a FastAPI dependency for rate-limiting API endpoints.
Uses Redis sorted sets for a precise sliding window algorithm.
"""

import time
from fastapi import HTTPException, Request, status


class RateLimiter:
    """Sliding window rate limiter backed by Redis."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        key_prefix: str = "ratelimit",
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.key_prefix = key_prefix
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis
                from app.core.config import settings
                self._redis = redis.from_url(settings.REDIS_URL)
            except Exception:
                return None
        return self._redis

    def _get_client_key(self, request: Request) -> str:
        """Extract a client identifier from the request."""
        # Use API key if present, otherwise IP
        api_key = request.headers.get("Authorization", "")
        if api_key.startswith("Bearer "):
            return api_key[7:20]  # Use prefix of the key
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client_host = request.client.host if request.client else "unknown"
        return client_host

    def check(self, request: Request) -> bool:
        """Check if the request is within rate limits. Raises 429 if exceeded."""
        r = self._get_redis()
        if r is None:
            return True  # Skip rate limiting if Redis is unavailable

        client_key = self._get_client_key(request)
        now = time.time()

        # Check per-minute limit
        minute_key = f"{self.key_prefix}:{client_key}:m"
        self._check_window(r, minute_key, now, window_seconds=60, max_requests=self.rpm)

        # Check per-hour limit
        hour_key = f"{self.key_prefix}:{client_key}:h"
        self._check_window(r, hour_key, now, window_seconds=3600, max_requests=self.rph)

        return True

    def _check_window(self, r, key: str, now: float, window_seconds: int, max_requests: int):
        """Sliding window check using Redis sorted sets."""
        pipeline = r.pipeline()

        # Remove entries outside the window
        pipeline.zremrangebyscore(key, 0, now - window_seconds)
        # Count entries in the window
        pipeline.zcard(key)
        # Add current request
        pipeline.zadd(key, {str(now): now})
        # Set expiry on the key
        pipeline.expire(key, window_seconds + 1)

        results = pipeline.execute()
        current_count = results[1]

        if current_count >= max_requests:
            retry_after = int(window_seconds - (now - float(r.zrange(key, 0, 0)[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after}s.",
                headers={"Retry-After": str(max(retry_after, 1))},
            )


# Pre-configured limiters
inference_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500, key_prefix="rl:inference")
api_limiter = RateLimiter(requests_per_minute=120, requests_per_hour=5000, key_prefix="rl:api")
