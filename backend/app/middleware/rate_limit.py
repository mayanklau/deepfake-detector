"""Rate limiting middleware."""
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter middleware."""

    def __init__(self, app, default_rate: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.default_rate = default_rate
        self.window_seconds = window_seconds
        self.buckets: Dict[str, list] = defaultdict(list)

    def _get_client_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - self.window_seconds
        self.buckets[key] = [t for t in self.buckets[key] if t > window_start]
        if len(self.buckets[key]) >= self.default_rate:
            return True, 0
        self.buckets[key].append(now)
        return False, self.default_rate - len(self.buckets[key])

    async def dispatch(self, request: Request, call_next):
        client_key = self._get_client_key(request)
        is_limited, remaining = self._is_rate_limited(client_key)

        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_key}")
            return Response(
                content='{"error": {"code": 429, "message": "Rate limit exceeded"}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(self.window_seconds), "X-RateLimit-Limit": str(self.default_rate), "X-RateLimit-Remaining": "0"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.default_rate)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
