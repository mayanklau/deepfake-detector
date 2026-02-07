"""Authentication middleware for extracting and validating tokens."""
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {"/health", "/health/live", "/health/ready", "/api/v1/auth/login", "/api/v1/auth/register", "/docs", "/openapi.json"}

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Extract and validate JWT tokens from requests."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            request.state.auth_type = "jwt"
            request.state.auth_token = token
        elif api_key_header:
            request.state.auth_type = "api_key"
            request.state.auth_token = api_key_header
        else:
            request.state.auth_type = None
            request.state.auth_token = None

        return await call_next(request)
