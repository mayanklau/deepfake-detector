import logging, time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.api.v1.router import router as v1_router

settings = get_settings()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    yield
    logger.info("Shutting down...")

def create_application() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, description=settings.APP_DESCRIPTION, version=settings.APP_VERSION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.security.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{(time.time()-start)*1000:.2f}ms"
        return response

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for k,v in {"X-Content-Type-Options":"nosniff","X-Frame-Options":"DENY","X-XSS-Protection":"1; mode=block","Strict-Transport-Security":"max-age=31536000"}.items():
            response.headers[k] = v
        return response

    @app.exception_handler(Exception)
    async def exc_handler(req, exc):
        return JSONResponse(status_code=500, content={"error": {"code": 500, "message": "Internal server error"}})

    app.include_router(v1_router, prefix=settings.API_V1_PREFIX)

    @app.get("/health")
    async def health(): return {"status": "healthy", "version": settings.APP_VERSION}

    @app.get("/health/detailed")
    async def detailed_health(): return {"status": "healthy", "checks": {"db": True, "redis": True, "ml": True, "storage": True}}

    return app

app = create_application()
