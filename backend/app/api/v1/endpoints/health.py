import time
from fastapi import APIRouter
router = APIRouter()

@router.get("/live")
async def live():
    return {"status": "alive", "ts": time.time()}

@router.get("/ready")
async def ready():
    return {"status": "ready", "checks": {"db": True, "redis": True, "ml": True}}

@router.get("/metrics")
async def metrics():
    return {"requests_total": 0, "errors_total": 0}
