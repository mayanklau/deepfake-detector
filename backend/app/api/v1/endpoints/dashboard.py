from fastapi import APIRouter
router = APIRouter()

@router.get("/summary")
async def summary(period: str = "7d"):
    return {"total_analyses": 0, "fake_detected": 0, "authentic": 0, "uncertain": 0, "avg_confidence": 0.0}

@router.get("/charts/verdict-distribution")
async def verdict_dist():
    return {"labels": ["Authentic","Likely Authentic","Uncertain","Likely Fake","Fake"], "data": [0,0,0,0,0]}

@router.get("/charts/timeline")
async def timeline():
    return {"labels": [], "datasets": []}

@router.get("/charts/manipulation-types")
async def manip_types():
    return {"labels": [], "data": []}

@router.get("/charts/model-performance")
async def model_perf():
    return {"models": [], "accuracy": []}

@router.get("/recent-analyses")
async def recent(limit: int = 10):
    return []

@router.get("/alerts")
async def alerts():
    return []
