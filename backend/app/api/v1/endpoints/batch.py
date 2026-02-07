import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter()

class BatchJobResponse(BaseModel):
    id: str
    name: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    progress: float
    created_at: datetime

@router.post("/", response_model=BatchJobResponse, status_code=202)
async def create_batch(files: List[UploadFile] = File(...), name: str = Form("Batch")):
    return BatchJobResponse(id=str(uuid.uuid4()), name=name, status="queued", total_items=len(files), completed_items=0, failed_items=0, progress=0, created_at=datetime.now(timezone.utc))

@router.get("/")
async def list_batches():
    return []

@router.get("/{batch_id}")
async def get_batch(batch_id: str):
    return {"id": batch_id, "status": "processing", "progress": 50}

@router.get("/{batch_id}/results")
async def get_batch_results(batch_id: str):
    return {"batch_id": batch_id, "results": []}

@router.post("/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    return {"message": "Cancelled"}
