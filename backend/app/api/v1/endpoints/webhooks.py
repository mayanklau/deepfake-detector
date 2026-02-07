import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class WebhookRequest(BaseModel):
    name: str
    url: str
    events: List[str]

@router.post("/", status_code=201)
async def create_webhook(req: WebhookRequest):
    return {"id": str(uuid.uuid4()), "name": req.name, "url": req.url}

@router.get("/")
async def list_webhooks():
    return []

@router.get("/{wh_id}")
async def get_webhook(wh_id: str):
    return {"id": wh_id}

@router.delete("/{wh_id}")
async def delete_webhook(wh_id: str):
    return {"message": "Deleted"}

@router.post("/{wh_id}/test")
async def test_webhook(wh_id: str):
    return {"status": 200}

@router.get("/{wh_id}/deliveries")
async def get_deliveries(wh_id: str):
    return []
