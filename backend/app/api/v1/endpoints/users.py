import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()

@router.get("/")
async def list_users(page: int = 1, role: Optional[str] = None):
    return {"items": [], "total": 0}

@router.get("/{user_id}")
async def get_user(user_id: str):
    return {"id": user_id, "email": "user@example.com", "role": "analyst", "status": "active"}

@router.put("/{user_id}")
async def update_user(user_id: str):
    return {"message": "Updated"}

@router.delete("/{user_id}")
async def delete_user(user_id: str):
    return {"message": "Deleted"}

@router.post("/{user_id}/suspend")
async def suspend_user(user_id: str):
    return {"message": "Suspended"}

@router.post("/{user_id}/activate")
async def activate_user(user_id: str):
    return {"message": "Activated"}

@router.get("/{user_id}/activity")
async def get_activity(user_id: str):
    return {"activities": []}
