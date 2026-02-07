import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.post("/", status_code=201)
async def create_key(name: str = "default"):
    return {"id": str(uuid.uuid4()), "key": "dfk_xxxxx", "name": name}

@router.get("/")
async def list_keys():
    return []

@router.delete("/{key_id}")
async def revoke_key(key_id: str):
    return {"message": "Revoked"}

@router.put("/{key_id}/rotate")
async def rotate_key(key_id: str):
    return {"new_key": "dfk_new"}
