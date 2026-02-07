import uuid
from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

router = APIRouter()

class ComparisonReq(BaseModel):
    name: str
    analysis_ids: List[str]

@router.post("/", status_code=201)
async def create(req: ComparisonReq):
    return {"id": str(uuid.uuid4()), "name": req.name}

@router.get("/")
async def list_comparisons():
    return []

@router.get("/{cid}")
async def get_comparison(cid: str):
    return {"id": cid, "items": []}

@router.delete("/{cid}")
async def delete_comparison(cid: str):
    return {"message": "Deleted"}
