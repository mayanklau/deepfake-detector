from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class TagReq(BaseModel):
    name: str
    color: str = "#3B82F6"

@router.get("/")
async def list_tags():
    return []

@router.post("/", status_code=201)
async def create_tag(req: TagReq):
    return {"name": req.name, "color": req.color}

@router.delete("/{tid}")
async def delete_tag(tid: str):
    return {"message": "Deleted"}

@router.post("/analysis/{aid}/{tid}")
async def tag_analysis(aid: str, tid: str):
    return {"message": "Tagged"}
