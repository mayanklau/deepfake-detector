from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def list_notifications():
    return {"items": [], "unread_count": 0}

@router.get("/unread-count")
async def unread_count():
    return {"count": 0}

@router.put("/{nid}/read")
async def mark_read(nid: str):
    return {"message": "Read"}

@router.put("/read-all")
async def mark_all_read():
    return {"message": "All read"}

@router.delete("/{nid}")
async def delete_notification(nid: str):
    return {"message": "Deleted"}

@router.get("/settings")
async def get_settings():
    return {"email": True, "in_app": True}

@router.put("/settings")
async def update_settings():
    return {"message": "Updated"}
