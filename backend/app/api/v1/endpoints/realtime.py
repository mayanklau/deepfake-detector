from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
router = APIRouter()

@router.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            await ws.send_json({"status": "analyzing", "progress": 50})
    except WebSocketDisconnect:
        pass

@router.websocket("/ws/status/{aid}")
async def ws_status(ws: WebSocket, aid: str):
    await ws.accept()
    try:
        while True:
            await ws.send_json({"id": aid, "status": "processing"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@router.post("/stream/start")
async def start_stream():
    return {"stream_id": "...", "ws_url": "ws://..."}
