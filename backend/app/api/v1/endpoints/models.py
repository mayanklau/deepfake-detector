from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def list_models():
    return {"models": [
        {"name": "efficientnet_b4_dfdc", "version": "3.0.0", "status": "loaded"},
        {"name": "xception_faceforensics", "version": "2.1.0", "status": "loaded"},
        {"name": "retinaface_resnet50", "version": "2.0.0", "status": "loaded"},
        {"name": "rawnet3_asvspoof", "version": "2.0.0", "status": "loaded"},
        {"name": "syncnet_v2", "version": "1.5.0", "status": "loaded"},
    ]}

@router.get("/{name}")
async def get_model(name: str):
    return {"name": name, "version": "1.0.0"}

@router.post("/{name}/reload")
async def reload_model(name: str):
    return {"message": f"Reloaded {name}"}

@router.get("/{name}/metrics")
async def model_metrics(name: str):
    return {"accuracy": 0.95, "precision": 0.93, "recall": 0.96, "f1": 0.945}
