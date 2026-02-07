import hashlib, logging, os, uuid
from pathlib import Path
from typing import Dict, Optional
logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self, backend="local", base_path="/data"):
        self.backend = backend
        self.base_path = Path(base_path)

    async def save_file(self, file_data, filename, subdir="uploads"):
        file_hash = hashlib.sha256(file_data).hexdigest()
        stored = f"{uuid.uuid4()}{Path(filename).suffix}"
        return {"stored_filename": stored, "file_hash_sha256": file_hash, "file_size_bytes": len(file_data)}

    async def get_file(self, path):
        p = Path(path)
        return p.read_bytes() if p.exists() else None

    async def delete_file(self, path):
        p = Path(path)
        if p.exists(): p.unlink(); return True
        return False

storage_service = StorageService()
