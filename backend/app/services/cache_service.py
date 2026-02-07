import hashlib, time, logging
from typing import Any, Dict, Optional
logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self._cache = {}

    async def get(self, key):
        if key in self._cache and self._cache[key]["exp"] > time.time():
            return self._cache[key]["val"]
        return None

    async def set(self, key, value, ttl=3600):
        self._cache[key] = {"val": value, "exp": time.time() + ttl}

    async def delete(self, key):
        self._cache.pop(key, None)

    async def clear(self):
        self._cache.clear()

cache_service = CacheService()
