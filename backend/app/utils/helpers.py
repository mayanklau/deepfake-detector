"""General utility helpers."""
import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid.uuid4())
    return f"{prefix}{uid}" if prefix else uid

def hash_file(data: bytes, algorithm: str = "sha256") -> str:
    """Compute hash of file data."""
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    name = re.sub(r"[^\w\s.-]", "", filename)
    name = re.sub(r"\s+", "_", name)
    return name[:255]

def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)

def parse_iso_datetime(s: str) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."

def mask_email(email: str) -> str:
    """Mask email for logging."""
    parts = email.split("@")
    if len(parts) != 2:
        return "***"
    local = parts[0]
    masked = local[0] + "***" + (local[-1] if len(local) > 1 else "")
    return f"{masked}@{parts[1]}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    if denominator == 0:
        return default
    return numerator / denominator
