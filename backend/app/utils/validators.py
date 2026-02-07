"""Input validation utilities."""
import os
from typing import List, Optional, Tuple

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp", "image/gif"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/ogg", "audio/x-wav"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES | ALLOWED_AUDIO_TYPES

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".wav", ".mp3", ".flac", ".ogg",
}

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

def validate_file_type(filename: str, content_type: Optional[str] = None) -> Tuple[bool, str]:
    """Validate file type by extension and MIME type."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File extension '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    if content_type and content_type not in ALLOWED_TYPES:
        return False, f"MIME type '{content_type}' not allowed"
    return True, "OK"

def validate_file_size(size_bytes: int, max_size: int = MAX_FILE_SIZE) -> Tuple[bool, str]:
    """Validate file size."""
    if size_bytes <= 0:
        return False, "File is empty"
    if size_bytes > max_size:
        return False, f"File too large ({size_bytes} bytes). Maximum: {max_size} bytes ({max_size // 1024 // 1024}MB)"
    return True, "OK"

def get_media_type(filename: str) -> str:
    """Determine media type from filename."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}:
        return "image"
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    if ext in {".wav", ".mp3", ".flac", ".ogg"}:
        return "audio"
    return "unknown"

def validate_analysis_config(config: dict) -> Tuple[bool, List[str]]:
    """Validate analysis configuration."""
    errors = []
    valid_methods = {"weighted_average", "voting", "stacking", "max_score"}
    if config.get("ensemble_method") and config["ensemble_method"] not in valid_methods:
        errors.append(f"Invalid ensemble method. Must be one of: {valid_methods}")
    priority = config.get("priority", 5)
    if not 1 <= priority <= 10:
        errors.append("Priority must be between 1 and 10")
    return len(errors) == 0, errors
