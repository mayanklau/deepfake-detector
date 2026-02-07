"""Tests for validators."""
import pytest
from app.utils.validators import validate_file_type, validate_file_size, get_media_type

class TestFileValidation:
    def test_valid_jpeg(self):
        ok, msg = validate_file_type("photo.jpg", "image/jpeg")
        assert ok

    def test_invalid_ext(self):
        ok, msg = validate_file_type("file.exe")
        assert not ok

    def test_valid_size(self):
        ok, msg = validate_file_size(1024 * 1024)
        assert ok

    def test_too_large(self):
        ok, msg = validate_file_size(600 * 1024 * 1024)
        assert not ok

    def test_empty(self):
        ok, msg = validate_file_size(0)
        assert not ok

    def test_media_type(self):
        assert get_media_type("video.mp4") == "video"
        assert get_media_type("song.wav") == "audio"
        assert get_media_type("pic.png") == "image"
