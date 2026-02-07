"""Tests for helper utilities."""
from app.utils.helpers import (
    generate_id, hash_file, format_file_size, sanitize_filename,
    mask_email, safe_divide, truncate_string,
)

def test_generate_id():
    assert len(generate_id()) == 36
    assert generate_id("pfx_").startswith("pfx_")

def test_hash_file():
    h = hash_file(b"hello world")
    assert len(h) == 64

def test_format_file_size():
    assert "KB" in format_file_size(2048)
    assert "MB" in format_file_size(5 * 1024 * 1024)

def test_sanitize_filename():
    assert sanitize_filename("hello world!@#.jpg") == "hello_world.jpg"

def test_mask_email():
    assert "@" in mask_email("test@example.com")

def test_safe_divide():
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 2) == 5.0

def test_truncate():
    assert truncate_string("short", 100) == "short"
    assert len(truncate_string("a" * 200, 50)) == 50
