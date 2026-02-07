"""Unit tests for authentication."""
import pytest
from app.core.auth import PasswordService, JWTService, TOTPService, APIKeyService

class TestPasswordService:
    def setup_method(self):
        self.svc = PasswordService(rounds=4)

    def test_hash_verify(self):
        h = self.svc.hash_password("TestP@ss123!")
        assert self.svc.verify_password("TestP@ss123!", h)
        assert not self.svc.verify_password("wrong", h)

    def test_strength_valid(self):
        ok, errs = self.svc.validate_password_strength("SecureP@ss123!")
        assert ok and not errs

    def test_strength_short(self):
        ok, errs = self.svc.validate_password_strength("Ab1!")
        assert not ok

class TestJWTService:
    def setup_method(self):
        self.svc = JWTService()

    def test_create_decode(self):
        token = self.svc.create_access_token("user1", role="analyst")
        payload = self.svc.decode_token(token)
        assert payload.sub == "user1"

    def test_token_pair(self):
        pair = self.svc.create_token_pair("user1")
        assert pair.access_token and pair.refresh_token

class TestTOTPService:
    def test_generate_verify(self):
        svc = TOTPService()
        secret = svc.generate_secret()
        code = svc.generate_totp(secret)
        assert svc.verify_totp(secret, code)

    def test_backup_codes(self):
        svc = TOTPService()
        codes = svc.generate_backup_codes(8)
        assert len(codes) == 8

class TestAPIKeyService:
    def test_generate(self):
        svc = APIKeyService()
        key, hsh = svc.generate_api_key()
        assert key.startswith("dfk_")
        assert len(hsh) == 64
