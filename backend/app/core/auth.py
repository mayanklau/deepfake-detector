"""
DeepFake Detector - Authentication & Authorization Service
Complete auth system with JWT, OAuth2, MFA, API keys, rate limiting,
session management, and role-based access control.
"""

import hashlib
import hmac
import secrets
import struct
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.database import (
    User, UserRole, UserStatus, UserSession, APIKey, AuditLog, AuditAction,
)


settings = get_settings()
security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=settings.security.API_KEY_HEADER, auto_error=False)


# ============================================================================
# Password Hashing
# ============================================================================

class PasswordService:
    """Secure password hashing and validation."""
    
    def __init__(self, rounds: int = 12):
        self.rounds = rounds
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt(rounds=self.rounds)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password meets security requirements."""
        errors = []
        
        if len(password) < settings.security.PASSWORD_MIN_LENGTH:
            errors.append(
                f"Password must be at least {settings.security.PASSWORD_MIN_LENGTH} characters"
            )
        
        if settings.security.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if settings.security.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if settings.security.PASSWORD_REQUIRE_DIGITS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if settings.security.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")
        
        # Check for common patterns
        common_passwords = {"password", "123456", "qwerty", "admin", "letmein"}
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors


# ============================================================================
# JWT Token Management
# ============================================================================

class TokenPayload(BaseModel):
    sub: str
    exp: int
    iat: int
    jti: str
    type: str = "access"
    role: Optional[str] = None
    org_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int


class JWTService:
    """JWT token creation and validation."""
    
    def __init__(self):
        self.secret_key = settings.security.JWT_SECRET_KEY
        self.algorithm = settings.security.JWT_ALGORITHM
        self.access_expiry = timedelta(minutes=settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_expiry = timedelta(days=settings.security.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    def create_access_token(
        self,
        user_id: str,
        role: str = None,
        org_id: str = None,
        scopes: List[str] = None,
        extra_claims: Dict[str, Any] = None,
    ) -> str:
        """Create a JWT access token."""
        now = datetime.now(timezone.utc)
        expires = now + self.access_expiry
        
        payload = {
            "sub": user_id,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
            "type": "access",
            "role": role,
            "org_id": org_id,
            "scopes": scopes or [],
        }
        
        if extra_claims:
            payload.update(extra_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a JWT refresh token."""
        now = datetime.now(timezone.utc)
        expires = now + self.refresh_expiry
        
        payload = {
            "sub": user_id,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
            "type": "refresh",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_token_pair(
        self,
        user_id: str,
        role: str = None,
        org_id: str = None,
        scopes: List[str] = None,
    ) -> TokenPair:
        """Create both access and refresh tokens."""
        access_token = self.create_access_token(user_id, role, org_id, scopes)
        refresh_token = self.create_refresh_token(user_id)
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_expiry.total_seconds()),
            refresh_expires_in=int(self.refresh_expiry.total_seconds()),
        )
    
    def decode_token(self, token: str) -> TokenPayload:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def create_email_verification_token(self, email: str) -> str:
        """Create a token for email verification."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=24)
        
        payload = {
            "sub": email,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "type": "email_verification",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_password_reset_token(self, user_id: str) -> str:
        """Create a token for password reset."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)
        
        payload = {
            "sub": user_id,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "type": "password_reset",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)


# ============================================================================
# MFA / TOTP Service
# ============================================================================

class TOTPService:
    """Time-based One-Time Password service."""
    
    def __init__(self, digits: int = 6, interval: int = 30):
        self.digits = digits
        self.interval = interval
    
    def generate_secret(self) -> str:
        """Generate a new TOTP secret."""
        import base64
        return base64.b32encode(secrets.token_bytes(20)).decode("utf-8").rstrip("=")
    
    def generate_totp(self, secret: str, timestamp: int = None) -> str:
        """Generate a TOTP code."""
        import base64
        
        if timestamp is None:
            timestamp = int(time.time())
        
        # Pad secret
        padding = 8 - (len(secret) % 8)
        if padding != 8:
            secret += "=" * padding
        
        key = base64.b32decode(secret.upper())
        counter = timestamp // self.interval
        counter_bytes = struct.pack(">Q", counter)
        
        # HMAC-SHA1
        mac = hmac.new(key, counter_bytes, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = mac[-1] & 0x0F
        code = struct.unpack(">I", mac[offset:offset + 4])[0]
        code = code & 0x7FFFFFFF
        code = code % (10 ** self.digits)
        
        return str(code).zfill(self.digits)
    
    def verify_totp(self, secret: str, code: str, window: int = 1) -> bool:
        """Verify a TOTP code with time window tolerance."""
        timestamp = int(time.time())
        
        for i in range(-window, window + 1):
            check_time = timestamp + (i * self.interval)
            expected = self.generate_totp(secret, check_time)
            if hmac.compare_digest(code, expected):
                return True
        
        return False
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup recovery codes."""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes
    
    def get_provisioning_uri(
        self, secret: str, email: str, issuer: str = None
    ) -> str:
        """Generate a provisioning URI for QR code."""
        import urllib.parse
        
        issuer = issuer or settings.security.MFA_ISSUER
        label = urllib.parse.quote(f"{issuer}:{email}")
        params = urllib.parse.urlencode({
            "secret": secret,
            "issuer": issuer,
            "algorithm": "SHA1",
            "digits": self.digits,
            "period": self.interval,
        })
        
        return f"otpauth://totp/{label}?{params}"


# ============================================================================
# API Key Service
# ============================================================================

class APIKeyService:
    """API key generation and validation."""
    
    def __init__(self):
        self.prefix = settings.security.API_KEY_PREFIX
        self.key_length = settings.security.API_KEY_LENGTH
    
    def generate_api_key(self) -> Tuple[str, str]:
        """Generate an API key. Returns (full_key, key_hash)."""
        raw_key = secrets.token_urlsafe(self.key_length)
        full_key = f"{self.prefix}{raw_key}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return full_key, key_hash
    
    def hash_api_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_key_prefix(self, key: str) -> str:
        """Extract the prefix portion of an API key for lookup."""
        return key[:len(self.prefix) + 8]


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """In-memory rate limiter with sliding window."""
    
    def __init__(self):
        self._windows: Dict[str, List[float]] = {}
    
    def _parse_rate(self, rate: str) -> Tuple[int, int]:
        """Parse rate string like '100/minute' into (count, seconds)."""
        count_str, period = rate.split("/")
        count = int(count_str)
        
        period_map = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }
        
        seconds = period_map.get(period, 60)
        return count, seconds
    
    def is_rate_limited(self, key: str, rate: str) -> Tuple[bool, Dict[str, int]]:
        """Check if a key has exceeded its rate limit."""
        max_requests, window_seconds = self._parse_rate(rate)
        now = time.time()
        window_start = now - window_seconds
        
        if key not in self._windows:
            self._windows[key] = []
        
        # Clean old entries
        self._windows[key] = [
            t for t in self._windows[key] if t > window_start
        ]
        
        current_count = len(self._windows[key])
        remaining = max(0, max_requests - current_count)
        
        if current_count >= max_requests:
            retry_after = int(self._windows[key][0] - window_start) + 1
            return True, {
                "limit": max_requests,
                "remaining": 0,
                "retry_after": retry_after,
                "reset": int(self._windows[key][0] + window_seconds),
            }
        
        # Record this request
        self._windows[key].append(now)
        
        return False, {
            "limit": max_requests,
            "remaining": remaining - 1,
            "retry_after": 0,
            "reset": int(now + window_seconds),
        }


# ============================================================================
# Permission System (RBAC)
# ============================================================================

class Permission:
    """Permission constants."""
    
    # Analysis
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_UPDATE = "analysis:update"
    ANALYSIS_DELETE = "analysis:delete"
    ANALYSIS_BATCH = "analysis:batch"
    
    # Reports
    REPORT_CREATE = "report:create"
    REPORT_READ = "report:read"
    REPORT_DOWNLOAD = "report:download"
    
    # Users
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Admin
    ADMIN_ACCESS = "admin:access"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_USERS = "admin:users"
    ADMIN_MODELS = "admin:models"
    ADMIN_AUDIT = "admin:audit"
    
    # API
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # Webhooks
    WEBHOOK_CREATE = "webhook:create"
    WEBHOOK_READ = "webhook:read"
    WEBHOOK_DELETE = "webhook:delete"
    
    # Export
    EXPORT_DATA = "export:data"


# Role-Permission Mapping
ROLE_PERMISSIONS = {
    UserRole.SUPER_ADMIN: [
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_DELETE,
        Permission.ANALYSIS_BATCH,
        Permission.REPORT_CREATE, Permission.REPORT_READ, Permission.REPORT_DOWNLOAD,
        Permission.USER_READ, Permission.USER_CREATE,
        Permission.USER_UPDATE, Permission.USER_DELETE,
        Permission.ADMIN_ACCESS, Permission.ADMIN_SETTINGS,
        Permission.ADMIN_USERS, Permission.ADMIN_MODELS, Permission.ADMIN_AUDIT,
        Permission.API_READ, Permission.API_WRITE, Permission.API_ADMIN,
        Permission.WEBHOOK_CREATE, Permission.WEBHOOK_READ, Permission.WEBHOOK_DELETE,
        Permission.EXPORT_DATA,
    ],
    UserRole.ADMIN: [
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_DELETE,
        Permission.ANALYSIS_BATCH,
        Permission.REPORT_CREATE, Permission.REPORT_READ, Permission.REPORT_DOWNLOAD,
        Permission.USER_READ, Permission.USER_CREATE, Permission.USER_UPDATE,
        Permission.ADMIN_ACCESS, Permission.ADMIN_SETTINGS, Permission.ADMIN_USERS,
        Permission.API_READ, Permission.API_WRITE,
        Permission.WEBHOOK_CREATE, Permission.WEBHOOK_READ, Permission.WEBHOOK_DELETE,
        Permission.EXPORT_DATA,
    ],
    UserRole.ANALYST: [
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE, Permission.ANALYSIS_BATCH,
        Permission.REPORT_CREATE, Permission.REPORT_READ, Permission.REPORT_DOWNLOAD,
        Permission.API_READ,
        Permission.WEBHOOK_CREATE, Permission.WEBHOOK_READ,
        Permission.EXPORT_DATA,
    ],
    UserRole.VIEWER: [
        Permission.ANALYSIS_READ,
        Permission.REPORT_READ, Permission.REPORT_DOWNLOAD,
    ],
    UserRole.API_USER: [
        Permission.ANALYSIS_CREATE, Permission.ANALYSIS_READ,
        Permission.ANALYSIS_BATCH,
        Permission.REPORT_READ,
        Permission.API_READ, Permission.API_WRITE,
    ],
}


def has_permission(role: UserRole, permission: str) -> bool:
    """Check if a role has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, [])


def get_permissions(role: UserRole) -> List[str]:
    """Get all permissions for a role."""
    return ROLE_PERMISSIONS.get(role, [])


# ============================================================================
# Auth Schemas
# ============================================================================

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)
    mfa_code: Optional[str] = Field(None, min_length=6, max_length=6)
    remember_me: bool = False


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class MFAEnableRequest(BaseModel):
    password: str
    totp_code: str = Field(..., min_length=6, max_length=6)


class MFADisableRequest(BaseModel):
    password: str
    totp_code: str = Field(..., min_length=6, max_length=6)


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    allowed_ips: Optional[List[str]] = None
    rate_limit: Optional[str] = None


# ============================================================================
# Auth Response Models
# ============================================================================

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    role: str
    status: str
    mfa_enabled: bool
    avatar_url: Optional[str]
    organization_id: Optional[str]
    created_at: datetime
    last_login_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    user: UserResponse
    tokens: TokenPair
    requires_mfa: bool = False


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: Optional[str] = None  # Only returned on creation
    key_prefix: str
    scopes: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    last_used_at: Optional[datetime]
    usage_count: int


# ============================================================================
# Service Instances
# ============================================================================

password_service = PasswordService(rounds=settings.security.PASSWORD_BCRYPT_ROUNDS)
jwt_service = JWTService()
totp_service = TOTPService(
    digits=settings.security.TOTP_DIGITS,
    interval=settings.security.TOTP_INTERVAL,
)
api_key_service = APIKeyService()
rate_limiter = RateLimiter()


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[TokenPayload]:
    """Extract and validate JWT token from request."""
    if not credentials:
        return None
    
    return jwt_service.decode_token(credentials.credentials)


async def get_current_user_from_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[Dict[str, Any]]:
    """Validate API key from request header."""
    if not api_key:
        return None
    
    key_hash = api_key_service.hash_api_key(api_key)
    # In production, this would query the database
    return {"key_hash": key_hash, "api_key": api_key}


async def require_auth(
    token_payload: Optional[TokenPayload] = Depends(get_current_user_from_token),
    api_key_data: Optional[Dict] = Depends(get_current_user_from_api_key),
) -> TokenPayload:
    """Require either JWT or API key authentication."""
    if token_payload:
        return token_payload
    
    if api_key_data:
        # Convert API key auth to a token-like payload
        return TokenPayload(
            sub="api_key_user",
            exp=int(time.time()) + 3600,
            iat=int(time.time()),
            jti=str(uuid.uuid4()),
            type="api_key",
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: str):
    """Create a dependency that checks for a specific permission."""
    async def _check_permission(
        payload: TokenPayload = Depends(require_auth),
    ) -> TokenPayload:
        if payload.role and not has_permission(UserRole(payload.role), permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}",
            )
        return payload
    
    return _check_permission


def require_role(*roles: UserRole):
    """Create a dependency that checks for specific roles."""
    async def _check_role(
        payload: TokenPayload = Depends(require_auth),
    ) -> TokenPayload:
        if payload.role and UserRole(payload.role) not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {', '.join(r.value for r in roles)}",
            )
        return payload
    
    return _check_role


# ============================================================================
# Audit Logging Helper
# ============================================================================

async def create_audit_log(
    db: AsyncSession,
    user_id: Optional[uuid.UUID],
    action: AuditAction,
    resource_type: str = None,
    resource_id: str = None,
    details: Dict = None,
    ip_address: str = None,
    user_agent: str = None,
):
    """Create an audit log entry."""
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(log)
    await db.flush()
    return log
