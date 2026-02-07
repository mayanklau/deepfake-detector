import uuid
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

router = APIRouter()

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)
    mfa_code: Optional[str] = None
    remember_me: bool = False

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str
    mfa_enabled: bool
    created_at: datetime

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    return TokenResponse(access_token="eyJ...", refresh_token="eyJ...", expires_in=1800)

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(request: RegisterRequest):
    return UserResponse(id=str(uuid.uuid4()), email=request.email, username=request.username, first_name=request.first_name, last_name=request.last_name, role="analyst", mfa_enabled=False, created_at=datetime.now(timezone.utc))

@router.post("/refresh")
async def refresh_token(refresh_token: str):
    return {"access_token": "new_token", "token_type": "bearer", "expires_in": 1800}

@router.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}

@router.post("/password/reset-request")
async def request_password_reset(email: EmailStr):
    return {"message": "If account exists, reset email sent"}

@router.post("/password/reset")
async def reset_password(token: str, new_password: str):
    return {"message": "Password reset successfully"}

@router.post("/password/change")
async def change_password(current_password: str, new_password: str):
    return {"message": "Password changed"}

@router.post("/mfa/enable")
async def enable_mfa():
    return {"secret": "JBSWY3DPEHPK3PXP", "provisioning_uri": "otpauth://...", "backup_codes": ["ABCD-1234"]}

@router.post("/mfa/disable")
async def disable_mfa(totp_code: str):
    return {"message": "MFA disabled"}

@router.post("/mfa/verify")
async def verify_mfa(code: str):
    return {"verified": True}

@router.get("/me", response_model=UserResponse)
async def get_current_user():
    return UserResponse(id=str(uuid.uuid4()), email="user@example.com", username="testuser", role="analyst", mfa_enabled=False, created_at=datetime.now(timezone.utc))

@router.put("/me")
async def update_profile(first_name: Optional[str] = None, last_name: Optional[str] = None):
    return {"message": "Profile updated"}

@router.get("/oauth/{provider}")
async def oauth_redirect(provider: str):
    return {"redirect_url": f"https://oauth.{provider}.com/authorize"}

@router.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, code: str):
    return {"access_token": "...", "token_type": "bearer"}
