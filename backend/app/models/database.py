"""
DeepFake Detector - Database Models
Comprehensive SQLAlchemy ORM models for the entire platform.
Includes user management, analysis tracking, reporting, audit logging,
API key management, webhooks, and multi-tenant support.
"""

import uuid
import enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SAEnum, Float, ForeignKey,
    Index, Integer, JSON, LargeBinary, String, Text, Table,
    UniqueConstraint, CheckConstraint, BigInteger, SmallInteger,
    event, func, text, Numeric,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, INET, TSVECTOR
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    validates, Session,
)


# ============================================================================
# Base Model
# ============================================================================

class Base(DeclarativeBase):
    """Base model with common fields."""
    pass


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )


# ============================================================================
# Enums
# ============================================================================

class UserRole(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"


class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    ANALYZING = "analyzing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class AnalysisType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    LIVE_STREAM = "live_stream"
    BATCH = "batch"


class MediaType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class VerdictLevel(str, enum.Enum):
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    FAKE = "fake"


class ManipulationType(str, enum.Enum):
    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    FACE_GENERATION = "face_generation"
    LIP_SYNC = "lip_sync"
    VOICE_CLONE = "voice_clone"
    AUDIO_SPLICE = "audio_splice"
    GAN_GENERATED = "gan_generated"
    DIFFUSION_GENERATED = "diffusion_generated"
    INPAINTING = "inpainting"
    ATTRIBUTE_MANIPULATION = "attribute_manipulation"
    EXPRESSION_MANIPULATION = "expression_manipulation"
    BACKGROUND_MANIPULATION = "background_manipulation"
    TEMPORAL_MANIPULATION = "temporal_manipulation"
    UNKNOWN = "unknown"


class NotificationType(str, enum.Enum):
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"
    HIGH_RISK_DETECTED = "high_risk_detected"
    BATCH_COMPLETE = "batch_complete"
    SYSTEM_ALERT = "system_alert"
    ACCOUNT_ACTIVITY = "account_activity"
    WEBHOOK_DELIVERY = "webhook_delivery"
    REPORT_READY = "report_ready"


class AuditAction(str, enum.Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLE = "mfa_enable"
    MFA_DISABLE = "mfa_disable"
    FILE_UPLOAD = "file_upload"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_VIEW = "analysis_view"
    REPORT_GENERATE = "report_generate"
    REPORT_DOWNLOAD = "report_download"
    SETTINGS_CHANGE = "settings_change"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    WEBHOOK_CREATE = "webhook_create"
    WEBHOOK_DELETE = "webhook_delete"
    EXPORT_DATA = "export_data"
    ADMIN_ACTION = "admin_action"


class ReportFormat(str, enum.Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"


class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class WebhookEvent(str, enum.Enum):
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    HIGH_RISK_DETECTED = "analysis.high_risk"
    BATCH_COMPLETED = "batch.completed"
    REPORT_READY = "report.ready"


# ============================================================================
# Association Tables
# ============================================================================

user_team_association = Table(
    "user_team_association",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")),
    Column("team_id", UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE")),
    Column("role", String(50), default="member"),
    Column("joined_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
    UniqueConstraint("user_id", "team_id", name="uq_user_team"),
)

analysis_tag_association = Table(
    "analysis_tag_association",
    Base.metadata,
    Column("analysis_id", UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")),
    Column("tag_id", UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE")),
    UniqueConstraint("analysis_id", "tag_id", name="uq_analysis_tag"),
)


# ============================================================================
# Organization & Team Models
# ============================================================================

class Organization(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "organizations"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Subscription
    subscription_tier: Mapped[SubscriptionTier] = mapped_column(
        SAEnum(SubscriptionTier), default=SubscriptionTier.FREE
    )
    subscription_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    
    # Limits
    max_users: Mapped[int] = mapped_column(Integer, default=5)
    max_analyses_per_month: Mapped[int] = mapped_column(Integer, default=100)
    max_storage_gb: Mapped[int] = mapped_column(Integer, default=10)
    current_storage_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    
    # Settings
    settings: Mapped[Optional[Dict]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    users = relationship("User", back_populates="organization", lazy="dynamic")
    teams = relationship("Team", back_populates="organization", lazy="dynamic")
    api_keys = relationship("APIKey", back_populates="organization", lazy="dynamic")
    webhooks = relationship("Webhook", back_populates="organization", lazy="dynamic")
    
    __table_args__ = (
        Index("ix_organizations_slug", "slug"),
        Index("ix_organizations_subscription", "subscription_tier"),
    )


class Team(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "teams"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    
    # Relationships
    organization = relationship("Organization", back_populates="teams")
    members = relationship("User", secondary=user_team_association, back_populates="teams")
    
    __table_args__ = (
        UniqueConstraint("name", "organization_id", name="uq_team_org"),
    )


# ============================================================================
# User Model
# ============================================================================

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Profile
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    locale: Mapped[str] = mapped_column(String(10), default="en")
    
    # Role & Status
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole), default=UserRole.ANALYST)
    status: Mapped[UserStatus] = mapped_column(SAEnum(UserStatus), default=UserStatus.PENDING_VERIFICATION)
    
    # Organization
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True
    )
    
    # Security
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mfa_backup_codes: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Login tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_ip: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Preferences
    preferences: Mapped[Optional[Dict]] = mapped_column(JSONB, default=dict)
    notification_settings: Mapped[Optional[Dict]] = mapped_column(JSONB, default=dict)
    
    # Usage
    analyses_count: Mapped[int] = mapped_column(Integer, default=0)
    storage_used_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    
    # OAuth
    oauth_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    oauth_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    teams = relationship("Team", secondary=user_team_association, back_populates="members")
    analyses = relationship("Analysis", back_populates="user", lazy="dynamic")
    api_keys = relationship("APIKey", back_populates="user", lazy="dynamic")
    notifications = relationship("Notification", back_populates="user", lazy="dynamic")
    audit_logs = relationship("AuditLog", back_populates="user", lazy="dynamic")
    reports = relationship("Report", back_populates="user", lazy="dynamic")
    sessions = relationship("UserSession", back_populates="user", lazy="dynamic")
    
    @hybrid_property
    def full_name(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or self.username
    
    @hybrid_property
    def is_active(self) -> bool:
        return self.status == UserStatus.ACTIVE and not self.is_deleted
    
    @hybrid_property
    def is_admin(self) -> bool:
        return self.role in (UserRole.ADMIN, UserRole.SUPER_ADMIN)
    
    __table_args__ = (
        Index("ix_users_email_status", "email", "status"),
        Index("ix_users_org", "organization_id"),
        Index("ix_users_role", "role"),
        Index("ix_users_created", "created_at"),
    )


class UserSession(Base, TimestampMixin):
    __tablename__ = "user_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    refresh_token_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device_info: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index("ix_sessions_user_active", "user_id", "is_active"),
        Index("ix_sessions_expires", "expires_at"),
    )


# ============================================================================
# Media & Upload Models
# ============================================================================

class MediaFile(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "media_files"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    
    # File Info
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_hash_md5: Mapped[str] = mapped_column(String(32), nullable=False)
    file_hash_perceptual: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    
    media_type: Mapped[MediaType] = mapped_column(SAEnum(MediaType), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # Image Properties
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    color_space: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    bit_depth: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    
    # Video Properties
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    codec: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    bitrate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Audio Properties
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channels: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    audio_codec: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    audio_bitrate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Metadata
    exif_data: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    xmp_data: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    raw_metadata: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Processing
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    preview_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Storage
    storage_backend: Mapped[str] = mapped_column(String(20), default="local")
    storage_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    
    # Relationships
    user = relationship("User")
    analyses = relationship("Analysis", back_populates="media_file", lazy="dynamic")
    
    __table_args__ = (
        Index("ix_media_user_type", "user_id", "media_type"),
        Index("ix_media_hash", "file_hash_sha256"),
        Index("ix_media_created", "created_at"),
    )


# ============================================================================
# Analysis Models
# ============================================================================

class Analysis(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "analyses"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    media_file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("media_files.id", ondelete="CASCADE")
    )
    
    # Analysis Info
    analysis_type: Mapped[AnalysisType] = mapped_column(SAEnum(AnalysisType), nullable=False)
    status: Mapped[AnalysisStatus] = mapped_column(
        SAEnum(AnalysisStatus), default=AnalysisStatus.PENDING
    )
    priority: Mapped[int] = mapped_column(SmallInteger, default=5, nullable=False)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Results
    verdict: Mapped[Optional[VerdictLevel]] = mapped_column(SAEnum(VerdictLevel), nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    manipulation_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detected_manipulations: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    
    # Detailed Results
    face_detection_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    manipulation_detection_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    frequency_analysis_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    gan_detection_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    noise_analysis_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    compression_analysis_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    metadata_analysis_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    temporal_analysis_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    audio_analysis_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    lip_sync_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Ensemble Results
    ensemble_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    model_versions: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Explainability
    grad_cam_paths: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    attention_map_paths: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    feature_attribution: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    explanation_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Batch reference
    batch_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("batch_jobs.id", ondelete="SET NULL"), nullable=True
    )
    
    # Task tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Configuration used
    analysis_config: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Notes
    user_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    analyst_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    media_file = relationship("MediaFile", back_populates="analyses")
    reports = relationship("Report", back_populates="analysis", lazy="dynamic")
    face_detections = relationship("FaceDetection", back_populates="analysis", lazy="dynamic")
    frame_analyses = relationship("FrameAnalysis", back_populates="analysis", lazy="dynamic")
    tags = relationship("Tag", secondary=analysis_tag_association, back_populates="analyses")
    batch = relationship("BatchJob", back_populates="analyses")
    comparison_items = relationship("ComparisonItem", back_populates="analysis", lazy="dynamic")
    
    __table_args__ = (
        Index("ix_analyses_user_status", "user_id", "status"),
        Index("ix_analyses_verdict", "verdict"),
        Index("ix_analyses_created", "created_at"),
        Index("ix_analyses_batch", "batch_id"),
        Index("ix_analyses_type_status", "analysis_type", "status"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="ck_confidence_range"),
        CheckConstraint("manipulation_probability >= 0 AND manipulation_probability <= 1", name="ck_manipulation_range"),
        CheckConstraint("priority >= 1 AND priority <= 10", name="ck_priority_range"),
    )


class FaceDetection(Base, TimestampMixin):
    __tablename__ = "face_detections"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    
    # Detection
    face_index: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Bounding Box
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_width: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_height: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Landmarks
    landmarks: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    landmark_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Face Analysis
    face_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_yaw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_pitch: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_roll: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Manipulation per face
    is_manipulated: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    manipulation_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    manipulation_type: Mapped[Optional[ManipulationType]] = mapped_column(
        SAEnum(ManipulationType), nullable=True
    )
    manipulation_details: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Embedding
    face_embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    face_crop_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Frame reference (for videos)
    frame_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    timestamp_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="face_detections")
    
    __table_args__ = (
        Index("ix_face_analysis", "analysis_id"),
        Index("ix_face_manipulated", "is_manipulated"),
    )


class FrameAnalysis(Base, TimestampMixin):
    __tablename__ = "frame_analyses"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Per-frame results
    manipulation_score: Mapped[float] = mapped_column(Float, nullable=False)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Detailed scores
    scores: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Anomalies
    has_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    anomaly_details: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Frame paths
    frame_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    heatmap_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="frame_analyses")
    
    __table_args__ = (
        Index("ix_frame_analysis", "analysis_id", "frame_number"),
        UniqueConstraint("analysis_id", "frame_number", name="uq_analysis_frame"),
    )


# ============================================================================
# Batch Processing
# ============================================================================

class BatchJob(Base, TimestampMixin):
    __tablename__ = "batch_jobs"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[AnalysisStatus] = mapped_column(
        SAEnum(AnalysisStatus), default=AnalysisStatus.PENDING
    )
    
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)
    
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    config: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    summary_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User")
    analyses = relationship("Analysis", back_populates="batch", lazy="dynamic")
    
    @hybrid_property
    def progress_percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return ((self.completed_items + self.failed_items) / self.total_items) * 100


# ============================================================================
# Comparison
# ============================================================================

class Comparison(Base, TimestampMixin):
    __tablename__ = "comparisons"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    comparison_results: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User")
    items = relationship("ComparisonItem", back_populates="comparison", lazy="dynamic")


class ComparisonItem(Base, TimestampMixin):
    __tablename__ = "comparison_items"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    comparison_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("comparisons.id", ondelete="CASCADE")
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    order: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    comparison = relationship("Comparison", back_populates="items")
    analysis = relationship("Analysis", back_populates="comparison_items")


# ============================================================================
# Reports
# ============================================================================

class Report(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "reports"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    report_format: Mapped[ReportFormat] = mapped_column(SAEnum(ReportFormat), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # Report Content
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    findings: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    recommendations: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    
    # Template
    template_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    custom_branding: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Access
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    share_token: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True)
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="reports")
    analysis = relationship("Analysis", back_populates="reports")
    
    __table_args__ = (
        Index("ix_reports_user", "user_id"),
        Index("ix_reports_analysis", "analysis_id"),
        Index("ix_reports_share", "share_token"),
    )


# ============================================================================
# Tags
# ============================================================================

class Tag(Base, TimestampMixin):
    __tablename__ = "tags"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    analyses = relationship("Analysis", secondary=analysis_tag_association, back_populates="tags")


# ============================================================================
# API Keys
# ============================================================================

class APIKey(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Permissions
    scopes: Mapped[List] = mapped_column(JSONB, default=list)
    rate_limit: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    allowed_ips: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    allowed_origins: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_count: Mapped[int] = mapped_column(BigInteger, default=0)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    organization = relationship("Organization", back_populates="api_keys")
    
    __table_args__ = (
        Index("ix_api_keys_hash", "key_hash"),
        Index("ix_api_keys_prefix", "key_prefix"),
        Index("ix_api_keys_user", "user_id"),
    )


# ============================================================================
# Webhooks
# ============================================================================

class Webhook(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "webhooks"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    secret: Mapped[str] = mapped_column(String(255), nullable=False)
    
    events: Mapped[List] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Status tracking
    last_triggered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    total_deliveries: Mapped[int] = mapped_column(Integer, default=0)
    
    # Headers
    custom_headers: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User")
    organization = relationship("Organization", back_populates="webhooks")
    deliveries = relationship("WebhookDelivery", back_populates="webhook", lazy="dynamic")


class WebhookDelivery(Base, TimestampMixin):
    __tablename__ = "webhook_deliveries"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("webhooks.id", ondelete="CASCADE")
    )
    
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[Dict] = mapped_column(JSONB, nullable=False)
    
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    is_successful: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    webhook = relationship("Webhook", back_populates="deliveries")
    
    __table_args__ = (
        Index("ix_deliveries_webhook", "webhook_id"),
        Index("ix_deliveries_created", "created_at"),
    )


# ============================================================================
# Notifications
# ============================================================================

class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    
    notification_type: Mapped[NotificationType] = mapped_column(
        SAEnum(NotificationType), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    
    __table_args__ = (
        Index("ix_notifications_user_read", "user_id", "is_read"),
        Index("ix_notifications_created", "created_at"),
    )


# ============================================================================
# Audit Log
# ============================================================================

class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    
    action: Mapped[AuditAction] = mapped_column(SAEnum(AuditAction), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    details: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    changes: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("ix_audit_user", "user_id"),
        Index("ix_audit_action", "action"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_created", "created_at"),
    )


# ============================================================================
# ML Model Registry
# ============================================================================

class MLModelRecord(Base, TimestampMixin):
    __tablename__ = "ml_model_records"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False)
    
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # Performance metrics
    metrics: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    training_config: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    loaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[List]] = mapped_column(JSONB, nullable=True)
    
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_version"),
        Index("ix_ml_models_active", "is_active"),
        Index("ix_ml_models_type", "model_type"),
    )


# ============================================================================
# System Configuration
# ============================================================================

class SystemConfig(Base, TimestampMixin):
    __tablename__ = "system_configs"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    value: Mapped[Dict] = mapped_column(JSONB, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    
    __table_args__ = (
        Index("ix_system_config_key", "key"),
    )


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimitRecord(Base):
    __tablename__ = "rate_limit_records"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    hits: Mapped[int] = mapped_column(Integer, default=1)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    __table_args__ = (
        Index("ix_rate_limit_key_window", "key", "window_start"),
    )
