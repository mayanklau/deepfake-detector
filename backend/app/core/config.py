"""
DeepFake Detector - Core Configuration Module
Production-grade configuration management with environment variable support,
validation, and multi-environment deployment configurations.
"""

import os
import secrets
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import lru_cache
from enum import Enum

from pydantic import (
    AnyHttpUrl,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"


class CacheBackend(str, Enum):
    REDIS = "redis"
    MEMCACHED = "memcached"
    LOCAL = "local"


class MLBackend(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"


# ============================================================================
# Base Settings
# ============================================================================

class BaseAppSettings(BaseSettings):
    """Base settings with common configuration."""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# ============================================================================
# Database Configuration
# ============================================================================

class DatabaseSettings(BaseAppSettings):
    """Database connection and pool configuration."""
    
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="deepfake_user", description="Database user")
    POSTGRES_PASSWORD: str = Field(default="changeme", description="Database password")
    POSTGRES_DB: str = Field(default="deepfake_detector", description="Database name")
    
    # Connection Pool Settings
    DB_POOL_SIZE: int = Field(default=20, ge=5, le=100)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=50)
    DB_POOL_TIMEOUT: int = Field(default=30, ge=10, le=120)
    DB_POOL_RECYCLE: int = Field(default=3600, ge=300)
    DB_ECHO: bool = Field(default=False)
    
    # Read Replicas
    DB_READ_REPLICAS: List[str] = Field(default_factory=list)
    DB_USE_READ_REPLICA: bool = Field(default=False)
    
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Redis Configuration
# ============================================================================

class RedisSettings(BaseAppSettings):
    """Redis connection configuration for caching and message broker."""
    
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_CACHE_DB: int = Field(default=1)
    REDIS_CELERY_DB: int = Field(default=2)
    REDIS_SESSION_DB: int = Field(default=3)
    REDIS_MAX_CONNECTIONS: int = Field(default=50)
    REDIS_SOCKET_TIMEOUT: int = Field(default=5)
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True)
    REDIS_SSL: bool = Field(default=False)
    
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        scheme = "rediss" if self.REDIS_SSL else "redis"
        return f"{scheme}://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def CELERY_BROKER_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        scheme = "rediss" if self.REDIS_SSL else "redis"
        return f"{scheme}://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_CELERY_DB}"
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Authentication & Security Configuration
# ============================================================================

class SecuritySettings(BaseAppSettings):
    """JWT, OAuth, and security configuration."""
    
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5, le=1440)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=90)
    
    # Password Policy
    PASSWORD_MIN_LENGTH: int = Field(default=12, ge=8)
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(default=True)
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(default=True)
    PASSWORD_REQUIRE_DIGITS: bool = Field(default=True)
    PASSWORD_REQUIRE_SPECIAL: bool = Field(default=True)
    PASSWORD_BCRYPT_ROUNDS: int = Field(default=12, ge=10, le=16)
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_DEFAULT: str = Field(default="100/minute")
    RATE_LIMIT_AUTH: str = Field(default="5/minute")
    RATE_LIMIT_UPLOAD: str = Field(default="10/minute")
    RATE_LIMIT_ANALYSIS: str = Field(default="20/minute")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    
    # OAuth2 Providers
    GOOGLE_CLIENT_ID: Optional[str] = Field(default=None)
    GOOGLE_CLIENT_SECRET: Optional[str] = Field(default=None)
    GITHUB_CLIENT_ID: Optional[str] = Field(default=None)
    GITHUB_CLIENT_SECRET: Optional[str] = Field(default=None)
    MICROSOFT_CLIENT_ID: Optional[str] = Field(default=None)
    MICROSOFT_CLIENT_SECRET: Optional[str] = Field(default=None)
    
    # API Keys
    API_KEY_HEADER: str = Field(default="X-API-Key")
    API_KEY_PREFIX: str = Field(default="dfk_")
    API_KEY_LENGTH: int = Field(default=48)
    
    # MFA
    MFA_ENABLED: bool = Field(default=True)
    MFA_ISSUER: str = Field(default="DeepFake Detector")
    TOTP_DIGITS: int = Field(default=6)
    TOTP_INTERVAL: int = Field(default=30)
    
    # Session
    SESSION_COOKIE_NAME: str = Field(default="dfdetector_session")
    SESSION_MAX_AGE: int = Field(default=86400)
    SESSION_SECURE: bool = Field(default=True)
    SESSION_HTTPONLY: bool = Field(default=True)
    SESSION_SAMESITE: str = Field(default="lax")
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Storage Configuration
# ============================================================================

class StorageSettings(BaseAppSettings):
    """File storage configuration for uploads and processed files."""
    
    STORAGE_BACKEND: StorageBackend = Field(default=StorageBackend.LOCAL)
    
    # Local Storage
    UPLOAD_DIR: str = Field(default="/data/uploads")
    PROCESSED_DIR: str = Field(default="/data/processed")
    TEMP_DIR: str = Field(default="/data/temp")
    REPORTS_DIR: str = Field(default="/data/reports")
    MODELS_DIR: str = Field(default="/data/models")
    CACHE_DIR: str = Field(default="/data/cache")
    
    # File Limits
    MAX_UPLOAD_SIZE_MB: int = Field(default=500, ge=1, le=5000)
    MAX_VIDEO_DURATION_SECONDS: int = Field(default=600, ge=10, le=3600)
    MAX_BATCH_SIZE: int = Field(default=50, ge=1, le=200)
    ALLOWED_IMAGE_EXTENSIONS: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    )
    ALLOWED_VIDEO_EXTENSIONS: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]
    )
    ALLOWED_AUDIO_EXTENSIONS: List[str] = Field(
        default=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
    )
    
    # S3 Configuration
    S3_BUCKET: Optional[str] = Field(default=None)
    S3_REGION: Optional[str] = Field(default=None)
    S3_ACCESS_KEY: Optional[str] = Field(default=None)
    S3_SECRET_KEY: Optional[str] = Field(default=None)
    S3_ENDPOINT_URL: Optional[str] = Field(default=None)
    S3_USE_SSL: bool = Field(default=True)
    S3_PREFIX: str = Field(default="deepfake-detector/")
    
    # GCS Configuration
    GCS_BUCKET: Optional[str] = Field(default=None)
    GCS_PROJECT_ID: Optional[str] = Field(default=None)
    GCS_CREDENTIALS_FILE: Optional[str] = Field(default=None)
    
    # Azure Blob Configuration
    AZURE_STORAGE_ACCOUNT: Optional[str] = Field(default=None)
    AZURE_STORAGE_KEY: Optional[str] = Field(default=None)
    AZURE_CONTAINER: Optional[str] = Field(default=None)
    
    # CDN
    CDN_ENABLED: bool = Field(default=False)
    CDN_BASE_URL: Optional[str] = Field(default=None)
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# ML Model Configuration
# ============================================================================

class MLSettings(BaseAppSettings):
    """Machine learning model and inference configuration."""
    
    ML_BACKEND: MLBackend = Field(default=MLBackend.PYTORCH)
    
    # Device Configuration
    USE_GPU: bool = Field(default=True)
    GPU_DEVICE_IDS: List[int] = Field(default=[0])
    GPU_MEMORY_FRACTION: float = Field(default=0.8, ge=0.1, le=1.0)
    MIXED_PRECISION: bool = Field(default=True)
    
    # Model Registry
    MODEL_REGISTRY_URL: Optional[str] = Field(default=None)
    MODEL_CACHE_DIR: str = Field(default="/data/model_cache")
    MODEL_AUTO_UPDATE: bool = Field(default=False)
    MODEL_VERSION_PINNING: bool = Field(default=True)
    
    # Face Detection
    FACE_DETECTION_MODEL: str = Field(default="retinaface_resnet50")
    FACE_DETECTION_CONFIDENCE: float = Field(default=0.7, ge=0.1, le=1.0)
    FACE_DETECTION_NMS_THRESHOLD: float = Field(default=0.4, ge=0.1, le=0.9)
    FACE_MIN_SIZE: int = Field(default=30, ge=10)
    FACE_MAX_FACES: int = Field(default=20, ge=1)
    FACE_ALIGNMENT: bool = Field(default=True)
    FACE_LANDMARKS: bool = Field(default=True)
    
    # Manipulation Detection
    MANIPULATION_MODEL: str = Field(default="efficientnet_b4_dfdc")
    MANIPULATION_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    MANIPULATION_ENSEMBLE: bool = Field(default=True)
    MANIPULATION_MODELS: List[str] = Field(
        default=[
            "efficientnet_b4_dfdc",
            "xception_faceforensics",
            "capsule_network_v2",
            "multi_attention_network",
            "frequency_aware_network",
        ]
    )
    
    # Audio Deepfake Detection
    AUDIO_MODEL: str = Field(default="rawnet3_asvspoof")
    AUDIO_SAMPLE_RATE: int = Field(default=16000)
    AUDIO_SEGMENT_LENGTH: float = Field(default=4.0)
    AUDIO_OVERLAP: float = Field(default=0.5)
    AUDIO_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    AUDIO_MODELS: List[str] = Field(
        default=[
            "rawnet3_asvspoof",
            "wav2vec2_deepfake",
            "aasist_antispoofing",
            "lcnn_audio_forensics",
        ]
    )
    
    # Video Analysis
    VIDEO_FRAME_SAMPLE_RATE: int = Field(default=5, ge=1, le=30)
    VIDEO_MAX_FRAMES: int = Field(default=300, ge=10, le=1000)
    VIDEO_TEMPORAL_MODEL: str = Field(default="slowfast_r50")
    VIDEO_TEMPORAL_WINDOW: int = Field(default=16)
    
    # GAN Detection
    GAN_DETECTION_MODEL: str = Field(default="spec_forensics_v3")
    GAN_DETECTION_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    GAN_FINGERPRINT_ENABLED: bool = Field(default=True)
    
    # Frequency Analysis
    FREQUENCY_ANALYSIS_ENABLED: bool = Field(default=True)
    DCT_BLOCK_SIZE: int = Field(default=8)
    FFT_ENABLED: bool = Field(default=True)
    WAVELET_ENABLED: bool = Field(default=True)
    SPECTRAL_BANDS: int = Field(default=64)
    
    # Compression Analysis
    COMPRESSION_ANALYSIS_ENABLED: bool = Field(default=True)
    JPEG_QUALITY_ESTIMATION: bool = Field(default=True)
    DOUBLE_COMPRESSION_DETECTION: bool = Field(default=True)
    
    # Noise Analysis
    NOISE_ANALYSIS_ENABLED: bool = Field(default=True)
    NOISE_ESTIMATION_METHOD: str = Field(default="wavelet")
    NOISE_INCONSISTENCY_THRESHOLD: float = Field(default=0.3)
    
    # Metadata Analysis
    METADATA_ANALYSIS_ENABLED: bool = Field(default=True)
    EXIF_ANALYSIS: bool = Field(default=True)
    XMP_ANALYSIS: bool = Field(default=True)
    
    # Lip Sync Detection
    LIP_SYNC_MODEL: str = Field(default="syncnet_v2")
    LIP_SYNC_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Ensemble Configuration
    ENSEMBLE_METHOD: str = Field(default="weighted_average")
    ENSEMBLE_WEIGHTS: Dict[str, float] = Field(
        default={
            "face_manipulation": 0.30,
            "frequency_analysis": 0.15,
            "gan_detection": 0.15,
            "temporal_consistency": 0.15,
            "noise_analysis": 0.10,
            "compression_analysis": 0.05,
            "metadata_analysis": 0.05,
            "lip_sync": 0.05,
        }
    )
    
    # Inference
    INFERENCE_BATCH_SIZE: int = Field(default=8, ge=1, le=64)
    INFERENCE_NUM_WORKERS: int = Field(default=4, ge=1, le=16)
    INFERENCE_TIMEOUT: int = Field(default=300, ge=30, le=1800)
    INFERENCE_RETRY_COUNT: int = Field(default=3, ge=0, le=5)
    
    # Explainability
    GRAD_CAM_ENABLED: bool = Field(default=True)
    ATTENTION_MAPS_ENABLED: bool = Field(default=True)
    FEATURE_ATTRIBUTION_ENABLED: bool = Field(default=True)
    LIME_ENABLED: bool = Field(default=False)
    SHAP_ENABLED: bool = Field(default=False)
    
    class Config(BaseAppSettings.Config):
        env_prefix = "ML_"


# ============================================================================
# Celery / Task Queue Configuration
# ============================================================================

class CelerySettings(BaseAppSettings):
    """Celery task queue configuration."""
    
    CELERY_WORKER_CONCURRENCY: int = Field(default=4, ge=1, le=32)
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=600)
    CELERY_TASK_HARD_TIME_LIMIT: int = Field(default=900)
    CELERY_TASK_MAX_RETRIES: int = Field(default=3)
    CELERY_TASK_RETRY_DELAY: int = Field(default=60)
    CELERY_RESULT_EXPIRES: int = Field(default=86400)
    CELERY_PREFETCH_MULTIPLIER: int = Field(default=1)
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = Field(default=100)
    CELERY_WORKER_MAX_MEMORY_PER_CHILD: int = Field(default=500000)  # KB
    
    # Priority Queues
    CELERY_QUEUES: Dict[str, Dict[str, Any]] = Field(
        default={
            "high_priority": {"routing_key": "high"},
            "default": {"routing_key": "default"},
            "low_priority": {"routing_key": "low"},
            "batch": {"routing_key": "batch"},
            "realtime": {"routing_key": "realtime"},
        }
    )
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Monitoring & Observability Configuration
# ============================================================================

class MonitoringSettings(BaseAppSettings):
    """Monitoring, metrics, and observability configuration."""
    
    # Prometheus
    PROMETHEUS_ENABLED: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(default=9090)
    
    # Grafana
    GRAFANA_ENABLED: bool = Field(default=True)
    GRAFANA_PORT: int = Field(default=3001)
    
    # OpenTelemetry
    OTEL_ENABLED: bool = Field(default=True)
    OTEL_EXPORTER_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_SERVICE_NAME: str = Field(default="deepfake-detector")
    OTEL_TRACE_SAMPLE_RATE: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Sentry
    SENTRY_DSN: Optional[str] = Field(default=None)
    SENTRY_ENVIRONMENT: Optional[str] = Field(default=None)
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1)
    
    # ELK Stack
    ELASTICSEARCH_URL: Optional[str] = Field(default=None)
    KIBANA_URL: Optional[str] = Field(default=None)
    LOGSTASH_URL: Optional[str] = Field(default=None)
    
    # Health Checks
    HEALTH_CHECK_INTERVAL: int = Field(default=30)
    HEALTH_CHECK_TIMEOUT: int = Field(default=10)
    HEALTH_CHECK_DB: bool = Field(default=True)
    HEALTH_CHECK_REDIS: bool = Field(default=True)
    HEALTH_CHECK_ML_MODELS: bool = Field(default=True)
    HEALTH_CHECK_STORAGE: bool = Field(default=True)
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Email / Notification Configuration
# ============================================================================

class NotificationSettings(BaseAppSettings):
    """Email and notification configuration."""
    
    # SMTP
    SMTP_HOST: Optional[str] = Field(default=None)
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    SMTP_TLS: bool = Field(default=True)
    SMTP_FROM_EMAIL: str = Field(default="noreply@deepfake-detector.com")
    SMTP_FROM_NAME: str = Field(default="DeepFake Detector")
    
    # Webhook
    WEBHOOK_ENABLED: bool = Field(default=True)
    WEBHOOK_TIMEOUT: int = Field(default=10)
    WEBHOOK_MAX_RETRIES: int = Field(default=3)
    
    # Slack
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None)
    SLACK_CHANNEL: Optional[str] = Field(default=None)
    
    # PagerDuty
    PAGERDUTY_API_KEY: Optional[str] = Field(default=None)
    PAGERDUTY_SERVICE_ID: Optional[str] = Field(default=None)
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Master Application Settings
# ============================================================================

class Settings(BaseAppSettings):
    """Master application settings aggregating all configuration modules."""
    
    # Application
    APP_NAME: str = Field(default="DeepFake Detector")
    APP_VERSION: str = Field(default="2.0.0")
    APP_DESCRIPTION: str = Field(
        default="Production-grade AI-powered DeepFake Detection Platform"
    )
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS: int = Field(default=4, ge=1, le=32)
    RELOAD: bool = Field(default=False)
    
    # API
    API_V1_PREFIX: str = Field(default="/api/v1")
    API_V2_PREFIX: str = Field(default="/api/v2")
    DOCS_URL: str = Field(default="/docs")
    REDOC_URL: str = Field(default="/redoc")
    OPENAPI_URL: str = Field(default="/openapi.json")
    
    # Feature Flags
    FEATURE_BATCH_PROCESSING: bool = Field(default=True)
    FEATURE_REALTIME_DETECTION: bool = Field(default=True)
    FEATURE_FORENSIC_REPORTS: bool = Field(default=True)
    FEATURE_API_ACCESS: bool = Field(default=True)
    FEATURE_WEBHOOK_NOTIFICATIONS: bool = Field(default=True)
    FEATURE_ADMIN_PANEL: bool = Field(default=True)
    FEATURE_MULTI_TENANT: bool = Field(default=False)
    FEATURE_AUDIT_LOG: bool = Field(default=True)
    FEATURE_EXPORT: bool = Field(default=True)
    FEATURE_COMPARISON_MODE: bool = Field(default=True)
    FEATURE_LIVE_STREAM: bool = Field(default=False)
    
    # Sub-configurations
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    
    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate critical settings for production environment."""
        if self.ENVIRONMENT == Environment.PRODUCTION:
            assert self.security.SECRET_KEY != "changeme", (
                "SECRET_KEY must be changed in production"
            )
            assert self.db.POSTGRES_PASSWORD != "changeme", (
                "Database password must be changed in production"
            )
            assert not self.DEBUG, "DEBUG must be False in production"
            assert self.security.SESSION_SECURE, (
                "Session cookies must be secure in production"
            )
        return self
    
    class Config(BaseAppSettings.Config):
        env_prefix = ""


# ============================================================================
# Settings Factory
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def get_test_settings() -> Settings:
    """Get settings configured for testing."""
    return Settings(
        ENVIRONMENT=Environment.TESTING,
        DEBUG=True,
        db=DatabaseSettings(
            POSTGRES_DB="deepfake_detector_test",
            DB_POOL_SIZE=5,
        ),
        redis=RedisSettings(
            REDIS_DB=15,
        ),
        ml=MLSettings(
            USE_GPU=False,
            INFERENCE_BATCH_SIZE=2,
        ),
    )
