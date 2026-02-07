"""Pydantic schemas for analysis requests and responses."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class AnalysisType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    ANALYZING = "analyzing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VerdictLevel(str, Enum):
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    FAKE = "fake"

class AnalysisConfigSchema(BaseModel):
    face_detection: bool = True
    manipulation_detection: bool = True
    frequency_analysis: bool = True
    gan_detection: bool = True
    noise_analysis: bool = True
    compression_analysis: bool = True
    metadata_analysis: bool = True
    temporal_analysis: bool = True
    audio_analysis: bool = True
    lip_sync_detection: bool = True
    ensemble_method: str = "weighted_average"
    priority: int = Field(default=5, ge=1, le=10)

class AnalysisCreateRequest(BaseModel):
    media_file_id: str
    config: Optional[AnalysisConfigSchema] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class AnalysisResponseSchema(BaseModel):
    id: str
    user_id: str
    analysis_type: AnalysisType
    status: AnalysisStatus
    verdict: Optional[VerdictLevel] = None
    confidence_score: Optional[float] = None
    manipulation_probability: Optional[float] = None
    processing_time_ms: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    explanation_text: Optional[str] = None
    component_scores: Optional[Dict[str, float]] = None
    class Config:
        from_attributes = True

class AnalysisListResponseSchema(BaseModel):
    items: List[AnalysisResponseSchema]
    total: int
    page: int
    page_size: int
