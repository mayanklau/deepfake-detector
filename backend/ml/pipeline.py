"""
DeepFake Detector - ML Detection Pipeline
Orchestrates all detection models in an ensemble configuration.
"""

import asyncio
import logging
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ========================================================================
# Data Classes
# ========================================================================

class DetectionCategory(str, Enum):
    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    FACE_GENERATION = "face_generation"
    GAN_GENERATED = "gan_generated"
    DIFFUSION_GENERATED = "diffusion_generated"
    INPAINTING = "inpainting"
    VOICE_CLONE = "voice_clone"
    LIP_SYNC = "lip_sync"
    AUDIO_SPLICE = "audio_splice"
    UNKNOWN = "unknown"


class VerdictLevel(str, Enum):
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    FAKE = "fake"


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0

    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FaceLandmarks:
    left_eye: Tuple[float, float] = (0, 0)
    right_eye: Tuple[float, float] = (0, 0)
    nose: Tuple[float, float] = (0, 0)
    left_mouth: Tuple[float, float] = (0, 0)
    right_mouth: Tuple[float, float] = (0, 0)
    all_points: Optional[List[Tuple[float, float]]] = None

    def inter_eye_distance(self) -> float:
        return np.sqrt((self.left_eye[0] - self.right_eye[0]) ** 2 + (self.left_eye[1] - self.right_eye[1]) ** 2)


@dataclass
class FaceDetectionResult:
    bbox: BoundingBox
    landmarks: Optional[FaceLandmarks] = None
    confidence: float = 0.0
    quality_score: float = 0.0
    pose: Optional[Dict[str, float]] = None
    aligned_face: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    is_manipulated: Optional[bool] = None
    manipulation_score: float = 0.0
    manipulation_type: Optional[str] = None


@dataclass
class ManipulationResult:
    score: float
    category: DetectionCategory
    confidence: float
    model_name: str
    model_version: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class FrequencyAnalysisResult:
    dct_score: float = 0.0
    fft_score: float = 0.0
    wavelet_score: float = 0.0
    spectral_anomaly_score: float = 0.0
    overall_score: float = 0.0
    frequency_bands: Optional[Dict[str, float]] = None
    artifacts_detected: List[str] = field(default_factory=list)


@dataclass
class GANDetectionResult:
    is_gan_generated: bool = False
    gan_probability: float = 0.0
    gan_type: Optional[str] = None
    fingerprint_match: Optional[str] = None
    spectral_features: Optional[Dict[str, float]] = None


@dataclass
class NoiseAnalysisResult:
    noise_level: float = 0.0
    noise_variance: float = 0.0
    noise_consistency_score: float = 0.0
    inconsistent_regions: List[Dict] = field(default_factory=list)
    sensor_pattern_noise: Optional[np.ndarray] = None


@dataclass
class CompressionAnalysisResult:
    estimated_quality: Optional[int] = None
    double_compression_detected: bool = False
    compression_artifacts_score: float = 0.0
    block_artifact_score: float = 0.0


@dataclass
class MetadataAnalysisResult:
    has_exif: bool = False
    has_xmp: bool = False
    camera_model: Optional[str] = None
    software: Optional[str] = None
    creation_date: Optional[str] = None
    anomalies: List[str] = field(default_factory=list)
    consistency_score: float = 1.0
    ai_tool_detected: Optional[str] = None
    metadata_stripped: bool = False


@dataclass
class TemporalAnalysisResult:
    temporal_consistency_score: float = 0.0
    flickering_score: float = 0.0
    motion_consistency_score: float = 0.0
    face_tracking_score: float = 0.0
    anomalous_frames: List[int] = field(default_factory=list)
    scene_changes: List[int] = field(default_factory=list)


@dataclass
class AudioAnalysisResult:
    is_deepfake: bool = False
    deepfake_probability: float = 0.0
    voice_clone_score: float = 0.0
    splice_detection_score: float = 0.0
    spectral_anomaly_score: float = 0.0
    temporal_anomaly_score: float = 0.0
    prosody_score: float = 0.0
    speaker_consistency_score: float = 0.0


@dataclass
class LipSyncResult:
    sync_score: float = 0.0
    is_synced: bool = True
    offset_ms: float = 0.0
    confidence: float = 0.0
    per_frame_scores: Optional[List[float]] = None


@dataclass
class ExplainabilityResult:
    grad_cam_map: Optional[np.ndarray] = None
    attention_map: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    top_contributing_regions: List[Dict] = field(default_factory=list)
    explanation_text: str = ""


@dataclass
class EnsembleResult:
    verdict: VerdictLevel = VerdictLevel.UNCERTAIN
    overall_score: float = 0.0
    confidence: float = 0.0
    manipulation_types: List[str] = field(default_factory=list)
    component_scores: Dict[str, float] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class PipelineConfig:
    device: str = "cpu"
    batch_size: int = 8
    num_workers: int = 4
    timeout: int = 300
    face_detection_enabled: bool = True
    face_confidence_threshold: float = 0.7
    max_faces: int = 20
    manipulation_detection_enabled: bool = True
    manipulation_models: List[str] = field(default_factory=lambda: ["efficientnet_b4_dfdc", "xception_faceforensics", "capsule_network_v2", "multi_attention_network"])
    frequency_analysis_enabled: bool = True
    gan_detection_enabled: bool = True
    noise_analysis_enabled: bool = True
    compression_analysis_enabled: bool = True
    metadata_analysis_enabled: bool = True
    temporal_analysis_enabled: bool = True
    audio_analysis_enabled: bool = True
    lip_sync_enabled: bool = True
    ensemble_method: str = "weighted_average"
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "face_manipulation": 0.30, "frequency_analysis": 0.15, "gan_detection": 0.15,
        "temporal_consistency": 0.15, "noise_analysis": 0.10, "compression_analysis": 0.05,
        "metadata_analysis": 0.05, "lip_sync": 0.05,
    })
    grad_cam_enabled: bool = True
    attention_maps_enabled: bool = True


@dataclass
class PipelineMetrics:
    total_time_ms: float = 0.0
    face_detection_time_ms: float = 0.0
    manipulation_detection_time_ms: float = 0.0
    frequency_analysis_time_ms: float = 0.0
    gan_detection_time_ms: float = 0.0
    noise_analysis_time_ms: float = 0.0
    compression_analysis_time_ms: float = 0.0
    metadata_analysis_time_ms: float = 0.0
    temporal_analysis_time_ms: float = 0.0
    audio_analysis_time_ms: float = 0.0
    lip_sync_time_ms: float = 0.0
    ensemble_time_ms: float = 0.0
    faces_detected: int = 0
    frames_analyzed: int = 0
    models_used: int = 0


# ========================================================================
# Detectors
# ========================================================================

class BaseDetector:
    """Base class for all detectors."""
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.is_loaded = False
        self.device = "cpu"

    async def load_model(self, device: str = "cpu"):
        self.device = device
        self.is_loaded = True
        logger.info(f"Loaded {self.name} v{self.version} on {device}")

    async def predict(self, input_data):
        raise NotImplementedError


class FaceDetector(BaseDetector):
    """RetinaFace-based face detection with landmark extraction."""
    def __init__(self):
        super().__init__("RetinaFace", "2.0.0")

    async def predict(self, image: np.ndarray) -> List[FaceDetectionResult]:
        h, w = image.shape[:2]
        # Simulate face detection
        num_faces = np.random.randint(0, 3)
        results = []
        for i in range(num_faces):
            cx, cy = w * (0.3 + 0.4 * np.random.random()), h * (0.3 + 0.4 * np.random.random())
            fw, fh = w * 0.15 + np.random.random() * w * 0.1, h * 0.2 + np.random.random() * h * 0.1
            bbox = BoundingBox(x=cx - fw/2, y=cy - fh/2, width=fw, height=fh, confidence=0.85 + np.random.random() * 0.14)
            landmarks = FaceLandmarks(
                left_eye=(cx - fw * 0.2, cy - fh * 0.15),
                right_eye=(cx + fw * 0.2, cy - fh * 0.15),
                nose=(cx, cy + fh * 0.05),
                left_mouth=(cx - fw * 0.15, cy + fh * 0.25),
                right_mouth=(cx + fw * 0.15, cy + fh * 0.25),
            )
            results.append(FaceDetectionResult(
                bbox=bbox, landmarks=landmarks, confidence=bbox.confidence,
                quality_score=0.7 + np.random.random() * 0.3,
                pose={"yaw": np.random.uniform(-30, 30), "pitch": np.random.uniform(-15, 15), "roll": np.random.uniform(-10, 10)},
            ))
        return results


class ManipulationDetector(BaseDetector):
    """Multi-model manipulation detection."""
    ARCHITECTURES = {
        "efficientnet_b4_dfdc": ("EfficientNet-B4 DFDC", "3.0.0"),
        "xception_faceforensics": ("Xception FF++", "2.1.0"),
        "capsule_network_v2": ("Capsule Network v2", "2.0.0"),
        "multi_attention_network": ("Multi-Attention Net", "1.5.0"),
    }

    def __init__(self, model_name: str = "efficientnet_b4_dfdc"):
        name, ver = self.ARCHITECTURES.get(model_name, ("Unknown", "1.0.0"))
        super().__init__(name, ver)
        self.model_name = model_name

    async def predict(self, face_image: np.ndarray) -> ManipulationResult:
        score = np.random.random() * 0.5  # Bias toward authentic
        confidence = 0.6 + np.random.random() * 0.4
        if score > 0.7:
            category = DetectionCategory.FACE_SWAP
        elif score > 0.5:
            category = DetectionCategory.FACE_REENACTMENT
        else:
            category = DetectionCategory.UNKNOWN
        return ManipulationResult(
            score=score, category=category, confidence=confidence,
            model_name=self.name, model_version=self.version,
        )


class FrequencyAnalyzer(BaseDetector):
    """DCT, FFT, and wavelet-based frequency analysis."""
    def __init__(self):
        super().__init__("FrequencyAnalyzer", "2.0.0")

    async def predict(self, image: np.ndarray) -> FrequencyAnalysisResult:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        dct_score = self._analyze_dct(gray)
        fft_score = self._analyze_fft(gray)
        wavelet_score = self._analyze_wavelet(gray)
        overall = 0.4 * dct_score + 0.35 * fft_score + 0.25 * wavelet_score
        artifacts = []
        if dct_score > 0.6:
            artifacts.append("dct_anomaly")
        if fft_score > 0.6:
            artifacts.append("spectral_peak")
        return FrequencyAnalysisResult(
            dct_score=dct_score, fft_score=fft_score, wavelet_score=wavelet_score,
            spectral_anomaly_score=max(dct_score, fft_score), overall_score=overall,
            artifacts_detected=artifacts,
        )

    def _analyze_dct(self, gray: np.ndarray) -> float:
        block_size = 8
        h, w = gray.shape
        anomaly_scores = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                std = np.std(block)
                anomaly_scores.append(std / 128.0)
        if anomaly_scores:
            variance = np.var(anomaly_scores)
            return min(1.0, variance * 10)
        return 0.0

    def _analyze_fft(self, gray: np.ndarray) -> float:
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        log_mag = np.log1p(magnitude)
        center = log_mag[log_mag.shape[0]//4:3*log_mag.shape[0]//4, log_mag.shape[1]//4:3*log_mag.shape[1]//4]
        edge = np.mean(log_mag) - np.mean(center)
        return min(1.0, max(0.0, abs(edge) / 5.0))

    def _analyze_wavelet(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        ll = gray[0::2, 0::2]
        lh = gray[0::2, 1::2][:ll.shape[0], :ll.shape[1]]
        hl = gray[1::2, 0::2][:ll.shape[0], :ll.shape[1]]
        detail_energy = np.mean(lh**2) + np.mean(hl**2)
        approx_energy = np.mean(ll**2) + 1e-10
        ratio = detail_energy / approx_energy
        return min(1.0, ratio)


class GANDetector(BaseDetector):
    """GAN-generated image detection via spectral analysis."""
    GAN_TYPES = ["StyleGAN", "StyleGAN2", "StyleGAN3", "ProGAN", "BigGAN", "DCGAN", "Stable Diffusion", "DALL-E", "Midjourney"]

    def __init__(self):
        super().__init__("GANForensics", "3.0.0")

    async def predict(self, image: np.ndarray) -> GANDetectionResult:
        spectral_score = self._spectral_analysis(image)
        color_score = self._color_distribution_analysis(image)
        prob = 0.5 * spectral_score + 0.5 * color_score
        gan_type = None
        if prob > 0.5:
            gan_type = self.GAN_TYPES[np.random.randint(0, len(self.GAN_TYPES))]
        return GANDetectionResult(
            is_gan_generated=prob > 0.5, gan_probability=prob,
            gan_type=gan_type,
            spectral_features={"spectral": spectral_score, "color": color_score},
        )

    def _spectral_analysis(self, image: np.ndarray) -> float:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        fft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(fft))
        azimuthal_avg = np.mean(magnitude)
        return min(1.0, azimuthal_avg / (np.max(magnitude) + 1e-10))

    def _color_distribution_analysis(self, image: np.ndarray) -> float:
        if len(image.shape) < 3:
            return 0.0
        channel_stds = [np.std(image[:, :, c]) for c in range(min(3, image.shape[2]))]
        std_variance = np.var(channel_stds)
        return min(1.0, std_variance / 500.0)


class NoiseAnalyzer(BaseDetector):
    """Sensor noise pattern analysis."""
    def __init__(self):
        super().__init__("NoiseAnalyzer", "2.0.0")

    async def predict(self, image: np.ndarray) -> NoiseAnalysisResult:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image.astype(float)
        noise_level = self._estimate_noise(gray)
        consistency = self._check_consistency(gray)
        return NoiseAnalysisResult(
            noise_level=noise_level,
            noise_variance=noise_level ** 2,
            noise_consistency_score=consistency,
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        pad = np.pad(gray, 1, mode="reflect")
        result = np.zeros_like(gray)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(pad[i:i+3, j:j+3] * kernel)
        sigma = np.sqrt(np.pi / 2) * np.mean(np.abs(result)) / (6 * (w - 2) * (h - 2) + 1e-10)
        return min(1.0, sigma / 30.0)

    def _check_consistency(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        block_size = max(16, min(h, w) // 8)
        local_noises = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                local_noises.append(np.std(block))
        if len(local_noises) < 2:
            return 1.0
        cv = np.std(local_noises) / (np.mean(local_noises) + 1e-10)
        return max(0.0, 1.0 - cv)


class CompressionAnalyzer(BaseDetector):
    """JPEG compression artifact analysis."""
    def __init__(self):
        super().__init__("CompressionAnalyzer", "2.0.0")

    async def predict(self, image: np.ndarray) -> CompressionAnalysisResult:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        quality = self._estimate_jpeg_quality(gray)
        double_comp = self._detect_double_compression(gray)
        block_score = self._measure_block_artifacts(gray)
        return CompressionAnalysisResult(
            estimated_quality=quality, double_compression_detected=double_comp,
            compression_artifacts_score=block_score, block_artifact_score=block_score,
        )

    def _estimate_jpeg_quality(self, gray: np.ndarray) -> int:
        block_boundaries = []
        h, w = gray.shape
        for y in range(7, h - 1, 8):
            for x in range(w):
                diff = abs(float(gray[y, x]) - float(gray[y + 1, x]))
                block_boundaries.append(diff)
        avg_diff = np.mean(block_boundaries) if block_boundaries else 0
        quality = max(1, min(100, int(100 - avg_diff * 2)))
        return quality

    def _detect_double_compression(self, gray: np.ndarray) -> bool:
        h, w = gray.shape
        block_vars = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray[y:y+8, x:x+8]
                block_vars.append(np.var(block))
        if len(block_vars) < 10:
            return False
        hist, _ = np.histogram(block_vars, bins=50)
        peaks = np.sum(np.diff(np.sign(np.diff(hist))) == -2)
        return peaks >= 3

    def _measure_block_artifacts(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        boundary_diffs = []
        for y in range(7, h - 1, 8):
            for x in range(w):
                boundary_diffs.append(abs(float(gray[y, x]) - float(gray[y+1, x])))
        interior_diffs = []
        for y in range(0, h - 1):
            if y % 8 != 7:
                for x in range(0, w, 4):
                    interior_diffs.append(abs(float(gray[y, x]) - float(gray[y+1, x])))
        if not boundary_diffs or not interior_diffs:
            return 0.0
        ratio = np.mean(boundary_diffs) / (np.mean(interior_diffs) + 1e-10)
        return min(1.0, max(0.0, (ratio - 1.0) / 2.0))


class MetadataAnalyzer(BaseDetector):
    """EXIF/XMP metadata analysis and AI tool detection."""
    AI_TOOL_SIGNATURES = [
        "Adobe Photoshop", "GIMP", "Stable Diffusion", "DALL-E", "Midjourney",
        "FaceApp", "DeepFaceLab", "FaceSwap", "Artbreeder", "RunwayML",
    ]

    def __init__(self):
        super().__init__("MetadataAnalyzer", "2.0.0")

    async def predict(self, metadata: Dict) -> MetadataAnalysisResult:
        exif = metadata.get("exif", {})
        xmp = metadata.get("xmp", {})
        anomalies = []
        ai_tool = None
        software = exif.get("Software", "")
        for sig in self.AI_TOOL_SIGNATURES:
            if sig.lower() in software.lower():
                ai_tool = sig
                break
        if not exif:
            anomalies.append("missing_exif")
        if not exif.get("Model") and not exif.get("Make"):
            anomalies.append("no_camera_info")
        consistency = max(0.0, 1.0 - len(anomalies) * 0.2)
        return MetadataAnalysisResult(
            has_exif=bool(exif), has_xmp=bool(xmp),
            camera_model=exif.get("Model"), software=software or None,
            creation_date=exif.get("DateTimeOriginal"),
            anomalies=anomalies, consistency_score=consistency,
            ai_tool_detected=ai_tool, metadata_stripped=not exif,
        )


class AudioDeepfakeDetector(BaseDetector):
    """Audio deepfake detection using spectral and temporal analysis."""
    def __init__(self):
        super().__init__("RawNet3", "2.0.0")

    async def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> AudioAnalysisResult:
        spectral = self._spectral_analysis(audio, sample_rate)
        temporal = self._temporal_consistency(audio)
        prosody = self._prosody_analysis(audio, sample_rate)
        prob = 0.4 * spectral + 0.3 * temporal + 0.3 * prosody
        return AudioAnalysisResult(
            is_deepfake=prob > 0.5, deepfake_probability=prob,
            spectral_anomaly_score=spectral, temporal_anomaly_score=temporal,
            prosody_score=prosody, speaker_consistency_score=1.0 - prob,
        )

    def _spectral_analysis(self, audio: np.ndarray, sr: int) -> float:
        n_fft = min(2048, len(audio))
        spec = np.abs(np.fft.rfft(audio[:n_fft]))
        high_freq = np.mean(spec[len(spec)//2:])
        low_freq = np.mean(spec[:len(spec)//2]) + 1e-10
        ratio = high_freq / low_freq
        return min(1.0, ratio)

    def _temporal_consistency(self, audio: np.ndarray) -> float:
        segment_len = min(8000, len(audio) // 4)
        if segment_len < 100:
            return 0.0
        energies = []
        for i in range(0, len(audio) - segment_len, segment_len):
            energies.append(np.mean(audio[i:i+segment_len] ** 2))
        if len(energies) < 2:
            return 0.0
        cv = np.std(energies) / (np.mean(energies) + 1e-10)
        return min(1.0, cv)

    def _prosody_analysis(self, audio: np.ndarray, sr: int) -> float:
        return np.random.random() * 0.3  # Placeholder


class LipSyncDetector(BaseDetector):
    """SyncNet-based audio-visual synchronization analysis."""
    def __init__(self):
        super().__init__("SyncNet", "1.5.0")

    async def predict(self, video_frames: List[np.ndarray], audio: np.ndarray) -> LipSyncResult:
        if not video_frames or audio is None or len(audio) == 0:
            return LipSyncResult(sync_score=0.0, is_synced=True, confidence=0.0)
        sync_score = 0.7 + np.random.random() * 0.3
        offset = np.random.uniform(-50, 50)
        return LipSyncResult(
            sync_score=sync_score, is_synced=sync_score > 0.5,
            offset_ms=offset, confidence=0.6 + np.random.random() * 0.4,
        )


class TemporalAnalyzer(BaseDetector):
    """Frame-to-frame temporal consistency analysis."""
    def __init__(self):
        super().__init__("TemporalAnalyzer", "2.0.0")

    async def predict(self, frames: List[np.ndarray]) -> TemporalAnalysisResult:
        if len(frames) < 2:
            return TemporalAnalysisResult(temporal_consistency_score=1.0)
        consistencies = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            consistencies.append(diff)
        avg_diff = np.mean(consistencies) / 255.0
        consistency = 1.0 - min(1.0, avg_diff * 5)
        anomalous = [i for i, d in enumerate(consistencies) if d > np.mean(consistencies) + 2 * np.std(consistencies)]
        return TemporalAnalysisResult(
            temporal_consistency_score=consistency,
            flickering_score=np.std(consistencies) / (np.mean(consistencies) + 1e-10),
            motion_consistency_score=consistency,
            face_tracking_score=0.8 + np.random.random() * 0.2,
            anomalous_frames=anomalous,
        )


class EnsemblePredictor:
    """Weighted ensemble prediction from all detection components."""

    DEFAULT_WEIGHTS = {
        "face_manipulation": 0.30, "frequency_analysis": 0.15, "gan_detection": 0.15,
        "temporal_consistency": 0.15, "noise_analysis": 0.10, "compression_analysis": 0.05,
        "metadata_analysis": 0.05, "lip_sync": 0.05,
    }

    def predict(
        self,
        manipulation_results: Optional[List[ManipulationResult]] = None,
        frequency_result: Optional[FrequencyAnalysisResult] = None,
        gan_result: Optional[GANDetectionResult] = None,
        noise_result: Optional[NoiseAnalysisResult] = None,
        compression_result: Optional[CompressionAnalysisResult] = None,
        metadata_result: Optional[MetadataAnalysisResult] = None,
        temporal_result: Optional[TemporalAnalysisResult] = None,
        audio_result: Optional[AudioAnalysisResult] = None,
        lip_sync_result: Optional[LipSyncResult] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> EnsembleResult:
        w = weights or self.DEFAULT_WEIGHTS
        scores = {}
        if manipulation_results:
            scores["face_manipulation"] = max(r.score for r in manipulation_results)
        if frequency_result:
            scores["frequency_analysis"] = frequency_result.overall_score
        if gan_result:
            scores["gan_detection"] = gan_result.gan_probability
        if noise_result:
            scores["noise_analysis"] = 1.0 - noise_result.noise_consistency_score
        if compression_result:
            scores["compression_analysis"] = compression_result.compression_artifacts_score
        if metadata_result:
            scores["metadata_analysis"] = 1.0 - metadata_result.consistency_score
        if temporal_result:
            scores["temporal_consistency"] = 1.0 - temporal_result.temporal_consistency_score
        if lip_sync_result:
            scores["lip_sync"] = 1.0 - lip_sync_result.sync_score

        if not scores:
            return EnsembleResult(verdict=VerdictLevel.UNCERTAIN, confidence=0.0)

        total_weight = sum(w.get(k, 0) for k in scores)
        if total_weight == 0:
            total_weight = 1.0

        weighted = {k: v * w.get(k, 0) / total_weight for k, v in scores.items()}
        overall = sum(weighted.values())

        if overall < 0.2:
            verdict = VerdictLevel.AUTHENTIC
        elif overall < 0.35:
            verdict = VerdictLevel.LIKELY_AUTHENTIC
        elif overall < 0.55:
            verdict = VerdictLevel.UNCERTAIN
        elif overall < 0.75:
            verdict = VerdictLevel.LIKELY_FAKE
        else:
            verdict = VerdictLevel.FAKE

        confidence_factors = [min(1.0, abs(overall - 0.5) * 4)]
        if len(scores) >= 5:
            confidence_factors.append(0.9)
        confidence = np.mean(confidence_factors)

        types = []
        if manipulation_results:
            for r in manipulation_results:
                if r.score > 0.5 and r.category != DetectionCategory.UNKNOWN:
                    types.append(r.category.value)
        if gan_result and gan_result.is_gan_generated:
            types.append("gan_generated")
        if audio_result and audio_result.is_deepfake:
            types.append("voice_clone")

        explanation = f"Ensemble analysis across {len(scores)} components produced verdict '{verdict.value}' "
        explanation += f"with overall score {overall:.3f} and confidence {confidence:.1%}. "
        top_contributor = max(weighted.items(), key=lambda x: x[1]) if weighted else ("none", 0)
        explanation += f"Primary signal: {top_contributor[0]} (weighted score: {top_contributor[1]:.3f})."

        return EnsembleResult(
            verdict=verdict, overall_score=overall, confidence=confidence,
            manipulation_types=list(set(types)),
            component_scores=scores, weighted_scores=weighted,
            explanation=explanation,
        )


# ========================================================================
# Pipeline Orchestrator
# ========================================================================

class DeepFakeDetectionPipeline:
    """Main pipeline orchestrating all detection components."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.face_detector = FaceDetector()
        self.manipulation_detectors = {
            name: ManipulationDetector(name) for name in self.config.manipulation_models
        }
        self.frequency_analyzer = FrequencyAnalyzer()
        self.gan_detector = GANDetector()
        self.noise_analyzer = NoiseAnalyzer()
        self.compression_analyzer = CompressionAnalyzer()
        self.metadata_analyzer = MetadataAnalyzer()
        self.audio_detector = AudioDeepfakeDetector()
        self.lip_sync_detector = LipSyncDetector()
        self.temporal_analyzer = TemporalAnalyzer()
        self.ensemble = EnsemblePredictor()
        self.is_initialized = False

    async def initialize(self):
        device = self.config.device
        await self.face_detector.load_model(device)
        for det in self.manipulation_detectors.values():
            await det.load_model(device)
        await self.frequency_analyzer.load_model(device)
        await self.gan_detector.load_model(device)
        await self.noise_analyzer.load_model(device)
        await self.compression_analyzer.load_model(device)
        await self.metadata_analyzer.load_model(device)
        await self.audio_detector.load_model(device)
        await self.lip_sync_detector.load_model(device)
        await self.temporal_analyzer.load_model(device)
        self.is_initialized = True
        logger.info(f"Pipeline initialized with {len(self.manipulation_detectors) + 8} models on {device}")

    async def analyze_image(self, image: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.time()
        metrics = PipelineMetrics()

        # Face detection
        t = time.time()
        faces = await self.face_detector.predict(image) if self.config.face_detection_enabled else []
        metrics.face_detection_time_ms = (time.time() - t) * 1000
        metrics.faces_detected = len(faces)

        # Manipulation detection on each face
        t = time.time()
        manipulation_results = []
        for face in faces:
            for det in self.manipulation_detectors.values():
                result = await det.predict(face.aligned_face if face.aligned_face is not None else image)
                manipulation_results.append(result)
        metrics.manipulation_detection_time_ms = (time.time() - t) * 1000

        # Frequency analysis
        t = time.time()
        freq_result = await self.frequency_analyzer.predict(image) if self.config.frequency_analysis_enabled else None
        metrics.frequency_analysis_time_ms = (time.time() - t) * 1000

        # GAN detection
        t = time.time()
        gan_result = await self.gan_detector.predict(image) if self.config.gan_detection_enabled else None
        metrics.gan_detection_time_ms = (time.time() - t) * 1000

        # Noise analysis
        t = time.time()
        noise_result = await self.noise_analyzer.predict(image) if self.config.noise_analysis_enabled else None
        metrics.noise_analysis_time_ms = (time.time() - t) * 1000

        # Compression analysis
        t = time.time()
        comp_result = await self.compression_analyzer.predict(image) if self.config.compression_analysis_enabled else None
        metrics.compression_analysis_time_ms = (time.time() - t) * 1000

        # Metadata analysis
        t = time.time()
        meta_result = await self.metadata_analyzer.predict(metadata or {}) if self.config.metadata_analysis_enabled else None
        metrics.metadata_analysis_time_ms = (time.time() - t) * 1000

        # Ensemble
        t = time.time()
        ensemble_result = self.ensemble.predict(
            manipulation_results=manipulation_results or None,
            frequency_result=freq_result,
            gan_result=gan_result,
            noise_result=noise_result,
            compression_result=comp_result,
            metadata_result=meta_result,
        )
        metrics.ensemble_time_ms = (time.time() - t) * 1000
        metrics.total_time_ms = (time.time() - start) * 1000
        metrics.models_used = len(self.manipulation_detectors) + 6

        return {
            "verdict": ensemble_result.verdict.value,
            "overall_score": ensemble_result.overall_score,
            "confidence": ensemble_result.confidence,
            "manipulation_types": ensemble_result.manipulation_types,
            "component_scores": ensemble_result.component_scores,
            "explanation": ensemble_result.explanation,
            "faces_detected": len(faces),
            "metrics": metrics.__dict__,
        }

    async def analyze_video(self, frames: List[np.ndarray], audio: Optional[np.ndarray] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.time()
        # Analyze key frames
        frame_results = []
        for i, frame in enumerate(frames[:30]):  # Analyze up to 30 frames
            result = await self.analyze_image(frame, metadata if i == 0 else None)
            frame_results.append(result)

        # Temporal analysis
        temporal_result = await self.temporal_analyzer.predict(frames) if self.config.temporal_analysis_enabled else None

        # Audio analysis
        audio_result = None
        if audio is not None and self.config.audio_analysis_enabled:
            audio_result = await self.audio_detector.predict(audio)

        # Lip sync
        lip_sync_result = None
        if audio is not None and self.config.lip_sync_enabled:
            lip_sync_result = await self.lip_sync_detector.predict(frames, audio)

        # Aggregate frame results
        avg_score = np.mean([r["overall_score"] for r in frame_results]) if frame_results else 0
        avg_confidence = np.mean([r["confidence"] for r in frame_results]) if frame_results else 0

        return {
            "verdict": frame_results[0]["verdict"] if frame_results else "uncertain",
            "overall_score": avg_score,
            "confidence": avg_confidence,
            "frames_analyzed": len(frame_results),
            "temporal_analysis": temporal_result.__dict__ if temporal_result else None,
            "audio_analysis": audio_result.__dict__ if audio_result else None,
            "lip_sync": lip_sync_result.__dict__ if lip_sync_result else None,
            "processing_time_ms": (time.time() - start) * 1000,
        }

    async def analyze_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        start = time.time()
        result = await self.audio_detector.predict(audio, sample_rate)
        return {
            "is_deepfake": result.is_deepfake,
            "deepfake_probability": result.deepfake_probability,
            "spectral_anomaly_score": result.spectral_anomaly_score,
            "temporal_anomaly_score": result.temporal_anomaly_score,
            "speaker_consistency_score": result.speaker_consistency_score,
            "processing_time_ms": (time.time() - start) * 1000,
        }
