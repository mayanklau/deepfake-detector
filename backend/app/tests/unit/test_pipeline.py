"""Unit tests for the ML detection pipeline."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from ml.pipeline import (
    FaceDetector, ManipulationDetector, FrequencyAnalyzer, GANDetector,
    NoiseAnalyzer, CompressionAnalyzer, MetadataAnalyzer, AudioDeepfakeDetector,
    LipSyncDetector, TemporalAnalyzer, EnsemblePredictor,
    ManipulationResult, DetectionCategory, FrequencyAnalysisResult,
    GANDetectionResult, NoiseAnalysisResult, CompressionAnalysisResult,
    MetadataAnalysisResult, DeepFakeDetectionPipeline,
    BoundingBox, FaceLandmarks, ImagePreprocessor, PipelineConfig,
)

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def sample_audio():
    return (np.random.randn(16000 * 4) * 0.5).astype(np.float32)

class TestBoundingBox:
    def test_area(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=50, confidence=0.9)
        assert bbox.area() == 5000

    def test_iou_same(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        assert bbox.iou(bbox) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = BoundingBox(x=0, y=0, width=50, height=50)
        b = BoundingBox(x=100, y=100, width=50, height=50)
        assert bbox_iou(a, b) == 0.0

def bbox_iou(a, b):
    return a.iou(b)

class TestFaceLandmarks:
    def test_inter_eye_distance(self):
        lm = FaceLandmarks(left_eye=(0, 0), right_eye=(60, 0))
        assert lm.inter_eye_distance() == pytest.approx(60.0)

@pytest.mark.asyncio
class TestFaceDetector:
    async def test_detect_faces(self, sample_image):
        det = FaceDetector()
        await det.load_model("cpu")
        results = await det.predict(sample_image)
        assert isinstance(results, list)
        for r in results:
            assert r.confidence > 0
            assert r.bbox.width > 0

@pytest.mark.asyncio
class TestManipulationDetector:
    async def test_predict(self, sample_image):
        det = ManipulationDetector("efficientnet_b4_dfdc")
        await det.load_model("cpu")
        result = await det.predict(sample_image)
        assert 0 <= result.score <= 1
        assert 0 <= result.confidence <= 1
        assert result.model_name == "EfficientNet-B4 DFDC"

@pytest.mark.asyncio
class TestFrequencyAnalyzer:
    async def test_analyze(self, sample_image):
        analyzer = FrequencyAnalyzer()
        await analyzer.load_model("cpu")
        result = await analyzer.predict(sample_image)
        assert 0 <= result.dct_score <= 1
        assert 0 <= result.fft_score <= 1
        assert 0 <= result.wavelet_score <= 1
        assert 0 <= result.overall_score <= 1

@pytest.mark.asyncio
class TestGANDetector:
    async def test_detect(self, sample_image):
        det = GANDetector()
        await det.load_model("cpu")
        result = await det.predict(sample_image)
        assert isinstance(result.is_gan_generated, bool)
        assert 0 <= result.gan_probability <= 1

@pytest.mark.asyncio
class TestNoiseAnalyzer:
    async def test_analyze(self, sample_image):
        analyzer = NoiseAnalyzer()
        await analyzer.load_model("cpu")
        result = await analyzer.predict(sample_image)
        assert 0 <= result.noise_consistency_score <= 1

@pytest.mark.asyncio
class TestCompressionAnalyzer:
    async def test_analyze(self, sample_image):
        analyzer = CompressionAnalyzer()
        await analyzer.load_model("cpu")
        result = await analyzer.predict(sample_image)
        assert 1 <= result.estimated_quality <= 100
        assert isinstance(result.double_compression_detected, bool)

@pytest.mark.asyncio
class TestAudioDetector:
    async def test_detect(self, sample_audio):
        det = AudioDeepfakeDetector()
        await det.load_model("cpu")
        result = await det.predict(sample_audio)
        assert isinstance(result.is_deepfake, bool)
        assert 0 <= result.deepfake_probability <= 1

class TestEnsemblePredictor:
    def test_predict_authentic(self):
        ensemble = EnsemblePredictor()
        result = ensemble.predict(
            manipulation_results=[ManipulationResult(score=0.1, category=DetectionCategory.UNKNOWN, confidence=0.8, model_name="test", model_version="1.0")],
            frequency_result=FrequencyAnalysisResult(dct_score=0.1, fft_score=0.1, overall_score=0.1),
            gan_result=GANDetectionResult(is_gan_generated=False, gan_probability=0.05),
            noise_result=NoiseAnalysisResult(noise_consistency_score=0.95),
            compression_result=CompressionAnalysisResult(compression_artifacts_score=0.05),
        )
        assert result.verdict.value in ("authentic", "likely_authentic")
        assert result.overall_score < 0.35

    def test_predict_fake(self):
        ensemble = EnsemblePredictor()
        result = ensemble.predict(
            manipulation_results=[ManipulationResult(score=0.9, category=DetectionCategory.FACE_SWAP, confidence=0.95, model_name="test", model_version="1.0")],
            frequency_result=FrequencyAnalysisResult(dct_score=0.8, fft_score=0.7, overall_score=0.75),
            gan_result=GANDetectionResult(is_gan_generated=True, gan_probability=0.85),
        )
        assert result.verdict.value in ("likely_fake", "fake")
        assert result.overall_score > 0.5

    def test_empty_scores(self):
        ensemble = EnsemblePredictor()
        result = ensemble.predict()
        assert result.verdict == "uncertain"
        assert result.confidence == 0.0

@pytest.mark.asyncio
class TestFullPipeline:
    async def test_image_analysis(self, sample_image):
        pipeline = DeepFakeDetectionPipeline(PipelineConfig(device="cpu"))
        await pipeline.initialize()
        result = await pipeline.analyze_image(sample_image)
        assert "verdict" in result
        assert "overall_score" in result
        assert "confidence" in result
        assert result["metrics"]["total_time_ms"] > 0

    async def test_audio_analysis(self, sample_audio):
        pipeline = DeepFakeDetectionPipeline(PipelineConfig(device="cpu"))
        await pipeline.initialize()
        result = await pipeline.analyze_audio(sample_audio)
        assert "is_deepfake" in result
        assert "deepfake_probability" in result
