"""Unit tests for CHIMERA evaluation models.

Tests cover:
- EvaluationResult creation and management
- Track-specific metrics (Calibration, ErrorDetection, etc.)
- TrackResult handling
- Metric computations and interpretations
- Serialization/deserialization
"""

import pytest
from pydantic import ValidationError

from chimera.models.evaluation import (
    CalibrationMetrics,
    ConfidenceBin,
    ErrorDetectionMetrics,
    EvaluationResult,
    KnowledgeBoundaryMetrics,
    SelfCorrectionMetrics,
    TrackResult,
)
from chimera.models.task import TrackType


class TestConfidenceBin:
    """Tests for ConfidenceBin model."""

    def test_basic_bin(self) -> None:
        """Test creating a basic confidence bin."""
        bin_ = ConfidenceBin(
            bin_lower=0.0,
            bin_upper=0.1,
            bin_center=0.05,
            count=50,
            accuracy=0.02,
            avg_confidence=0.05,
            calibration_error=0.03,
        )
        assert bin_.bin_center == 0.05
        assert bin_.count == 50

    def test_bin_validation_bounds(self) -> None:
        """Test bin bounds validation."""
        # Valid bin
        bin_ = ConfidenceBin(
            bin_lower=0.8,
            bin_upper=0.9,
            bin_center=0.85,
            count=100,
            accuracy=0.82,
            avg_confidence=0.85,
            calibration_error=0.03,
        )
        assert bin_.bin_lower < bin_.bin_upper

    def test_bin_invalid_bounds(self) -> None:
        """Test that invalid bounds fail validation."""
        with pytest.raises(ValidationError):
            ConfidenceBin(
                bin_lower=1.5,  # > 1.0
                bin_upper=0.9,
                bin_center=0.85,
                count=100,
                accuracy=0.82,
                avg_confidence=0.85,
                calibration_error=0.03,
            )


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics model."""

    @pytest.fixture
    def sample_calibration_metrics(self) -> CalibrationMetrics:
        """Create sample calibration metrics."""
        return CalibrationMetrics(
            ece=0.08,
            mce=0.15,
            ace=0.07,
            brier_score=0.12,
            overconfidence_rate=0.35,
            underconfidence_rate=0.15,
            accuracy=0.78,
            avg_confidence=0.82,
            n_samples=1000,
            confidence_bins=[
                ConfidenceBin(
                    bin_lower=0.0,
                    bin_upper=0.2,
                    bin_center=0.1,
                    count=50,
                    accuracy=0.06,
                    avg_confidence=0.1,
                    calibration_error=0.04,
                ),
                ConfidenceBin(
                    bin_lower=0.8,
                    bin_upper=1.0,
                    bin_center=0.9,
                    count=300,
                    accuracy=0.85,
                    avg_confidence=0.9,
                    calibration_error=0.05,
                ),
            ],
        )

    def test_basic_creation(self, sample_calibration_metrics: CalibrationMetrics) -> None:
        """Test creating calibration metrics."""
        assert sample_calibration_metrics.ece == 0.08
        assert sample_calibration_metrics.n_samples == 1000

    def test_confidence_accuracy_gap(self, sample_calibration_metrics: CalibrationMetrics) -> None:
        """Test confidence-accuracy gap computation."""
        gap = sample_calibration_metrics.confidence_accuracy_gap
        expected = abs(0.82 - 0.78)
        assert abs(gap - expected) < 0.001

    def test_is_overconfident(self, sample_calibration_metrics: CalibrationMetrics) -> None:
        """Test overconfidence detection."""
        # avg_confidence (0.82) > accuracy (0.78)
        assert sample_calibration_metrics.is_overconfident is True

    def test_is_underconfident(self) -> None:
        """Test underconfidence detection."""
        metrics = CalibrationMetrics(
            ece=0.05,
            mce=0.10,
            ace=0.04,
            brier_score=0.08,
            overconfidence_rate=0.10,
            underconfidence_rate=0.40,
            accuracy=0.85,
            avg_confidence=0.70,  # < accuracy
            n_samples=500,
        )
        assert metrics.is_overconfident is False

    def test_interpretation_excellent(self) -> None:
        """Test interpretation for excellent calibration."""
        metrics = CalibrationMetrics(
            ece=0.03,  # < 0.05
            mce=0.08,
            ace=0.02,
            brier_score=0.05,
            overconfidence_rate=0.05,
            underconfidence_rate=0.05,
            accuracy=0.90,
            avg_confidence=0.88,
            n_samples=1000,
        )
        interpretation = metrics.get_interpretation()
        assert "Excellent" in interpretation

    def test_interpretation_poor(self) -> None:
        """Test interpretation for poor calibration."""
        metrics = CalibrationMetrics(
            ece=0.25,  # >= 0.20
            mce=0.40,
            ace=0.22,
            brier_score=0.30,
            overconfidence_rate=0.60,
            underconfidence_rate=0.05,
            accuracy=0.60,
            avg_confidence=0.85,
            n_samples=500,
        )
        interpretation = metrics.get_interpretation()
        assert "Poor" in interpretation

    def test_metric_bounds(self) -> None:
        """Test that metrics must be in valid bounds."""
        with pytest.raises(ValidationError):
            CalibrationMetrics(
                ece=1.5,  # > 1.0, invalid
                mce=0.10,
                ace=0.05,
                brier_score=0.08,
                overconfidence_rate=0.10,
                underconfidence_rate=0.10,
                accuracy=0.80,
                avg_confidence=0.75,
                n_samples=100,
            )


class TestErrorDetectionMetrics:
    """Tests for ErrorDetectionMetrics model."""

    @pytest.fixture
    def sample_error_metrics(self) -> ErrorDetectionMetrics:
        """Create sample error detection metrics."""
        return ErrorDetectionMetrics(
            precision=0.85,
            recall=0.72,
            f1_score=0.78,
            false_positive_rate=0.15,
            false_negative_rate=0.28,
            false_humility_rate=0.08,
            recovery_rate=0.65,
            n_samples=500,
            n_total_errors=100,
            n_detected_errors=72,
        )

    def test_basic_creation(self, sample_error_metrics: ErrorDetectionMetrics) -> None:
        """Test creating error detection metrics."""
        assert sample_error_metrics.f1_score == 0.78
        assert sample_error_metrics.n_detected_errors == 72

    def test_interpretation_good(self, sample_error_metrics: ErrorDetectionMetrics) -> None:
        """Test interpretation for good error detection."""
        interpretation = sample_error_metrics.get_interpretation()
        assert "Good" in interpretation

    def test_interpretation_excellent(self) -> None:
        """Test interpretation for excellent error detection."""
        metrics = ErrorDetectionMetrics(
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            false_positive_rate=0.08,
            false_negative_rate=0.12,
            false_humility_rate=0.03,
            recovery_rate=0.85,
            n_samples=500,
            n_total_errors=100,
            n_detected_errors=88,
        )
        interpretation = metrics.get_interpretation()
        assert "Excellent" in interpretation

    def test_f1_consistency(self) -> None:
        """Test that F1 is consistent with precision and recall."""
        # F1 = 2 * (precision * recall) / (precision + recall)
        precision = 0.8
        recall = 0.6
        expected_f1 = 2 * (precision * recall) / (precision + recall)

        metrics = ErrorDetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=expected_f1,  # Should match formula
            false_positive_rate=0.2,
            false_negative_rate=0.4,
            false_humility_rate=0.1,
            recovery_rate=0.5,
            n_samples=100,
            n_total_errors=50,
            n_detected_errors=30,
        )
        assert abs(metrics.f1_score - expected_f1) < 0.001


class TestKnowledgeBoundaryMetrics:
    """Tests for KnowledgeBoundaryMetrics model."""

    @pytest.fixture
    def sample_kb_metrics(self) -> KnowledgeBoundaryMetrics:
        """Create sample knowledge boundary metrics."""
        return KnowledgeBoundaryMetrics(
            abstention_accuracy=0.82,
            appropriate_abstention_rate=0.75,
            inappropriate_abstention_rate=0.08,
            overconfidence_on_unknown=0.65,
            auroc=0.88,
            auprc=0.85,
            n_samples=400,
            n_answerable=300,
            n_unanswerable=100,
        )

    def test_basic_creation(self, sample_kb_metrics: KnowledgeBoundaryMetrics) -> None:
        """Test creating knowledge boundary metrics."""
        assert sample_kb_metrics.appropriate_abstention_rate == 0.75
        assert sample_kb_metrics.auroc == 0.88

    def test_interpretation_good(self, sample_kb_metrics: KnowledgeBoundaryMetrics) -> None:
        """Test interpretation for good boundary recognition."""
        interpretation = sample_kb_metrics.get_interpretation()
        assert "Good" in interpretation

    def test_interpretation_excellent(self) -> None:
        """Test interpretation for excellent boundary recognition."""
        metrics = KnowledgeBoundaryMetrics(
            abstention_accuracy=0.92,
            appropriate_abstention_rate=0.88,
            inappropriate_abstention_rate=0.03,
            overconfidence_on_unknown=0.25,
            auroc=0.95,
            auprc=0.93,
            n_samples=500,
            n_answerable=350,
            n_unanswerable=150,
        )
        interpretation = metrics.get_interpretation()
        assert "Excellent" in interpretation


class TestSelfCorrectionMetrics:
    """Tests for SelfCorrectionMetrics model."""

    @pytest.fixture
    def sample_sc_metrics(self) -> SelfCorrectionMetrics:
        """Create sample self-correction metrics."""
        return SelfCorrectionMetrics(
            corruption_detection_rate=0.78,
            corruption_detection_auc=0.82,
            correction_accuracy=0.70,
            sycophancy_index=0.12,
            false_alarm_rate=0.08,
            n_samples=300,
            n_corrupted=150,
            n_detected=117,
            n_corrected=105,
        )

    def test_basic_creation(self, sample_sc_metrics: SelfCorrectionMetrics) -> None:
        """Test creating self-correction metrics."""
        assert sample_sc_metrics.corruption_detection_rate == 0.78
        assert sample_sc_metrics.sycophancy_index == 0.12

    def test_resilience_score(self, sample_sc_metrics: SelfCorrectionMetrics) -> None:
        """Test resilience score computation."""
        expected = (0.78 + 0.70) / 2  # (detection + correction) / 2
        assert abs(sample_sc_metrics.resilience_score - expected) < 0.001

    def test_interpretation_good(self, sample_sc_metrics: SelfCorrectionMetrics) -> None:
        """Test interpretation for good self-correction."""
        interpretation = sample_sc_metrics.get_interpretation()
        assert "Good" in interpretation

    def test_low_sycophancy_is_good(self) -> None:
        """Test that low sycophancy index is desirable."""
        # A model that rarely changes correct answers to incorrect ones
        metrics = SelfCorrectionMetrics(
            corruption_detection_rate=0.85,
            corruption_detection_auc=0.90,
            correction_accuracy=0.80,
            sycophancy_index=0.02,  # Very low - good!
            false_alarm_rate=0.05,
            n_samples=200,
            n_corrupted=100,
            n_detected=85,
            n_corrected=68,
        )
        assert metrics.sycophancy_index < 0.1


class TestTrackResult:
    """Tests for TrackResult model."""

    def test_calibration_track_result(self) -> None:
        """Test creating calibration track result."""
        metrics = CalibrationMetrics(
            ece=0.08,
            mce=0.15,
            ace=0.07,
            brier_score=0.12,
            overconfidence_rate=0.25,
            underconfidence_rate=0.15,
            accuracy=0.78,
            avg_confidence=0.80,
            n_samples=500,
        )
        result = TrackResult(
            track=TrackType.CALIBRATION,
            metrics=metrics,
            n_samples=500,
            evaluation_time_seconds=45.5,
        )
        assert result.track == TrackType.CALIBRATION
        assert isinstance(result.metrics, CalibrationMetrics)

    def test_error_detection_track_result(self) -> None:
        """Test creating error detection track result."""
        metrics = ErrorDetectionMetrics(
            precision=0.85,
            recall=0.72,
            f1_score=0.78,
            false_positive_rate=0.15,
            false_negative_rate=0.28,
            false_humility_rate=0.08,
            recovery_rate=0.65,
            n_samples=300,
            n_total_errors=60,
            n_detected_errors=43,
        )
        result = TrackResult(
            track=TrackType.ERROR_DETECTION,
            metrics=metrics,
            n_samples=300,
            evaluation_time_seconds=30.0,
            errors=["Some samples failed to parse"],
        )
        assert len(result.errors) == 1


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    @pytest.fixture
    def calibration_track_result(self) -> TrackResult:
        """Create a calibration track result."""
        return TrackResult(
            track=TrackType.CALIBRATION,
            metrics=CalibrationMetrics(
                ece=0.08,
                mce=0.15,
                ace=0.07,
                brier_score=0.12,
                overconfidence_rate=0.25,
                underconfidence_rate=0.15,
                accuracy=0.78,
                avg_confidence=0.80,
                n_samples=500,
            ),
            n_samples=500,
            evaluation_time_seconds=45.0,
        )

    @pytest.fixture
    def error_track_result(self) -> TrackResult:
        """Create an error detection track result."""
        return TrackResult(
            track=TrackType.ERROR_DETECTION,
            metrics=ErrorDetectionMetrics(
                precision=0.85,
                recall=0.72,
                f1_score=0.78,
                false_positive_rate=0.15,
                false_negative_rate=0.28,
                false_humility_rate=0.08,
                recovery_rate=0.65,
                n_samples=300,
                n_total_errors=60,
                n_detected_errors=43,
            ),
            n_samples=300,
            evaluation_time_seconds=30.0,
        )

    def test_basic_creation(self) -> None:
        """Test creating basic evaluation result."""
        result = EvaluationResult(model_name="gemini-2.0-flash")
        assert result.model_name == "gemini-2.0-flash"
        assert result.id is not None
        assert result.started_at is not None

    def test_with_model_version(self) -> None:
        """Test evaluation result with model version."""
        result = EvaluationResult(
            model_name="gpt-4",
            model_version="turbo-2024-04-09",
            config_name="full_evaluation",
        )
        assert result.model_version == "turbo-2024-04-09"
        assert result.config_name == "full_evaluation"

    def test_add_track_result(
        self,
        calibration_track_result: TrackResult,
    ) -> None:
        """Test adding track results."""
        result = EvaluationResult(model_name="test-model")
        result.add_track_result(calibration_track_result)

        assert "calibration" in result.tracks
        assert result.total_samples == 500
        assert result.total_time_seconds == 45.0

    def test_add_multiple_tracks(
        self,
        calibration_track_result: TrackResult,
        error_track_result: TrackResult,
    ) -> None:
        """Test adding multiple track results."""
        result = EvaluationResult(model_name="test-model")
        result.add_track_result(calibration_track_result)
        result.add_track_result(error_track_result)

        assert len(result.tracks) == 2
        assert result.total_samples == 800
        assert result.total_time_seconds == 75.0

    def test_get_track_result(
        self,
        calibration_track_result: TrackResult,
    ) -> None:
        """Test getting specific track result."""
        result = EvaluationResult(model_name="test-model")
        result.add_track_result(calibration_track_result)

        retrieved = result.get_track_result(TrackType.CALIBRATION)
        assert retrieved is not None
        assert retrieved.track == TrackType.CALIBRATION

        missing = result.get_track_result(TrackType.ERROR_DETECTION)
        assert missing is None

    def test_mark_completed(self) -> None:
        """Test marking evaluation as completed."""
        result = EvaluationResult(model_name="test-model")
        assert result.completed_at is None

        result.mark_completed()
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

    def test_duration_seconds(self) -> None:
        """Test duration computation."""
        result = EvaluationResult(model_name="test-model")
        assert result.duration_seconds is None  # Not completed

        result.mark_completed()
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    def test_get_summary(
        self,
        calibration_track_result: TrackResult,
        error_track_result: TrackResult,
    ) -> None:
        """Test getting evaluation summary."""
        result = EvaluationResult(model_name="gemini-2.0-flash")
        result.add_track_result(calibration_track_result)
        result.add_track_result(error_track_result)
        result.mark_completed()

        summary = result.get_summary()

        assert summary["model"] == "gemini-2.0-flash"
        assert summary["total_samples"] == 800
        assert "calibration_ece" in summary
        assert "error_detection_f1" in summary

    def test_serialization(
        self,
        calibration_track_result: TrackResult,
    ) -> None:
        """Test JSON serialization."""
        result = EvaluationResult(model_name="test-model")
        result.add_track_result(calibration_track_result)

        json_str = result.model_dump_json()
        assert "test-model" in json_str
        assert "calibration" in json_str

    def test_evaluation_with_errors(self) -> None:
        """Test evaluation result with errors."""
        result = EvaluationResult(
            model_name="test-model",
            errors=[
                "API rate limit hit at sample 245",
                "Invalid response format at sample 312",
            ],
        )
        assert len(result.errors) == 2

    def test_evaluation_metadata(self) -> None:
        """Test evaluation result with custom metadata."""
        result = EvaluationResult(
            model_name="test-model",
            metadata={
                "environment": "production",
                "run_id": "abc123",
                "notes": "Initial benchmark run",
            },
        )
        assert result.metadata["environment"] == "production"
        assert result.metadata["run_id"] == "abc123"


class TestEvaluationEdgeCases:
    """Edge case tests for evaluation models."""

    def test_zero_samples(self) -> None:
        """Test metrics with zero samples."""
        metrics = CalibrationMetrics(
            ece=0.0,
            mce=0.0,
            ace=0.0,
            brier_score=0.0,
            overconfidence_rate=0.0,
            underconfidence_rate=0.0,
            accuracy=0.0,
            avg_confidence=0.0,
            n_samples=0,
        )
        assert metrics.n_samples == 0

    def test_perfect_calibration(self) -> None:
        """Test perfectly calibrated model."""
        metrics = CalibrationMetrics(
            ece=0.0,
            mce=0.0,
            ace=0.0,
            brier_score=0.0,
            overconfidence_rate=0.0,
            underconfidence_rate=0.0,
            accuracy=0.90,
            avg_confidence=0.90,
            n_samples=1000,
        )
        assert metrics.confidence_accuracy_gap == 0.0
        interpretation = metrics.get_interpretation()
        assert "Excellent" in interpretation

    def test_completely_wrong_predictions(self) -> None:
        """Test model that's always wrong with high confidence."""
        metrics = CalibrationMetrics(
            ece=0.90,
            mce=0.95,
            ace=0.88,
            brier_score=0.95,
            overconfidence_rate=0.95,
            underconfidence_rate=0.0,
            accuracy=0.05,
            avg_confidence=0.95,
            n_samples=500,
        )
        assert metrics.is_overconfident is True
        assert "Poor" in metrics.get_interpretation()
