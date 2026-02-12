"""Unit tests for error detection metrics.

Tests cover:
- Detection result creation
- Error detection metrics computation
- Classification metrics (precision, recall, F1)
- Detection calibration
"""

from uuid import uuid4

import pytest

from chimera.metrics.error_detection import (
    DetectionCalibrationMetrics,
    DetectionOutcome,
    DetectionResult,
    ErrorDetectionMetrics,
    ErrorDetectionMetricsConfig,
    ErrorDetectionSummary,
)
from chimera.models.response import (
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ResponseMetadata,
)


def _test_metadata() -> ResponseMetadata:
    """Create test response metadata."""
    return ResponseMetadata(
        model_name="test-model",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_true_positive(self) -> None:
        """Test true positive result."""
        result = DetectionResult(
            task_id="task1",
            has_actual_error=True,
            detected_error=True,
            outcome=DetectionOutcome.TRUE_POSITIVE,
            confidence=0.9,
        )

        assert result.outcome == DetectionOutcome.TRUE_POSITIVE

    def test_true_negative(self) -> None:
        """Test true negative result."""
        result = DetectionResult(
            task_id="task2",
            has_actual_error=False,
            detected_error=False,
            outcome=DetectionOutcome.TRUE_NEGATIVE,
        )

        assert result.outcome == DetectionOutcome.TRUE_NEGATIVE

    def test_false_positive(self) -> None:
        """Test false positive result."""
        result = DetectionResult(
            task_id="task3",
            has_actual_error=False,
            detected_error=True,
            outcome=DetectionOutcome.FALSE_POSITIVE,
        )

        assert result.outcome == DetectionOutcome.FALSE_POSITIVE

    def test_false_negative(self) -> None:
        """Test false negative result."""
        result = DetectionResult(
            task_id="task4",
            has_actual_error=True,
            detected_error=False,
            outcome=DetectionOutcome.FALSE_NEGATIVE,
        )

        assert result.outcome == DetectionOutcome.FALSE_NEGATIVE


class TestErrorDetectionSummary:
    """Tests for ErrorDetectionSummary."""

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        summary = ErrorDetectionSummary(
            total_tasks=100,
            true_positives=40,
            true_negatives=30,
            false_positives=15,
            false_negatives=15,
            accuracy=0.7,
            precision=0.73,
            recall=0.73,
            f1_score=0.73,
        )

        d = summary.to_dict()

        assert d["total_tasks"] == 100
        assert d["accuracy"] == 0.7
        assert d["f1_score"] == 0.73


class TestErrorDetectionMetrics:
    """Tests for ErrorDetectionMetrics."""

    @pytest.fixture
    def metrics(self) -> ErrorDetectionMetrics:
        """Create metrics instance."""
        return ErrorDetectionMetrics()

    @pytest.fixture
    def sample_results(self) -> list[DetectionResult]:
        """Create sample detection results."""
        return [
            DetectionResult("1", True, True, DetectionOutcome.TRUE_POSITIVE, 0.9),
            DetectionResult("2", True, True, DetectionOutcome.TRUE_POSITIVE, 0.8),
            DetectionResult("3", False, False, DetectionOutcome.TRUE_NEGATIVE, 0.85),
            DetectionResult("4", False, False, DetectionOutcome.TRUE_NEGATIVE, 0.7),
            DetectionResult("5", True, False, DetectionOutcome.FALSE_NEGATIVE, 0.4),
            DetectionResult("6", False, True, DetectionOutcome.FALSE_POSITIVE, 0.6),
        ]

    def test_compute_empty(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test computation with empty results."""
        summary = metrics.compute([])

        assert summary.total_tasks == 0
        assert summary.accuracy == 0.0

    def test_compute_accuracy(
        self,
        metrics: ErrorDetectionMetrics,
        sample_results: list[DetectionResult],
    ) -> None:
        """Test accuracy computation."""
        summary = metrics.compute(sample_results)

        # 4 correct (2 TP + 2 TN) out of 6
        expected_accuracy = 4 / 6
        assert abs(summary.accuracy - expected_accuracy) < 0.01

    def test_compute_precision(
        self,
        metrics: ErrorDetectionMetrics,
        sample_results: list[DetectionResult],
    ) -> None:
        """Test precision computation."""
        summary = metrics.compute(sample_results)

        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.667
        expected_precision = 2 / 3
        assert abs(summary.precision - expected_precision) < 0.01

    def test_compute_recall(
        self,
        metrics: ErrorDetectionMetrics,
        sample_results: list[DetectionResult],
    ) -> None:
        """Test recall computation."""
        summary = metrics.compute(sample_results)

        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 0.667
        expected_recall = 2 / 3
        assert abs(summary.recall - expected_recall) < 0.01

    def test_compute_f1(
        self,
        metrics: ErrorDetectionMetrics,
        sample_results: list[DetectionResult],
    ) -> None:
        """Test F1 score computation."""
        summary = metrics.compute(sample_results)

        # F1 = 2 * P * R / (P + R) = 2 * 0.667 * 0.667 / 1.333 = 0.667
        assert abs(summary.f1_score - 0.667) < 0.01

    def test_compute_specificity(
        self,
        metrics: ErrorDetectionMetrics,
        sample_results: list[DetectionResult],
    ) -> None:
        """Test specificity computation."""
        summary = metrics.compute(sample_results)

        # Specificity = TN / (TN + FP) = 2 / (2 + 1) = 0.667
        expected_specificity = 2 / 3
        assert abs(summary.specificity - expected_specificity) < 0.01

    def test_compute_mean_confidence(
        self,
        metrics: ErrorDetectionMetrics,
        sample_results: list[DetectionResult],
    ) -> None:
        """Test mean confidence computation."""
        summary = metrics.compute(sample_results)

        expected_mean = (0.9 + 0.8 + 0.85 + 0.7 + 0.4 + 0.6) / 6
        assert abs(summary.mean_confidence - expected_mean) < 0.01

    def test_perfect_detection(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test perfect detection scenario."""
        results = [
            DetectionResult("1", True, True, DetectionOutcome.TRUE_POSITIVE, 1.0),
            DetectionResult("2", False, False, DetectionOutcome.TRUE_NEGATIVE, 1.0),
        ]

        summary = metrics.compute(results)

        assert summary.accuracy == 1.0
        assert summary.precision == 1.0
        assert summary.recall == 1.0
        assert summary.f1_score == 1.0

    def test_all_false_positives(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test all false positives scenario."""
        results = [
            DetectionResult("1", False, True, DetectionOutcome.FALSE_POSITIVE, 0.9),
            DetectionResult("2", False, True, DetectionOutcome.FALSE_POSITIVE, 0.8),
        ]

        summary = metrics.compute(results)

        assert summary.accuracy == 0.0
        assert summary.precision == 0.0
        assert summary.specificity == 0.0


class TestErrorDetectionMetricsConfig:
    """Tests for ErrorDetectionMetricsConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ErrorDetectionMetricsConfig()

        assert config.confidence_threshold == 0.5
        assert config.use_soft_matching is True
        assert config.localization_tolerance == 20

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ErrorDetectionMetricsConfig(
            confidence_threshold=0.7,
            use_soft_matching=False,
        )

        assert config.confidence_threshold == 0.7
        assert config.use_soft_matching is False


class TestEvaluateResponse:
    """Tests for evaluating model responses."""

    @pytest.fixture
    def metrics(self) -> ErrorDetectionMetrics:
        return ErrorDetectionMetrics()

    def test_evaluate_correct_detection(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test evaluating correct error detection."""
        response = ModelResponse(
            task_id=uuid4(),
            raw_text="Yes, there is an error in this response.",
            parsed_answer=ParsedAnswer(raw_answer="Yes", normalized="yes"),
            confidence=ConfidenceScore(numeric=0.9),
            metadata=_test_metadata(),
        )

        result = metrics.evaluate_response(response, has_actual_error=True)

        assert result.detected_error is True
        assert result.outcome == DetectionOutcome.TRUE_POSITIVE

    def test_evaluate_correct_no_error(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test evaluating correct no-error detection."""
        response = ModelResponse(
            task_id=uuid4(),
            raw_text="No errors found in this response.",
            parsed_answer=ParsedAnswer(raw_answer="No", normalized="no"),
            confidence=ConfidenceScore(numeric=0.85),
            metadata=_test_metadata(),
        )

        result = metrics.evaluate_response(response, has_actual_error=False)

        assert result.detected_error is False
        assert result.outcome == DetectionOutcome.TRUE_NEGATIVE

    def test_evaluate_missed_error(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test evaluating missed error (false negative)."""
        response = ModelResponse(
            task_id=uuid4(),
            raw_text="The response looks correct to me.",
            parsed_answer=ParsedAnswer(raw_answer="correct", normalized="correct"),
            confidence=ConfidenceScore(numeric=0.6),
            metadata=_test_metadata(),
        )

        result = metrics.evaluate_response(response, has_actual_error=True)

        assert result.detected_error is False
        assert result.outcome == DetectionOutcome.FALSE_NEGATIVE

    def test_evaluate_false_alarm(
        self,
        metrics: ErrorDetectionMetrics,
    ) -> None:
        """Test evaluating false alarm (false positive)."""
        response = ModelResponse(
            task_id=uuid4(),
            raw_text="There is an error: the calculation is wrong.",
            parsed_answer=ParsedAnswer(raw_answer="error", normalized="error"),
            confidence=ConfidenceScore(numeric=0.7),
            metadata=_test_metadata(),
        )

        result = metrics.evaluate_response(response, has_actual_error=False)

        assert result.detected_error is True
        assert result.outcome == DetectionOutcome.FALSE_POSITIVE


class TestDetectionCalibrationMetrics:
    """Tests for DetectionCalibrationMetrics."""

    @pytest.fixture
    def calibration(self) -> DetectionCalibrationMetrics:
        return DetectionCalibrationMetrics(n_bins=5)

    def test_empty_results(
        self,
        calibration: DetectionCalibrationMetrics,
    ) -> None:
        """Test with empty results."""
        result = calibration.compute([])

        assert result["ece"] == 0.0
        assert result["bins"] == []

    def test_compute_ece(
        self,
        calibration: DetectionCalibrationMetrics,
    ) -> None:
        """Test ECE computation."""
        results = [
            DetectionResult("1", True, True, DetectionOutcome.TRUE_POSITIVE, 0.9),
            DetectionResult("2", True, True, DetectionOutcome.TRUE_POSITIVE, 0.85),
            DetectionResult("3", False, False, DetectionOutcome.TRUE_NEGATIVE, 0.8),
            DetectionResult("4", False, True, DetectionOutcome.FALSE_POSITIVE, 0.7),
            DetectionResult("5", True, False, DetectionOutcome.FALSE_NEGATIVE, 0.3),
        ]

        result = calibration.compute(results)

        assert "ece" in result
        assert 0 <= result["ece"] <= 1

    def test_bin_statistics(
        self,
        calibration: DetectionCalibrationMetrics,
    ) -> None:
        """Test bin statistics are computed."""
        results = [
            DetectionResult("1", True, True, DetectionOutcome.TRUE_POSITIVE, 0.9),
            DetectionResult("2", True, True, DetectionOutcome.TRUE_POSITIVE, 0.95),
            DetectionResult("3", False, False, DetectionOutcome.TRUE_NEGATIVE, 0.1),
            DetectionResult("4", False, False, DetectionOutcome.TRUE_NEGATIVE, 0.15),
        ]

        result = calibration.compute(results)

        assert len(result["bins"]) > 0
        for bin_stat in result["bins"]:
            assert "mean_confidence" in bin_stat
            assert "accuracy" in bin_stat
            assert "count" in bin_stat

    def test_perfect_calibration(
        self,
        calibration: DetectionCalibrationMetrics,
    ) -> None:
        """Test ECE is 0 with perfect calibration."""
        # Create results where confidence matches accuracy
        results = [
            DetectionResult("1", True, True, DetectionOutcome.TRUE_POSITIVE, 0.9),
            DetectionResult("2", True, True, DetectionOutcome.TRUE_POSITIVE, 0.92),
            DetectionResult("3", True, True, DetectionOutcome.TRUE_POSITIVE, 0.88),
            DetectionResult("4", True, True, DetectionOutcome.TRUE_POSITIVE, 0.91),
            DetectionResult("5", True, True, DetectionOutcome.TRUE_POSITIVE, 0.89),
        ]

        result = calibration.compute(results)

        # All correct with ~0.9 confidence = good calibration
        assert result["ece"] < 0.15

    def test_overconfidence_metric(
        self,
        calibration: DetectionCalibrationMetrics,
    ) -> None:
        """Test overconfidence metric."""
        # High confidence but wrong
        results = [
            DetectionResult("1", True, False, DetectionOutcome.FALSE_NEGATIVE, 0.9),
            DetectionResult("2", True, False, DetectionOutcome.FALSE_NEGATIVE, 0.85),
            DetectionResult("3", False, True, DetectionOutcome.FALSE_POSITIVE, 0.8),
        ]

        result = calibration.compute(results)

        # Should show overconfidence
        assert result["overconfidence"] > 0


class TestMetricResult:
    """Tests for MetricResult generation."""

    def test_compute_result(self) -> None:
        """Test computing metric result."""
        metrics = ErrorDetectionMetrics()

        results = [
            DetectionResult("1", True, True, DetectionOutcome.TRUE_POSITIVE, 0.9),
            DetectionResult("2", False, False, DetectionOutcome.TRUE_NEGATIVE, 0.8),
        ]

        metric_result = metrics.compute_result(results)

        assert metric_result.name == "error_detection"
        assert metric_result.value == 1.0  # Perfect F1
        assert "summary" in metric_result.metadata
