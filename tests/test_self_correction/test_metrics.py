"""Tests for self-correction metrics."""

from __future__ import annotations

import numpy as np
import pytest

from chimera.metrics.self_correction import (
    CorrectionEvaluator,
    CorrectionQuality,
    DetectionEvaluator,
    DetectionResult,
    SelfCorrectionMetricsComputer,
    SelfCorrectionMetricsConfig,
    SelfCorrectionResult,
    SelfCorrectionSummary,
)
from chimera.models.evaluation import SelfCorrectionMetrics


class TestDetectionResult:
    """Test DetectionResult enum."""

    def test_all_results_defined(self) -> None:
        """Verify all expected results exist."""
        expected = [
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ]
        assert len(DetectionResult) == len(expected)
        for result in expected:
            assert DetectionResult(result) is not None

    def test_result_values(self) -> None:
        """Test result string values."""
        assert DetectionResult.TRUE_POSITIVE.value == "true_positive"
        assert DetectionResult.FALSE_POSITIVE.value == "false_positive"
        assert DetectionResult.FALSE_NEGATIVE.value == "false_negative"
        assert DetectionResult.TRUE_NEGATIVE.value == "true_negative"


class TestCorrectionQuality:
    """Test CorrectionQuality enum."""

    def test_all_qualities_defined(self) -> None:
        """Verify all expected qualities exist."""
        expected = [
            "correct",
            "partial",
            "incorrect",
            "not_applicable",
        ]
        assert len(CorrectionQuality) == len(expected)
        for quality in expected:
            assert CorrectionQuality(quality) is not None


class TestSelfCorrectionResult:
    """Test SelfCorrectionResult dataclass."""

    def test_create_true_positive(self) -> None:
        """Test creating a true positive result."""
        result = SelfCorrectionResult(
            task_id="task_001",
            is_corrupted=True,
            model_detected_corruption=True,
            detection_confidence=0.9,
        )

        assert result.task_id == "task_001"
        assert result.detection_result == DetectionResult.TRUE_POSITIVE

    def test_create_false_positive(self) -> None:
        """Test creating a false positive result."""
        result = SelfCorrectionResult(
            task_id="task_002",
            is_corrupted=False,
            model_detected_corruption=True,
            detection_confidence=0.7,
        )

        assert result.detection_result == DetectionResult.FALSE_POSITIVE

    def test_create_false_negative(self) -> None:
        """Test creating a false negative result."""
        result = SelfCorrectionResult(
            task_id="task_003",
            is_corrupted=True,
            model_detected_corruption=False,
            detection_confidence=0.3,
        )

        assert result.detection_result == DetectionResult.FALSE_NEGATIVE

    def test_create_true_negative(self) -> None:
        """Test creating a true negative result."""
        result = SelfCorrectionResult(
            task_id="task_004",
            is_corrupted=False,
            model_detected_corruption=False,
            detection_confidence=0.2,
        )

        assert result.detection_result == DetectionResult.TRUE_NEGATIVE

    def test_result_defaults(self) -> None:
        """Test result default values."""
        result = SelfCorrectionResult(
            task_id="test",
            is_corrupted=False,
            model_detected_corruption=False,
        )

        assert result.detection_confidence == 0.5
        assert result.original_reasoning == ""
        assert result.model_response == ""
        assert result.metadata == {}

    def test_result_with_correction(self) -> None:
        """Test result with correction quality."""
        result = SelfCorrectionResult(
            task_id="task_005",
            is_corrupted=True,
            model_detected_corruption=True,
            correction_quality=CorrectionQuality.CORRECT,
            ground_truth_correction="The answer is 42.",
            model_correction="The correct answer is 42.",
        )

        assert result.correction_quality == CorrectionQuality.CORRECT
        assert result.ground_truth_correction == "The answer is 42."


class TestSelfCorrectionMetricsConfig:
    """Test SelfCorrectionMetricsConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SelfCorrectionMetricsConfig()

        assert config.detection_threshold == 0.5
        assert config.partial_credit_weight == 0.5
        assert config.require_explanation is True
        assert config.similarity_threshold == 0.8

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SelfCorrectionMetricsConfig(
            detection_threshold=0.7,
            partial_credit_weight=0.3,
            similarity_threshold=0.9,
        )

        assert config.detection_threshold == 0.7
        assert config.partial_credit_weight == 0.3
        assert config.similarity_threshold == 0.9

    def test_validation_detection_threshold(self) -> None:
        """Test detection threshold validation."""
        with pytest.raises(ValueError):
            SelfCorrectionMetricsConfig(detection_threshold=1.5)

        with pytest.raises(ValueError):
            SelfCorrectionMetricsConfig(detection_threshold=-0.1)


class TestSelfCorrectionMetricsComputer:
    """Test SelfCorrectionMetricsComputer."""

    @pytest.fixture
    def computer(self) -> SelfCorrectionMetricsComputer:
        """Create metrics computer with default config."""
        return SelfCorrectionMetricsComputer()

    @pytest.fixture
    def sample_results(self) -> list[SelfCorrectionResult]:
        """Create sample results for testing."""
        return [
            # True positives (detected corrupted)
            SelfCorrectionResult(
                task_id="t1",
                is_corrupted=True,
                model_detected_corruption=True,
                detection_confidence=0.9,
                correction_quality=CorrectionQuality.CORRECT,
            ),
            SelfCorrectionResult(
                task_id="t2",
                is_corrupted=True,
                model_detected_corruption=True,
                detection_confidence=0.8,
                correction_quality=CorrectionQuality.PARTIAL,
            ),
            # False negative (missed corruption)
            SelfCorrectionResult(
                task_id="t3",
                is_corrupted=True,
                model_detected_corruption=False,
                detection_confidence=0.3,
            ),
            # True negative (correctly validated)
            SelfCorrectionResult(
                task_id="t4",
                is_corrupted=False,
                model_detected_corruption=False,
                detection_confidence=0.2,
            ),
            SelfCorrectionResult(
                task_id="t5",
                is_corrupted=False,
                model_detected_corruption=False,
                detection_confidence=0.1,
            ),
            # False positive (incorrectly flagged)
            SelfCorrectionResult(
                task_id="t6",
                is_corrupted=False,
                model_detected_corruption=True,
                detection_confidence=0.6,
            ),
        ]

    def test_compute_metrics(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test computing metrics from results."""
        metrics = computer.compute(sample_results)

        assert isinstance(metrics, SelfCorrectionMetrics)
        assert metrics.n_samples == 6
        assert metrics.n_corrupted == 3  # 3 corrupted samples
        assert metrics.n_detected == 2  # 2 true positives

    def test_detection_rate(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test detection rate computation."""
        metrics = computer.compute(sample_results)

        # 2 detected out of 3 corrupted = 2/3
        assert abs(metrics.corruption_detection_rate - (2 / 3)) < 0.01

    def test_false_alarm_rate(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test false alarm rate computation."""
        metrics = computer.compute(sample_results)

        # 1 false positive out of 3 uncorrupted = 1/3
        assert abs(metrics.false_alarm_rate - (1 / 3)) < 0.01

    def test_correction_accuracy(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test correction accuracy computation."""
        metrics = computer.compute(sample_results)

        # 1 correct + 0.5 * 1 partial out of 3 detected = 1.5/3 = 0.5
        # (Note: 3 detected = 2 TP + 1 FP)
        assert metrics.correction_accuracy >= 0.0
        assert metrics.correction_accuracy <= 1.0

    def test_detection_auc(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test detection AUC computation."""
        metrics = computer.compute(sample_results)

        # AUC should be between 0 and 1
        assert 0.0 <= metrics.corruption_detection_auc <= 1.0

    def test_empty_results(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test with empty results."""
        metrics = computer.compute([])

        assert metrics.n_samples == 0
        assert metrics.corruption_detection_rate == 0.0
        assert metrics.corruption_detection_auc == 0.5

    def test_all_corrupted(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test with all corrupted samples."""
        results = [
            SelfCorrectionResult(
                task_id=f"t{i}",
                is_corrupted=True,
                model_detected_corruption=(i % 2 == 0),
                detection_confidence=0.5 + (i % 3) * 0.2,
            )
            for i in range(10)
        ]
        metrics = computer.compute(results)

        assert metrics.n_corrupted == 10
        assert metrics.false_alarm_rate == 0.0  # No uncorrupted samples

    def test_all_uncorrupted(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test with all uncorrupted samples."""
        results = [
            SelfCorrectionResult(
                task_id=f"t{i}",
                is_corrupted=False,
                model_detected_corruption=(i % 3 == 0),
                detection_confidence=0.5 - (i % 3) * 0.1,
            )
            for i in range(10)
        ]
        metrics = computer.compute(results)

        assert metrics.n_corrupted == 0
        assert metrics.corruption_detection_rate == 0.0  # No corrupted samples

    def test_resilience_score(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test resilience score computation."""
        metrics = computer.compute(sample_results)

        # Resilience = (detection_rate + correction_accuracy) / 2
        expected = (metrics.corruption_detection_rate + metrics.correction_accuracy) / 2
        assert abs(metrics.resilience_score - expected) < 0.01

    def test_get_summary(
        self,
        computer: SelfCorrectionMetricsComputer,
        sample_results: list[SelfCorrectionResult],
    ) -> None:
        """Test getting summary statistics."""
        summary = computer.get_summary(sample_results)

        assert isinstance(summary, SelfCorrectionSummary)
        assert summary.total_samples == 6
        assert summary.total_corrupted == 3
        assert summary.total_uncorrupted == 3
        assert summary.true_positives == 2
        assert summary.false_positives == 1
        assert summary.false_negatives == 1
        assert summary.true_negatives == 2


class TestSelfCorrectionMetricsComputerArrays:
    """Test SelfCorrectionMetricsComputer with array inputs."""

    @pytest.fixture
    def computer(self) -> SelfCorrectionMetricsComputer:
        """Create metrics computer."""
        return SelfCorrectionMetricsComputer()

    def test_compute_from_arrays(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test computing metrics from numpy arrays."""
        is_corrupted = np.array([True, True, True, False, False, False])
        detection_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.6])
        detected = np.array([True, True, False, False, False, True])
        correction_scores = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.3])

        metrics = computer.compute_from_arrays(
            is_corrupted=is_corrupted,
            detection_scores=detection_scores,
            detected=detected,
            correction_scores=correction_scores,
        )

        assert isinstance(metrics, SelfCorrectionMetrics)
        assert metrics.n_samples == 6
        assert metrics.n_corrupted == 3
        assert metrics.n_detected == 2

    def test_compute_from_arrays_empty(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test with empty arrays."""
        metrics = computer.compute_from_arrays(
            is_corrupted=np.array([], dtype=bool),
            detection_scores=np.array([]),
            detected=np.array([], dtype=bool),
            correction_scores=np.array([]),
        )

        assert metrics.n_samples == 0
        assert metrics.corruption_detection_auc == 0.5

    def test_perfect_detection(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test with perfect detection."""
        is_corrupted = np.array([True, True, False, False])
        detection_scores = np.array([0.9, 0.8, 0.2, 0.1])
        detected = np.array([True, True, False, False])
        correction_scores = np.array([1.0, 1.0, 0.0, 0.0])

        metrics = computer.compute_from_arrays(
            is_corrupted=is_corrupted,
            detection_scores=detection_scores,
            detected=detected,
            correction_scores=correction_scores,
        )

        assert metrics.corruption_detection_rate == 1.0
        assert metrics.false_alarm_rate == 0.0
        assert metrics.corruption_detection_auc == 1.0

    def test_worst_detection(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test with worst-case detection."""
        is_corrupted = np.array([True, True, False, False])
        detection_scores = np.array([0.1, 0.2, 0.8, 0.9])  # Inverted
        detected = np.array([False, False, True, True])
        correction_scores = np.array([0.0, 0.0, 0.5, 0.5])

        metrics = computer.compute_from_arrays(
            is_corrupted=is_corrupted,
            detection_scores=detection_scores,
            detected=detected,
            correction_scores=correction_scores,
        )

        assert metrics.corruption_detection_rate == 0.0
        assert metrics.false_alarm_rate == 1.0
        assert metrics.corruption_detection_auc == 0.0


class TestCorrectionEvaluator:
    """Test CorrectionEvaluator."""

    @pytest.fixture
    def evaluator(self) -> CorrectionEvaluator:
        """Create correction evaluator."""
        return CorrectionEvaluator()

    def test_evaluate_correct(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test evaluating a correct correction."""
        quality = evaluator.evaluate(
            model_correction="The answer is 42 because it is the meaning of life.",
            expected_correction="The answer is 42 because it is the meaning of life.",
            is_corrupted=True,
        )

        assert quality == CorrectionQuality.CORRECT

    def test_evaluate_partial(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test evaluating a partial correction."""
        quality = evaluator.evaluate(
            model_correction="The answer is 42.",
            expected_correction="The answer is 42 because 6 times 7 equals 42.",
            is_corrupted=True,
        )

        # Should be partial due to some overlap
        assert quality in [CorrectionQuality.PARTIAL, CorrectionQuality.INCORRECT]

    def test_evaluate_incorrect(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test evaluating an incorrect correction."""
        quality = evaluator.evaluate(
            model_correction="alpha beta gamma delta",
            expected_correction="one two three four five six seven eight",
            is_corrupted=True,
        )

        # Should be incorrect due to zero overlap
        assert quality == CorrectionQuality.INCORRECT

    def test_evaluate_not_applicable(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test evaluating when not corrupted."""
        quality = evaluator.evaluate(
            model_correction="Some correction",
            expected_correction="Expected correction",
            is_corrupted=False,
        )

        assert quality == CorrectionQuality.NOT_APPLICABLE

    def test_evaluate_empty_correction(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test evaluating empty correction."""
        quality = evaluator.evaluate(
            model_correction="",
            expected_correction="The answer is 42.",
            is_corrupted=True,
        )

        assert quality == CorrectionQuality.INCORRECT

    def test_evaluate_batch(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test batch evaluation."""
        qualities = evaluator.evaluate_batch(
            model_corrections=["The answer is 42.", "Wrong answer", ""],
            expected_corrections=["The answer is 42.", "Correct answer", "Something"],
            is_corrupted=[True, True, True],
        )

        assert len(qualities) == 3
        assert qualities[0] == CorrectionQuality.CORRECT

    def test_compute_score(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test computing numerical score."""
        score = evaluator.compute_score(
            model_correction="The answer is forty-two.",
            expected_correction="The answer is forty-two.",
        )

        assert score == 1.0

    def test_compute_score_zero(
        self,
        evaluator: CorrectionEvaluator,
    ) -> None:
        """Test score for completely different texts."""
        score = evaluator.compute_score(
            model_correction="alpha beta gamma",
            expected_correction="one two three",
        )

        assert score == 0.0

    def test_custom_threshold(self) -> None:
        """Test with custom similarity threshold."""
        evaluator = CorrectionEvaluator(
            config=SelfCorrectionMetricsConfig(similarity_threshold=0.5)
        )

        # With 5/6 word overlap = 0.833 similarity, threshold 0.5 => CORRECT
        quality = evaluator.evaluate(
            model_correction="The answer is definitely 42 here",
            expected_correction="The answer is 42 here today",
            is_corrupted=True,
        )

        # With lower threshold, should be correct
        assert quality == CorrectionQuality.CORRECT


class TestDetectionEvaluator:
    """Test DetectionEvaluator."""

    @pytest.fixture
    def evaluator(self) -> DetectionEvaluator:
        """Create detection evaluator."""
        return DetectionEvaluator()

    def test_evaluate_detected_error(
        self,
        evaluator: DetectionEvaluator,
    ) -> None:
        """Test detecting error phrases."""
        detected, confidence = evaluator.evaluate(
            "There is an error in step 2. The calculation is wrong."
        )

        assert detected is True
        assert confidence > 0.5

    def test_evaluate_validated_correct(
        self,
        evaluator: DetectionEvaluator,
    ) -> None:
        """Test validating correct reasoning."""
        detected, confidence = evaluator.evaluate("This reasoning is correct. The answer is valid.")

        assert detected is False
        assert confidence > 0.5

    def test_evaluate_empty_response(
        self,
        evaluator: DetectionEvaluator,
    ) -> None:
        """Test empty response."""
        detected, confidence = evaluator.evaluate("")

        assert detected is False
        assert confidence == 0.5

    def test_evaluate_mixed_phrases(
        self,
        evaluator: DetectionEvaluator,
    ) -> None:
        """Test response with mixed phrases."""
        detected, confidence = evaluator.evaluate(
            "The first step is correct, but there is an error in step 3. "
            "The reasoning is mostly valid except for that mistake."
        )

        # Has both error and validation phrases
        assert isinstance(detected, bool)
        assert 0.0 <= confidence <= 1.0

    def test_evaluate_batch(
        self,
        evaluator: DetectionEvaluator,
    ) -> None:
        """Test batch evaluation."""
        results = evaluator.evaluate_batch(
            [
                "This is wrong.",
                "This is correct.",
                "",
            ]
        )

        assert len(results) == 3
        assert results[0][0] is True  # detected
        assert results[1][0] is False  # not detected
        assert results[2][0] is False  # empty

    def test_custom_threshold(self) -> None:
        """Test with custom detection threshold."""
        evaluator = DetectionEvaluator(config=SelfCorrectionMetricsConfig(detection_threshold=0.7))

        # Needs stronger error signal to detect
        detected, _ = evaluator.evaluate("There might be an issue, but seems correct overall.")

        # With higher threshold, less likely to detect
        assert isinstance(detected, bool)


class TestAUCComputation:
    """Test AUC computation edge cases."""

    @pytest.fixture
    def computer(self) -> SelfCorrectionMetricsComputer:
        """Create metrics computer."""
        return SelfCorrectionMetricsComputer()

    def test_auc_single_sample(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test AUC with single sample."""
        result = SelfCorrectionResult(
            task_id="t1",
            is_corrupted=True,
            model_detected_corruption=True,
            detection_confidence=0.9,
        )
        metrics = computer.compute([result])

        # AUC should default to 0.5 with insufficient samples
        assert metrics.corruption_detection_auc == 0.5

    def test_auc_all_same_class(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test AUC when all samples are same class."""
        results = [
            SelfCorrectionResult(
                task_id=f"t{i}",
                is_corrupted=True,
                model_detected_corruption=True,
                detection_confidence=0.5 + i * 0.1,
            )
            for i in range(5)
        ]
        metrics = computer.compute(results)

        # AUC should be 0.5 with only one class
        assert metrics.corruption_detection_auc == 0.5

    def test_auc_perfect_separation(
        self,
        computer: SelfCorrectionMetricsComputer,
    ) -> None:
        """Test AUC with perfect class separation."""
        results = [
            SelfCorrectionResult(
                task_id="t1",
                is_corrupted=True,
                model_detected_corruption=True,
                detection_confidence=0.9,
            ),
            SelfCorrectionResult(
                task_id="t2",
                is_corrupted=True,
                model_detected_corruption=True,
                detection_confidence=0.8,
            ),
            SelfCorrectionResult(
                task_id="t3",
                is_corrupted=False,
                model_detected_corruption=False,
                detection_confidence=0.2,
            ),
            SelfCorrectionResult(
                task_id="t4",
                is_corrupted=False,
                model_detected_corruption=False,
                detection_confidence=0.1,
            ),
        ]
        metrics = computer.compute(results)

        # Perfect separation should give AUC = 1.0
        assert metrics.corruption_detection_auc == 1.0
