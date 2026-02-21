"""Self-correction metrics for CHIMERA benchmark.

This module implements metrics for evaluating a model's ability to
detect and correct corrupted reasoning in its own responses.

Metrics:
    - Corruption Detection Rate: How often the model identifies corrupted reasoning
    - Corruption Detection AUC: Area under ROC curve for detection
    - Correction Accuracy: How accurately the model fixes detected errors
    - Sycophancy Index: Rate of incorrectly changing correct answers
    - False Alarm Rate: Rate of flagging uncorrupted reasoning as wrong

Self-Correction Track Evaluation Flow:
1. Present model with reasoning traces (some corrupted, some correct)
2. Model indicates whether reasoning is valid
3. If corrupted, model attempts to correct
4. Compare model's assessment and corrections to ground truth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from chimera.models.evaluation import SelfCorrectionMetrics


class DetectionResult(str, Enum):
    """Result of corruption detection."""

    # Model correctly identified corrupted reasoning
    TRUE_POSITIVE = "true_positive"

    # Model incorrectly flagged uncorrupted reasoning
    FALSE_POSITIVE = "false_positive"

    # Model missed corrupted reasoning
    FALSE_NEGATIVE = "false_negative"

    # Model correctly validated uncorrupted reasoning
    TRUE_NEGATIVE = "true_negative"


class CorrectionQuality(str, Enum):
    """Quality of the correction provided."""

    # Fully correct correction
    CORRECT = "correct"

    # Partially correct correction
    PARTIAL = "partial"

    # Incorrect or no correction
    INCORRECT = "incorrect"

    # Not applicable (no corruption present)
    NOT_APPLICABLE = "not_applicable"


@dataclass
class SelfCorrectionResult:
    """Result of a single self-correction evaluation.

    Attributes:
        task_id: Identifier of the task
        is_corrupted: Whether the original reasoning was corrupted
        model_detected_corruption: Whether model flagged as corrupted
        detection_confidence: Model's confidence in detection (0-1)
        detection_result: Classification of detection outcome
        correction_quality: Quality of the correction if applicable
        original_reasoning: The presented reasoning trace
        model_response: The model's full response
        ground_truth_correction: The expected correction
        model_correction: The model's correction (if any)
        metadata: Additional evaluation metadata
    """

    task_id: str
    is_corrupted: bool
    model_detected_corruption: bool
    detection_confidence: float = 0.5
    detection_result: DetectionResult = DetectionResult.TRUE_NEGATIVE
    correction_quality: CorrectionQuality = CorrectionQuality.NOT_APPLICABLE
    original_reasoning: str = ""
    model_response: str = ""
    ground_truth_correction: str = ""
    model_correction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute detection result based on corruption status and detection."""
        if self.is_corrupted and self.model_detected_corruption:
            self.detection_result = DetectionResult.TRUE_POSITIVE
        elif self.is_corrupted and not self.model_detected_corruption:
            self.detection_result = DetectionResult.FALSE_NEGATIVE
        elif not self.is_corrupted and self.model_detected_corruption:
            self.detection_result = DetectionResult.FALSE_POSITIVE
        else:
            self.detection_result = DetectionResult.TRUE_NEGATIVE


@dataclass
class SelfCorrectionSummary:
    """Summary statistics for self-correction evaluation.

    Attributes:
        total_samples: Total number of samples evaluated
        total_corrupted: Number of corrupted samples
        total_uncorrupted: Number of uncorrupted samples
        true_positives: Correctly detected corruptions
        false_positives: Incorrectly flagged as corrupted
        false_negatives: Missed corruptions
        true_negatives: Correctly validated as uncorrupted
        correct_corrections: Number of fully correct corrections
        partial_corrections: Number of partially correct corrections
        incorrect_corrections: Number of incorrect corrections
        detection_rate: Rate of detecting corruption
        false_alarm_rate: Rate of false alarms
        correction_accuracy: Accuracy of corrections
        sycophancy_index: Rate of changing correct to incorrect
    """

    total_samples: int = 0
    total_corrupted: int = 0
    total_uncorrupted: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    correct_corrections: int = 0
    partial_corrections: int = 0
    incorrect_corrections: int = 0
    detection_rate: float = 0.0
    false_alarm_rate: float = 0.0
    correction_accuracy: float = 0.0
    sycophancy_index: float = 0.0


class SelfCorrectionMetricsConfig(BaseModel):
    """Configuration for self-correction metrics computation.

    Attributes:
        detection_threshold: Confidence threshold for detection
        partial_credit_weight: Weight for partial corrections
        require_explanation: Whether corrections must include explanation
        similarity_threshold: Threshold for correction similarity matching
    """

    detection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for corruption detection",
    )

    partial_credit_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for partial corrections (0-1)",
    )

    require_explanation: bool = Field(
        default=True,
        description="Whether corrections must include explanation",
    )

    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for correction similarity matching",
    )


class SelfCorrectionMetricsComputer:
    """Compute self-correction metrics from evaluation results.

    This class processes self-correction evaluation results and computes
    comprehensive metrics for the Self-Correction track.

    Example:
        >>> computer = SelfCorrectionMetricsComputer()
        >>> results = [result1, result2, result3, ...]
        >>> metrics = computer.compute(results)
        >>> print(f"Detection Rate: {metrics.corruption_detection_rate:.3f}")
    """

    def __init__(
        self,
        config: SelfCorrectionMetricsConfig | None = None,
    ) -> None:
        """Initialize the metrics computer.

        Args:
            config: Configuration for metrics computation
        """
        self.config = config or SelfCorrectionMetricsConfig()

    def compute(
        self,
        results: list[SelfCorrectionResult],
    ) -> SelfCorrectionMetrics:
        """Compute self-correction metrics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            SelfCorrectionMetrics with all computed metrics
        """
        if not results:
            return self._empty_metrics()

        summary = self._compute_summary(results)

        # Compute AUC for detection
        detection_auc = self._compute_detection_auc(results)

        return SelfCorrectionMetrics(
            corruption_detection_rate=summary.detection_rate,
            corruption_detection_auc=detection_auc,
            correction_accuracy=summary.correction_accuracy,
            sycophancy_index=summary.sycophancy_index,
            false_alarm_rate=summary.false_alarm_rate,
            n_samples=summary.total_samples,
            n_corrupted=summary.total_corrupted,
            n_detected=summary.true_positives,
            n_corrected=summary.correct_corrections,
        )

    def compute_from_arrays(
        self,
        is_corrupted: NDArray[np.bool_],
        detection_scores: NDArray[np.floating[Any]],
        detected: NDArray[np.bool_],
        correction_scores: NDArray[np.floating[Any]],
    ) -> SelfCorrectionMetrics:
        """Compute metrics from numpy arrays.

        This is an alternative interface for when results are already
        in array format.

        Args:
            is_corrupted: Boolean array indicating corruption ground truth
            detection_scores: Confidence scores for corruption detection
            detected: Boolean array of detection decisions
            correction_scores: Scores for correction quality (0-1)

        Returns:
            SelfCorrectionMetrics with all computed metrics
        """
        n_samples = len(is_corrupted)
        if n_samples == 0:
            return self._empty_metrics()

        # Detection metrics
        true_positives = int(np.sum(is_corrupted & detected))
        false_positives = int(np.sum(~is_corrupted & detected))
        _false_negatives = int(np.sum(is_corrupted & ~detected))  # noqa: F841
        _true_negatives = int(np.sum(~is_corrupted & ~detected))  # noqa: F841

        n_corrupted = int(np.sum(is_corrupted))
        n_uncorrupted = n_samples - n_corrupted

        # Detection rate
        detection_rate = true_positives / n_corrupted if n_corrupted > 0 else 0.0

        # False alarm rate
        false_alarm_rate = false_positives / n_uncorrupted if n_uncorrupted > 0 else 0.0

        # Sycophancy index (changing correct answers)
        # Measured as false positive rate + incorrect corrections on uncorrupted
        sycophancy_index = false_alarm_rate

        # Correction accuracy (among detected corruptions)
        n_detected = true_positives + false_positives
        if n_detected > 0:
            # Average correction score for detected items
            detected_corrections = correction_scores[detected]
            correction_accuracy = float(np.mean(detected_corrections))
            n_corrected = int(np.sum(detected_corrections >= self.config.similarity_threshold))
        else:
            correction_accuracy = 0.0
            n_corrected = 0

        # Compute AUC
        detection_auc = self._compute_auc(
            is_corrupted.astype(float),
            detection_scores,
        )

        return SelfCorrectionMetrics(
            corruption_detection_rate=detection_rate,
            corruption_detection_auc=detection_auc,
            correction_accuracy=correction_accuracy,
            sycophancy_index=sycophancy_index,
            false_alarm_rate=false_alarm_rate,
            n_samples=n_samples,
            n_corrupted=n_corrupted,
            n_detected=true_positives,
            n_corrected=n_corrected,
        )

    def _compute_summary(
        self,
        results: list[SelfCorrectionResult],
    ) -> SelfCorrectionSummary:
        """Compute summary statistics from results.

        Args:
            results: List of evaluation results

        Returns:
            Summary statistics
        """
        summary = SelfCorrectionSummary()
        summary.total_samples = len(results)

        for result in results:
            if result.is_corrupted:
                summary.total_corrupted += 1
            else:
                summary.total_uncorrupted += 1

            if result.detection_result == DetectionResult.TRUE_POSITIVE:
                summary.true_positives += 1
            elif result.detection_result == DetectionResult.FALSE_POSITIVE:
                summary.false_positives += 1
            elif result.detection_result == DetectionResult.FALSE_NEGATIVE:
                summary.false_negatives += 1
            else:
                summary.true_negatives += 1

            if result.correction_quality == CorrectionQuality.CORRECT:
                summary.correct_corrections += 1
            elif result.correction_quality == CorrectionQuality.PARTIAL:
                summary.partial_corrections += 1
            elif result.correction_quality == CorrectionQuality.INCORRECT:
                summary.incorrect_corrections += 1

        # Compute rates
        if summary.total_corrupted > 0:
            summary.detection_rate = summary.true_positives / summary.total_corrupted
        else:
            summary.detection_rate = 0.0

        if summary.total_uncorrupted > 0:
            summary.false_alarm_rate = summary.false_positives / summary.total_uncorrupted
        else:
            summary.false_alarm_rate = 0.0

        # Correction accuracy among detected
        n_detected = summary.true_positives + summary.false_positives
        if n_detected > 0:
            weighted_correct = (
                summary.correct_corrections
                + self.config.partial_credit_weight * summary.partial_corrections
            )
            summary.correction_accuracy = weighted_correct / n_detected
        else:
            summary.correction_accuracy = 0.0

        # Sycophancy: changing correct answers incorrectly
        # Measured by false positive rate
        summary.sycophancy_index = summary.false_alarm_rate

        return summary

    def _compute_detection_auc(
        self,
        results: list[SelfCorrectionResult],
    ) -> float:
        """Compute AUC for corruption detection.

        Args:
            results: List of evaluation results

        Returns:
            AUC score between 0 and 1
        """
        if len(results) < 2:
            return 0.5

        labels = np.array([r.is_corrupted for r in results], dtype=float)
        scores = np.array([r.detection_confidence for r in results], dtype=float)

        return self._compute_auc(labels, scores)

    def _compute_auc(
        self,
        labels: NDArray[np.floating[Any]],
        scores: NDArray[np.floating[Any]],
    ) -> float:
        """Compute AUC using the trapezoidal rule.

        Implements AUC computation without sklearn dependency.

        Args:
            labels: Binary labels (0 or 1)
            scores: Continuous scores

        Returns:
            AUC score between 0 and 1
        """
        # Handle edge cases
        n_positive = np.sum(labels)
        n_negative = len(labels) - n_positive

        if n_positive == 0 or n_negative == 0:
            return 0.5

        # Sort by descending score
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]

        # Compute TPR and FPR at each threshold
        tprs: list[float] = [0.0]
        fprs: list[float] = [0.0]
        tp_cumsum = 0.0
        fp_cumsum = 0.0

        for label in sorted_labels:
            if label > 0.5:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            tprs.append(tp_cumsum / n_positive)
            fprs.append(fp_cumsum / n_negative)

        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(fprs) - 1):
            auc += (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2

        return float(auc)

    def _empty_metrics(self) -> SelfCorrectionMetrics:
        """Return empty metrics for edge cases.

        Returns:
            SelfCorrectionMetrics with zero values
        """
        return SelfCorrectionMetrics(
            corruption_detection_rate=0.0,
            corruption_detection_auc=0.5,
            correction_accuracy=0.0,
            sycophancy_index=0.0,
            false_alarm_rate=0.0,
            n_samples=0,
            n_corrupted=0,
            n_detected=0,
            n_corrected=0,
        )

    def get_summary(
        self,
        results: list[SelfCorrectionResult],
    ) -> SelfCorrectionSummary:
        """Get summary statistics without full metrics computation.

        Args:
            results: List of evaluation results

        Returns:
            Summary statistics
        """
        return self._compute_summary(results)


class CorrectionEvaluator:
    """Evaluate the quality of model corrections.

    This class provides methods for assessing how well a model's
    correction matches the expected correction.

    Example:
        >>> evaluator = CorrectionEvaluator()
        >>> quality = evaluator.evaluate(model_correction, expected_correction)
        >>> print(f"Quality: {quality}")
    """

    def __init__(
        self,
        config: SelfCorrectionMetricsConfig | None = None,
    ) -> None:
        """Initialize the correction evaluator.

        Args:
            config: Configuration for evaluation
        """
        self.config = config or SelfCorrectionMetricsConfig()

    def evaluate(
        self,
        model_correction: str,
        expected_correction: str,
        is_corrupted: bool = True,
    ) -> CorrectionQuality:
        """Evaluate the quality of a model's correction.

        Args:
            model_correction: The model's correction text
            expected_correction: The expected correction text
            is_corrupted: Whether the original was corrupted

        Returns:
            CorrectionQuality classification
        """
        if not is_corrupted:
            return CorrectionQuality.NOT_APPLICABLE

        if not model_correction:
            return CorrectionQuality.INCORRECT

        # Compute similarity
        similarity = self._compute_similarity(model_correction, expected_correction)

        if similarity >= self.config.similarity_threshold:
            return CorrectionQuality.CORRECT
        elif similarity >= self.config.similarity_threshold * 0.5:
            return CorrectionQuality.PARTIAL
        else:
            return CorrectionQuality.INCORRECT

    def evaluate_batch(
        self,
        model_corrections: list[str],
        expected_corrections: list[str],
        is_corrupted: list[bool],
    ) -> list[CorrectionQuality]:
        """Evaluate a batch of corrections.

        Args:
            model_corrections: List of model corrections
            expected_corrections: List of expected corrections
            is_corrupted: List of corruption flags

        Returns:
            List of CorrectionQuality classifications
        """
        return [
            self.evaluate(m, e, c)
            for m, e, c in zip(
                model_corrections,
                expected_corrections,
                is_corrupted,
                strict=True,
            )
        ]

    def compute_score(
        self,
        model_correction: str,
        expected_correction: str,
    ) -> float:
        """Compute a numerical correction score.

        Args:
            model_correction: The model's correction text
            expected_correction: The expected correction text

        Returns:
            Score between 0 and 1
        """
        return self._compute_similarity(model_correction, expected_correction)

    def _compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute text similarity using word overlap.

        Uses Jaccard similarity on word tokens as a simple baseline.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1_normalized = text1.lower().strip()
        text2_normalized = text2.lower().strip()

        # Exact match
        if text1_normalized == text2_normalized:
            return 1.0

        # Tokenize
        words1 = set(text1_normalized.split())
        words2 = set(text2_normalized.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class DetectionEvaluator:
    """Evaluate corruption detection responses.

    This class provides methods for determining whether a model
    detected corruption in reasoning.

    Example:
        >>> evaluator = DetectionEvaluator()
        >>> detected, confidence = evaluator.evaluate(model_response)
        >>> print(f"Detected: {detected}, Confidence: {confidence}")
    """

    # Phrases indicating detection of errors
    DETECTION_PHRASES = [
        "error",
        "mistake",
        "incorrect",
        "wrong",
        "invalid",
        "flaw",
        "issue",
        "problem",
        "not correct",
        "not right",
        "false",
        "inaccurate",
        "should be",
        "needs correction",
        "actually",
        "however",
    ]

    # Phrases indicating validation/acceptance
    VALIDATION_PHRASES = [
        "correct",
        "valid",
        "accurate",
        "right",
        "proper",
        "sound reasoning",
        "well done",
        "looks good",
        "no issues",
        "no errors",
        "is correct",
        "is right",
        "is valid",
    ]

    def __init__(
        self,
        config: SelfCorrectionMetricsConfig | None = None,
    ) -> None:
        """Initialize the detection evaluator.

        Args:
            config: Configuration for evaluation
        """
        self.config = config or SelfCorrectionMetricsConfig()

    def evaluate(
        self,
        model_response: str,
    ) -> tuple[bool, float]:
        """Evaluate whether model detected corruption.

        Args:
            model_response: The model's response text

        Returns:
            Tuple of (detected, confidence)
        """
        if not model_response:
            return False, 0.5

        response_lower = model_response.lower()

        # Count detection and validation phrase matches
        detection_count = sum(1 for phrase in self.DETECTION_PHRASES if phrase in response_lower)
        validation_count = sum(1 for phrase in self.VALIDATION_PHRASES if phrase in response_lower)

        # Compute confidence based on phrase density
        total_matches = detection_count + validation_count
        if total_matches == 0:
            return False, 0.5

        # Detection confidence
        detection_ratio = detection_count / total_matches

        # Apply threshold
        detected = detection_ratio > self.config.detection_threshold

        # Confidence is how strongly one way or another
        confidence = max(detection_ratio, 1 - detection_ratio)

        return detected, confidence

    def evaluate_batch(
        self,
        model_responses: list[str],
    ) -> list[tuple[bool, float]]:
        """Evaluate a batch of responses.

        Args:
            model_responses: List of model responses

        Returns:
            List of (detected, confidence) tuples
        """
        return [self.evaluate(response) for response in model_responses]
