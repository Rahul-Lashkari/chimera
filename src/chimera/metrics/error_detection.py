"""Error detection metrics for CHIMERA benchmark.

This module provides metrics for evaluating a model's ability to detect
errors in responses, including:

- Detection accuracy (binary: does the response contain errors?)
- Identification accuracy (did the model correctly identify the error?)
- Localization accuracy (did the model correctly locate the error?)
- False positive/negative rates
- Detection calibration (confidence vs detection accuracy)

These metrics are specific to the Error Detection track and complement
the general calibration metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import Field

from chimera.metrics.base import MetricConfig, MetricResult
from chimera.models.response import ModelResponse


class DetectionOutcome(str, Enum):
    """Possible outcomes for error detection."""

    TRUE_POSITIVE = "true_positive"  # Correctly detected error
    TRUE_NEGATIVE = "true_negative"  # Correctly identified no error
    FALSE_POSITIVE = "false_positive"  # Falsely claimed error exists
    FALSE_NEGATIVE = "false_negative"  # Missed actual error


@dataclass
class DetectionResult:
    """Result of a single error detection evaluation.

    Attributes:
        task_id: ID of the task
        has_actual_error: Whether the response actually has an error
        detected_error: Whether the model detected an error
        outcome: Detection outcome (TP, TN, FP, FN)
        confidence: Model's confidence in detection
        correct_error_type: If detected, was error type correct?
        correct_location: If detected, was error location correct?
        metadata: Additional metadata
    """

    task_id: str
    has_actual_error: bool
    detected_error: bool
    outcome: DetectionOutcome
    confidence: float = 0.0
    correct_error_type: bool | None = None
    correct_location: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorDetectionSummary:
    """Summary statistics for error detection performance.

    Attributes:
        total_tasks: Total number of tasks evaluated
        true_positives: Correctly detected errors
        true_negatives: Correctly identified no errors
        false_positives: Falsely claimed errors
        false_negatives: Missed errors
        accuracy: Overall detection accuracy
        precision: Precision (TP / (TP + FP))
        recall: Recall / sensitivity (TP / (TP + FN))
        f1_score: F1 score (harmonic mean of precision and recall)
        specificity: True negative rate (TN / (TN + FP))
        mean_confidence: Average confidence score
        confidence_calibration: Error between confidence and accuracy
    """

    total_tasks: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0

    mean_confidence: float = 0.0
    confidence_calibration: float = 0.0

    # Additional metrics
    error_type_accuracy: float | None = None
    localization_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "mean_confidence": self.mean_confidence,
            "confidence_calibration": self.confidence_calibration,
            "error_type_accuracy": self.error_type_accuracy,
            "localization_accuracy": self.localization_accuracy,
        }


class ErrorDetectionMetricsConfig(MetricConfig):
    """Configuration for error detection metrics.

    Attributes:
        confidence_threshold: Threshold for considering detection confident
        use_soft_matching: Use soft matching for error identification
        localization_tolerance: Tolerance for location matching (chars)
    """

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_soft_matching: bool = True
    localization_tolerance: int = Field(default=20, ge=0)


class ErrorDetectionMetrics:
    """Compute error detection performance metrics.

    This metric evaluates how well a model can detect errors in responses,
    computing standard classification metrics (precision, recall, F1) as
    well as calibration metrics specific to error detection.

    Example:
        >>> metrics = ErrorDetectionMetrics()
        >>> results = [
        ...     DetectionResult(
        ...         task_id="1",
        ...         has_actual_error=True,
        ...         detected_error=True,
        ...         outcome=DetectionOutcome.TRUE_POSITIVE,
        ...         confidence=0.85,
        ...     ),
        ...     # ... more results
        ... ]
        >>> summary = metrics.compute(results)
        >>> print(f"F1 Score: {summary.f1_score:.3f}")
    """

    name = "error_detection"

    def __init__(
        self,
        config: ErrorDetectionMetricsConfig | None = None,
    ) -> None:
        """Initialize error detection metrics.

        Args:
            config: Metrics configuration
        """
        self.config = config or ErrorDetectionMetricsConfig()

    def compute(
        self,
        results: list[DetectionResult],
    ) -> ErrorDetectionSummary:
        """Compute error detection metrics.

        Args:
            results: List of detection results

        Returns:
            Summary of error detection performance
        """
        if not results:
            return ErrorDetectionSummary()

        # Count outcomes
        tp = sum(1 for r in results if r.outcome == DetectionOutcome.TRUE_POSITIVE)
        tn = sum(1 for r in results if r.outcome == DetectionOutcome.TRUE_NEGATIVE)
        fp = sum(1 for r in results if r.outcome == DetectionOutcome.FALSE_POSITIVE)
        fn = sum(1 for r in results if r.outcome == DetectionOutcome.FALSE_NEGATIVE)

        total = len(results)

        # Compute metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Confidence metrics
        confidences = [r.confidence for r in results]
        mean_conf_val: float = float(np.mean(confidences)) if confidences else 0.0

        # Calibration: difference between confidence and accuracy
        confidence_calibration: float = abs(mean_conf_val - accuracy)

        # Error type accuracy (for true positives with type info)
        type_results = [r for r in results if r.correct_error_type is not None]
        error_type_accuracy = None
        if type_results:
            error_type_accuracy = sum(1 for r in type_results if r.correct_error_type) / len(
                type_results
            )

        # Localization accuracy
        loc_results = [r for r in results if r.correct_location is not None]
        localization_accuracy = None
        if loc_results:
            localization_accuracy = sum(1 for r in loc_results if r.correct_location) / len(
                loc_results
            )

        return ErrorDetectionSummary(
            total_tasks=total,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            mean_confidence=mean_conf_val,
            confidence_calibration=confidence_calibration,
            error_type_accuracy=error_type_accuracy,
            localization_accuracy=localization_accuracy,
        )

    def evaluate_response(
        self,
        response: ModelResponse,
        has_actual_error: bool,
        actual_error_type: str | None = None,
        actual_error_location: str | None = None,
    ) -> DetectionResult:
        """Evaluate a single model response.

        Args:
            response: The model's response
            has_actual_error: Whether the task actually has an error
            actual_error_type: The actual error type (if any)
            actual_error_location: The actual error location (if any)

        Returns:
            Detection result for this response
        """
        # Parse detection from response
        detected = self._parse_detection(response)
        confidence = response.confidence.numeric if response.confidence else 0.5

        # Determine outcome
        if has_actual_error and detected:
            outcome = DetectionOutcome.TRUE_POSITIVE
        elif not has_actual_error and not detected:
            outcome = DetectionOutcome.TRUE_NEGATIVE
        elif not has_actual_error and detected:
            outcome = DetectionOutcome.FALSE_POSITIVE
        else:  # has_actual_error and not detected
            outcome = DetectionOutcome.FALSE_NEGATIVE

        # Check error type if applicable
        correct_error_type = None
        if outcome == DetectionOutcome.TRUE_POSITIVE and actual_error_type:
            parsed_type = self._parse_error_type(response)
            correct_error_type = self._match_error_type(parsed_type, actual_error_type)

        # Check location if applicable
        correct_location = None
        if outcome == DetectionOutcome.TRUE_POSITIVE and actual_error_location:
            parsed_location = self._parse_error_location(response)
            correct_location = self._match_location(
                parsed_location,
                actual_error_location,
                self.config.localization_tolerance,
            )

        return DetectionResult(
            task_id=str(response.task_id),
            has_actual_error=has_actual_error,
            detected_error=detected,
            outcome=outcome,
            confidence=confidence,
            correct_error_type=correct_error_type,
            correct_location=correct_location,
        )

    def _parse_detection(self, response: ModelResponse) -> bool:
        """Parse whether the model detected an error.

        Looks for explicit yes/no answers or error identification language.
        """
        if not response.raw_text:
            return False

        text = response.raw_text.lower()

        # Check for explicit yes/no
        if response.parsed_answer:
            answer = response.parsed_answer.normalized.lower()
            # "yes" = there is an error, "no" = no error
            if answer in ("yes", "true", "error", "errors"):
                return True
            if answer in ("no", "false", "correct", "no error", "no errors"):
                return False

        # Check for error detection language
        error_indicators = [
            "error found",
            "errors found",
            "contains an error",
            "there is an error",
            "there are errors",
            "incorrect",
            "mistake",
            "wrong",
            "should be",
            "the error is",
        ]

        no_error_indicators = [
            "no error",
            "no errors",
            "correct",
            "accurate",
            "no mistake",
            "no correction needed",
        ]

        for indicator in no_error_indicators:
            if indicator in text:
                return False

        return any(indicator in text for indicator in error_indicators)

    def _parse_error_type(self, response: ModelResponse) -> str | None:
        """Parse the error type from response."""
        if not response.raw_text:
            return None

        text = response.raw_text.lower()

        error_type_keywords = {
            "factual": ["factual", "fact", "incorrect fact", "wrong fact"],
            "computational": ["computational", "calculation", "math", "arithmetic"],
            "logical": ["logical", "logic", "reasoning", "contradiction"],
            "temporal": ["temporal", "date", "year", "time", "when"],
            "magnitude": ["magnitude", "order of magnitude", "scale"],
            "hallucination": ["hallucination", "made up", "fabricated", "invented"],
        }

        for error_type, keywords in error_type_keywords.items():
            if any(kw in text for kw in keywords):
                return error_type

        return None

    def _parse_error_location(self, response: ModelResponse) -> str | None:
        """Parse the error location from response."""
        if not response.raw_text:
            return None

        import re

        # Look for quoted text
        quotes: list[str] = re.findall(r'"([^"]+)"', response.raw_text)
        if quotes:
            return str(quotes[0])

        quotes = re.findall(r"'([^']+)'", response.raw_text)
        if quotes:
            return str(quotes[0])

        return None

    def _match_error_type(
        self,
        parsed: str | None,
        actual: str,
    ) -> bool:
        """Check if parsed error type matches actual."""
        if not parsed:
            return False

        if self.config.use_soft_matching:
            # Allow partial matches
            return parsed.lower() in actual.lower() or actual.lower() in parsed.lower()

        return parsed.lower() == actual.lower()

    def _match_location(
        self,
        parsed: str | None,
        actual: str,
        tolerance: int,
    ) -> bool:
        """Check if parsed location matches actual."""
        if not parsed:
            return False

        parsed_lower = parsed.lower()
        actual_lower = actual.lower()

        # Exact match
        if parsed_lower == actual_lower:
            return True

        # Substring match
        if parsed_lower in actual_lower or actual_lower in parsed_lower:
            return True

        # Fuzzy match within tolerance
        if self.config.use_soft_matching:
            # Check if significant overlap
            parsed_words = set(parsed_lower.split())
            actual_words = set(actual_lower.split())
            overlap = len(parsed_words & actual_words)
            total = len(parsed_words | actual_words)
            if total > 0 and overlap / total >= 0.5:
                return True

        return False

    def compute_result(
        self,
        results: list[DetectionResult],
    ) -> MetricResult:
        """Compute metric result in standard format.

        Args:
            results: Detection results

        Returns:
            MetricResult with summary
        """
        summary = self.compute(results)

        return MetricResult(
            name=self.name,
            value=summary.f1_score,  # Primary metric is F1
            metadata={
                "summary": summary.to_dict(),
                "n_samples": summary.total_tasks,
            },
        )


class DetectionCalibrationMetrics:
    """Compute calibration metrics specific to error detection.

    This measures how well a model's confidence in error detection
    correlates with its actual detection accuracy.
    """

    def __init__(self, n_bins: int = 10) -> None:
        """Initialize with number of bins.

        Args:
            n_bins: Number of confidence bins
        """
        self.n_bins = n_bins

    def compute(
        self,
        results: list[DetectionResult],
    ) -> dict[str, Any]:
        """Compute detection calibration.

        Args:
            results: Detection results with confidence scores

        Returns:
            Calibration statistics
        """
        if not results:
            return {"ece": 0.0, "bins": []}

        # Group by confidence bins
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bins_data: list[list[tuple[float, bool]]] = [[] for _ in range(self.n_bins)]

        for result in results:
            bin_idx = min(int(result.confidence * self.n_bins), self.n_bins - 1)
            is_correct = result.outcome in (
                DetectionOutcome.TRUE_POSITIVE,
                DetectionOutcome.TRUE_NEGATIVE,
            )
            bins_data[bin_idx].append((result.confidence, is_correct))

        # Compute per-bin statistics
        bin_stats = []
        ece = 0.0
        total = len(results)

        for i, bin_data in enumerate(bins_data):
            if not bin_data:
                continue

            confidences, correctness = zip(*bin_data, strict=True)
            mean_conf = np.mean(confidences)
            accuracy = np.mean(correctness)
            count = len(bin_data)

            # ECE contribution
            ece += count / total * abs(mean_conf - accuracy)

            bin_stats.append(
                {
                    "bin_lower": float(bin_edges[i]),
                    "bin_upper": float(bin_edges[i + 1]),
                    "mean_confidence": float(mean_conf),
                    "accuracy": float(accuracy),
                    "count": count,
                    "gap": float(abs(mean_conf - accuracy)),
                }
            )

        return {
            "ece": float(ece),
            "n_bins": self.n_bins,
            "bins": bin_stats,
            "overconfidence": self._compute_overconfidence(bin_stats),
        }

    def _compute_overconfidence(
        self,
        bin_stats: list[dict],
    ) -> float:
        """Compute average overconfidence (confidence - accuracy when positive)."""
        overconf_gaps = [
            b["mean_confidence"] - b["accuracy"]
            for b in bin_stats
            if b["mean_confidence"] > b["accuracy"]
        ]
        return float(np.mean(overconf_gaps)) if overconf_gaps else 0.0
