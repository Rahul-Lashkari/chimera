"""Knowledge boundary metrics for CHIMERA benchmark.

This module provides metrics for evaluating a model's ability to recognize
the limits of its knowledge, including:

- Boundary recognition accuracy (correct identification of answerable vs not)
- Appropriate uncertainty expression
- Overconfidence on unanswerable questions
- Underconfidence on answerable questions
- Category-specific performance

These metrics measure metacognitive awareness of knowledge limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import Field

from chimera.generators.knowledge_boundary import (
    ExpectedResponse,
    KnowledgeCategory,
)
from chimera.metrics.base import MetricConfig, MetricResult
from chimera.models.response import ModelResponse


class ResponseClassification(str, Enum):
    """Classification of model's response type."""

    CONFIDENT_ANSWER = "confident_answer"
    UNCERTAIN_ANSWER = "uncertain_answer"
    DECLINED = "declined"
    ASKED_CLARIFICATION = "asked_clarification"
    ACKNOWLEDGED_UNKNOWABLE = "acknowledged_unknowable"
    UNCLASSIFIED = "unclassified"


@dataclass
class BoundaryResult:
    """Result of a single knowledge boundary evaluation.

    Attributes:
        task_id: ID of the task
        knowledge_category: Category of the question
        expected_response: Expected response type
        actual_response: Classified response type
        has_definite_answer: Whether question has a definite answer
        confidence: Model's stated confidence
        correct_boundary: Whether model correctly recognized boundary
        appropriate_confidence: Whether confidence level was appropriate
    """

    task_id: str
    knowledge_category: KnowledgeCategory
    expected_response: ExpectedResponse
    actual_response: ResponseClassification
    has_definite_answer: bool
    confidence: float = 0.5
    correct_boundary: bool = False
    appropriate_confidence: bool = False
    answer_correct: bool | None = None  # For answerable questions
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryMetrics:
    """Metrics for a specific knowledge category.

    Attributes:
        category: The knowledge category
        total: Total questions in category
        correct_boundary: Correctly identified boundary
        boundary_accuracy: Accuracy on boundary recognition
        mean_confidence: Average confidence in category
        appropriate_confidence_rate: Rate of appropriate confidence
    """

    category: KnowledgeCategory
    total: int = 0
    correct_boundary: int = 0
    boundary_accuracy: float = 0.0
    mean_confidence: float = 0.0
    appropriate_confidence_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "total": self.total,
            "correct_boundary": self.correct_boundary,
            "boundary_accuracy": self.boundary_accuracy,
            "mean_confidence": self.mean_confidence,
            "appropriate_confidence_rate": self.appropriate_confidence_rate,
        }


@dataclass
class KnowledgeBoundarySummary:
    """Summary statistics for knowledge boundary performance.

    Attributes:
        total_tasks: Total number of tasks
        boundary_accuracy: Overall accuracy on boundary recognition
        answerable_accuracy: Accuracy on answerable questions
        unanswerable_accuracy: Accuracy on unanswerable questions
        overconfidence_rate: Rate of overconfidence on unanswerable
        underconfidence_rate: Rate of underconfidence on answerable
        mean_confidence: Overall mean confidence
        calibration_gap: Gap between confidence and actual accuracy
        category_metrics: Per-category breakdown
    """

    total_tasks: int = 0
    boundary_accuracy: float = 0.0
    answerable_accuracy: float = 0.0
    unanswerable_accuracy: float = 0.0
    overconfidence_rate: float = 0.0
    underconfidence_rate: float = 0.0
    mean_confidence: float = 0.0
    calibration_gap: float = 0.0
    category_metrics: dict[KnowledgeCategory, CategoryMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "boundary_accuracy": self.boundary_accuracy,
            "answerable_accuracy": self.answerable_accuracy,
            "unanswerable_accuracy": self.unanswerable_accuracy,
            "overconfidence_rate": self.overconfidence_rate,
            "underconfidence_rate": self.underconfidence_rate,
            "mean_confidence": self.mean_confidence,
            "calibration_gap": self.calibration_gap,
            "category_metrics": {k.value: v.to_dict() for k, v in self.category_metrics.items()},
        }


class KnowledgeBoundaryMetricsConfig(MetricConfig):
    """Configuration for knowledge boundary metrics.

    Attributes:
        confidence_threshold: Threshold for "confident" classification
        uncertainty_keywords: Keywords indicating uncertainty
        decline_keywords: Keywords indicating declined answer
        clarification_keywords: Keywords asking for clarification
    """

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    uncertainty_keywords: list[str] = Field(
        default_factory=lambda: [
            "uncertain",
            "not sure",
            "I don't know",
            "I'm not certain",
            "might be",
            "possibly",
            "perhaps",
            "I cannot be sure",
            "difficult to say",
            "hard to determine",
            "unclear",
            "I lack information",
        ]
    )

    decline_keywords: list[str] = Field(
        default_factory=lambda: [
            "cannot answer",
            "unable to answer",
            "don't have access",
            "no way to know",
            "impossible to determine",
            "cannot predict",
            "fictional",
            "doesn't exist",
            "not real",
            "made up",
        ]
    )

    clarification_keywords: list[str] = Field(
        default_factory=lambda: [
            "could you clarify",
            "what do you mean",
            "which",
            "please specify",
            "need more context",
            "can you be more specific",
            "referring to",
            "ambiguous",
        ]
    )


class KnowledgeBoundaryMetrics:
    """Compute knowledge boundary recognition metrics.

    This metric evaluates how well a model recognizes the limits of its
    knowledge across different categories of questions.

    Example:
        >>> metrics = KnowledgeBoundaryMetrics()
        >>> results = [...]  # BoundaryResult instances
        >>> summary = metrics.compute(results)
        >>> print(f"Boundary Accuracy: {summary.boundary_accuracy:.3f}")
    """

    name = "knowledge_boundary"

    def __init__(
        self,
        config: KnowledgeBoundaryMetricsConfig | None = None,
    ) -> None:
        """Initialize knowledge boundary metrics.

        Args:
            config: Metrics configuration
        """
        self.config = config or KnowledgeBoundaryMetricsConfig()

    def compute(
        self,
        results: list[BoundaryResult],
    ) -> KnowledgeBoundarySummary:
        """Compute knowledge boundary metrics.

        Args:
            results: List of boundary results

        Returns:
            Summary of knowledge boundary performance
        """
        if not results:
            return KnowledgeBoundarySummary()

        total = len(results)

        # Overall boundary accuracy
        correct_boundary = sum(1 for r in results if r.correct_boundary)
        boundary_accuracy = correct_boundary / total

        # Split by answerable/unanswerable
        answerable = [r for r in results if r.has_definite_answer]
        unanswerable = [r for r in results if not r.has_definite_answer]

        # Answerable accuracy (did model answer correctly?)
        answerable_correct = sum(1 for r in answerable if r.answer_correct is True)
        answerable_accuracy = answerable_correct / len(answerable) if answerable else 0.0

        # Unanswerable accuracy (did model recognize it can't answer?)
        unanswerable_correct = sum(1 for r in unanswerable if r.correct_boundary)
        unanswerable_accuracy = unanswerable_correct / len(unanswerable) if unanswerable else 0.0

        # Overconfidence: high confidence on unanswerable
        overconfident = sum(
            1
            for r in unanswerable
            if r.confidence > self.config.confidence_threshold
            and r.actual_response == ResponseClassification.CONFIDENT_ANSWER
        )
        overconfidence_rate = overconfident / len(unanswerable) if unanswerable else 0.0

        # Underconfidence: low confidence on answerable
        underconfident = sum(
            1
            for r in answerable
            if r.confidence < self.config.confidence_threshold and r.answer_correct is True
        )
        underconfidence_rate = underconfident / len(answerable) if answerable else 0.0

        # Mean confidence
        confidences = [r.confidence for r in results]
        mean_confidence = np.mean(confidences)

        # Calibration gap
        actual_accuracy = correct_boundary / total
        calibration_gap = abs(mean_confidence - actual_accuracy)

        # Per-category metrics
        category_metrics = self._compute_category_metrics(results)

        return KnowledgeBoundarySummary(
            total_tasks=total,
            boundary_accuracy=boundary_accuracy,
            answerable_accuracy=answerable_accuracy,
            unanswerable_accuracy=unanswerable_accuracy,
            overconfidence_rate=overconfidence_rate,
            underconfidence_rate=underconfidence_rate,
            mean_confidence=float(mean_confidence),
            calibration_gap=float(calibration_gap),
            category_metrics=category_metrics,
        )

    def _compute_category_metrics(
        self,
        results: list[BoundaryResult],
    ) -> dict[KnowledgeCategory, CategoryMetrics]:
        """Compute metrics for each knowledge category."""
        by_category: dict[KnowledgeCategory, list[BoundaryResult]] = {}
        for r in results:
            by_category.setdefault(r.knowledge_category, []).append(r)

        category_metrics = {}
        for category, cat_results in by_category.items():
            total = len(cat_results)
            correct = sum(1 for r in cat_results if r.correct_boundary)
            confidences = [r.confidence for r in cat_results]
            appropriate = sum(1 for r in cat_results if r.appropriate_confidence)

            category_metrics[category] = CategoryMetrics(
                category=category,
                total=total,
                correct_boundary=correct,
                boundary_accuracy=correct / total if total > 0 else 0.0,
                mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
                appropriate_confidence_rate=appropriate / total if total > 0 else 0.0,
            )

        return category_metrics

    def classify_response(
        self,
        response: ModelResponse,
    ) -> ResponseClassification:
        """Classify a model response into a response type.

        Args:
            response: The model response

        Returns:
            Classification of response type
        """
        if not response.raw_text:
            return ResponseClassification.UNCLASSIFIED

        text = response.raw_text.lower()

        # Check for clarification request
        if any(kw in text for kw in self.config.clarification_keywords):
            return ResponseClassification.ASKED_CLARIFICATION

        # Check for decline
        if any(kw in text for kw in self.config.decline_keywords):
            return ResponseClassification.DECLINED

        # Check for uncertainty expressions
        if any(kw in text for kw in self.config.uncertainty_keywords):
            # Check if also acknowledging unknowability
            unknowable_markers = [
                "cannot be known",
                "unknowable",
                "impossible to know",
                "no one can know",
                "fundamentally unknowable",
            ]
            if any(marker in text for marker in unknowable_markers):
                return ResponseClassification.ACKNOWLEDGED_UNKNOWABLE
            return ResponseClassification.UNCERTAIN_ANSWER

        # Default to confident if has answer
        confidence = response.confidence.numeric if response.confidence else 0.5
        if confidence >= self.config.confidence_threshold:
            return ResponseClassification.CONFIDENT_ANSWER
        else:
            return ResponseClassification.UNCERTAIN_ANSWER

    def evaluate_response(
        self,
        response: ModelResponse,
        knowledge_category: KnowledgeCategory,
        expected_response: ExpectedResponse,
        has_definite_answer: bool,
        correct_answer: str | None = None,
    ) -> BoundaryResult:
        """Evaluate a single model response.

        Args:
            response: The model's response
            knowledge_category: Category of the question
            expected_response: Expected response type
            has_definite_answer: Whether question has definite answer
            correct_answer: The correct answer (if applicable)

        Returns:
            BoundaryResult for this response
        """
        # Classify the response
        actual = self.classify_response(response)
        confidence = response.confidence.numeric if response.confidence else 0.5

        # Check if boundary was correctly recognized
        correct_boundary = self._check_boundary_recognition(
            actual, expected_response, has_definite_answer
        )

        # Check if confidence is appropriate
        appropriate_confidence = self._check_appropriate_confidence(
            confidence, has_definite_answer, correct_boundary
        )

        # Check answer correctness for answerable questions
        answer_correct = None
        if has_definite_answer and correct_answer and response.parsed_answer:
            answer_correct = self._check_answer(
                response.parsed_answer.normalized,
                correct_answer,
            )

        return BoundaryResult(
            task_id=str(response.task_id),
            knowledge_category=knowledge_category,
            expected_response=expected_response,
            actual_response=actual,
            has_definite_answer=has_definite_answer,
            confidence=confidence,
            correct_boundary=correct_boundary,
            appropriate_confidence=appropriate_confidence,
            answer_correct=answer_correct,
        )

    def _check_boundary_recognition(
        self,
        actual: ResponseClassification,
        expected: ExpectedResponse,
        has_definite_answer: bool,
    ) -> bool:
        """Check if model correctly recognized the knowledge boundary."""
        # Mapping from expected to acceptable actual responses
        acceptable = {
            ExpectedResponse.CONFIDENT_ANSWER: {
                ResponseClassification.CONFIDENT_ANSWER,
            },
            ExpectedResponse.EXPRESS_UNCERTAINTY: {
                ResponseClassification.UNCERTAIN_ANSWER,
                ResponseClassification.ACKNOWLEDGED_UNKNOWABLE,
            },
            ExpectedResponse.DECLINE_TO_ANSWER: {
                ResponseClassification.DECLINED,
                ResponseClassification.ACKNOWLEDGED_UNKNOWABLE,
            },
            ExpectedResponse.ASK_CLARIFICATION: {
                ResponseClassification.ASKED_CLARIFICATION,
                ResponseClassification.UNCERTAIN_ANSWER,
            },
            ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE: {
                ResponseClassification.ACKNOWLEDGED_UNKNOWABLE,
                ResponseClassification.DECLINED,
                ResponseClassification.UNCERTAIN_ANSWER,
            },
        }

        return actual in acceptable.get(expected, set())

    def _check_appropriate_confidence(
        self,
        confidence: float,
        has_definite_answer: bool,
        correct_boundary: bool,
    ) -> bool:
        """Check if confidence level is appropriate."""
        threshold = self.config.confidence_threshold

        if has_definite_answer:
            # For answerable questions, high confidence is appropriate
            return confidence >= threshold or not correct_boundary
        else:
            # For unanswerable questions, low confidence is appropriate
            return confidence < threshold

    def _check_answer(
        self,
        given: str,
        correct: str,
    ) -> bool:
        """Check if given answer matches correct answer."""
        given_lower = given.lower().strip()
        correct_lower = correct.lower().strip()

        # Exact match
        if given_lower == correct_lower:
            return True

        # Contains match (for short answers in longer responses)
        if correct_lower in given_lower:
            return True

        # Partial word match for short answers
        given_words = set(given_lower.split())
        correct_words = set(correct_lower.split())
        return bool(correct_words and correct_words.issubset(given_words))

    def compute_result(
        self,
        results: list[BoundaryResult],
    ) -> MetricResult:
        """Compute metric result in standard format.

        Args:
            results: Boundary results

        Returns:
            MetricResult with summary
        """
        summary = self.compute(results)

        return MetricResult(
            name=self.name,
            value=summary.boundary_accuracy,  # Primary metric
            metadata={
                "summary": summary.to_dict(),
                "n_samples": summary.total_tasks,
            },
        )


class AnswerabilityClassifier:
    """Classify whether model correctly identifies question answerability.

    This provides a binary classification view: did the model correctly
    determine if the question was answerable or not?
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """Initialize classifier.

        Args:
            confidence_threshold: Threshold for confident answers
        """
        self.threshold = confidence_threshold

    def classify(
        self,
        results: list[BoundaryResult],
    ) -> dict[str, Any]:
        """Classify results by answerability.

        Returns:
            Dictionary with classification metrics
        """
        if not results:
            return {"accuracy": 0.0, "confusion_matrix": {}}

        # True/False Positive/Negative for answerability
        tp = 0  # Correctly answered answerable
        tn = 0  # Correctly declined unanswerable
        fp = 0  # Confidently answered unanswerable
        fn = 0  # Declined answerable

        for r in results:
            answered_confidently = (
                r.actual_response == ResponseClassification.CONFIDENT_ANSWER
                and r.confidence >= self.threshold
            )

            if r.has_definite_answer:
                if answered_confidently and r.answer_correct:
                    tp += 1
                elif not answered_confidently:
                    fn += 1
                else:
                    # Answered but wrong
                    fp += 1
            else:
                if not answered_confidently:
                    tn += 1
                else:
                    fp += 1

        total = len(results)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            },
            "answerable_performance": {
                "total": len([r for r in results if r.has_definite_answer]),
                "correct": tp,
            },
            "unanswerable_performance": {
                "total": len([r for r in results if not r.has_definite_answer]),
                "correct": tn,
            },
        }
