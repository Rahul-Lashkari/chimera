"""Tests for knowledge boundary metrics."""

from __future__ import annotations

from uuid import uuid4

import pytest

from chimera.generators.knowledge_boundary import (
    ExpectedResponse,
    KnowledgeCategory,
)
from chimera.metrics.knowledge_boundary import (
    AnswerabilityClassifier,
    BoundaryResult,
    CategoryMetrics,
    KnowledgeBoundaryMetrics,
    KnowledgeBoundaryMetricsConfig,
    KnowledgeBoundarySummary,
    ResponseClassification,
)
from chimera.models.response import (
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ResponseMetadata,
)


def create_model_response(
    raw_text: str,
    confidence: ConfidenceScore,
    task_id: str | None = None,
    parsed_answer: ParsedAnswer | None = None,
) -> ModelResponse:
    """Helper to create a ModelResponse with all required fields."""
    return ModelResponse(
        task_id=uuid4(),
        raw_text=raw_text,
        confidence=confidence,
        parsed_answer=parsed_answer
        or ParsedAnswer.from_raw(raw_text.split(".")[0] if "." in raw_text else raw_text),
        metadata=ResponseMetadata(
            model_name="test-model",
            latency_ms=100,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        ),
    )


class TestResponseClassification:
    """Test ResponseClassification enum."""

    def test_all_classifications_defined(self) -> None:
        """Verify all expected classifications exist."""
        expected = [
            "confident_answer",
            "uncertain_answer",
            "declined",
            "asked_clarification",
            "acknowledged_unknowable",
            "unclassified",
        ]
        assert len(ResponseClassification) == len(expected)


class TestBoundaryResult:
    """Test BoundaryResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a boundary result."""
        result = BoundaryResult(
            task_id="task-1",
            knowledge_category=KnowledgeCategory.FACTUAL_KNOWN,
            expected_response=ExpectedResponse.CONFIDENT_ANSWER,
            actual_response=ResponseClassification.CONFIDENT_ANSWER,
            has_definite_answer=True,
            confidence=0.9,
            correct_boundary=True,
            appropriate_confidence=True,
            answer_correct=True,
        )

        assert result.task_id == "task-1"
        assert result.correct_boundary is True
        assert result.answer_correct is True

    def test_result_defaults(self) -> None:
        """Test default values."""
        result = BoundaryResult(
            task_id="task-1",
            knowledge_category=KnowledgeCategory.UNKNOWABLE,
            expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
            actual_response=ResponseClassification.ACKNOWLEDGED_UNKNOWABLE,
            has_definite_answer=False,
        )

        assert result.confidence == 0.5
        assert result.correct_boundary is False
        assert result.metadata == {}


class TestCategoryMetrics:
    """Test CategoryMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Test creating category metrics."""
        metrics = CategoryMetrics(
            category=KnowledgeCategory.FACTUAL_KNOWN,
            total=10,
            correct_boundary=8,
            boundary_accuracy=0.8,
            mean_confidence=0.85,
            appropriate_confidence_rate=0.7,
        )

        assert metrics.total == 10
        assert metrics.boundary_accuracy == 0.8

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = CategoryMetrics(
            category=KnowledgeCategory.SUBJECTIVE,
            total=5,
            correct_boundary=3,
            boundary_accuracy=0.6,
        )

        d = metrics.to_dict()
        assert d["category"] == "subjective"
        assert d["total"] == 5
        assert d["boundary_accuracy"] == 0.6


class TestKnowledgeBoundarySummary:
    """Test KnowledgeBoundarySummary dataclass."""

    def test_create_summary(self) -> None:
        """Test creating a summary."""
        summary = KnowledgeBoundarySummary(
            total_tasks=100,
            boundary_accuracy=0.75,
            answerable_accuracy=0.8,
            unanswerable_accuracy=0.7,
            overconfidence_rate=0.15,
            underconfidence_rate=0.1,
        )

        assert summary.total_tasks == 100
        assert summary.boundary_accuracy == 0.75

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        summary = KnowledgeBoundarySummary(
            total_tasks=50,
            boundary_accuracy=0.8,
        )

        d = summary.to_dict()
        assert d["total_tasks"] == 50
        assert d["boundary_accuracy"] == 0.8


class TestKnowledgeBoundaryMetricsConfig:
    """Test KnowledgeBoundaryMetricsConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = KnowledgeBoundaryMetricsConfig()

        assert config.confidence_threshold == 0.7
        assert len(config.uncertainty_keywords) > 0
        assert len(config.decline_keywords) > 0
        assert len(config.clarification_keywords) > 0

    def test_custom_threshold(self) -> None:
        """Test custom confidence threshold."""
        config = KnowledgeBoundaryMetricsConfig(confidence_threshold=0.5)

        assert config.confidence_threshold == 0.5


class TestKnowledgeBoundaryMetrics:
    """Test KnowledgeBoundaryMetrics class."""

    @pytest.fixture
    def metrics(self) -> KnowledgeBoundaryMetrics:
        """Create metrics instance."""
        return KnowledgeBoundaryMetrics()

    @pytest.fixture
    def sample_results(self) -> list[BoundaryResult]:
        """Create sample results for testing."""
        return [
            # Correct on answerable
            BoundaryResult(
                task_id="1",
                knowledge_category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                actual_response=ResponseClassification.CONFIDENT_ANSWER,
                has_definite_answer=True,
                confidence=0.9,
                correct_boundary=True,
                appropriate_confidence=True,
                answer_correct=True,
            ),
            # Correct on unanswerable
            BoundaryResult(
                task_id="2",
                knowledge_category=KnowledgeCategory.UNKNOWABLE,
                expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
                actual_response=ResponseClassification.ACKNOWLEDGED_UNKNOWABLE,
                has_definite_answer=False,
                confidence=0.3,
                correct_boundary=True,
                appropriate_confidence=True,
            ),
            # Overconfident on unanswerable
            BoundaryResult(
                task_id="3",
                knowledge_category=KnowledgeCategory.TEMPORAL_FUTURE,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                actual_response=ResponseClassification.CONFIDENT_ANSWER,
                has_definite_answer=False,
                confidence=0.85,
                correct_boundary=False,
                appropriate_confidence=False,
            ),
            # Underconfident on answerable
            BoundaryResult(
                task_id="4",
                knowledge_category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                actual_response=ResponseClassification.UNCERTAIN_ANSWER,
                has_definite_answer=True,
                confidence=0.4,
                correct_boundary=False,
                appropriate_confidence=False,
                answer_correct=True,
            ),
        ]

    def test_compute_empty_results(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test computing with empty results."""
        summary = metrics.compute([])

        assert summary.total_tasks == 0
        assert summary.boundary_accuracy == 0.0

    def test_compute_boundary_accuracy(
        self,
        metrics: KnowledgeBoundaryMetrics,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test boundary accuracy computation."""
        summary = metrics.compute(sample_results)

        # 2 correct out of 4
        assert summary.boundary_accuracy == 0.5
        assert summary.total_tasks == 4

    def test_compute_overconfidence_rate(
        self,
        metrics: KnowledgeBoundaryMetrics,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test overconfidence rate computation."""
        summary = metrics.compute(sample_results)

        # 1 overconfident out of 2 unanswerable
        assert summary.overconfidence_rate == 0.5

    def test_compute_underconfidence_rate(
        self,
        metrics: KnowledgeBoundaryMetrics,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test underconfidence rate computation."""
        summary = metrics.compute(sample_results)

        # 1 underconfident out of 2 answerable
        assert summary.underconfidence_rate == 0.5

    def test_compute_category_metrics(
        self,
        metrics: KnowledgeBoundaryMetrics,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test per-category metrics computation."""
        summary = metrics.compute(sample_results)

        # Check factual known category
        factual = summary.category_metrics.get(KnowledgeCategory.FACTUAL_KNOWN)
        assert factual is not None
        assert factual.total == 2
        assert factual.correct_boundary == 1

    def test_classify_confident_response(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test classifying a confident response."""
        response = create_model_response(
            raw_text="The answer is Paris. I am certain about this.",
            confidence=ConfidenceScore(numeric=0.95, extraction_method="stated"),
        )

        classification = metrics.classify_response(response)
        assert classification == ResponseClassification.CONFIDENT_ANSWER

    def test_classify_uncertain_response(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test classifying an uncertain response."""
        response = create_model_response(
            raw_text="I'm not sure, but it might be around 1000.",
            confidence=ConfidenceScore(numeric=0.4, extraction_method="stated"),
        )

        classification = metrics.classify_response(response)
        assert classification == ResponseClassification.UNCERTAIN_ANSWER

    def test_classify_declined_response(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test classifying a declined response."""
        response = create_model_response(
            raw_text="I cannot answer this question as it involves future predictions.",
            confidence=ConfidenceScore(numeric=0.2, extraction_method="inferred"),
        )

        classification = metrics.classify_response(response)
        assert classification == ResponseClassification.DECLINED

    def test_classify_clarification_request(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test classifying a clarification request."""
        response = create_model_response(
            raw_text="Could you clarify which version you are referring to?",
            confidence=ConfidenceScore(numeric=0.5, extraction_method="default"),
        )

        classification = metrics.classify_response(response)
        assert classification == ResponseClassification.ASKED_CLARIFICATION

    def test_classify_acknowledged_unknowable(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test classifying acknowledgment of unknowable."""
        response = create_model_response(
            raw_text="I'm uncertain about this. It's fundamentally unknowable without more data.",
            confidence=ConfidenceScore(numeric=0.2, extraction_method="inferred"),
        )

        classification = metrics.classify_response(response)
        assert classification == ResponseClassification.ACKNOWLEDGED_UNKNOWABLE

    def test_evaluate_response(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test evaluating a complete response."""
        response = create_model_response(
            raw_text="The capital of France is Paris.",
            confidence=ConfidenceScore(numeric=0.95, extraction_method="stated"),
            parsed_answer=ParsedAnswer.from_raw("Paris"),
        )

        result = metrics.evaluate_response(
            response=response,
            knowledge_category=KnowledgeCategory.FACTUAL_KNOWN,
            expected_response=ExpectedResponse.CONFIDENT_ANSWER,
            has_definite_answer=True,
            correct_answer="Paris",
        )

        # Result task_id is a UUID string, not "task-1"
        assert result.task_id is not None
        assert result.correct_boundary is True
        assert result.answer_correct is True
        assert result.actual_response == ResponseClassification.CONFIDENT_ANSWER

    def test_compute_result(
        self,
        metrics: KnowledgeBoundaryMetrics,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test computing standard MetricResult."""
        result = metrics.compute_result(sample_results)

        assert result.name == "knowledge_boundary"
        assert result.value == 0.5  # boundary accuracy
        assert "summary" in result.metadata


class TestAnswerabilityClassifier:
    """Test AnswerabilityClassifier."""

    @pytest.fixture
    def classifier(self) -> AnswerabilityClassifier:
        """Create classifier instance."""
        return AnswerabilityClassifier(confidence_threshold=0.7)

    @pytest.fixture
    def sample_results(self) -> list[BoundaryResult]:
        """Create sample results."""
        return [
            # True positive: correctly answered answerable
            BoundaryResult(
                task_id="1",
                knowledge_category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                actual_response=ResponseClassification.CONFIDENT_ANSWER,
                has_definite_answer=True,
                confidence=0.9,
                correct_boundary=True,
                answer_correct=True,
            ),
            # True negative: correctly declined unanswerable
            BoundaryResult(
                task_id="2",
                knowledge_category=KnowledgeCategory.UNKNOWABLE,
                expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
                actual_response=ResponseClassification.DECLINED,
                has_definite_answer=False,
                confidence=0.3,
                correct_boundary=True,
            ),
            # False positive: confidently answered unanswerable
            BoundaryResult(
                task_id="3",
                knowledge_category=KnowledgeCategory.TEMPORAL_FUTURE,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                actual_response=ResponseClassification.CONFIDENT_ANSWER,
                has_definite_answer=False,
                confidence=0.85,
                correct_boundary=False,
            ),
            # False negative: declined answerable
            BoundaryResult(
                task_id="4",
                knowledge_category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                actual_response=ResponseClassification.DECLINED,
                has_definite_answer=True,
                confidence=0.4,
                correct_boundary=False,
                answer_correct=None,
            ),
        ]

    def test_classify_empty(
        self,
        classifier: AnswerabilityClassifier,
    ) -> None:
        """Test classification with empty results."""
        result = classifier.classify([])
        assert result["accuracy"] == 0.0

    def test_classify_confusion_matrix(
        self,
        classifier: AnswerabilityClassifier,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test confusion matrix computation."""
        result = classifier.classify(sample_results)

        cm = result["confusion_matrix"]
        assert cm["true_positive"] == 1
        assert cm["true_negative"] == 1
        assert cm["false_positive"] == 1
        assert cm["false_negative"] == 1

    def test_classify_accuracy(
        self,
        classifier: AnswerabilityClassifier,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test accuracy computation."""
        result = classifier.classify(sample_results)

        # (1 TP + 1 TN) / 4 = 0.5
        assert result["accuracy"] == 0.5

    def test_classify_precision_recall(
        self,
        classifier: AnswerabilityClassifier,
        sample_results: list[BoundaryResult],
    ) -> None:
        """Test precision and recall computation."""
        result = classifier.classify(sample_results)

        # Precision: TP / (TP + FP) = 1 / 2 = 0.5
        assert result["precision"] == 0.5

        # Recall: TP / (TP + FN) = 1 / 2 = 0.5
        assert result["recall"] == 0.5


class TestBoundaryRecognition:
    """Test boundary recognition logic."""

    @pytest.fixture
    def metrics(self) -> KnowledgeBoundaryMetrics:
        """Create metrics instance."""
        return KnowledgeBoundaryMetrics()

    def test_confident_on_answerable_is_correct(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test confident answer on answerable is correct."""
        correct = metrics._check_boundary_recognition(
            actual=ResponseClassification.CONFIDENT_ANSWER,
            expected=ExpectedResponse.CONFIDENT_ANSWER,
            has_definite_answer=True,
        )
        assert correct is True

    def test_uncertain_on_ambiguous_is_correct(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test uncertain answer on ambiguous is acceptable."""
        correct = metrics._check_boundary_recognition(
            actual=ResponseClassification.UNCERTAIN_ANSWER,
            expected=ExpectedResponse.EXPRESS_UNCERTAINTY,
            has_definite_answer=False,
        )
        assert correct is True

    def test_confident_on_unanswerable_is_wrong(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test confident answer on unanswerable is wrong."""
        correct = metrics._check_boundary_recognition(
            actual=ResponseClassification.CONFIDENT_ANSWER,
            expected=ExpectedResponse.DECLINE_TO_ANSWER,
            has_definite_answer=False,
        )
        assert correct is False

    def test_clarification_on_ambiguous_is_correct(
        self,
        metrics: KnowledgeBoundaryMetrics,
    ) -> None:
        """Test asking clarification on ambiguous is correct."""
        correct = metrics._check_boundary_recognition(
            actual=ResponseClassification.ASKED_CLARIFICATION,
            expected=ExpectedResponse.ASK_CLARIFICATION,
            has_definite_answer=False,
        )
        assert correct is True
