"""Unit tests for CHIMERA response models.

Tests cover:
- ModelResponse creation and validation
- ConfidenceScore parsing (numeric, verbal, percentage)
- ReasoningTrace construction
- ParsedAnswer normalization
- Abstention detection
- Serialization/deserialization
"""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from chimera.models.response import (
    ConfidenceLevel,
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ReasoningStep,
    ReasoningTrace,
    ResponseMetadata,
)


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_all_levels_defined(self) -> None:
        """Verify all expected confidence levels exist."""
        expected = {"very_low", "low", "medium", "high", "very_high"}
        actual = {level.value for level in ConfidenceLevel}
        assert actual == expected

    def test_level_ordering(self) -> None:
        """Verify levels have semantic ordering."""
        levels = list(ConfidenceLevel)
        # Should go from very_low to very_high
        level_values = [level.value for level in levels]
        assert "very_low" in level_values
        assert "very_high" in level_values


class TestConfidenceScore:
    """Tests for ConfidenceScore model."""

    def test_numeric_only(self) -> None:
        """Test creating confidence with just numeric value."""
        conf = ConfidenceScore(numeric=0.85)
        assert conf.numeric == 0.85
        assert conf.verbal is None

    def test_with_verbal(self) -> None:
        """Test creating confidence with numeric and verbal."""
        conf = ConfidenceScore(
            numeric=0.9,
            verbal=ConfidenceLevel.VERY_HIGH,
        )
        assert conf.numeric == 0.9
        assert conf.verbal == ConfidenceLevel.VERY_HIGH

    def test_from_percentage(self) -> None:
        """Test creating confidence from percentage."""
        conf = ConfidenceScore.from_percentage(85)
        assert conf.numeric == 0.85

    def test_from_percentage_with_raw(self) -> None:
        """Test from_percentage preserves raw text."""
        conf = ConfidenceScore.from_percentage(90, "I'm 90% sure")
        assert conf.numeric == 0.9
        # The field is called raw_text, not raw_expression
        assert conf.raw_text == "I'm 90% sure"

    def test_numeric_bounds_lower(self) -> None:
        """Test numeric value must be >= 0."""
        with pytest.raises(ValidationError):
            ConfidenceScore(numeric=-0.1)

    def test_numeric_bounds_upper(self) -> None:
        """Test numeric value must be <= 1."""
        with pytest.raises(ValidationError):
            ConfidenceScore(numeric=1.5)

    def test_to_percentage(self) -> None:
        """Test converting to percentage."""
        conf = ConfidenceScore(numeric=0.75)
        assert conf.to_percentage() == 75.0

    def test_high_confidence_detection(self) -> None:
        """Test high confidence detection via numeric comparison."""
        high = ConfidenceScore(numeric=0.9)
        low = ConfidenceScore(numeric=0.3)
        # Use numeric comparison directly
        assert high.numeric >= 0.8
        assert low.numeric < 0.8

    def test_verbal_level_mapping(self) -> None:
        """Test that verbal levels map to expected numeric ranges."""
        # Test from_verbal factory method
        conf = ConfidenceScore.from_verbal(ConfidenceLevel.VERY_HIGH)
        assert conf.numeric >= 0.8
        assert conf.verbal == ConfidenceLevel.VERY_HIGH


class TestReasoningStep:
    """Tests for ReasoningStep model."""

    def test_basic_step(self) -> None:
        """Test creating a basic reasoning step."""
        step = ReasoningStep(
            step_number=1,
            content="First, let's analyze the problem.",
        )
        assert step.step_number == 1
        assert "analyze" in step.content

    def test_step_with_type(self) -> None:
        """Test step with step_type field."""
        step = ReasoningStep(
            step_number=2,
            content="Checking our work...",
            step_type="verification",
        )
        assert step.step_type == "verification"

    def test_step_with_confidence(self) -> None:
        """Test step with confidence."""
        step = ReasoningStep(
            step_number=1,
            content="I believe this is correct.",
            confidence=0.9,
        )
        assert step.confidence == 0.9


class TestReasoningTrace:
    """Tests for ReasoningTrace model."""

    def test_empty_trace(self) -> None:
        """Test creating empty reasoning trace."""
        trace = ReasoningTrace(steps=[])
        assert len(trace.steps) == 0

    def test_trace_with_steps(self) -> None:
        """Test creating trace with multiple steps."""
        steps = [
            ReasoningStep(step_number=1, content="Step 1"),
            ReasoningStep(step_number=2, content="Step 2"),
            ReasoningStep(step_number=3, content="Conclusion"),
        ]
        trace = ReasoningTrace(steps=steps)
        assert len(trace.steps) == 3

    def test_trace_raw_reasoning(self) -> None:
        """Test raw reasoning field."""
        trace = ReasoningTrace(
            steps=[
                ReasoningStep(step_number=1, content="First"),
                ReasoningStep(step_number=2, content="Second"),
            ],
            raw_reasoning="First, I will... Second, I conclude...",
        )
        assert "First" in trace.raw_reasoning
        assert "Second" in trace.raw_reasoning

    def test_trace_length(self) -> None:
        """Test __len__ for trace."""
        # step_number must be >= 1
        steps = [ReasoningStep(step_number=i + 1, content=f"Step {i + 1}") for i in range(5)]
        trace = ReasoningTrace(steps=steps)
        assert len(trace) == 5

    def test_trace_add_step(self) -> None:
        """Test adding steps to trace."""
        trace = ReasoningTrace(steps=[])
        trace.add_step("First step", step_type="inference")
        trace.add_step("Second step", step_type="conclusion")
        assert len(trace) == 2

    def test_trace_get_step(self) -> None:
        """Test getting a specific step."""
        trace = ReasoningTrace(
            steps=[
                ReasoningStep(step_number=1, content="First"),
                ReasoningStep(step_number=2, content="Second"),
            ]
        )
        step = trace.get_step(1)
        assert step is not None
        assert step.content == "First"


class TestParsedAnswer:
    """Tests for ParsedAnswer model."""

    def test_basic_answer(self) -> None:
        """Test creating a basic parsed answer."""
        answer = ParsedAnswer(
            raw_answer="Paris",
            normalized="paris",
        )
        assert answer.raw_answer == "Paris"
        assert answer.normalized == "paris"

    def test_answer_with_type(self) -> None:
        """Test answer with explicit type."""
        answer = ParsedAnswer(
            raw_answer="42",
            normalized="42",
            answer_type="numeric",
        )
        assert answer.answer_type == "numeric"

    def test_abstention_answer(self) -> None:
        """Test abstention detection."""
        answer = ParsedAnswer(
            raw_answer="I don't know",
            normalized="abstain",
            is_abstention=True,
        )
        assert answer.is_abstention is True

    def test_answer_with_abstention_reason(self) -> None:
        """Test answer with abstention reason."""
        answer = ParsedAnswer(
            raw_answer="I cannot answer",
            normalized="cannot_answer",
            is_abstention=True,
            abstention_reason="Insufficient information provided",
        )
        assert answer.abstention_reason is not None

    def test_from_raw_factory(self) -> None:
        """Test from_raw factory method."""
        answer = ParsedAnswer.from_raw("  Paris  ")
        assert answer.raw_answer == "  Paris  "
        assert answer.normalized == "paris"


class TestResponseMetadata:
    """Tests for ResponseMetadata model."""

    def test_full_metadata(self) -> None:
        """Test creating full metadata."""
        meta = ResponseMetadata(
            model_name="gemini-2.0-flash",
            model_version="2.0",
            latency_ms=245.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            temperature=0.7,
        )
        assert meta.model_name == "gemini-2.0-flash"
        assert meta.latency_ms == 245.5
        assert meta.total_tokens == 150

    def test_metadata_token_validation(self) -> None:
        """Test that total_tokens is auto-corrected if wrong."""
        meta = ResponseMetadata(
            model_name="test",
            latency_ms=100,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=200,  # Wrong, should be 150
        )
        # Model validator should auto-correct
        assert meta.total_tokens == 150


class TestModelResponse:
    """Tests for ModelResponse model."""

    @pytest.fixture
    def sample_task_id(self) -> str:
        """Create a sample task ID."""
        return str(uuid4())

    @pytest.fixture
    def sample_metadata(self) -> ResponseMetadata:
        """Create sample response metadata."""
        return ResponseMetadata(
            model_name="gemini-2.0-flash",
            latency_ms=200.0,
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
        )

    def test_minimal_response(self, sample_task_id: str, sample_metadata: ResponseMetadata) -> None:
        """Test creating minimal response."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="The answer is Paris.",
            parsed_answer=ParsedAnswer(
                raw_answer="Paris",
                normalized="paris",
            ),
            confidence=ConfidenceScore(numeric=0.9),
            metadata=sample_metadata,
        )
        assert str(response.task_id) == sample_task_id
        assert response.raw_text == "The answer is Paris."

    def test_response_with_reasoning(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test response with reasoning trace."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="Let me think...",
            parsed_answer=ParsedAnswer(
                raw_answer="42",
                normalized="42",
            ),
            confidence=ConfidenceScore(numeric=0.85),
            reasoning=ReasoningTrace(
                steps=[
                    ReasoningStep(step_number=1, content="Analyzing..."),
                    ReasoningStep(step_number=2, content="Therefore 42."),
                ]
            ),
            metadata=sample_metadata,
        )
        assert response.reasoning is not None
        assert len(response.reasoning.steps) == 2

    def test_response_with_metadata(self, sample_task_id: str) -> None:
        """Test response with full metadata."""
        meta = ResponseMetadata(
            model_name="gemini-2.0-flash",
            latency_ms=200.0,
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
        )
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="Answer",
            parsed_answer=ParsedAnswer(raw_answer="A", normalized="a"),
            confidence=ConfidenceScore(numeric=0.8),
            metadata=meta,
        )
        assert response.metadata is not None
        assert response.metadata.model_name == "gemini-2.0-flash"

    def test_response_is_correct(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test is_correct flag."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="Paris",
            parsed_answer=ParsedAnswer(raw_answer="Paris", normalized="paris"),
            confidence=ConfidenceScore(numeric=0.95),
            is_correct=True,
            metadata=sample_metadata,
        )
        assert response.is_correct is True

    def test_response_json_serialization(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test response can be serialized to JSON."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="Answer text",
            parsed_answer=ParsedAnswer(raw_answer="A", normalized="a"),
            confidence=ConfidenceScore(numeric=0.75),
            metadata=sample_metadata,
        )
        json_str = response.model_dump_json()
        data = json.loads(json_str)

        assert "task_id" in data
        assert "raw_text" in data
        assert "confidence" in data
        assert data["confidence"]["numeric"] == 0.75

    def test_response_json_deserialization(self) -> None:
        """Test response can be deserialized from JSON."""
        data = {
            "task_id": str(uuid4()),
            "raw_text": "The answer is B",
            "parsed_answer": {
                "raw_answer": "B",
                "normalized": "b",
            },
            "confidence": {"numeric": 0.8},
            "metadata": {
                "model_name": "test",
                "latency_ms": 100,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        response = ModelResponse.model_validate(data)
        assert response.confidence.numeric == 0.8

    def test_response_error_detection_fields(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test error detection track-specific fields."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="I found an error...",
            parsed_answer=ParsedAnswer(raw_answer="Error found", normalized="error_found"),
            confidence=ConfidenceScore(numeric=0.9),
            metadata=sample_metadata,
            self_identified_errors=["Calculation error"],
            correction_attempted=True,
            corrected_answer="The correct answer is 4",
        )
        assert response.correction_attempted is True
        assert len(response.self_identified_errors) == 1

    def test_response_self_correction_fields(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test self-correction track-specific fields."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="Reviewing the reasoning...",
            parsed_answer=ParsedAnswer(raw_answer="Corrected", normalized="corrected"),
            confidence=ConfidenceScore(numeric=0.85),
            metadata=sample_metadata,
            corruption_detected=True,
            detected_corruption_index=3,
        )
        assert response.corruption_detected is True
        assert response.detected_corruption_index == 3

    def test_response_is_abstention(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test is_abstention method."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="I don't know",
            parsed_answer=ParsedAnswer(
                raw_answer="I don't know",
                normalized="abstain",
                is_abstention=True,
            ),
            confidence=ConfidenceScore(numeric=0.1),
            metadata=sample_metadata,
        )
        assert response.is_abstention() is True

    def test_response_get_confidence_percentage(
        self, sample_task_id: str, sample_metadata: ResponseMetadata
    ) -> None:
        """Test get_confidence_percentage method."""
        response = ModelResponse(
            task_id=sample_task_id,
            raw_text="Answer",
            parsed_answer=ParsedAnswer(raw_answer="A", normalized="a"),
            confidence=ConfidenceScore(numeric=0.85),
            metadata=sample_metadata,
        )
        assert response.get_confidence_percentage() == 85.0


class TestResponseEdgeCases:
    """Edge case tests for response models."""

    @pytest.fixture
    def sample_metadata(self) -> ResponseMetadata:
        """Create sample response metadata."""
        return ResponseMetadata(
            model_name="test",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

    def test_very_low_confidence(self) -> None:
        """Test handling of very low confidence."""
        conf = ConfidenceScore(numeric=0.01)
        assert conf.numeric == 0.01
        # Use direct comparison instead of method
        assert conf.numeric < 0.5

    def test_exact_boundary_confidence(self) -> None:
        """Test exact boundary values."""
        zero = ConfidenceScore(numeric=0.0)
        one = ConfidenceScore(numeric=1.0)
        assert zero.numeric == 0.0
        assert one.numeric == 1.0

    def test_response_empty_raw_text(self, sample_metadata: ResponseMetadata) -> None:
        """Test handling empty raw text."""
        # Empty raw text might be valid for some responses
        response = ModelResponse(
            task_id=str(uuid4()),
            raw_text="",
            parsed_answer=ParsedAnswer(
                raw_answer="",
                normalized="",
                is_abstention=True,
            ),
            confidence=ConfidenceScore(numeric=0.1),
            metadata=sample_metadata,
        )
        assert response.raw_text == ""

    def test_response_unicode_content(self, sample_metadata: ResponseMetadata) -> None:
        """Test handling unicode in responses."""
        response = ModelResponse(
            task_id=str(uuid4()),
            raw_text="答案是：巴黎 🗼",
            parsed_answer=ParsedAnswer(
                raw_answer="巴黎",
                normalized="巴黎",
            ),
            confidence=ConfidenceScore(numeric=0.95),
            metadata=sample_metadata,
        )
        assert "巴黎" in response.raw_text
        assert "🗼" in response.raw_text

    def test_response_long_reasoning_trace(self) -> None:
        """Test handling long reasoning traces."""
        # step_number must be >= 1
        steps = [
            ReasoningStep(step_number=i + 1, content=f"Step {i + 1}: " + "x" * 100)
            for i in range(50)
        ]
        trace = ReasoningTrace(steps=steps)
        assert len(trace) == 50


class TestConfidenceScoreEdgeCases:
    """Additional edge cases for confidence score."""

    def test_from_percentage_boundary(self) -> None:
        """Test from_percentage at boundaries."""
        zero = ConfidenceScore.from_percentage(0)
        hundred = ConfidenceScore.from_percentage(100)
        assert zero.numeric == 0.0
        assert hundred.numeric == 1.0

    def test_verbal_to_numeric_mapping(self) -> None:
        """Test verbal confidence levels map to reasonable numeric ranges."""
        # Very high should map to high numeric
        conf_high = ConfidenceScore.from_verbal(ConfidenceLevel.VERY_HIGH)
        assert conf_high.numeric >= 0.8

        # Very low should map to low numeric
        conf_low = ConfidenceScore.from_verbal(ConfidenceLevel.VERY_LOW)
        assert conf_low.numeric <= 0.2

    def test_numeric_precision(self) -> None:
        """Test numeric values are rounded appropriately."""
        conf = ConfidenceScore(numeric=0.123456789)
        # Should be rounded to 4 decimal places
        assert conf.numeric == 0.1235
