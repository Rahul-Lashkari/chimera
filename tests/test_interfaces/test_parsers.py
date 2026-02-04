"""Unit tests for CHIMERA response and confidence parsers.

Tests cover:
- ConfidenceParser: Percentage, fraction, decimal, verbal extraction
- ResponseParser: Answer extraction, reasoning, abstention detection
- Edge cases and error handling
"""

from uuid import uuid4

import pytest

from chimera.interfaces.parsers import (
    ConfidenceParser,
    ParserConfig,
    ResponseParser,
    parse_confidence,
    parse_response,
)
from chimera.models.response import ConfidenceLevel


class TestConfidenceParser:
    """Tests for ConfidenceParser."""

    @pytest.fixture
    def parser(self) -> ConfidenceParser:
        """Create a default confidence parser."""
        return ConfidenceParser()

    # ==================== Percentage Extraction ====================

    def test_parse_percentage_basic(self, parser: ConfidenceParser) -> None:
        """Test parsing basic percentage format."""
        result = parser.parse("I am 85% confident")
        assert abs(result.numeric - 0.85) < 0.01

    def test_parse_percentage_with_space(self, parser: ConfidenceParser) -> None:
        """Test parsing percentage with space before %."""
        result = parser.parse("Confidence: 90 %")
        assert abs(result.numeric - 0.90) < 0.01

    def test_parse_percentage_word(self, parser: ConfidenceParser) -> None:
        """Test parsing 'percent' word format."""
        result = parser.parse("I am 75 percent sure")
        assert abs(result.numeric - 0.75) < 0.01

    def test_parse_confidence_label(self, parser: ConfidenceParser) -> None:
        """Test parsing 'Confidence: X' format."""
        result = parser.parse("Answer: Paris\nConfidence: 95\nReasoning: Capital of France")
        assert abs(result.numeric - 0.95) < 0.01

    def test_parse_confidence_level_label(self, parser: ConfidenceParser) -> None:
        """Test parsing 'Confidence level: X' format."""
        result = parser.parse("Confidence level: 80")
        assert abs(result.numeric - 0.80) < 0.01

    def test_parse_100_percent(self, parser: ConfidenceParser) -> None:
        """Test parsing 100% confidence."""
        result = parser.parse("I am 100% certain")
        assert result.numeric == 1.0

    def test_parse_0_percent(self, parser: ConfidenceParser) -> None:
        """Test parsing 0% confidence."""
        result = parser.parse("My confidence is 0%")
        assert result.numeric == 0.0

    # ==================== Fraction Extraction ====================

    def test_parse_fraction_slash(self, parser: ConfidenceParser) -> None:
        """Test parsing X/Y format."""
        result = parser.parse("My confidence is 8/10")
        assert abs(result.numeric - 0.8) < 0.01

    def test_parse_fraction_out_of(self, parser: ConfidenceParser) -> None:
        """Test parsing 'X out of Y' format."""
        result = parser.parse("I'd rate my confidence 9 out of 10")
        assert abs(result.numeric - 0.9) < 0.01

    def test_parse_fraction_five_scale(self, parser: ConfidenceParser) -> None:
        """Test parsing X/5 format."""
        result = parser.parse("Confidence: 4/5")
        assert abs(result.numeric - 0.8) < 0.01

    # ==================== Verbal Expressions ====================

    def test_parse_verbal_very_confident(self, parser: ConfidenceParser) -> None:
        """Test parsing 'very confident' expression."""
        result = parser.parse("I am very confident in this answer")
        assert result.numeric >= 0.85
        assert result.verbal == ConfidenceLevel.VERY_HIGH

    def test_parse_verbal_certain(self, parser: ConfidenceParser) -> None:
        """Test parsing 'certain' expression."""
        result = parser.parse("I am certain the answer is Paris")
        assert result.numeric >= 0.85
        assert result.verbal == ConfidenceLevel.VERY_HIGH

    def test_parse_verbal_pretty_sure(self, parser: ConfidenceParser) -> None:
        """Test parsing 'pretty sure' expression."""
        result = parser.parse("I'm pretty sure it's correct")
        assert result.numeric >= 0.70
        assert result.verbal == ConfidenceLevel.HIGH

    def test_parse_verbal_think(self, parser: ConfidenceParser) -> None:
        """Test parsing 'I think' expression."""
        result = parser.parse("I think the answer is 42")
        assert result.numeric >= 0.50
        assert result.verbal == ConfidenceLevel.MEDIUM

    def test_parse_verbal_unsure(self, parser: ConfidenceParser) -> None:
        """Test parsing 'unsure' expression."""
        result = parser.parse("I'm unsure about this answer")
        assert result.numeric <= 0.40
        assert result.verbal == ConfidenceLevel.LOW

    def test_parse_verbal_dont_know(self, parser: ConfidenceParser) -> None:
        """Test parsing 'don't know' expression."""
        result = parser.parse("I don't know the answer")
        assert result.numeric <= 0.20
        assert result.verbal == ConfidenceLevel.VERY_LOW

    def test_parse_verbal_guessing(self, parser: ConfidenceParser) -> None:
        """Test parsing 'guessing' expression."""
        result = parser.parse("I'm just guessing here")
        assert result.numeric <= 0.25
        assert result.verbal == ConfidenceLevel.VERY_LOW

    # ==================== Priority and Edge Cases ====================

    def test_percentage_takes_priority_over_verbal(self, parser: ConfidenceParser) -> None:
        """Test that explicit percentage overrides verbal."""
        result = parser.parse("I'm not very confident, maybe 85%")
        # Percentage should take priority
        assert abs(result.numeric - 0.85) < 0.01

    def test_no_confidence_returns_default(self, parser: ConfidenceParser) -> None:
        """Test that missing confidence returns default."""
        result = parser.parse("The answer is Paris.")
        assert result.numeric == 0.5  # Default

    def test_custom_default_confidence(self) -> None:
        """Test custom default confidence value."""
        config = ParserConfig(default_confidence=0.3)
        parser = ConfidenceParser(config)
        result = parser.parse("The answer is 42")
        assert result.numeric == 0.3

    def test_raw_text_captured(self, parser: ConfidenceParser) -> None:
        """Test that raw text is captured."""
        result = parser.parse("Answer: Paris\nConfidence: 90%")
        assert result.raw_text is not None
        assert "90%" in result.raw_text or "Confidence" in result.raw_text

    def test_multiline_response(self, parser: ConfidenceParser) -> None:
        """Test parsing multiline response."""
        text = """
        Answer: Paris

        Confidence: 85%

        Reasoning: Paris is the capital of France.
        """
        result = parser.parse(text)
        assert abs(result.numeric - 0.85) < 0.01


class TestResponseParser:
    """Tests for ResponseParser."""

    @pytest.fixture
    def parser(self) -> ResponseParser:
        """Create a default response parser."""
        return ResponseParser()

    @pytest.fixture
    def task_id(self) -> str:
        """Create a sample task ID."""
        return str(uuid4())

    # ==================== Answer Extraction ====================

    def test_extract_answer_label_format(self, parser: ResponseParser) -> None:
        """Test extracting 'Answer: X' format."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: Paris\nConfidence: 90%",
        )
        assert result.parsed_answer.raw_answer == "Paris"

    def test_extract_answer_sentence_format(self, parser: ResponseParser) -> None:
        """Test extracting 'The answer is X' format."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="The answer is 42. I am 80% confident.",
        )
        assert "42" in result.parsed_answer.raw_answer

    def test_extract_answer_my_answer_format(self, parser: ResponseParser) -> None:
        """Test extracting 'My answer: X' format."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="My answer: Berlin",
        )
        assert result.parsed_answer.raw_answer == "Berlin"

    def test_answer_normalization(self, parser: ResponseParser) -> None:
        """Test that answers are normalized."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: PARIS",
        )
        assert result.parsed_answer.normalized == "paris"

    # ==================== Confidence Integration ====================

    def test_confidence_extracted(self, parser: ResponseParser) -> None:
        """Test that confidence is extracted."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: Paris\nConfidence: 85%",
        )
        assert abs(result.confidence.numeric - 0.85) < 0.01

    def test_verbal_confidence_extracted(self, parser: ResponseParser) -> None:
        """Test that verbal confidence is extracted."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="I'm very confident the answer is Paris.",
        )
        assert result.confidence.numeric >= 0.85

    # ==================== Reasoning Extraction ====================

    def test_reasoning_extracted(self, parser: ResponseParser) -> None:
        """Test that reasoning is extracted."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: Paris\nConfidence: 90%\nReasoning: Paris is the capital of France.",
        )
        assert result.reasoning is not None
        assert len(result.reasoning.steps) > 0
        assert "capital" in result.reasoning.steps[0].content.lower()

    def test_explanation_extracted(self, parser: ResponseParser) -> None:
        """Test that explanation is extracted."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: 42\nExplanation: This is the answer to everything.",
        )
        assert result.reasoning is not None

    def test_no_reasoning(self, parser: ResponseParser) -> None:
        """Test response without reasoning."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="The answer is Paris.",
        )
        # Reasoning may or may not be None depending on text content
        # Just verify parsing works
        assert result.parsed_answer is not None

    # ==================== Abstention Detection ====================

    def test_abstention_dont_know(self, parser: ResponseParser) -> None:
        """Test abstention with 'don't know'."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="I don't know the answer to this question.",
        )
        assert result.parsed_answer.is_abstention is True

    def test_abstention_cannot_answer(self, parser: ResponseParser) -> None:
        """Test abstention with 'cannot answer'."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="I cannot answer this question with certainty.",
        )
        assert result.parsed_answer.is_abstention is True

    def test_abstention_unsure(self, parser: ResponseParser) -> None:
        """Test abstention with 'unsure'."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="I'm unsure and cannot provide an answer.",
        )
        assert result.parsed_answer.is_abstention is True

    def test_abstention_insufficient_info(self, parser: ResponseParser) -> None:
        """Test abstention with 'insufficient information'."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="There is insufficient information to answer this.",
        )
        assert result.parsed_answer.is_abstention is True

    def test_no_abstention(self, parser: ResponseParser) -> None:
        """Test non-abstention response."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: Paris\nConfidence: 95%",
        )
        assert result.parsed_answer.is_abstention is False

    # ==================== Multiple Choice ====================

    def test_multiple_choice_letter(self, parser: ResponseParser) -> None:
        """Test multiple choice letter extraction."""
        result = parser.parse_multiple_choice(
            "The answer is B.",
            options=["A", "B", "C", "D"],
        )
        assert result == "B"

    def test_multiple_choice_answer_format(self, parser: ResponseParser) -> None:
        """Test multiple choice with 'Answer: X' format."""
        result = parser.parse_multiple_choice(
            "Answer: C",
            options=["A", "B", "C", "D"],
        )
        assert result == "C"

    def test_multiple_choice_standalone(self, parser: ResponseParser) -> None:
        """Test multiple choice with standalone letter."""
        result = parser.parse_multiple_choice(
            "A",
            options=["A", "B", "C", "D"],
        )
        assert result == "A"

    def test_multiple_choice_not_found(self, parser: ResponseParser) -> None:
        """Test multiple choice when no valid option found."""
        result = parser.parse_multiple_choice(
            "I think it might be either option.",
            options=["A", "B", "C", "D"],
        )
        assert result is None

    # ==================== Metadata Handling ====================

    def test_metadata_preserved(self, parser: ResponseParser) -> None:
        """Test that metadata is preserved in response."""
        from chimera.models.response import ResponseMetadata

        metadata = ResponseMetadata(
            model_name="test-model",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        result = parser.parse(
            task_id=uuid4(),
            raw_text="Answer: Paris",
            metadata=metadata,
        )
        assert result.metadata is not None
        assert result.metadata.model_name == "test-model"

    # ==================== Complex Responses ====================

    def test_complex_response(self, parser: ResponseParser) -> None:
        """Test parsing a complex multi-part response."""
        text = """
        Let me think about this question carefully.

        Answer: The capital of France is Paris.

        Confidence: 95%

        Reasoning: Paris has been the capital of France since the late 10th century.
        It is the largest city in France and serves as the country's political,
        economic, and cultural center.
        """
        result = parser.parse(task_id=uuid4(), raw_text=text)

        assert "Paris" in result.parsed_answer.raw_answer
        assert abs(result.confidence.numeric - 0.95) < 0.01
        assert result.reasoning is not None


class TestConvenienceFunctions:
    """Tests for convenience parsing functions."""

    def test_parse_confidence_function(self) -> None:
        """Test standalone parse_confidence function."""
        result = parse_confidence("I am 80% confident")
        assert abs(result.numeric - 0.80) < 0.01

    def test_parse_response_function(self) -> None:
        """Test standalone parse_response function."""
        result = parse_response(
            task_id=uuid4(),
            raw_text="Answer: 42\nConfidence: 75%",
        )
        assert result.parsed_answer is not None
        assert abs(result.confidence.numeric - 0.75) < 0.01


class TestEdgeCases:
    """Edge case tests for parsers."""

    @pytest.fixture
    def parser(self) -> ResponseParser:
        """Create a default response parser."""
        return ResponseParser()

    def test_empty_response(self, parser: ResponseParser) -> None:
        """Test parsing empty response."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="",
        )
        assert result.parsed_answer.raw_answer == ""

    def test_whitespace_only_response(self, parser: ResponseParser) -> None:
        """Test parsing whitespace-only response."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="   \n\n   ",
        )
        assert result.parsed_answer.raw_answer.strip() == ""

    def test_unicode_content(self, parser: ResponseParser) -> None:
        """Test parsing unicode content."""
        result = parser.parse(
            task_id=uuid4(),
            raw_text="答案是：巴黎\nConfidence: 90%",
        )
        assert result.confidence.numeric >= 0.85

    def test_very_long_response(self, parser: ResponseParser) -> None:
        """Test parsing very long response."""
        long_text = "Answer: Paris\n" + "This is additional context. " * 1000
        result = parser.parse(
            task_id=uuid4(),
            raw_text=long_text,
        )
        assert result.parsed_answer.raw_answer == "Paris"

    def test_malformed_percentage(self) -> None:
        """Test parsing malformed percentage."""
        parser = ConfidenceParser()
        # Should not crash, should return default
        result = parser.parse("Confidence: abc%")
        assert result.numeric == 0.5  # Default

    def test_percentage_out_of_range(self) -> None:
        """Test percentage outside valid range."""
        parser = ConfidenceParser()
        # 150% should not be accepted as valid
        result = parser.parse("I am 150% sure")
        # Should either cap at 1.0 or return default
        assert result.numeric <= 1.0
