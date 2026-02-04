"""Response and confidence parsing utilities for CHIMERA.

This module provides parsers for extracting structured information
from raw model responses, including confidence levels and answers.
"""

import re
from uuid import UUID

from pydantic import BaseModel, Field

from chimera.models.response import (
    ConfidenceLevel,
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ReasoningStep,
    ReasoningTrace,
    ResponseMetadata,
)


class ParserConfig(BaseModel):
    """Configuration for response parsing.

    Attributes:
        default_confidence: Default confidence if none can be extracted
        confidence_keywords: Keywords that indicate confidence statements
        abstention_keywords: Keywords that indicate abstention
        normalize_answers: Whether to normalize extracted answers
    """

    default_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default confidence if none detected",
    )
    confidence_keywords: list[str] = Field(
        default_factory=lambda: [
            "confidence",
            "confident",
            "certain",
            "sure",
            "probability",
            "likely",
        ],
        description="Keywords indicating confidence statements",
    )
    abstention_keywords: list[str] = Field(
        default_factory=lambda: [
            "i don't know",
            "i do not know",
            "i'm not sure",
            "i am not sure",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "no answer",
            "unknown",
            "unsure",
            "i cannot determine",
            "insufficient information",
        ],
        description="Keywords indicating abstention",
    )
    normalize_answers: bool = Field(
        default=True,
        description="Whether to normalize extracted answers",
    )


class ConfidenceParser:
    """Parser for extracting confidence from model responses.

    This parser uses multiple strategies to extract confidence:
    1. Explicit percentage (e.g., "85%", "0.85", "85 percent")
    2. Verbal expressions (e.g., "very confident", "somewhat sure")
    3. Fraction format (e.g., "8/10", "9 out of 10")
    4. Confidence scale (e.g., "confidence: 4/5")

    Example:
        >>> parser = ConfidenceParser()
        >>> score = parser.parse("I am 85% confident the answer is Paris")
        >>> print(score.numeric)  # 0.85
    """

    # Verbal confidence mappings
    VERBAL_CONFIDENCE_MAP: dict[str, tuple[float, ConfidenceLevel]] = {
        # Very high confidence
        "absolutely certain": (0.99, ConfidenceLevel.VERY_HIGH),
        "100% certain": (1.0, ConfidenceLevel.VERY_HIGH),
        "completely certain": (0.98, ConfidenceLevel.VERY_HIGH),
        "extremely confident": (0.95, ConfidenceLevel.VERY_HIGH),
        "very confident": (0.90, ConfidenceLevel.VERY_HIGH),
        "highly confident": (0.90, ConfidenceLevel.VERY_HIGH),
        "certain": (0.90, ConfidenceLevel.VERY_HIGH),
        # High confidence
        "quite confident": (0.80, ConfidenceLevel.HIGH),
        "fairly confident": (0.75, ConfidenceLevel.HIGH),
        "confident": (0.75, ConfidenceLevel.HIGH),
        "pretty sure": (0.75, ConfidenceLevel.HIGH),
        "reasonably confident": (0.70, ConfidenceLevel.HIGH),
        "reasonably certain": (0.70, ConfidenceLevel.HIGH),
        # Medium confidence
        "somewhat confident": (0.60, ConfidenceLevel.MEDIUM),
        "moderately confident": (0.55, ConfidenceLevel.MEDIUM),
        "think": (0.55, ConfidenceLevel.MEDIUM),
        "believe": (0.55, ConfidenceLevel.MEDIUM),
        "probably": (0.60, ConfidenceLevel.MEDIUM),
        "likely": (0.60, ConfidenceLevel.MEDIUM),
        # Low confidence
        "not very confident": (0.35, ConfidenceLevel.LOW),
        "not entirely sure": (0.40, ConfidenceLevel.LOW),
        "somewhat unsure": (0.35, ConfidenceLevel.LOW),
        "uncertain": (0.30, ConfidenceLevel.LOW),
        "unsure": (0.30, ConfidenceLevel.LOW),
        # Very low confidence
        "very unsure": (0.15, ConfidenceLevel.VERY_LOW),
        "not sure at all": (0.10, ConfidenceLevel.VERY_LOW),
        "don't know": (0.10, ConfidenceLevel.VERY_LOW),
        "do not know": (0.10, ConfidenceLevel.VERY_LOW),
        "no idea": (0.05, ConfidenceLevel.VERY_LOW),
        "guessing": (0.20, ConfidenceLevel.VERY_LOW),
        "just a guess": (0.15, ConfidenceLevel.VERY_LOW),
    }

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize the confidence parser.

        Args:
            config: Parser configuration. Uses defaults if None.
        """
        self.config = config or ParserConfig()

    def parse(self, text: str) -> ConfidenceScore:
        """Parse confidence from text.

        Tries multiple extraction strategies in order of preference.

        Args:
            text: The text to parse.

        Returns:
            Extracted ConfidenceScore.
        """
        text_lower = text.lower()

        # Strategy 1: Look for fraction format (8/10, 9 out of 10)
        # Check fractions first as they're more specific patterns
        fraction = self._extract_fraction(text)
        if fraction is not None:
            return ConfidenceScore(
                numeric=fraction,
                raw_text=self._find_confidence_expression(text),
            )

        # Strategy 2: Look for explicit percentage
        percentage = self._extract_percentage(text)
        if percentage is not None:
            return ConfidenceScore(
                numeric=percentage / 100.0,
                raw_text=self._find_confidence_expression(text),
            )

        # Strategy 3: Look for decimal confidence (0.85)
        decimal = self._extract_decimal(text)
        if decimal is not None:
            return ConfidenceScore(
                numeric=decimal,
                raw_text=self._find_confidence_expression(text),
            )

        # Strategy 4: Look for verbal expressions
        verbal_result = self._extract_verbal(text_lower)
        if verbal_result is not None:
            numeric, level = verbal_result
            return ConfidenceScore(
                numeric=numeric,
                verbal=level,
                raw_text=self._find_confidence_expression(text),
            )

        # Default confidence
        return ConfidenceScore(
            numeric=self.config.default_confidence,
            raw_text=None,
        )

    def _extract_percentage(self, text: str) -> float | None:
        """Extract percentage from text.

        Handles formats like:
        - 85%
        - 85 percent
        - 85 per cent
        - confidence: 85%
        - confidence level: 85

        Args:
            text: Text to parse.

        Returns:
            Percentage value (0-100) or None.
        """
        patterns = [
            # "85%" or "85 %"
            r"(\d{1,3})\s*%",
            # "85 percent" or "85 per cent"
            r"(\d{1,3})\s*(?:percent|per\s*cent)",
            # "confidence: 85" or "confidence level: 85"
            r"confidence(?:\s+level)?[:\s]+(\d{1,3})",
            # After "Confidence:" on its own line
            r"(?:^|\n)\s*confidence[:\s]+(\d{1,3})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if 0 <= value <= 100:
                    return value

        return None

    def _extract_fraction(self, text: str) -> float | None:
        """Extract fraction confidence from text.

        Handles formats like:
        - 8/10
        - 9 out of 10
        - 4/5 confident

        Args:
            text: Text to parse.

        Returns:
            Numeric confidence (0-1) or None.
        """
        patterns = [
            # "8/10" or "8 / 10"
            r"(\d+)\s*/\s*(\d+)",
            # "8 out of 10"
            r"(\d+)\s*out\s*of\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                if denominator > 0:
                    value = numerator / denominator
                    if 0 <= value <= 1:
                        return value

        return None

    def _extract_decimal(self, text: str) -> float | None:
        """Extract decimal confidence from text.

        Handles formats like:
        - 0.85
        - confidence: 0.9

        Args:
            text: Text to parse.

        Returns:
            Numeric confidence (0-1) or None.
        """
        # Look for decimal in confidence context
        pattern = r"confidence[:\s]+0\.(\d+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(f"0.{match.group(1)}")
            if 0 <= value <= 1:
                return value

        return None

    def _extract_verbal(self, text: str) -> tuple[float, ConfidenceLevel] | None:
        """Extract verbal confidence from text.

        Args:
            text: Lowercase text to parse.

        Returns:
            Tuple of (numeric, level) or None.
        """
        # Check for verbal expressions (longest matches first)
        sorted_expressions = sorted(
            self.VERBAL_CONFIDENCE_MAP.keys(),
            key=len,
            reverse=True,
        )

        for expression in sorted_expressions:
            if expression in text:
                return self.VERBAL_CONFIDENCE_MAP[expression]

        return None

    def _find_confidence_expression(self, text: str) -> str | None:
        """Find the raw confidence expression in text.

        Args:
            text: Text to search.

        Returns:
            The found expression or None.
        """
        # Look for lines containing confidence
        for line in text.split("\n"):
            if any(kw in line.lower() for kw in self.config.confidence_keywords):
                return line.strip()
        return None


class ResponseParser:
    """Parser for extracting structured responses from model output.

    This parser extracts:
    - The answer portion of the response
    - Confidence level
    - Reasoning/explanation
    - Abstention detection

    Example:
        >>> parser = ResponseParser()
        >>> response = parser.parse(
        ...     task_id=uuid4(),
        ...     raw_text="Answer: Paris\\nConfidence: 90%\\nReasoning: France's capital",
        ... )
        >>> print(response.parsed_answer.normalized)  # "paris"
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize the response parser.

        Args:
            config: Parser configuration. Uses defaults if None.
        """
        self.config = config or ParserConfig()
        self.confidence_parser = ConfidenceParser(config)

    def parse(
        self,
        task_id: UUID,
        raw_text: str,
        metadata: ResponseMetadata | None = None,
    ) -> ModelResponse:
        """Parse a raw model response into structured format.

        Args:
            task_id: ID of the task this response is for.
            raw_text: Raw text from the model.
            metadata: Optional response metadata.

        Returns:
            Parsed ModelResponse.
        """
        # Extract components
        answer = self._extract_answer(raw_text)
        confidence = self.confidence_parser.parse(raw_text)
        reasoning = self._extract_reasoning(raw_text)
        is_abstention = self._detect_abstention(raw_text)

        # Normalize answer
        normalized = self._normalize_answer(answer) if self.config.normalize_answers else answer

        parsed_answer = ParsedAnswer(
            raw_answer=answer,
            normalized=normalized,
            is_abstention=is_abstention,
        )

        # Use provided metadata or create a default one
        response_metadata = metadata or ResponseMetadata(
            model_name="unknown",
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        return ModelResponse(
            task_id=task_id,
            raw_text=raw_text,
            parsed_answer=parsed_answer,
            confidence=confidence,
            reasoning=reasoning,
            metadata=response_metadata,
        )

    def _extract_answer(self, text: str) -> str:
        """Extract the answer portion from text.

        Looks for common answer formats:
        - "Answer: X"
        - "The answer is X"
        - "A: X"

        Args:
            text: Text to parse.

        Returns:
            Extracted answer string.
        """
        patterns = [
            # "Answer: X" format
            r"(?:^|\n)\s*answer[:\s]+(.+?)(?:\n|$)",
            # "The answer is X" format
            r"the\s+answer\s+is[:\s]+(.+?)(?:\.|,|\n|$)",
            # "A: X" format
            r"(?:^|\n)\s*a[:\s]+(.+?)(?:\n|$)",
            # "My answer: X" format
            r"my\s+answer[:\s]+(.+?)(?:\.|,|\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Clean up common artifacts
                answer = re.sub(r"\s*\[.*?\]", "", answer)  # Remove bracketed content
                answer = re.sub(
                    r"\s*\(.*?\)", "", answer
                )  # Remove parenthetical content (optional)
                return answer.strip()

        # Fallback: return first sentence if no pattern matches
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if sentences:
            return sentences[0].strip()

        return text.strip()

    def _extract_reasoning(self, text: str) -> ReasoningTrace | None:
        """Extract reasoning/explanation from text.

        Args:
            text: Text to parse.

        Returns:
            ReasoningTrace or None.
        """
        patterns = [
            # "Reasoning: X" format
            r"(?:^|\n)\s*reasoning[:\s]+(.+?)(?:\n\n|$)",
            # "Explanation: X" format
            r"(?:^|\n)\s*explanation[:\s]+(.+?)(?:\n\n|$)",
            # "Because: X" format
            r"(?:^|\n)\s*because[:\s]+(.+?)(?:\n\n|$)",
            # "This is because X" format
            r"this\s+is\s+because\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning_text = match.group(1).strip()
                if reasoning_text:
                    steps = [
                        ReasoningStep(
                            step_number=1,
                            content=reasoning_text,
                        )
                    ]
                    return ReasoningTrace(steps=steps)

        return None

    def _detect_abstention(self, text: str) -> bool:
        """Detect if the response is an abstention.

        Args:
            text: Text to check.

        Returns:
            True if abstention detected.
        """
        text_lower = text.lower()

        return any(keyword in text_lower for keyword in self.config.abstention_keywords)

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison.

        Normalization includes:
        - Lowercase
        - Strip whitespace
        - Remove punctuation
        - Normalize numbers

        Args:
            answer: Answer to normalize.

        Returns:
            Normalized answer.
        """
        normalized = answer.lower().strip()

        # Remove common punctuation
        normalized = re.sub(r"[.,!?;:]$", "", normalized)

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized

    def parse_multiple_choice(
        self,
        text: str,
        options: list[str],
    ) -> str | None:
        """Parse a multiple choice response.

        Args:
            text: Text to parse.
            options: List of valid options (e.g., ["A", "B", "C", "D"]).

        Returns:
            Selected option or None if not found.
        """
        text_upper = text.upper()

        # Look for standalone letter
        for option in options:
            patterns = [
                rf"(?:^|\s){option}(?:\s|$|\.|\))",  # " A " or "A." or "A)"
                rf"answer[:\s]+{option}(?:\s|$|\.)",  # "Answer: A"
                rf"(?:^|\n)\s*{option}(?:\s|$|\.)",  # Line starting with A
            ]
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    return option

        return None


# Convenience functions


def parse_confidence(text: str) -> ConfidenceScore:
    """Parse confidence from text using default settings.

    Args:
        text: Text to parse.

    Returns:
        Extracted ConfidenceScore.
    """
    parser = ConfidenceParser()
    return parser.parse(text)


def parse_response(
    task_id: UUID,
    raw_text: str,
    metadata: ResponseMetadata | None = None,
) -> ModelResponse:
    """Parse a model response using default settings.

    Args:
        task_id: Task ID.
        raw_text: Raw model output.
        metadata: Optional metadata.

    Returns:
        Parsed ModelResponse.
    """
    parser = ResponseParser()
    return parser.parse(task_id, raw_text, metadata)
