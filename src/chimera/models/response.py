"""Response data models for CHIMERA benchmark.

This module defines models for capturing and analyzing model responses:
- ModelResponse: Complete response from an LLM including answer and confidence
- ConfidenceScore: Structured confidence representation
- ReasoningTrace: Step-by-step reasoning analysis
- ResponseMetadata: Timing, tokens, and other response metadata
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfidenceLevel(str, Enum):
    """Verbal confidence levels for qualitative analysis."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ConfidenceScore(BaseModel):
    """Structured confidence representation with multiple formats.

    Supports both numeric (0-100) and verbal (low/medium/high) confidence,
    enabling analysis across different elicitation strategies.

    Attributes:
        numeric: Numeric confidence score (0.0 to 1.0)
        verbal: Optional verbal confidence level
        raw_text: Raw confidence text from model response
        extraction_method: How confidence was extracted
    """

    model_config = ConfigDict(use_enum_values=True)

    numeric: float = Field(
        ge=0.0,
        le=1.0,
        description="Numeric confidence score (0.0 to 1.0)",
    )
    verbal: ConfidenceLevel | None = Field(
        default=None,
        description="Verbal confidence level",
    )
    raw_text: str | None = Field(
        default=None,
        description="Raw confidence text from model response",
    )
    extraction_method: str = Field(
        default="parsed",
        description="How confidence was extracted (parsed, inferred, default)",
    )

    @field_validator("numeric")
    @classmethod
    def validate_numeric_range(cls, v: float) -> float:
        """Ensure numeric confidence is in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Numeric confidence must be between 0.0 and 1.0")
        return round(v, 4)  # Limit precision

    @classmethod
    def from_percentage(cls, percentage: float, raw_text: str | None = None) -> "ConfidenceScore":
        """Create confidence score from percentage (0-100).

        Args:
            percentage: Confidence as percentage (0-100).
            raw_text: Optional raw text from model.

        Returns:
            ConfidenceScore with normalized numeric value.
        """
        numeric = percentage / 100.0
        verbal = cls._numeric_to_verbal(numeric)
        return cls(
            numeric=numeric,
            verbal=verbal,
            raw_text=raw_text,
            extraction_method="parsed",
        )

    @classmethod
    def from_verbal(cls, level: ConfidenceLevel) -> "ConfidenceScore":
        """Create confidence score from verbal level.

        Args:
            level: Verbal confidence level.

        Returns:
            ConfidenceScore with estimated numeric value.
        """
        verbal_to_numeric = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        return cls(
            numeric=verbal_to_numeric[level],
            verbal=level,
            extraction_method="inferred",
        )

    @staticmethod
    def _numeric_to_verbal(numeric: float) -> ConfidenceLevel:
        """Convert numeric confidence to verbal level."""
        if numeric < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif numeric < 0.4:
            return ConfidenceLevel.LOW
        elif numeric < 0.6:
            return ConfidenceLevel.MEDIUM
        elif numeric < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def to_percentage(self) -> float:
        """Convert to percentage (0-100)."""
        return self.numeric * 100.0


class ReasoningStep(BaseModel):
    """Individual step in a reasoning trace.

    Attributes:
        step_number: Order of this step in the reasoning chain
        content: The reasoning content for this step
        step_type: Type of reasoning (inference, calculation, etc.)
        confidence: Optional confidence for this specific step
    """

    step_number: int = Field(ge=1, description="Order of this step")
    content: str = Field(description="The reasoning content")
    step_type: str = Field(
        default="inference",
        description="Type of reasoning step",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence for this step",
    )


class ReasoningTrace(BaseModel):
    """Step-by-step reasoning analysis.

    Captures the model's reasoning process for analysis of
    error detection and self-correction capabilities.

    Attributes:
        steps: List of reasoning steps
        raw_reasoning: Raw reasoning text from model
        is_complete: Whether the reasoning chain is complete
        identified_uncertainties: Steps where model expressed uncertainty
    """

    steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="List of reasoning steps",
    )
    raw_reasoning: str | None = Field(
        default=None,
        description="Raw reasoning text from model",
    )
    is_complete: bool = Field(
        default=True,
        description="Whether the reasoning chain is complete",
    )
    identified_uncertainties: list[int] = Field(
        default_factory=list,
        description="Step numbers where model expressed uncertainty",
    )

    def __len__(self) -> int:
        """Return number of reasoning steps."""
        return len(self.steps)

    def add_step(
        self,
        content: str,
        step_type: str = "inference",
        confidence: float | None = None,
    ) -> None:
        """Add a reasoning step.

        Args:
            content: The reasoning content.
            step_type: Type of reasoning step.
            confidence: Optional confidence for this step.
        """
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            content=content,
            step_type=step_type,
            confidence=confidence,
        )
        self.steps.append(step)

    def get_step(self, step_number: int) -> ReasoningStep | None:
        """Get a specific reasoning step.

        Args:
            step_number: The step number (1-indexed).

        Returns:
            The reasoning step or None if not found.
        """
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None


class ParsedAnswer(BaseModel):
    """Parsed and normalized answer from model response.

    Attributes:
        raw_answer: The raw answer text from the model
        normalized: Normalized answer for comparison
        answer_type: Detected answer type
        is_abstention: Whether the model abstained from answering
        abstention_reason: Reason for abstention if applicable
    """

    raw_answer: str = Field(description="Raw answer text from model")
    normalized: str = Field(description="Normalized answer for comparison")
    answer_type: str = Field(
        default="text",
        description="Detected answer type",
    )
    is_abstention: bool = Field(
        default=False,
        description="Whether the model abstained",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Reason for abstention if applicable",
    )

    @classmethod
    def from_raw(cls, raw_answer: str) -> "ParsedAnswer":
        """Create ParsedAnswer from raw text.

        Args:
            raw_answer: Raw answer text from model.

        Returns:
            ParsedAnswer with normalized text.
        """
        normalized = raw_answer.strip().lower()

        # Detect abstention patterns
        abstention_patterns = [
            "i don't know",
            "i cannot answer",
            "i'm not sure",
            "i am not sure",
            "i cannot determine",
            "insufficient information",
            "cannot be answered",
            "unknown",
            "n/a",
        ]

        is_abstention = any(pattern in normalized for pattern in abstention_patterns)

        return cls(
            raw_answer=raw_answer,
            normalized=normalized,
            is_abstention=is_abstention,
            abstention_reason="Model expressed uncertainty" if is_abstention else None,
        )


class ResponseMetadata(BaseModel):
    """Metadata about the model response.

    Captures timing, token usage, and other response characteristics
    for analysis of model behavior.

    Attributes:
        model_name: Name of the model that generated the response
        model_version: Version of the model
        latency_ms: Response latency in milliseconds
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
        temperature: Temperature setting used
        timestamp: When the response was generated
        request_id: Unique identifier for the API request
    """

    model_name: str = Field(description="Name of the model")
    model_version: str | None = Field(
        default=None,
        description="Version of the model",
    )
    latency_ms: float = Field(
        ge=0,
        description="Response latency in milliseconds",
    )
    prompt_tokens: int = Field(
        ge=0,
        description="Number of tokens in the prompt",
    )
    completion_tokens: int = Field(
        ge=0,
        description="Number of tokens in the completion",
    )
    total_tokens: int = Field(
        ge=0,
        description="Total tokens used",
    )
    temperature: float = Field(
        ge=0.0,
        le=2.0,
        default=0.0,
        description="Temperature setting used",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the response was generated",
    )
    request_id: str | None = Field(
        default=None,
        description="Unique identifier for the API request",
    )

    @model_validator(mode="after")
    def validate_token_counts(self) -> "ResponseMetadata":
        """Ensure total tokens equals sum of prompt and completion."""
        expected_total = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected_total:
            # Auto-correct if not set properly
            self.total_tokens = expected_total
        return self


class ModelResponse(BaseModel):
    """Complete response from an LLM including answer and confidence.

    This is the primary data structure for capturing model outputs
    across all CHIMERA evaluation tracks.

    Attributes:
        id: Unique response identifier
        task_id: ID of the task this responds to
        raw_text: Complete raw response text
        parsed_answer: Parsed and normalized answer
        confidence: Structured confidence score
        reasoning: Optional reasoning trace
        metadata: Response metadata (timing, tokens, etc.)
        is_correct: Whether the answer is correct (set during evaluation)
        error_flags: Any errors or issues detected

    Example:
        >>> response = ModelResponse(
        ...     task_id=task.id,
        ...     raw_text="The capital of France is Paris. I'm 95% confident.",
        ...     parsed_answer=ParsedAnswer.from_raw("Paris"),
        ...     confidence=ConfidenceScore.from_percentage(95),
        ...     metadata=ResponseMetadata(
        ...         model_name="gemini-2.0-flash",
        ...         latency_ms=245,
        ...         prompt_tokens=50,
        ...         completion_tokens=20,
        ...         total_tokens=70,
        ...     ),
        ... )
    """

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique response identifier",
    )
    task_id: UUID = Field(description="ID of the task this responds to")
    raw_text: str = Field(description="Complete raw response text")
    parsed_answer: ParsedAnswer = Field(description="Parsed and normalized answer")
    confidence: ConfidenceScore = Field(description="Structured confidence score")
    reasoning: ReasoningTrace | None = Field(
        default=None,
        description="Optional reasoning trace",
    )
    metadata: ResponseMetadata = Field(description="Response metadata")
    is_correct: bool | None = Field(
        default=None,
        description="Whether the answer is correct (set during evaluation)",
    )
    error_flags: list[str] = Field(
        default_factory=list,
        description="Any errors or issues detected",
    )

    # Track-specific fields
    # For error_detection track
    self_identified_errors: list[str] = Field(
        default_factory=list,
        description="Errors the model identified in its own response",
    )
    correction_attempted: bool = Field(
        default=False,
        description="Whether the model attempted to correct errors",
    )
    corrected_answer: str | None = Field(
        default=None,
        description="Corrected answer if correction was attempted",
    )

    # For self_correction track
    corruption_detected: bool | None = Field(
        default=None,
        description="Whether the model detected reasoning corruption",
    )
    detected_corruption_index: int | None = Field(
        default=None,
        description="Index of corruption detected by model",
    )

    def is_abstention(self) -> bool:
        """Check if the response is an abstention.

        Returns:
            True if the model abstained from answering.
        """
        return self.parsed_answer.is_abstention

    def get_confidence_percentage(self) -> float:
        """Get confidence as a percentage.

        Returns:
            Confidence score as percentage (0-100).
        """
        return self.confidence.to_percentage()

    def mark_correct(self, is_correct: bool) -> None:
        """Mark whether the response is correct.

        Args:
            is_correct: Whether the answer is correct.
        """
        self.is_correct = is_correct

    def add_error_flag(self, error: str) -> None:
        """Add an error flag to the response.

        Args:
            error: Error description to add.
        """
        if error not in self.error_flags:
            self.error_flags.append(error)
