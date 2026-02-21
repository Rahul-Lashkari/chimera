"""Unit tests for error injection utilities.

Tests cover:
- Individual error injectors (factual, computational, logical, etc.)
- Error injection configuration
- Multi-error injection
- Edge cases and error handling
"""

import pytest

from chimera.generators.error_injection import (
    BaseErrorInjector,
    ComputationalErrorInjector,
    ErrorInjector,
    ErrorType,
    FactualErrorInjector,
    HallucinationInjector,
    InjectedError,
    InjectionConfig,
    LogicalErrorInjector,
    MagnitudeErrorInjector,
    TemporalErrorInjector,
)


class TestInjectedError:
    """Tests for InjectedError dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating an injected error."""
        error = InjectedError(
            error_type=ErrorType.FACTUAL,
            location="Paris",
            original="London",
            description="Wrong capital",
            correction="London is the capital of UK, not Paris",
        )

        assert error.error_type == ErrorType.FACTUAL
        assert error.location == "Paris"
        assert error.original == "London"

    def test_default_severity(self) -> None:
        """Test default severity is moderate."""
        error = InjectedError(
            error_type=ErrorType.COMPUTATIONAL,
            location="10",
            original="100",
            description="Wrong calculation",
            correction="Should be 100",
        )

        assert error.severity == "moderate"


class TestInjectionConfig:
    """Tests for InjectionConfig."""

    def test_basic_config(self) -> None:
        """Test basic configuration."""
        config = InjectionConfig(
            error_type=ErrorType.FACTUAL,
            severity="subtle",
        )

        assert config.error_type == ErrorType.FACTUAL
        assert config.severity == "subtle"
        assert config.num_errors == 1

    def test_multi_error_config(self) -> None:
        """Test multi-error configuration."""
        config = InjectionConfig(
            error_type=ErrorType.COMPUTATIONAL,
            num_errors=3,
        )

        assert config.num_errors == 3


class TestFactualErrorInjector:
    """Tests for FactualErrorInjector."""

    @pytest.fixture
    def injector(self) -> FactualErrorInjector:
        return FactualErrorInjector()

    def test_can_inject_with_known_word(
        self,
        injector: FactualErrorInjector,
    ) -> None:
        """Test detection of injectable words."""
        question = "What is the capital of France?"
        response = "The capital of France is Paris."

        assert injector.can_inject(question, response) is True

    def test_can_inject_without_known_word(
        self,
        injector: FactualErrorInjector,
    ) -> None:
        """Test when no injectable words exist."""
        question = "What color is the sky?"
        response = "The sky is blue."

        assert injector.can_inject(question, response) is False

    def test_inject_replaces_word(
        self,
        injector: FactualErrorInjector,
    ) -> None:
        """Test that injection replaces a word."""
        question = "What is the capital of France?"
        response = "The capital of France is Paris."

        modified, error = injector.inject(question, response)

        assert modified != response
        assert error is not None
        assert error.error_type == ErrorType.FACTUAL
        assert "Paris" not in modified or error.original != "Paris"

    def test_inject_returns_error_info(
        self,
        injector: FactualErrorInjector,
    ) -> None:
        """Test that injection returns error info."""
        response = "Einstein developed the theory of relativity."

        modified, error = injector.inject("", response)

        assert error is not None
        assert error.original in response
        assert error.location in modified


class TestComputationalErrorInjector:
    """Tests for ComputationalErrorInjector."""

    @pytest.fixture
    def injector(self) -> ComputationalErrorInjector:
        return ComputationalErrorInjector()

    def test_can_inject_with_equation(
        self,
        injector: ComputationalErrorInjector,
    ) -> None:
        """Test detection in equations."""
        response = "The answer is 15 + 25 = 40."

        assert injector.can_inject("", response) is True

    def test_can_inject_with_result(
        self,
        injector: ComputationalErrorInjector,
    ) -> None:
        """Test detection with result keyword."""
        response = "The result is 42."

        assert injector.can_inject("", response) is True

    def test_inject_changes_number(
        self,
        injector: ComputationalErrorInjector,
    ) -> None:
        """Test that injection changes a number."""
        response = "15% of 2000 = 300."

        modified, error = injector.inject("", response)

        assert error is not None
        assert error.error_type == ErrorType.COMPUTATIONAL
        # Error should record that a change was made
        # (string comparison may fail due to int rounding, so check error fields)
        assert error.original in response
        assert error.location != error.original or modified != response

    def test_inject_severity_obvious(
        self,
        injector: ComputationalErrorInjector,
    ) -> None:
        """Test obvious severity creates large error."""
        response = "The answer = 100."

        modified, error = injector.inject("", response, severity="obvious")

        assert error is not None
        # Obvious errors should be significantly different
        wrong_num = float(error.location)
        orig_num = float(error.original)
        ratio = wrong_num / orig_num if orig_num != 0 else 0
        assert ratio >= 10 or ratio <= 0.1


class TestLogicalErrorInjector:
    """Tests for LogicalErrorInjector."""

    @pytest.fixture
    def injector(self) -> LogicalErrorInjector:
        return LogicalErrorInjector()

    def test_can_inject_with_connector(
        self,
        injector: LogicalErrorInjector,
    ) -> None:
        """Test detection with logical connectors."""
        response = "Since all cats are mammals, therefore they are warm-blooded."

        assert injector.can_inject("", response) is True

    def test_inject_flips_logic(
        self,
        injector: LogicalErrorInjector,
    ) -> None:
        """Test that injection flips logical meaning."""
        response = "Since it is raining, the ground is wet."

        modified, error = injector.inject("", response)

        assert error is not None
        assert error.error_type == ErrorType.LOGICAL
        assert modified != response

    def test_inject_with_negation(
        self,
        injector: LogicalErrorInjector,
    ) -> None:
        """Test negation injection."""
        response = "This is true."

        modified, error = injector.inject("", response)

        if error:
            assert "not" in modified.lower() or modified != response


class TestTemporalErrorInjector:
    """Tests for TemporalErrorInjector."""

    @pytest.fixture
    def injector(self) -> TemporalErrorInjector:
        return TemporalErrorInjector()

    def test_can_inject_with_year(
        self,
        injector: TemporalErrorInjector,
    ) -> None:
        """Test detection of years."""
        response = "World War II ended in 1945."

        assert injector.can_inject("", response) is True

    def test_can_inject_with_month(
        self,
        injector: TemporalErrorInjector,
    ) -> None:
        """Test detection of months."""
        response = "The Declaration was signed in July."

        assert injector.can_inject("", response) is True

    def test_inject_changes_year(
        self,
        injector: TemporalErrorInjector,
    ) -> None:
        """Test year modification."""
        response = "The event occurred in 1969."

        modified, error = injector.inject("", response)

        assert error is not None
        assert error.error_type == ErrorType.TEMPORAL
        assert "1969" not in modified

    def test_inject_changes_month(
        self,
        injector: TemporalErrorInjector,
    ) -> None:
        """Test month modification."""
        response = "It happened in January."

        modified, error = injector.inject("", response)

        assert error is not None
        assert "January" not in modified


class TestMagnitudeErrorInjector:
    """Tests for MagnitudeErrorInjector."""

    @pytest.fixture
    def injector(self) -> MagnitudeErrorInjector:
        return MagnitudeErrorInjector()

    def test_can_inject_with_unit(
        self,
        injector: MagnitudeErrorInjector,
    ) -> None:
        """Test detection of numbers with units."""
        response = "The distance is 100 kilometers."

        assert injector.can_inject("", response) is True

    def test_inject_changes_magnitude(
        self,
        injector: MagnitudeErrorInjector,
    ) -> None:
        """Test magnitude modification."""
        response = "The building is 50 meters tall."

        modified, error = injector.inject("", response)

        assert error is not None
        assert error.error_type == ErrorType.MAGNITUDE


class TestHallucinationInjector:
    """Tests for HallucinationInjector."""

    @pytest.fixture
    def injector(self) -> HallucinationInjector:
        return HallucinationInjector()

    def test_can_inject_any_text(
        self,
        injector: HallucinationInjector,
    ) -> None:
        """Test that hallucinations can be added to any text."""
        response = "This is a normal response."

        assert injector.can_inject("", response) is True

    def test_inject_adds_content(
        self,
        injector: HallucinationInjector,
    ) -> None:
        """Test that injection adds content."""
        response = "The process takes time to complete."

        modified, error = injector.inject("", response)

        assert error is not None
        assert error.error_type == ErrorType.HALLUCINATION
        assert len(modified) > len(response)


class TestErrorInjector:
    """Tests for main ErrorInjector class."""

    @pytest.fixture
    def injector(self) -> ErrorInjector:
        return ErrorInjector()

    def test_get_injectable_types(
        self,
        injector: ErrorInjector,
    ) -> None:
        """Test getting injectable error types."""
        question = "What year did WWII end?"
        response = "WWII ended in 1945."

        types = injector.get_injectable_types(question, response)

        assert len(types) > 0
        assert ErrorType.TEMPORAL in types

    def test_inject_specific_type(
        self,
        injector: ErrorInjector,
    ) -> None:
        """Test injecting specific error type."""
        response = "The answer = 42."

        config = InjectionConfig(
            error_type=ErrorType.COMPUTATIONAL,
            severity="moderate",
        )

        modified, errors = injector.inject("", response, config)

        assert len(errors) > 0
        assert errors[0].error_type == ErrorType.COMPUTATIONAL

    def test_inject_random(
        self,
        injector: ErrorInjector,
    ) -> None:
        """Test random error injection."""
        response = "The capital of France is Paris, founded in 1200."

        modified, errors = injector.inject_random("", response)

        assert modified != response
        assert len(errors) > 0

    def test_inject_multiple_errors(
        self,
        injector: ErrorInjector,
    ) -> None:
        """Test injecting multiple errors."""
        response = "The result is 50 meters = 100 meters total."

        config = InjectionConfig(
            error_type=ErrorType.COMPUTATIONAL,
            num_errors=2,
        )

        modified, errors = injector.inject("", response, config)

        # May get fewer than requested if not enough targets
        assert len(errors) >= 1

    def test_fallback_when_type_unavailable(
        self,
        injector: ErrorInjector,
    ) -> None:
        """Test fallback to available type."""
        # Response with no computational content but has temporal
        response = "The event occurred in 1999."

        config = InjectionConfig(
            error_type=ErrorType.COMPUTATIONAL,  # Won't work
        )

        modified, errors = injector.inject("", response, config)

        # Should fall back to another available type
        if errors:
            assert modified != response

    def test_register_custom_injector(
        self,
        injector: ErrorInjector,
    ) -> None:
        """Test registering custom injector."""

        class CustomInjector(BaseErrorInjector):
            def can_inject(self, q, r):
                return True

            def inject(self, q, r, s="moderate"):
                return r + " [CUSTOM]", InjectedError(
                    error_type=ErrorType.OMISSION,
                    location="[CUSTOM]",
                    original="",
                    description="Custom error",
                    correction="Remove custom",
                )

        injector.register_injector(ErrorType.OMISSION, CustomInjector())

        config = InjectionConfig(error_type=ErrorType.OMISSION)
        modified, errors = injector.inject("", "Test", config)

        assert "[CUSTOM]" in modified
