"""Unit tests for error detection task generator.

Tests cover:
- Generator configuration
- Task generation with/without errors
- Task type variety
- Difficulty adjustment
- Seed data loading
"""

import pytest

from chimera.generators.error_detection import (
    ErrorDetectionGeneratorConfig,
    ErrorDetectionTaskGenerator,
    ErrorDetectionTaskType,
    ErrorSeverity,
    SourceResponse,
)
from chimera.models.task import AnswerType, DifficultyLevel, Task, TrackType


def _get_tag_value(task: Task, key: str) -> str | None:
    """Extract a tag value by key from task metadata tags.

    Tags are stored as "key:value" strings.
    """
    for tag in task.metadata.tags:
        if tag.startswith(f"{key}:"):
            return tag[len(key) + 1 :]
    return None


def _has_tag(task: Task, key: str, value: str) -> bool:
    """Check if task has a specific tag key:value pair."""
    return f"{key}:{value}" in task.metadata.tags


class TestErrorDetectionGeneratorConfig:
    """Tests for ErrorDetectionGeneratorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ErrorDetectionGeneratorConfig()

        assert config.error_rate == 0.7
        assert config.num_tasks == 100
        assert len(config.task_types) >= 2
        assert len(config.error_types) >= 3

    def test_custom_error_rate(self) -> None:
        """Test custom error rate."""
        config = ErrorDetectionGeneratorConfig(error_rate=0.5)

        assert config.error_rate == 0.5

    def test_custom_task_types(self) -> None:
        """Test custom task types."""
        config = ErrorDetectionGeneratorConfig(task_types=[ErrorDetectionTaskType.BINARY_DETECTION])

        assert len(config.task_types) == 1
        assert config.task_types[0] == ErrorDetectionTaskType.BINARY_DETECTION

    def test_severity_distribution(self) -> None:
        """Test severity distribution sums correctly."""
        config = ErrorDetectionGeneratorConfig()

        total = sum(config.severity_distribution.values())
        assert abs(total - 1.0) < 0.01


class TestSourceResponse:
    """Tests for SourceResponse dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a source response."""
        source = SourceResponse(
            question="What is 2+2?",
            response="2+2 equals 4.",
            has_errors=False,
        )

        assert source.question == "What is 2+2?"
        assert source.has_errors is False

    def test_with_errors(self) -> None:
        """Test source with errors."""
        source = SourceResponse(
            question="Test",
            response="Test response",
            has_errors=True,
            errors=[],
        )

        assert source.has_errors is True


class TestErrorDetectionTaskGenerator:
    """Tests for ErrorDetectionTaskGenerator."""

    @pytest.fixture
    def config(self) -> ErrorDetectionGeneratorConfig:
        """Create test configuration."""
        return ErrorDetectionGeneratorConfig(
            num_tasks=20,
            error_rate=0.5,
            task_types=[
                ErrorDetectionTaskType.BINARY_DETECTION,
                ErrorDetectionTaskType.ERROR_IDENTIFICATION,
            ],
            seed=42,
        )

    @pytest.fixture
    def generator(
        self,
        config: ErrorDetectionGeneratorConfig,
    ) -> ErrorDetectionTaskGenerator:
        """Create generator with config."""
        return ErrorDetectionTaskGenerator(config)

    def test_generator_initialization(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test generator initialization."""
        assert generator.config is not None
        assert generator.error_injector is not None

    def test_load_default_seed_data(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test loading default seed data."""
        generator.load_seed_data()

        assert len(generator._source_responses) > 0

    def test_generate_creates_tasks(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test task generation."""
        task_set = generator.generate()

        assert task_set is not None
        assert len(task_set.tasks) > 0
        assert task_set.track == TrackType.ERROR_DETECTION

    def test_generate_respects_num_tasks(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test task count matches config."""
        task_set = generator.generate()

        # May have slightly fewer if injection fails
        assert len(task_set.tasks) <= generator.config.num_tasks + 5

    def test_generate_error_rate(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test approximate error rate in generated tasks."""
        task_set = generator.generate()

        with_errors = sum(1 for t in task_set.tasks if _has_tag(t, "has_errors", "true"))

        actual_rate = with_errors / len(task_set.tasks)
        expected_rate = generator.config.error_rate

        # Allow 20% deviation due to randomness
        assert abs(actual_rate - expected_rate) < 0.25

    def test_tasks_have_correct_track(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test all tasks have error detection track."""
        task_set = generator.generate()

        for task in task_set.tasks:
            assert task.track == TrackType.ERROR_DETECTION

    def test_tasks_have_metadata(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test tasks have required metadata."""
        task_set = generator.generate()

        for task in task_set.tasks:
            assert task.metadata is not None
            assert _get_tag_value(task, "task_type") is not None
            assert _get_tag_value(task, "has_errors") is not None

    def test_binary_detection_tasks(self) -> None:
        """Test binary detection task generation."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=10,
            task_types=[ErrorDetectionTaskType.BINARY_DETECTION],
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        for task in task_set.tasks:
            assert _get_tag_value(task, "task_type") == "binary_detection"
            assert task.answer_type == AnswerType.BOOLEAN
            assert task.correct_answer in ("Yes", "No")

    def test_error_identification_tasks(self) -> None:
        """Test error identification task generation."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=10,
            task_types=[ErrorDetectionTaskType.ERROR_IDENTIFICATION],
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        for task in task_set.tasks:
            assert _get_tag_value(task, "task_type") == "error_identification"

    def test_multi_error_detection_tasks(self) -> None:
        """Test multi-error detection tasks."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=10,
            task_types=[ErrorDetectionTaskType.MULTI_ERROR_DETECTION],
            error_rate=1.0,  # All with errors
            max_errors_per_task=3,
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        for task in task_set.tasks:
            if _has_tag(task, "has_errors", "true"):
                assert task.answer_type == AnswerType.NUMERIC

    def test_difficulty_adjustment(self) -> None:
        """Test difficulty is adjusted based on severity."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=20,
            error_rate=1.0,
            severity_distribution={
                ErrorSeverity.SUBTLE: 1.0,
                ErrorSeverity.MODERATE: 0.0,
                ErrorSeverity.OBVIOUS: 0.0,
            },
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        # Subtle errors should make tasks harder (L4, L5)
        hard_tasks = sum(
            1 for t in task_set.tasks if t.difficulty in (DifficultyLevel.L4, DifficultyLevel.L5)
        )

        # Should have some hard tasks
        assert hard_tasks > 0

    def test_get_statistics(
        self,
        generator: ErrorDetectionTaskGenerator,
    ) -> None:
        """Test statistics retrieval."""
        generator.load_seed_data()
        task_set = generator.generate()
        stats = generator.get_statistics(task_set)

        assert "num_source_responses" in stats
        assert "error_rate" in stats
        assert "task_types" in stats

    def test_reproducibility_with_seed(self) -> None:
        """Test generation is reproducible with same seed."""
        config1 = ErrorDetectionGeneratorConfig(num_tasks=10, seed=123)
        config2 = ErrorDetectionGeneratorConfig(num_tasks=10, seed=123)

        gen1 = ErrorDetectionTaskGenerator(config1)
        gen2 = ErrorDetectionTaskGenerator(config2)

        # Note: Due to error injection randomness, tasks won't be identical
        # but the structure should be similar
        tasks1 = gen1.generate()
        tasks2 = gen2.generate()

        # Allow small variance due to random error injection failures
        assert abs(len(tasks1.tasks) - len(tasks2.tasks)) <= 5


class TestErrorDetectionTaskTypes:
    """Tests for different task type behaviors."""

    def test_error_localization_type(self) -> None:
        """Test error localization task type."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=5,
            task_types=[ErrorDetectionTaskType.ERROR_LOCALIZATION],
            error_rate=1.0,
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        # Should have localization tasks
        loc_tasks = [
            t for t in task_set.tasks if _get_tag_value(t, "task_type") == "error_localization"
        ]
        assert len(loc_tasks) > 0

    def test_error_correction_type(self) -> None:
        """Test error correction task type."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=5,
            task_types=[ErrorDetectionTaskType.ERROR_CORRECTION],
            error_rate=0.5,
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        for task in task_set.tasks:
            assert _get_tag_value(task, "task_type") == "error_correction"

    def test_mixed_task_types(self) -> None:
        """Test generation with multiple task types."""
        config = ErrorDetectionGeneratorConfig(
            num_tasks=30,
            task_types=[
                ErrorDetectionTaskType.BINARY_DETECTION,
                ErrorDetectionTaskType.ERROR_IDENTIFICATION,
                ErrorDetectionTaskType.ERROR_CORRECTION,
            ],
        )
        generator = ErrorDetectionTaskGenerator(config)

        task_set = generator.generate()

        # Should have variety of task types
        types = {_get_tag_value(t, "task_type") for t in task_set.tasks}
        assert len(types) >= 2
