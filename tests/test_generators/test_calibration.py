"""Unit tests for CHIMERA calibration task generator.

Tests cover:
- CalibrationTaskGenerator creation and configuration
- Task generation (single and batch)
- Template-based generation
- Difficulty stratification
- Trick question generation
- JSONL export
"""

import json
import tempfile
from pathlib import Path

import pytest

from chimera.generators.calibration import (
    CalibrationGeneratorConfig,
    CalibrationTaskGenerator,
)
from chimera.generators.difficulty import DifficultyStratifier, StratificationConfig
from chimera.generators.templates import (
    QuestionTemplate,
    TemplateRegistry,
    create_default_calibration_templates,
)
from chimera.models.task import (
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskSet,
    TrackType,
)


class TestCalibrationGeneratorConfig:
    """Tests for CalibrationGeneratorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CalibrationGeneratorConfig()
        assert config.n_tasks == 100
        assert config.seed is None
        assert config.use_default_templates is True
        assert config.require_confidence_elicitation is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CalibrationGeneratorConfig(
            n_tasks=50,
            seed=42,
            use_default_templates=True,
            include_trick_questions=True,
            trick_question_ratio=0.2,
        )
        assert config.n_tasks == 50
        assert config.seed == 42
        assert config.trick_question_ratio == 0.2

    def test_difficulty_distribution(self) -> None:
        """Test difficulty distribution configuration."""
        config = CalibrationGeneratorConfig(
            n_tasks=100,
            difficulty_distribution={
                "L1": 0.20,
                "L2": 0.20,
                "L3": 0.20,
                "L4": 0.20,
                "L5": 0.20,
            },
        )
        counts = config.get_difficulty_counts()
        assert sum(counts.values()) == 100

    def test_category_distribution(self) -> None:
        """Test category distribution configuration."""
        config = CalibrationGeneratorConfig(
            n_tasks=100,
            category_distribution={
                "factual": 0.50,
                "reasoning": 0.30,
                "numerical": 0.20,
            },
        )
        counts = config.get_category_counts()
        assert sum(counts.values()) == 100


class TestCalibrationTaskGenerator:
    """Tests for CalibrationTaskGenerator."""

    @pytest.fixture
    def generator(self) -> CalibrationTaskGenerator:
        """Create a generator with fixed seed for reproducibility."""
        config = CalibrationGeneratorConfig(
            n_tasks=50,
            seed=42,
        )
        return CalibrationTaskGenerator(config)

    @pytest.fixture
    def minimal_generator(self) -> CalibrationTaskGenerator:
        """Create a minimal generator."""
        config = CalibrationGeneratorConfig(
            n_tasks=10,
            seed=123,
            require_confidence_elicitation=False,
        )
        return CalibrationTaskGenerator(config)

    def test_generator_creation(self, generator: CalibrationTaskGenerator) -> None:
        """Test generator initialization."""
        assert generator.track == TrackType.CALIBRATION
        assert generator.config.n_tasks == 50
        assert generator.config.seed == 42

    def test_generate_single_task(self, generator: CalibrationTaskGenerator) -> None:
        """Test generating a single task."""
        task = generator.generate_task(
            difficulty=DifficultyLevel.L2,
            category=TaskCategory.FACTUAL,
        )

        assert isinstance(task, Task)
        assert task.track == TrackType.CALIBRATION
        assert task.difficulty == DifficultyLevel.L2
        assert task.category == TaskCategory.FACTUAL
        assert task.question is not None
        assert len(task.question) > 0

    def test_generate_task_has_answer(self, generator: CalibrationTaskGenerator) -> None:
        """Test that generated tasks have correct answers."""
        task = generator.generate_task(
            difficulty=DifficultyLevel.L1,
            category=TaskCategory.NUMERICAL,
        )
        assert task.correct_answer is not None
        assert len(task.correct_answer) > 0

    def test_generate_task_includes_confidence_prompt(
        self, generator: CalibrationTaskGenerator
    ) -> None:
        """Test that confidence elicitation is added."""
        task = generator.generate_task(
            difficulty=DifficultyLevel.L1,
            category=TaskCategory.FACTUAL,
        )
        # Should contain confidence-related text
        assert any(
            keyword in task.question.lower()
            for keyword in ["confidence", "confident", "0%", "100%"]
        )

    def test_generate_task_without_confidence_prompt(
        self, minimal_generator: CalibrationTaskGenerator
    ) -> None:
        """Test generation without confidence elicitation."""
        task = minimal_generator.generate_task(
            difficulty=DifficultyLevel.L1,
            category=TaskCategory.FACTUAL,
        )
        # Original question without confidence prompt
        # (may or may not contain confidence keywords depending on template)
        assert task.question is not None

    def test_generate_batch(self, generator: CalibrationTaskGenerator) -> None:
        """Test generating a batch of tasks."""
        tasks = generator.generate_batch(n_tasks=10)

        assert len(tasks) == 10
        assert all(isinstance(t, Task) for t in tasks)
        assert all(t.track == TrackType.CALIBRATION for t in tasks)

    def test_generate_batch_with_difficulty(self, generator: CalibrationTaskGenerator) -> None:
        """Test generating batch with specific difficulty."""
        tasks = generator.generate_batch(
            n_tasks=5,
            difficulty=DifficultyLevel.L3,
        )

        assert len(tasks) == 5
        assert all(t.difficulty == DifficultyLevel.L3 for t in tasks)

    def test_generate_batch_with_category(self, generator: CalibrationTaskGenerator) -> None:
        """Test generating batch with specific category."""
        tasks = generator.generate_batch(
            n_tasks=5,
            category=TaskCategory.NUMERICAL,
        )

        assert len(tasks) == 5
        assert all(t.category == TaskCategory.NUMERICAL for t in tasks)

    def test_generate_all(self, generator: CalibrationTaskGenerator) -> None:
        """Test generating all tasks according to config."""
        taskset = generator.generate_all()

        assert isinstance(taskset, TaskSet)
        assert len(taskset) == generator.config.n_tasks

    def test_generate_all_difficulty_distribution(
        self, generator: CalibrationTaskGenerator
    ) -> None:
        """Test that generated tasks follow difficulty distribution."""
        taskset = generator.generate_all()

        # Count by difficulty
        counts: dict[str, int] = {}
        for task in taskset:
            # difficulty may be enum or string depending on Pydantic config
            level = (
                task.difficulty.value if hasattr(task.difficulty, "value") else str(task.difficulty)
            )
            counts[level] = counts.get(level, 0) + 1

        # Check that we have tasks at multiple difficulty levels
        assert len(counts) >= 3

    def test_generate_trick_question(self, generator: CalibrationTaskGenerator) -> None:
        """Test generating trick questions."""
        task = generator.generate_trick_question(DifficultyLevel.L3)

        assert isinstance(task, Task)
        assert task.track == TrackType.CALIBRATION
        assert task.metadata is not None
        assert "trick_question" in task.metadata.tags

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same results."""
        config1 = CalibrationGeneratorConfig(n_tasks=10, seed=999)
        config2 = CalibrationGeneratorConfig(n_tasks=10, seed=999)

        gen1 = CalibrationTaskGenerator(config1)
        gen2 = CalibrationTaskGenerator(config2)

        tasks1 = gen1.generate_batch(5)
        tasks2 = gen2.generate_batch(5)

        # Questions should be identical with same seed
        for t1, t2 in zip(tasks1, tasks2, strict=True):
            assert t1.question == t2.question
            assert t1.correct_answer == t2.correct_answer

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        config1 = CalibrationGeneratorConfig(n_tasks=10, seed=111)
        config2 = CalibrationGeneratorConfig(n_tasks=10, seed=222)

        gen1 = CalibrationTaskGenerator(config1)
        gen2 = CalibrationTaskGenerator(config2)

        tasks1 = gen1.generate_batch(10)
        tasks2 = gen2.generate_batch(10)

        # At least some questions should differ
        questions1 = [t.question for t in tasks1]
        questions2 = [t.question for t in tasks2]

        assert questions1 != questions2

    def test_save_tasks(self, generator: CalibrationTaskGenerator) -> None:
        """Test saving generated tasks to JSONL."""
        taskset = generator.generate_all()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tasks.jsonl"
            generator.save(taskset, output_path)

            assert output_path.exists()

            # Verify file contents
            with open(output_path) as f:
                lines = f.readlines()

            assert len(lines) == len(taskset)

            for line in lines:
                data = json.loads(line)
                assert "track" in data
                assert data["track"] == "calibration"

    def test_get_statistics(self, generator: CalibrationTaskGenerator) -> None:
        """Test getting generation statistics."""
        taskset = generator.generate_all()
        stats = generator.get_statistics(taskset)

        assert stats["total_tasks"] == len(taskset)
        assert stats["track"] == "calibration"
        assert "difficulty_distribution" in stats
        assert "category_distribution" in stats

    def test_get_coverage_report(self, generator: CalibrationTaskGenerator) -> None:
        """Test getting template coverage report."""
        report = generator.get_coverage_report()

        assert "total_templates" in report
        assert report["total_templates"] > 0
        assert "by_category" in report
        assert "by_difficulty" in report

    def test_validate_task(self, generator: CalibrationTaskGenerator) -> None:
        """Test task validation."""
        valid_task = generator.generate_task(
            difficulty=DifficultyLevel.L2,
            category=TaskCategory.FACTUAL,
        )
        errors = generator.validate_task(valid_task)
        assert len(errors) == 0

    def test_validate_invalid_task(self, generator: CalibrationTaskGenerator) -> None:
        """Test validation catches invalid tasks."""
        # Create a task with wrong track
        invalid_task = Task(
            track=TrackType.ERROR_DETECTION,  # Wrong track
            question="Test question",
            correct_answer="Test answer",
        )
        errors = generator.validate_task(invalid_task)
        assert len(errors) > 0
        assert any("track" in e.lower() for e in errors)


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    def test_empty_registry(self) -> None:
        """Test empty registry."""
        registry = TemplateRegistry()
        assert registry.count() == 0

    def test_register_template(self) -> None:
        """Test registering a template."""
        registry = TemplateRegistry()
        template = QuestionTemplate(
            template="What is {a} + {b}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            variables={"a": [1, 2, 3], "b": [1, 2, 3]},
            answer_func=lambda a, b: str(a + b),
        )
        registry.register(template)
        assert registry.count() == 1

    def test_get_template(self) -> None:
        """Test getting a template."""
        import random

        registry = TemplateRegistry()
        template = QuestionTemplate(
            template="What is {a} + {b}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            variables={"a": [1, 2, 3], "b": [1, 2, 3]},
            answer_func=lambda a, b: str(a + b),
        )
        registry.register(template)

        rng = random.Random(42)
        retrieved = registry.get_template(
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            rng=rng,
        )
        assert retrieved is not None
        assert retrieved.category == TaskCategory.NUMERICAL

    def test_get_nonexistent_template(self) -> None:
        """Test getting template for category with no templates."""
        import random

        registry = TemplateRegistry()
        rng = random.Random()

        result = registry.get_template(
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L1,
            rng=rng,
        )
        assert result is None

    def test_default_templates(self) -> None:
        """Test that default templates are created."""
        registry = create_default_calibration_templates()

        assert registry.count() > 0

        # Should have templates for multiple categories
        factual = registry.get_all_templates(category=TaskCategory.FACTUAL)
        math = registry.get_all_templates(category=TaskCategory.NUMERICAL)
        reasoning = registry.get_all_templates(category=TaskCategory.REASONING)

        assert len(factual) > 0
        assert len(math) > 0
        assert len(reasoning) > 0

    def test_template_generation(self) -> None:
        """Test generating from a template."""
        import random

        template = QuestionTemplate(
            template="What is {a} × {b}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            variables={"a": [5], "b": [3]},
            answer_func=lambda a, b: str(a * b),
        )

        rng = random.Random(42)
        question, answer = template.generate(rng)

        assert question == "What is 5 × 3?"
        assert answer == "15"


class TestDifficultyStratifier:
    """Tests for DifficultyStratifier."""

    def test_default_stratifier(self) -> None:
        """Test stratifier with default config."""
        stratifier = DifficultyStratifier()
        counts = stratifier.get_target_counts(100)

        assert sum(counts.values()) == 100
        # Should have counts for all difficulty levels
        assert len(counts) == 5

    def test_custom_distribution(self) -> None:
        """Test stratifier with custom distribution."""
        config = StratificationConfig(
            target_distribution={
                DifficultyLevel.L1: 0.5,
                DifficultyLevel.L2: 0.3,
                DifficultyLevel.L3: 0.2,
                DifficultyLevel.L4: 0.0,
                DifficultyLevel.L5: 0.0,
            }
        )
        stratifier = DifficultyStratifier(config)
        counts = stratifier.get_target_counts(100)

        assert counts[DifficultyLevel.L1] >= 45  # ~50%
        assert counts[DifficultyLevel.L2] >= 25  # ~30%

    def test_validate_distribution(self) -> None:
        """Test validating task distribution."""
        stratifier = DifficultyStratifier()

        # Create tasks with even distribution
        tasks = []
        for level in DifficultyLevel:
            for _ in range(20):
                tasks.append(
                    Task(
                        track=TrackType.CALIBRATION,
                        question="Test",
                        correct_answer="Test",
                        difficulty=level,
                    )
                )

        is_valid, details = stratifier.validate_distribution(tasks)

        assert "actual_distribution" in details
        assert "target_distribution" in details

    def test_suggest_adjustments(self) -> None:
        """Test suggesting adjustments."""
        stratifier = DifficultyStratifier()

        current_counts = {
            DifficultyLevel.L1: 10,
            DifficultyLevel.L2: 10,
            DifficultyLevel.L3: 10,
            DifficultyLevel.L4: 10,
            DifficultyLevel.L5: 10,
        }

        adjustments = stratifier.suggest_adjustments(current_counts, 100)

        # Should suggest adding more to some levels
        assert isinstance(adjustments, dict)
        assert sum(adjustments.values()) == 50  # Need 50 more total

    def test_get_difficulty_for_next_task(self) -> None:
        """Test getting recommended difficulty for next task."""
        stratifier = DifficultyStratifier()

        # Start with empty counts
        current_counts = dict.fromkeys(DifficultyLevel, 0)

        # First recommendation should be for the level with highest target
        recommended = stratifier.get_difficulty_for_next_task(current_counts, 100)

        assert isinstance(recommended, DifficultyLevel)


class TestQuestionTemplate:
    """Tests for QuestionTemplate."""

    def test_template_with_answer_template(self) -> None:
        """Test template with answer_template."""
        import random

        template = QuestionTemplate(
            template="What is the capital of {country}?",
            answer_template="{capital}",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L1,
            variables={
                "country": ["France"],
                "capital": ["Paris"],
            },
        )

        rng = random.Random(42)
        question, answer = template.generate(rng)

        assert "France" in question
        assert answer == "Paris"

    def test_template_with_answer_func(self) -> None:
        """Test template with answer function."""
        import random

        template = QuestionTemplate(
            template="What is {a} + {b}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            variables={"a": [7], "b": [8]},
            answer_func=lambda a, b: str(a + b),
        )

        rng = random.Random(42)
        question, answer = template.generate(rng)

        assert "7" in question
        assert "8" in question
        assert answer == "15"

    def test_template_with_tags(self) -> None:
        """Test template tags are preserved."""
        template = QuestionTemplate(
            template="Test",
            answer_template="Answer",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L1,
            variables={},
            tags=["tag1", "tag2"],
        )

        assert template.tags == ["tag1", "tag2"]
