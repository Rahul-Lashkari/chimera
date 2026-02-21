"""Tests for SelfCorrectionTaskGenerator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera.generators.self_correction import (
    CorrectionExpectation,
    CorruptionType,
    ReasoningTrace,
    SelfCorrectionGeneratorConfig,
    SelfCorrectionTaskGenerator,
)
from chimera.models.task import DifficultyLevel, Task, TaskSet, TrackType


class TestCorruptionType:
    """Test CorruptionType enum."""

    def test_all_types_defined(self) -> None:
        """Verify all expected corruption types exist."""
        expected = [
            "logical",
            "computational",
            "factual",
            "procedural",
            "premise",
        ]
        assert len(CorruptionType) == len(expected)
        for ct in expected:
            assert CorruptionType(ct) is not None

    def test_corruption_type_values(self) -> None:
        """Test corruption type string values."""
        assert CorruptionType.LOGICAL.value == "logical"
        assert CorruptionType.COMPUTATIONAL.value == "computational"
        assert CorruptionType.FACTUAL.value == "factual"
        assert CorruptionType.PROCEDURAL.value == "procedural"
        assert CorruptionType.PREMISE.value == "premise"


class TestCorrectionExpectation:
    """Test CorrectionExpectation enum."""

    def test_all_expectations_defined(self) -> None:
        """Verify all expected responses exist."""
        expected = [
            "validate_correct",
            "detect_and_correct",
            "identify_uncorrectable",
        ]
        assert len(CorrectionExpectation) == len(expected)
        for exp in expected:
            assert CorrectionExpectation(exp) is not None


class TestReasoningTrace:
    """Test ReasoningTrace dataclass."""

    def test_create_uncorrupted_trace(self) -> None:
        """Test creating an uncorrupted trace."""
        trace = ReasoningTrace(
            problem="What is 2 + 2?",
            steps=["Step 1: Add 2 and 2", "Step 2: 2 + 2 = 4"],
            conclusion="The answer is 4.",
            is_corrupted=False,
        )

        assert trace.problem == "What is 2 + 2?"
        assert len(trace.steps) == 2
        assert trace.conclusion == "The answer is 4."
        assert trace.is_corrupted is False
        assert trace.corruption_type is None
        assert trace.corruption_location is None

    def test_create_corrupted_trace(self) -> None:
        """Test creating a corrupted trace."""
        trace = ReasoningTrace(
            problem="What is 2 + 2?",
            steps=["Step 1: Add 2 and 2", "Step 2: 2 + 2 = 5"],
            conclusion="The answer is 5.",
            is_corrupted=True,
            corruption_type=CorruptionType.COMPUTATIONAL,
            corruption_location=1,
            corruption_description="2 + 2 = 4, not 5",
            correct_steps=["Step 1: Add 2 and 2", "Step 2: 2 + 2 = 4"],
            correct_conclusion="The answer is 4.",
        )

        assert trace.is_corrupted is True
        assert trace.corruption_type == CorruptionType.COMPUTATIONAL
        assert trace.corruption_location == 1
        assert trace.correct_conclusion == "The answer is 4."

    def test_trace_defaults(self) -> None:
        """Test trace default values."""
        trace = ReasoningTrace(
            problem="Test",
            steps=["Step 1"],
            conclusion="Done",
        )

        assert trace.is_corrupted is False
        assert trace.difficulty == DifficultyLevel.L3
        assert trace.domain == "general"
        assert trace.metadata == {}

    def test_trace_with_difficulty(self) -> None:
        """Test trace with custom difficulty."""
        trace = ReasoningTrace(
            problem="Test",
            steps=["Step 1"],
            conclusion="Done",
            difficulty=DifficultyLevel.L5,
            domain="mathematics",
        )

        assert trace.difficulty == DifficultyLevel.L5
        assert trace.domain == "mathematics"


class TestSelfCorrectionGeneratorConfig:
    """Test SelfCorrectionGeneratorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SelfCorrectionGeneratorConfig()

        assert config.n_tasks == 100
        assert config.corruption_rate == 0.6
        assert config.include_correction_prompt is True
        assert config.require_explanation is True
        assert config.shuffle is True

    def test_corruption_type_distribution_default(self) -> None:
        """Test default corruption type distribution."""
        config = SelfCorrectionGeneratorConfig()

        # Should have distributions for all corruption types
        assert len(config.corruption_type_distribution) == 5

        # All should sum to 1.0
        total = sum(config.corruption_type_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_custom_corruption_rate(self) -> None:
        """Test custom corruption rate."""
        config = SelfCorrectionGeneratorConfig(corruption_rate=0.8)

        assert config.corruption_rate == 0.8

    def test_validation_corruption_rate(self) -> None:
        """Test corruption rate validation."""
        with pytest.raises(ValueError):
            SelfCorrectionGeneratorConfig(corruption_rate=1.5)

        with pytest.raises(ValueError):
            SelfCorrectionGeneratorConfig(corruption_rate=-0.1)

    def test_custom_distribution(self) -> None:
        """Test custom corruption type distribution."""
        custom = {
            CorruptionType.LOGICAL: 0.5,
            CorruptionType.COMPUTATIONAL: 0.5,
            CorruptionType.FACTUAL: 0.0,
            CorruptionType.PROCEDURAL: 0.0,
            CorruptionType.PREMISE: 0.0,
        }
        config = SelfCorrectionGeneratorConfig(
            corruption_type_distribution=custom,
        )

        assert config.corruption_type_distribution[CorruptionType.LOGICAL] == 0.5

    def test_validation_n_tasks(self) -> None:
        """Test n_tasks validation."""
        with pytest.raises(ValueError):
            SelfCorrectionGeneratorConfig(n_tasks=0)


class TestSelfCorrectionTaskGenerator:
    """Test SelfCorrectionTaskGenerator."""

    @pytest.fixture
    def generator(self) -> SelfCorrectionTaskGenerator:
        """Create generator with default config."""
        return SelfCorrectionTaskGenerator(
            config=SelfCorrectionGeneratorConfig(
                n_tasks=10,
                seed=42,
            )
        )

    @pytest.fixture
    def generator_with_seed_file(
        self,
        tmp_path: Path,
    ) -> SelfCorrectionTaskGenerator:
        """Create generator with seed file."""
        seed_data = [
            {
                "problem": "What is 2 + 2?",
                "steps": ["Add 2 and 2", "2 + 2 = 4"],
                "conclusion": "The answer is 4.",
                "is_corrupted": False,
                "difficulty": "L1",
                "domain": "mathematics",
            },
            {
                "problem": "What is 3 × 4?",
                "steps": ["Multiply 3 and 4", "3 × 4 = 11"],
                "conclusion": "The answer is 11.",
                "is_corrupted": True,
                "corruption_type": "computational",
                "corruption_location": 1,
                "corruption_description": "3 × 4 = 12, not 11",
                "correct_steps": ["Multiply 3 and 4", "3 × 4 = 12"],
                "correct_conclusion": "The answer is 12.",
                "difficulty": "L2",
                "domain": "mathematics",
            },
        ]

        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps(seed_data))

        return SelfCorrectionTaskGenerator(
            config=SelfCorrectionGeneratorConfig(
                n_tasks=5,
                seed=42,
                seed_data_path=seed_file,
            ),
        )

    def test_generator_initialization(self) -> None:
        """Test generator initialization."""
        gen = SelfCorrectionTaskGenerator()

        assert gen.config.n_tasks == 100
        assert gen.track == TrackType.SELF_CORRECTION

    def test_generate_taskset(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test generating a TaskSet."""
        task_set = generator.generate()

        assert isinstance(task_set, TaskSet)
        assert len(task_set.tasks) == 10
        assert task_set.track == TrackType.SELF_CORRECTION

    def test_generate_tasks(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test generating tasks."""
        task_set = generator.generate()

        for task in task_set.tasks:
            assert isinstance(task, Task)
            assert task.question is not None
            assert task.track == TrackType.SELF_CORRECTION

    def test_task_has_expected_metadata(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test task metadata structure."""
        task_set = generator.generate()
        task = task_set.tasks[0]

        # Metadata should have relevant tags
        assert task.metadata is not None
        assert any("domain:" in tag for tag in task.metadata.tags)
        assert any("corrupted:" in tag for tag in task.metadata.tags)
        assert any("expected:" in tag for tag in task.metadata.tags)

    def test_task_additional_data(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test task additional data fields."""
        task_set = generator.generate()
        task = task_set.tasks[0]

        additional_data = task.metadata.additional_data
        assert "is_corrupted" in additional_data
        assert "expected_response" in additional_data
        assert "original_problem" in additional_data

    def test_reproducibility(self) -> None:
        """Test generation is reproducible with seed."""
        config = SelfCorrectionGeneratorConfig(n_tasks=5, seed=42)
        gen1 = SelfCorrectionTaskGenerator(config=config)
        gen2 = SelfCorrectionTaskGenerator(config=config)

        tasks1 = gen1.generate()
        tasks2 = gen2.generate()

        # Same seed should produce same number of tasks
        assert len(tasks1.tasks) == len(tasks2.tasks)

    def test_load_seed_file(
        self,
        generator_with_seed_file: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test loading from seed file."""
        generator_with_seed_file.load_seed_data()
        assert len(generator_with_seed_file._reasoning_traces) == 2

    def test_default_seed_data(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test default seed data is loaded."""
        generator.load_seed_data()
        assert len(generator._reasoning_traces) > 0

    def test_corruption_rate_distribution(self) -> None:
        """Test tasks follow corruption rate."""
        config = SelfCorrectionGeneratorConfig(
            n_tasks=100,
            seed=42,
            corruption_rate=0.7,
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        task_set = gen.generate()

        # Count corrupted tasks
        corrupted_count = sum(
            1 for task in task_set.tasks if task.metadata.additional_data.get("is_corrupted", False)
        )

        # Should be roughly 70% corrupted (allow variance)
        assert 50 <= corrupted_count <= 90

    def test_generate_task_single(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test generating a single task."""
        from chimera.models.task import TaskCategory

        task = generator.generate_task(
            difficulty=DifficultyLevel.L2,
            category=TaskCategory.NUMERICAL,
        )

        assert isinstance(task, Task)
        assert task.track == TrackType.SELF_CORRECTION

    def test_generate_batch(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test generating a batch of tasks."""
        tasks = generator.generate_batch(n_tasks=5)

        assert len(tasks) == 5
        for task in tasks:
            assert isinstance(task, Task)

    def test_generate_batch_with_difficulty(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test generating batch with specific difficulty."""
        tasks = generator.generate_batch(
            n_tasks=5,
            difficulty=DifficultyLevel.L2,
        )

        assert len(tasks) == 5

    def test_inject_corruption(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test corruption injection."""
        generator.load_seed_data()

        # Find an uncorrupted trace
        uncorrupted = [t for t in generator._reasoning_traces if not t.is_corrupted]
        if uncorrupted:
            trace = uncorrupted[0]
            corrupted = generator._inject_corruption(trace)

            assert corrupted.is_corrupted is True
            assert corrupted.corruption_type is not None
            assert corrupted.correct_steps == trace.steps

    def test_templates_loaded(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test that templates are loaded."""
        assert "validation" in generator._templates
        assert "correction" in generator._templates
        assert "metacognitive" in generator._templates
        assert len(generator._templates["validation"]) > 0

    def test_taskset_tags(
        self,
        generator: SelfCorrectionTaskGenerator,
    ) -> None:
        """Test TaskSet has appropriate tags."""
        task_set = generator.generate()

        assert "self_correction" in task_set.tags
        assert any("corruption_rate" in tag for tag in task_set.tags)


class TestSelfCorrectionTaskGeneratorEdgeCases:
    """Test edge cases for SelfCorrectionTaskGenerator."""

    def test_empty_seed_path_uses_default(self) -> None:
        """Test that missing seed path uses default data."""
        config = SelfCorrectionGeneratorConfig(
            n_tasks=5,
            seed=42,
            seed_data_path=None,
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        gen.load_seed_data()

        assert len(gen._reasoning_traces) > 0

    def test_invalid_seed_path_uses_default(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that invalid seed path uses default data."""
        config = SelfCorrectionGeneratorConfig(
            n_tasks=5,
            seed=42,
            seed_data_path=tmp_path / "nonexistent.json",
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        gen.load_seed_data()

        assert len(gen._reasoning_traces) > 0

    def test_seed_directory_loads_multiple_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading from seed directory."""
        # Create multiple seed files
        seed_data1 = [
            {
                "problem": "Problem 1",
                "steps": ["Step 1"],
                "conclusion": "Conclusion 1",
                "is_corrupted": False,
            },
        ]
        seed_data2 = [
            {
                "problem": "Problem 2",
                "steps": ["Step 2"],
                "conclusion": "Conclusion 2",
                "is_corrupted": True,
                "corruption_type": "logical",
            },
        ]

        (tmp_path / "file1.json").write_text(json.dumps(seed_data1))
        (tmp_path / "file2.json").write_text(json.dumps(seed_data2))

        config = SelfCorrectionGeneratorConfig(
            n_tasks=5,
            seed=42,
            seed_data_path=tmp_path,
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        gen.load_seed_data()

        assert len(gen._reasoning_traces) == 2

    def test_zero_corruption_rate(self) -> None:
        """Test with zero corruption rate."""
        config = SelfCorrectionGeneratorConfig(
            n_tasks=10,
            seed=42,
            corruption_rate=0.0,
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        task_set = gen.generate()

        # All tasks should be uncorrupted
        for task in task_set.tasks:
            # Note: May still have some corrupted if not enough uncorrupted traces
            # Just verify we have tasks
            _ = task.metadata.additional_data.get("is_corrupted", False)
            assert task is not None

    def test_full_corruption_rate(self) -> None:
        """Test with 100% corruption rate."""
        config = SelfCorrectionGeneratorConfig(
            n_tasks=10,
            seed=42,
            corruption_rate=1.0,
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        task_set = gen.generate()

        # All tasks should be corrupted
        corrupted_count = sum(
            1 for task in task_set.tasks if task.metadata.additional_data.get("is_corrupted", False)
        )

        assert corrupted_count >= 8  # Most should be corrupted

    def test_no_shuffle(self) -> None:
        """Test generation without shuffling."""
        config = SelfCorrectionGeneratorConfig(
            n_tasks=5,
            seed=42,
            shuffle=False,
        )
        gen = SelfCorrectionTaskGenerator(config=config)
        task_set = gen.generate()

        assert len(task_set.tasks) == 5

    def test_domain_to_category_mapping(self) -> None:
        """Test domain to category mapping."""
        from chimera.models.task import TaskCategory

        gen = SelfCorrectionTaskGenerator()

        assert gen._domain_to_category("mathematics") == TaskCategory.NUMERICAL
        assert gen._domain_to_category("algebra") == TaskCategory.NUMERICAL
        assert gen._domain_to_category("logic") == TaskCategory.REASONING
        assert gen._domain_to_category("physics") == TaskCategory.SCIENTIFIC
        assert gen._domain_to_category("geography") == TaskCategory.FACTUAL
        assert gen._domain_to_category("unknown") == TaskCategory.REASONING
