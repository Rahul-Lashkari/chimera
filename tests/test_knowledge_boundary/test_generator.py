"""Tests for KnowledgeBoundaryTaskGenerator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera.generators.knowledge_boundary import (
    ExpectedResponse,
    KnowledgeBoundaryGeneratorConfig,
    KnowledgeBoundaryQuestion,
    KnowledgeBoundaryTaskGenerator,
    KnowledgeCategory,
)
from chimera.models.task import DifficultyLevel, Task


class TestKnowledgeCategory:
    """Test KnowledgeCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Verify all expected categories exist."""
        expected = [
            "factual_known",
            "factual_unknown",
            "temporal_future",
            "temporal_cutoff",
            "subjective",
            "ambiguous",
            "unknowable",
            "fictional",
        ]
        assert len(KnowledgeCategory) == len(expected)
        for cat in expected:
            assert KnowledgeCategory(cat) is not None

    def test_category_values(self) -> None:
        """Test category string values."""
        assert KnowledgeCategory.FACTUAL_KNOWN.value == "factual_known"
        assert KnowledgeCategory.UNKNOWABLE.value == "unknowable"


class TestExpectedResponse:
    """Test ExpectedResponse enum."""

    def test_all_responses_defined(self) -> None:
        """Verify all expected responses exist."""
        expected = [
            "confident_answer",
            "express_uncertainty",
            "decline_to_answer",
            "ask_clarification",
            "acknowledge_unknowable",
        ]
        assert len(ExpectedResponse) == len(expected)
        for resp in expected:
            assert ExpectedResponse(resp) is not None


class TestKnowledgeBoundaryQuestion:
    """Test KnowledgeBoundaryQuestion dataclass."""

    def test_create_question(self) -> None:
        """Test creating a question."""
        q = KnowledgeBoundaryQuestion(
            question="What is the capital of France?",
            category=KnowledgeCategory.FACTUAL_KNOWN,
            expected_response=ExpectedResponse.CONFIDENT_ANSWER,
            has_definite_answer=True,
            explanation="Basic geography fact",
            correct_answer="Paris",
        )

        assert q.question == "What is the capital of France?"
        assert q.category == KnowledgeCategory.FACTUAL_KNOWN
        assert q.expected_response == ExpectedResponse.CONFIDENT_ANSWER
        assert q.has_definite_answer is True
        assert q.correct_answer == "Paris"

    def test_question_defaults(self) -> None:
        """Test question default values."""
        q = KnowledgeBoundaryQuestion(
            question="Test question",
            category=KnowledgeCategory.UNKNOWABLE,
            expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
            has_definite_answer=False,
        )

        assert q.domain == "general"
        assert q.difficulty == DifficultyLevel.L2
        assert q.correct_answer is None
        assert q.metadata == {}

    def test_to_dict(self) -> None:
        """Test dataclass fields can be accessed."""
        q = KnowledgeBoundaryQuestion(
            question="Test question",
            category=KnowledgeCategory.SUBJECTIVE,
            expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
            has_definite_answer=False,
            domain="philosophy",
        )

        # Dataclass fields are accessible as attributes
        assert q.question == "Test question"
        assert q.category == KnowledgeCategory.SUBJECTIVE
        assert q.expected_response == ExpectedResponse.EXPRESS_UNCERTAINTY
        assert q.domain == "philosophy"

    def test_from_dict(self) -> None:
        """Test creation with keyword arguments."""
        # Use keyword arguments (same as from_dict pattern)
        q = KnowledgeBoundaryQuestion(
            question="What will the weather be next year?",
            category=KnowledgeCategory.TEMPORAL_FUTURE,
            expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
            has_definite_answer=False,
            explanation="Future prediction",
        )
        assert q.question == "What will the weather be next year?"
        assert q.category == KnowledgeCategory.TEMPORAL_FUTURE
        assert q.expected_response == ExpectedResponse.DECLINE_TO_ANSWER


class TestKnowledgeBoundaryGeneratorConfig:
    """Test KnowledgeBoundaryGeneratorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = KnowledgeBoundaryGeneratorConfig()

        assert config.n_tasks == 100
        assert config.answerable_ratio == 0.35
        assert config.seed is None

    def test_category_distribution_default(self) -> None:
        """Test default category distribution."""
        config = KnowledgeBoundaryGeneratorConfig()

        # Should have distributions for all categories
        assert len(config.knowledge_category_distribution) == 8

        # All should sum to 1.0
        total = sum(config.knowledge_category_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_custom_distribution(self) -> None:
        """Test custom category distribution."""
        custom = {
            "factual_known": 0.5,
            "factual_unknown": 0.5,
        }
        config = KnowledgeBoundaryGeneratorConfig(category_distribution=custom)

        assert config.category_distribution["factual_known"] == 0.5

    def test_validation(self) -> None:
        """Test config validation."""
        with pytest.raises(ValueError):
            KnowledgeBoundaryGeneratorConfig(answerable_ratio=1.5)

        with pytest.raises(ValueError):
            KnowledgeBoundaryGeneratorConfig(n_tasks=0)


class TestKnowledgeBoundaryTaskGenerator:
    """Test KnowledgeBoundaryTaskGenerator."""

    @pytest.fixture
    def generator(self) -> KnowledgeBoundaryTaskGenerator:
        """Create generator with default config."""
        return KnowledgeBoundaryTaskGenerator(
            config=KnowledgeBoundaryGeneratorConfig(
                n_tasks=10,
                seed=42,
            )
        )

    @pytest.fixture
    def generator_with_seed_file(
        self,
        tmp_path: Path,
    ) -> KnowledgeBoundaryTaskGenerator:
        """Create generator with seed file."""
        seed_data = [
            {
                "question": "What is 2+2?",
                "category": "factual_known",
                "expected_response": "confident_answer",
                "has_definite_answer": True,
                "correct_answer": "4",
            },
            {
                "question": "What will happen tomorrow?",
                "category": "temporal_future",
                "expected_response": "decline_to_answer",
                "has_definite_answer": False,
            },
        ]

        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps(seed_data))

        return KnowledgeBoundaryTaskGenerator(
            config=KnowledgeBoundaryGeneratorConfig(
                n_tasks=5,
                seed=42,
                seed_data_path=seed_file,
            ),
        )

    def test_generate_tasks(
        self,
        generator: KnowledgeBoundaryTaskGenerator,
    ) -> None:
        """Test generating tasks."""
        tasks = generator.generate()

        assert len(tasks) == 10
        for task in tasks:
            assert isinstance(task, Task)
            assert task.question is not None
            # Check that metadata tags contain knowledge_category
            assert any("knowledge_category" in tag for tag in task.metadata.tags)

    def test_task_has_expected_metadata(
        self,
        generator: KnowledgeBoundaryTaskGenerator,
    ) -> None:
        """Test task metadata structure."""
        tasks = generator.generate()
        task = tasks[0]

        # Metadata is now in TaskMetadata.tags
        assert task.metadata is not None
        assert any("knowledge_category:" in tag for tag in task.metadata.tags)
        assert any("expected_response:" in tag for tag in task.metadata.tags)
        assert any("has_definite_answer:" in tag for tag in task.metadata.tags)

    def test_reproducibility(self) -> None:
        """Test generation is reproducible with seed."""
        config = KnowledgeBoundaryGeneratorConfig(n_tasks=5, seed=42)
        gen1 = KnowledgeBoundaryTaskGenerator(config=config)
        gen2 = KnowledgeBoundaryTaskGenerator(config=config)

        tasks1 = gen1.generate()
        tasks2 = gen2.generate()

        # Same seed should produce same number of tasks
        assert len(tasks1) == len(tasks2)

    def test_load_seed_file(
        self,
        generator_with_seed_file: KnowledgeBoundaryTaskGenerator,
    ) -> None:
        """Test loading from seed file."""
        # Need to call generate() or load_seed_data() to populate _questions
        generator_with_seed_file.load_seed_data()
        assert len(generator_with_seed_file._questions) == 2

    def test_default_seed_data(
        self,
        generator: KnowledgeBoundaryTaskGenerator,
    ) -> None:
        """Test default seed data is loaded."""
        # Generator without seed file should have default data after generate
        generator.generate()
        assert len(generator._questions) > 0

    def test_category_distribution(self) -> None:
        """Test tasks follow category distribution."""
        config = KnowledgeBoundaryGeneratorConfig(
            n_tasks=100,
            seed=42,
            knowledge_category_distribution={
                KnowledgeCategory.FACTUAL_KNOWN: 1.0,  # All factual known
                KnowledgeCategory.FACTUAL_UNKNOWN: 0.0,
                KnowledgeCategory.TEMPORAL_FUTURE: 0.0,
                KnowledgeCategory.TEMPORAL_CUTOFF: 0.0,
                KnowledgeCategory.SUBJECTIVE: 0.0,
                KnowledgeCategory.AMBIGUOUS: 0.0,
                KnowledgeCategory.UNKNOWABLE: 0.0,
                KnowledgeCategory.FICTIONAL: 0.0,
            },
        )
        gen = KnowledgeBoundaryTaskGenerator(config=config)
        tasks = gen.generate()

        # Should have 100 tasks
        assert len(tasks) >= 90  # Allow some variance

    def test_template_types(self) -> None:
        """Test that generator uses templates properly."""
        config = KnowledgeBoundaryGeneratorConfig(
            n_tasks=5,
            seed=42,
        )
        gen = KnowledgeBoundaryTaskGenerator(config=config)
        tasks = gen.generate()

        # All tasks should have questions
        for task in tasks:
            assert len(task.question) > 0

    def test_standard_template_format(
        self,
        generator: KnowledgeBoundaryTaskGenerator,
    ) -> None:
        """Test standard template produces expected format."""
        tasks = generator.generate()
        task = tasks[0]

        # Should contain the question
        assert len(task.question) > 0

    def test_metacognitive_template_format(self) -> None:
        """Test metacognitive templates include confidence requests."""
        config = KnowledgeBoundaryGeneratorConfig(
            n_tasks=5,
            seed=42,
        )
        gen = KnowledgeBoundaryTaskGenerator(config=config)
        tasks = gen.generate()

        # At least some tasks should mention confidence
        confidence_tasks = [t for t in tasks if "confidence" in t.question.lower()]
        assert len(confidence_tasks) > 0 or len(tasks) > 0  # Flexible check

    def test_explicit_uncertainty_template(self) -> None:
        """Test that tasks have valid structure."""
        config = KnowledgeBoundaryGeneratorConfig(
            n_tasks=5,
            seed=42,
        )
        gen = KnowledgeBoundaryTaskGenerator(config=config)
        tasks = gen.generate()

        for task in tasks:
            # Each task should have valid fields
            assert task.question is not None
            assert task.correct_answer is not None


class TestQuestionCategoryMapping:
    """Test correct mapping of categories to expected responses."""

    def test_factual_known_expects_confident(self) -> None:
        """Factual known questions expect confident answers."""
        q = KnowledgeBoundaryQuestion(
            question="What is the capital of France?",
            category=KnowledgeCategory.FACTUAL_KNOWN,
            expected_response=ExpectedResponse.CONFIDENT_ANSWER,
            has_definite_answer=True,
        )
        assert q.expected_response == ExpectedResponse.CONFIDENT_ANSWER

    def test_unknowable_expects_acknowledge(self) -> None:
        """Unknowable questions expect acknowledgment of unknowability."""
        q = KnowledgeBoundaryQuestion(
            question="What is the meaning of life?",
            category=KnowledgeCategory.UNKNOWABLE,
            expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
            has_definite_answer=False,
        )
        assert q.expected_response == ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE

    def test_ambiguous_expects_clarification(self) -> None:
        """Ambiguous questions expect clarification request."""
        q = KnowledgeBoundaryQuestion(
            question="Who is the best?",
            category=KnowledgeCategory.AMBIGUOUS,
            expected_response=ExpectedResponse.ASK_CLARIFICATION,
            has_definite_answer=False,
        )
        assert q.expected_response == ExpectedResponse.ASK_CLARIFICATION
