"""Unit tests for CHIMERA task models.

Tests cover:
- Task creation and validation
- TaskSet operations (filtering, statistics)
- TaskMetadata handling
- Serialization/deserialization (JSON, JSONL)
- Track-specific fields validation
"""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from chimera.models.task import (
    AnswerType,
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskMetadata,
    TaskSet,
    TrackType,
)


class TestTrackType:
    """Tests for TrackType enum."""

    def test_all_tracks_defined(self) -> None:
        """Verify all expected tracks are defined."""
        expected = {
            "calibration",
            "error_detection",
            "knowledge_boundary",
            "self_correction",
        }
        actual = {t.value for t in TrackType}
        assert actual == expected

    def test_track_string_values(self) -> None:
        """Verify track values are lowercase strings."""
        for track in TrackType:
            assert track.value == track.value.lower()
            assert isinstance(track.value, str)


class TestDifficultyLevel:
    """Tests for DifficultyLevel enum."""

    def test_difficulty_ordering(self) -> None:
        """Verify difficulty levels have proper ordering semantics."""
        levels = list(DifficultyLevel)
        # Difficulty values are uppercase L1-L5
        expected_order = ["L1", "L2", "L3", "L4", "L5"]
        assert [level.value for level in levels] == expected_order

    def test_difficulty_descriptions(self) -> None:
        """Verify difficulty descriptions are accessible."""
        # L1 should be easiest
        assert DifficultyLevel.L1 is not None
        # L5 should be hardest
        assert DifficultyLevel.L5 is not None


class TestTaskCategory:
    """Tests for TaskCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Verify expected categories exist."""
        categories = {c.value for c in TaskCategory}
        # Should have these core categories (NUMERICAL not MATHEMATICS)
        assert "factual" in categories
        assert "reasoning" in categories
        assert "numerical" in categories

    def test_category_string_values(self) -> None:
        """Verify category values are strings."""
        for category in TaskCategory:
            assert isinstance(category.value, str)


class TestTaskMetadata:
    """Tests for TaskMetadata model."""

    def test_default_metadata(self) -> None:
        """Test metadata with default values."""
        meta = TaskMetadata()
        # Default source is "synthetic"
        assert meta.source == "synthetic"
        assert meta.verified is False
        assert meta.tags == []

    def test_full_metadata(self) -> None:
        """Test creation with all fields."""
        meta = TaskMetadata(
            source="test_source",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L3,
            verified=True,
            tags=["tag1", "tag2"],
            notes="Some notes about the task",
        )
        assert meta.source == "test_source"
        assert meta.verified is True
        assert len(meta.tags) == 2
        assert meta.notes == "Some notes about the task"

    def test_metadata_category_and_difficulty(self) -> None:
        """Test metadata with category and difficulty."""
        meta = TaskMetadata(
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.L4,
        )
        # Check the values (with use_enum_values=True they become strings)
        assert meta.category in (TaskCategory.REASONING, "reasoning")
        assert meta.difficulty in (DifficultyLevel.L4, "L4")


class TestTask:
    """Tests for Task model."""

    def test_minimal_task(self) -> None:
        """Test task creation with minimal required fields."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="What is 2 + 2?",
            correct_answer="4",
        )
        assert task.track == TrackType.CALIBRATION
        assert task.question == "What is 2 + 2?"
        assert task.correct_answer == "4"
        assert task.id is not None  # Auto-generated

    def test_task_auto_id_generation(self) -> None:
        """Test that each task gets a unique ID."""
        task1 = Task(
            track=TrackType.CALIBRATION,
            question="Q1",
            correct_answer="A1",
        )
        task2 = Task(
            track=TrackType.CALIBRATION,
            question="Q2",
            correct_answer="A2",
        )
        assert task1.id != task2.id

    def test_task_explicit_id(self) -> None:
        """Test task creation with explicit ID."""
        explicit_id = uuid4()
        task = Task(
            id=explicit_id,
            track=TrackType.CALIBRATION,
            question="Q",
            correct_answer="A",
        )
        assert task.id == explicit_id

    def test_full_task(self) -> None:
        """Test task creation with all fields."""
        task = Task(
            track=TrackType.ERROR_DETECTION,
            question="Identify the error in: 2 + 2 = 5",
            correct_answer="The sum should be 4, not 5",
            answer_type=AnswerType.FREE_FORM,
            difficulty=DifficultyLevel.L2,
            category=TaskCategory.NUMERICAL,  # Use NUMERICAL not MATHEMATICS
            metadata=TaskMetadata(
                source="synthetic",
                verified=True,
            ),
            # Track-specific field
            initial_response="2 + 2 = 5",
        )
        assert task.track == TrackType.ERROR_DETECTION
        assert task.initial_response == "2 + 2 = 5"

    def test_task_calibration_track(self) -> None:
        """Test calibration track specific task."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="What is the capital of France?",
            correct_answer="Paris",
            difficulty=DifficultyLevel.L1,
        )
        # Calibration tasks have difficulty for expected confidence proxy
        assert task.difficulty == DifficultyLevel.L1

    def test_task_knowledge_boundary_track(self) -> None:
        """Test knowledge boundary track specific fields."""
        task = Task(
            track=TrackType.KNOWLEDGE_BOUNDARY,
            question="What will the stock market do tomorrow?",
            correct_answer=None,  # Unanswerable
            is_answerable=False,
        )
        assert task.is_answerable is False

    def test_task_self_correction_track(self) -> None:
        """Test self-correction track specific fields."""
        task = Task(
            track=TrackType.SELF_CORRECTION,
            question="Review and correct if needed",
            correct_answer="Corrected reasoning",
            reasoning_chain=["Step 1", "Step 2 (flawed)", "Conclusion"],
            perturbation_index=1,
            perturbation_type="logical_fallacy",
        )
        assert task.perturbation_index == 1
        assert task.perturbation_type == "logical_fallacy"

    def test_task_json_serialization(self) -> None:
        """Test task can be serialized to JSON."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="Q",
            correct_answer="A",
            difficulty=DifficultyLevel.L3,
        )
        json_str = task.model_dump_json()
        data = json.loads(json_str)

        assert data["question"] == "Q"
        assert data["correct_answer"] == "A"
        assert data["track"] == "calibration"
        # Difficulty uses uppercase
        assert data["difficulty"] == "L3"

    def test_task_json_deserialization(self) -> None:
        """Test task can be deserialized from JSON."""
        data = {
            "track": "calibration",
            "question": "What is 1 + 1?",
            "correct_answer": "2",
            "difficulty": "L1",  # Uppercase
        }
        task = Task.model_validate(data)
        assert task.track == TrackType.CALIBRATION
        assert task.difficulty == DifficultyLevel.L1

    def test_task_invalid_track(self) -> None:
        """Test validation fails for invalid track."""
        with pytest.raises(ValidationError):
            Task(
                track="invalid_track",  # type: ignore
                question="Q",
                correct_answer="A",
            )

    def test_task_get_all_acceptable_answers(self) -> None:
        """Test get_all_acceptable_answers method."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="What is 2+2?",
            correct_answer="4",
            acceptable_answers=["four", "Four"],
        )
        answers = task.get_all_acceptable_answers()
        assert "4" in answers
        assert "four" in answers
        assert len(answers) == 3

    def test_task_is_track_method(self) -> None:
        """Test is_track method."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="Q",
            correct_answer="A",
        )
        assert task.is_track(TrackType.CALIBRATION)
        assert not task.is_track(TrackType.ERROR_DETECTION)


class TestTaskSet:
    """Tests for TaskSet model."""

    @pytest.fixture
    def sample_tasks(self) -> list[Task]:
        """Create a sample set of tasks for testing."""
        return [
            Task(
                track=TrackType.CALIBRATION,
                question="Q1",
                correct_answer="A1",
                difficulty=DifficultyLevel.L1,
                category=TaskCategory.FACTUAL,
            ),
            Task(
                track=TrackType.CALIBRATION,
                question="Q2",
                correct_answer="A2",
                difficulty=DifficultyLevel.L3,
                category=TaskCategory.REASONING,
            ),
            Task(
                track=TrackType.CALIBRATION,
                question="Q3",
                correct_answer="A3",
                difficulty=DifficultyLevel.L2,
                category=TaskCategory.FACTUAL,
            ),
            Task(
                track=TrackType.CALIBRATION,
                question="Q4",
                correct_answer="A4",
                difficulty=DifficultyLevel.L4,
                category=TaskCategory.NUMERICAL,
            ),
        ]

    def test_taskset_creation(self, sample_tasks: list[Task]) -> None:
        """Test TaskSet creation."""
        taskset = TaskSet(
            name="test_set",
            track=TrackType.CALIBRATION,
            tasks=sample_tasks,
        )
        assert taskset.name == "test_set"
        assert len(taskset.tasks) == 4

    def test_taskset_len(self, sample_tasks: list[Task]) -> None:
        """Test TaskSet __len__ method."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        assert len(taskset) == 4

    def test_taskset_iter(self, sample_tasks: list[Task]) -> None:
        """Test TaskSet iteration."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        tasks_from_iter = list(taskset)
        assert len(tasks_from_iter) == 4
        assert all(isinstance(t, Task) for t in tasks_from_iter)

    def test_taskset_getitem(self, sample_tasks: list[Task]) -> None:
        """Test TaskSet indexing."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        first_task = taskset[0]
        assert first_task.question == "Q1"

    def test_taskset_filter_by_difficulty(self, sample_tasks: list[Task]) -> None:
        """Test filtering tasks by difficulty."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        filtered = taskset.filter_by_difficulty(DifficultyLevel.L1)
        assert len(filtered) == 1
        assert filtered[0].difficulty == DifficultyLevel.L1

    def test_taskset_filter_by_category(self, sample_tasks: list[Task]) -> None:
        """Test filtering tasks by category."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        filtered = taskset.filter_by_category(TaskCategory.FACTUAL)
        assert len(filtered) == 2

    def test_taskset_get_difficulty_distribution(self, sample_tasks: list[Task]) -> None:
        """Test getting difficulty distribution."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        dist = taskset.get_difficulty_distribution()

        assert "L1" in dist
        assert dist["L1"] == 1
        assert dist["L3"] == 1

    def test_taskset_get_category_distribution(self, sample_tasks: list[Task]) -> None:
        """Test getting category distribution."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        dist = taskset.get_category_distribution()

        assert "factual" in dist
        assert dist["factual"] == 2

    def test_taskset_to_jsonl(self, sample_tasks: list[Task]) -> None:
        """Test serializing TaskSet to JSONL."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=sample_tasks)
        jsonl = taskset.to_jsonl()

        lines = jsonl.strip().split("\n")
        assert len(lines) == 4

        for line in lines:
            data = json.loads(line)
            assert "track" in data
            assert "question" in data

    def test_taskset_from_jsonl(self, sample_tasks: list[Task]) -> None:
        """Test deserializing TaskSet from JSONL."""
        # Create JSONL content
        jsonl_content = "\n".join(task.model_dump_json() for task in sample_tasks)

        # Load it back
        taskset = TaskSet.from_jsonl(
            jsonl_content=jsonl_content,
            name="loaded_set",
            track=TrackType.CALIBRATION,
        )
        assert len(taskset.tasks) == 4
        assert taskset.name == "loaded_set"

    def test_taskset_empty(self) -> None:
        """Test creating an empty TaskSet."""
        taskset = TaskSet(name="empty", track=TrackType.CALIBRATION, tasks=[])
        assert len(taskset) == 0
        assert taskset.get_difficulty_distribution() == {}
        assert taskset.get_category_distribution() == {}

    def test_taskset_add_task(self) -> None:
        """Test adding task to TaskSet."""
        taskset = TaskSet(name="test", track=TrackType.CALIBRATION, tasks=[])
        task = Task(
            track=TrackType.CALIBRATION,
            question="New Q",
            correct_answer="New A",
        )
        taskset.add_task(task)
        assert len(taskset) == 1


class TestTaskEdgeCases:
    """Edge case tests for Task models."""

    def test_task_empty_question(self) -> None:
        """Test that empty questions are rejected."""
        with pytest.raises(ValidationError):
            Task(
                track=TrackType.CALIBRATION,
                question="",
                correct_answer="A",
            )

    def test_task_whitespace_question(self) -> None:
        """Test that whitespace-only questions are rejected."""
        with pytest.raises(ValidationError):
            Task(
                track=TrackType.CALIBRATION,
                question="   ",
                correct_answer="A",
            )

    def test_task_very_long_question(self) -> None:
        """Test handling of very long questions."""
        long_question = "Q" * 10000
        task = Task(
            track=TrackType.CALIBRATION,
            question=long_question,
            correct_answer="A",
        )
        assert len(task.question) == 10000

    def test_task_unicode_content(self) -> None:
        """Test handling of unicode content."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="什么是人工智能？🤖",
            correct_answer="人工智能是...",
        )
        assert "🤖" in task.question

    def test_task_none_answer_for_unanswerable(self) -> None:
        """Test that unanswerable questions can have None answers."""
        task = Task(
            track=TrackType.KNOWLEDGE_BOUNDARY,
            question="What will happen next year?",
            correct_answer=None,
            is_answerable=False,
        )
        assert task.correct_answer is None
        assert task.is_answerable is False

    def test_task_error_detection_fields(self) -> None:
        """Test error detection track fields."""
        task = Task(
            track=TrackType.ERROR_DETECTION,
            question="Find the error",
            correct_answer="The error is...",
            initial_response="Wrong response here",
            error_locations=["Line 1", "Line 3"],
        )
        assert task.initial_response == "Wrong response here"
        assert len(task.error_locations) == 2
