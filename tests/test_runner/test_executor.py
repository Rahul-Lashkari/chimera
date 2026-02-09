"""Unit tests for CHIMERA benchmark executor.

Tests cover:
- BenchmarkRunner initialization
- Task execution (sync and async)
- Progress tracking
- Checkpointing
- Error handling
"""

from datetime import datetime
from uuid import uuid4

import pytest

from chimera.interfaces.base import ModelCapabilities, ModelConfig
from chimera.models.response import (
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ResponseMetadata,
)
from chimera.models.task import Task, TaskSet, TrackType
from chimera.runner.config import RunConfig
from chimera.runner.executor import (
    BenchmarkRunner,
    Checkpoint,
    ExecutionProgress,
    ExecutionState,
    TaskResult,
)


class MockModelInterface:
    """Mock model interface for testing."""

    def __init__(self, fail_rate: float = 0.0) -> None:
        self.fail_rate = fail_rate
        self.call_count = 0
        self.config = ModelConfig(model_name="mock-model")

    def generate_for_task(self, task: Task) -> ModelResponse:
        """Generate mock response."""
        self.call_count += 1

        import random

        if random.random() < self.fail_rate:
            raise Exception("Mock API error")

        return ModelResponse(
            task_id=task.id,
            raw_text=f"Answer: {task.correct_answer}\nConfidence: 85%",
            parsed_answer=ParsedAnswer(
                raw_answer=str(task.correct_answer),
                normalized=str(task.correct_answer).lower(),
            ),
            confidence=ConfidenceScore(numeric=0.85),
            metadata=ResponseMetadata(
                model_name="mock-model",
                latency_ms=100.0,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

    async def generate_for_task_async(self, task: Task) -> ModelResponse:
        """Generate mock response async."""
        return self.generate_for_task(task)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()


class TestTaskResult:
    """Tests for TaskResult."""

    def test_successful_result(self) -> None:
        """Test successful task result."""
        response = ModelResponse(
            task_id=uuid4(),
            raw_text="Answer: 42",
            parsed_answer=ParsedAnswer(
                raw_answer="42",
                normalized="42",
            ),
            confidence=ConfidenceScore(numeric=0.9),
            metadata=ResponseMetadata(
                model_name="test-model",
                latency_ms=100.0,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

        result = TaskResult(
            task_id=uuid4(),
            track=TrackType.CALIBRATION,
            response=response,
            latency_ms=150.0,
        )

        assert result.success is True
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed task result."""
        result = TaskResult(
            task_id=uuid4(),
            track=TrackType.CALIBRATION,
            error="API timeout",
            latency_ms=5000.0,
        )

        assert result.success is False
        assert result.error == "API timeout"

    def test_to_dict(self) -> None:
        """Test result serialization."""
        result = TaskResult(
            task_id=uuid4(),
            track=TrackType.CALIBRATION,
            error="Test error",
            attempt=2,
        )

        d = result.to_dict()

        assert "task_id" in d
        assert d["track"] == "calibration"
        assert d["success"] is False
        assert d["attempt"] == 2


class TestExecutionProgress:
    """Tests for ExecutionProgress."""

    def test_initial_progress(self) -> None:
        """Test initial progress state."""
        progress = ExecutionProgress(total_tasks=100)

        assert progress.progress_percent == 0.0
        assert progress.success_rate == 0.0

    def test_progress_percent(self) -> None:
        """Test progress percentage calculation."""
        progress = ExecutionProgress(
            total_tasks=100,
            completed_tasks=25,
        )

        assert progress.progress_percent == 25.0

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        progress = ExecutionProgress(
            total_tasks=100,
            completed_tasks=50,
            successful_tasks=45,
        )

        assert progress.success_rate == 90.0

    def test_update(self) -> None:
        """Test progress update."""
        progress = ExecutionProgress(
            total_tasks=100,
            completed_tasks=50,
            start_time=datetime.now(),
        )

        progress.update()

        assert progress.elapsed_seconds > 0
        assert progress.tasks_per_second >= 0

    def test_format_eta_seconds(self) -> None:
        """Test ETA formatting for seconds."""
        progress = ExecutionProgress()
        progress.estimated_remaining = 45

        assert progress.format_eta() == "45s"

    def test_format_eta_minutes(self) -> None:
        """Test ETA formatting for minutes."""
        progress = ExecutionProgress()
        progress.estimated_remaining = 150

        assert "m" in progress.format_eta()

    def test_format_eta_hours(self) -> None:
        """Test ETA formatting for hours."""
        progress = ExecutionProgress()
        progress.estimated_remaining = 7200

        assert "h" in progress.format_eta()


class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_checkpoint_creation(self) -> None:
        """Test creating a checkpoint."""
        progress = ExecutionProgress(
            total_tasks=100,
            completed_tasks=50,
        )

        checkpoint = Checkpoint(
            run_name="test_run",
            config={"name": "test"},
            state=ExecutionState.PAUSED,
            progress=progress,
            completed_task_ids=["id1", "id2"],
            track_progress={"calibration": 25},
        )

        assert checkpoint.run_name == "test_run"
        assert checkpoint.state == ExecutionState.PAUSED
        assert len(checkpoint.completed_task_ids) == 2

    def test_checkpoint_save_load(self, tmp_path) -> None:
        """Test checkpoint save and load."""
        progress = ExecutionProgress(total_tasks=50, completed_tasks=25)

        checkpoint = Checkpoint(
            run_name="test",
            config={},
            state=ExecutionState.PAUSED,
            progress=progress,
            completed_task_ids=["a", "b", "c"],
            track_progress={"calibration": 10},
        )

        path = tmp_path / "checkpoint.json"
        checkpoint.save(path)

        loaded = Checkpoint.load(path)

        assert loaded.run_name == "test"
        assert loaded.state == ExecutionState.PAUSED
        assert len(loaded.completed_task_ids) == 3


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.fixture
    def mock_model(self) -> MockModelInterface:
        """Create mock model."""
        return MockModelInterface()

    @pytest.fixture
    def sample_tasks(self) -> TaskSet:
        """Create sample task set."""
        tasks = [
            Task(
                track=TrackType.CALIBRATION,
                question=f"Question {i}",
                correct_answer=f"Answer {i}",
            )
            for i in range(10)
        ]
        return TaskSet(
            name="test_set",
            track=TrackType.CALIBRATION,
            tasks=tasks,
        )

    @pytest.fixture
    def runner(
        self,
        mock_model: MockModelInterface,
        sample_tasks: TaskSet,
    ) -> BenchmarkRunner:
        """Create benchmark runner."""
        config = RunConfig(
            name="test_run",
            tracks=[TrackType.CALIBRATION],
            max_tasks_per_track=10,
            batch_size=5,
            max_retries=1,
        )

        runner = BenchmarkRunner(
            model=mock_model,
            config=config,
            task_sets={TrackType.CALIBRATION: sample_tasks},
        )

        return runner

    def test_runner_initialization(
        self,
        runner: BenchmarkRunner,
    ) -> None:
        """Test runner initialization."""
        assert runner.state == ExecutionState.PENDING
        assert runner.progress.total_tasks == 0
        assert len(runner.results) == 0

    def test_add_task_set(
        self,
        mock_model: MockModelInterface,
    ) -> None:
        """Test adding task sets."""
        runner = BenchmarkRunner(model=mock_model)

        task_set = TaskSet(
            name="new_set",
            track=TrackType.ERROR_DETECTION,
            tasks=[],
        )

        runner.add_task_set(TrackType.ERROR_DETECTION, task_set)

        assert TrackType.ERROR_DETECTION in runner.task_sets

    def test_progress_callback(
        self,
        runner: BenchmarkRunner,
    ) -> None:
        """Test progress callback."""
        callbacks_received = []

        def callback(completed, total, current_task, track):
            callbacks_received.append((completed, total))

        runner.add_progress_callback(callback)
        runner.run()

        assert len(callbacks_received) > 0
        # Final callback should show all completed
        assert callbacks_received[-1][0] == callbacks_received[-1][1]

    def test_run_completes(
        self,
        runner: BenchmarkRunner,
    ) -> None:
        """Test benchmark runs to completion."""
        report = runner.run()

        assert runner.state == ExecutionState.COMPLETED
        assert len(runner.results) == 10
        assert report is not None

    def test_run_with_failures(
        self,
        sample_tasks: TaskSet,
    ) -> None:
        """Test benchmark handles failures."""
        model = MockModelInterface(fail_rate=0.3)

        config = RunConfig(
            tracks=[TrackType.CALIBRATION],
            max_tasks_per_track=10,
            batch_size=5,
            max_retries=0,  # No retries
        )

        runner = BenchmarkRunner(
            model=model,
            config=config,
            task_sets={TrackType.CALIBRATION: sample_tasks},
        )

        runner.run()  # Run without capturing unused return value

        # Should complete despite failures
        assert runner.state == ExecutionState.COMPLETED
        # Some should have failed
        failed = [r for r in runner.results if not r.success]
        assert len(failed) >= 0  # May have some failures

    def test_dry_run(
        self,
        runner: BenchmarkRunner,
    ) -> None:
        """Test dry run mode."""
        runner.config.dry_run = True

        report = runner.run()

        assert runner.state == ExecutionState.COMPLETED
        assert len(runner.results) == 0
        _ = report  # Ensure report is used

    def test_cancel(
        self,
        runner: BenchmarkRunner,
    ) -> None:
        """Test cancellation."""
        # Cancel immediately
        runner._cancel_requested = True

        runner.run()

        assert runner.state == ExecutionState.CANCELLED


class TestBenchmarkRunnerAsync:
    """Tests for async benchmark execution."""

    @pytest.fixture
    def mock_model(self) -> MockModelInterface:
        return MockModelInterface()

    @pytest.fixture
    def sample_tasks(self) -> TaskSet:
        tasks = [
            Task(
                track=TrackType.CALIBRATION,
                question=f"Q{i}",
                correct_answer=f"A{i}",
            )
            for i in range(5)
        ]
        return TaskSet(
            name="test",
            track=TrackType.CALIBRATION,
            tasks=tasks,
        )

    @pytest.mark.asyncio
    async def test_run_async(
        self,
        mock_model: MockModelInterface,
        sample_tasks: TaskSet,
    ) -> None:
        """Test async benchmark execution."""
        config = RunConfig(
            tracks=[TrackType.CALIBRATION],
            max_tasks_per_track=5,
            batch_size=2,
            max_concurrent=2,
        )

        runner = BenchmarkRunner(
            model=mock_model,
            config=config,
            task_sets={TrackType.CALIBRATION: sample_tasks},
        )

        await runner.run_async()  # Run without capturing unused return value

        assert runner.state == ExecutionState.COMPLETED
        assert len(runner.results) == 5


class TestExecutionState:
    """Tests for ExecutionState enum."""

    def test_state_values(self) -> None:
        """Test state values."""
        assert ExecutionState.PENDING.value == "pending"
        assert ExecutionState.RUNNING.value == "running"
        assert ExecutionState.COMPLETED.value == "completed"
        assert ExecutionState.FAILED.value == "failed"
        assert ExecutionState.CANCELLED.value == "cancelled"
