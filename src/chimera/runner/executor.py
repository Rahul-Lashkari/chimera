"""Benchmark execution engine for CHIMERA.

This module provides the core BenchmarkRunner class that orchestrates
benchmark execution across multiple tracks, manages progress, handles
errors, and coordinates with model interfaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from chimera.interfaces.base import BaseModelInterface
from chimera.models.response import ModelResponse
from chimera.models.task import Task, TaskSet, TrackType
from chimera.runner.config import ProgressCallback, RunConfig

if TYPE_CHECKING:
    from chimera.runner.report import BenchmarkReport

logger = logging.getLogger(__name__)


class ExecutionState(str, Enum):
    """State of benchmark execution.

    Attributes:
        PENDING: Not yet started.
        RUNNING: Currently executing.
        PAUSED: Execution paused (can resume).
        COMPLETED: Successfully completed.
        FAILED: Execution failed with error.
        CANCELLED: Execution was cancelled.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of evaluating a single task.

    Attributes:
        task_id: ID of the evaluated task.
        track: Track the task belongs to.
        response: Model's response (if successful).
        error: Error message (if failed).
        latency_ms: Time taken for evaluation.
        attempt: Attempt number (for retried tasks).
        timestamp: When the result was recorded.
    """

    task_id: UUID
    track: TrackType
    response: ModelResponse | None = None
    error: str | None = None
    latency_ms: float = 0.0
    attempt: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        """Check if task evaluation was successful."""
        return self.response is not None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": str(self.task_id),
            "track": self.track.value,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "attempt": self.attempt,
            "timestamp": self.timestamp.isoformat(),
        }


class ExecutionProgress(BaseModel):
    """Progress tracking for benchmark execution.

    Attributes:
        total_tasks: Total number of tasks to evaluate.
        completed_tasks: Number of completed tasks.
        successful_tasks: Number of successful evaluations.
        failed_tasks: Number of failed evaluations.
        current_track: Currently executing track.
        start_time: When execution started.
        elapsed_seconds: Time elapsed since start.
        estimated_remaining: Estimated seconds remaining.
        tasks_per_second: Current throughput.
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    current_track: TrackType | None = None
    current_task_id: str | None = None
    start_time: datetime | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining: float | None = None
    tasks_per_second: float = 0.0

    model_config = {"extra": "allow"}

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.completed_tasks == 0:
            return 0.0
        return (self.successful_tasks / self.completed_tasks) * 100

    def update(self) -> None:
        """Update elapsed time and estimates."""
        if self.start_time:
            self.elapsed_seconds = (datetime.now() - self.start_time).total_seconds()

            if self.completed_tasks > 0:
                self.tasks_per_second = self.completed_tasks / self.elapsed_seconds
                remaining = self.total_tasks - self.completed_tasks
                if self.tasks_per_second > 0:
                    self.estimated_remaining = remaining / self.tasks_per_second

    def format_eta(self) -> str:
        """Format estimated time remaining."""
        if self.estimated_remaining is None:
            return "calculating..."

        seconds = int(self.estimated_remaining)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"


class Checkpoint(BaseModel):
    """Checkpoint for resumable benchmark runs.

    Stores state needed to resume a paused or interrupted run.
    """

    run_name: str
    config: dict[str, Any]
    state: ExecutionState
    progress: ExecutionProgress
    completed_task_ids: list[str]
    track_progress: dict[str, int]
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"extra": "allow"}

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load checkpoint from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class BenchmarkRunner:
    """Main orchestrator for CHIMERA benchmark execution.

    Handles task execution, progress tracking, error handling,
    checkpointing, and result collection.

    Example:
        >>> from chimera import GeminiModel, GeminiConfig
        >>> from chimera.runner import BenchmarkRunner, RunConfig
        >>>
        >>> model = GeminiModel(GeminiConfig())
        >>> config = RunConfig.quick(tracks=["calibration"], max_tasks=50)
        >>> runner = BenchmarkRunner(model, config)
        >>>
        >>> # Run with progress callback
        >>> def on_progress(completed, total, task, track):
        ...     print(f"Progress: {completed}/{total}")
        >>>
        >>> report = runner.run(progress_callback=on_progress)
        >>> print(report.summary())
    """

    def __init__(
        self,
        model: BaseModelInterface,
        config: RunConfig | None = None,
        task_sets: dict[TrackType, TaskSet] | None = None,
    ) -> None:
        """Initialize benchmark runner.

        Args:
            model: Model interface to evaluate.
            config: Run configuration.
            task_sets: Pre-loaded task sets (optional).
        """
        self.model = model
        self.config = config or RunConfig()
        self.task_sets = task_sets or {}

        # Execution state
        self.state = ExecutionState.PENDING
        self.progress = ExecutionProgress()
        self.results: list[TaskResult] = []
        self.completed_task_ids: set[UUID] = set()

        # Callbacks
        self._progress_callbacks: list[ProgressCallback] = []
        self._cancel_requested = False

        # Setup logging
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)

    def add_task_set(self, track: TrackType, task_set: TaskSet) -> None:
        """Add a task set for a specific track.

        Args:
            track: Track type.
            task_set: TaskSet containing tasks.
        """
        self.task_sets[track] = task_set

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback.

        Args:
            callback: Function to call on progress updates.
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(
        self,
        current_task: str | None = None,
        track: str | None = None,
    ) -> None:
        """Notify all progress callbacks."""
        self.progress.update()
        for callback in self._progress_callbacks:
            try:
                callback(
                    completed=self.progress.completed_tasks,
                    total=self.progress.total_tasks,
                    current_task=current_task,
                    track=track,
                )
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _get_tasks_for_track(self, track: TrackType) -> list[Task]:
        """Get tasks for a specific track.

        Args:
            track: Track to get tasks for.

        Returns:
            List of tasks for the track.
        """
        if track not in self.task_sets:
            logger.warning(f"No task set for track: {track}")
            return []

        task_set = self.task_sets[track]
        tasks = list(task_set.tasks)

        # Apply track-specific filters
        track_config = self.config.get_track_config(track)

        if track_config.difficulty_filter:
            tasks = [
                t
                for t in tasks
                if t.metadata and t.metadata.difficulty in track_config.difficulty_filter
            ]

        if track_config.categories:
            tasks = [
                t
                for t in tasks
                if t.metadata
                and t.metadata.category
                and t.metadata.category.value in track_config.categories
            ]

        # Apply limit
        max_tasks = self.config.get_max_tasks(track)
        if len(tasks) > max_tasks:
            import random

            # Using random for reproducible shuffling, not security
            rng = random.Random(track_config.seed or self.config.seed)  # nosec B311
            if self.config.shuffle_tasks:
                rng.shuffle(tasks)
            tasks = tasks[:max_tasks]

        # Exclude already completed tasks
        tasks = [t for t in tasks if t.id not in self.completed_task_ids]

        return tasks

    def _execute_task(
        self,
        task: Task,
    ) -> TaskResult:
        """Execute a single task.

        Args:
            task: Task to execute.

        Returns:
            TaskResult with response or error.
        """
        start_time = time.time()
        last_error: str | None = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self.model.generate_for_task(task)
                latency = (time.time() - start_time) * 1000

                return TaskResult(
                    task_id=task.id,
                    track=task.track,
                    response=response,
                    latency_ms=latency,
                    attempt=attempt,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task {task.id} attempt {attempt} failed: {e}")

                if attempt <= self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        # All retries exhausted
        latency = (time.time() - start_time) * 1000
        return TaskResult(
            task_id=task.id,
            track=task.track,
            error=last_error,
            latency_ms=latency,
            attempt=self.config.max_retries + 1,
        )

    async def _execute_task_async(
        self,
        task: Task,
        semaphore: asyncio.Semaphore,
    ) -> TaskResult:
        """Execute a single task asynchronously.

        Args:
            task: Task to execute.
            semaphore: Semaphore for concurrency control.

        Returns:
            TaskResult with response or error.
        """
        async with semaphore:
            start_time = time.time()
            last_error: str | None = None

            for attempt in range(1, self.config.max_retries + 2):
                try:
                    response = await self.model.generate_for_task_async(task)
                    latency = (time.time() - start_time) * 1000

                    return TaskResult(
                        task_id=task.id,
                        track=task.track,
                        response=response,
                        latency_ms=latency,
                        attempt=attempt,
                    )

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Task {task.id} attempt {attempt} failed: {e}")

                    if attempt <= self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay * attempt)

            latency = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.id,
                track=task.track,
                error=last_error,
                latency_ms=latency,
                attempt=self.config.max_retries + 1,
            )

    def _execute_batch(
        self,
        tasks: list[Task],
    ) -> list[TaskResult]:
        """Execute a batch of tasks.

        Args:
            tasks: Tasks to execute.

        Returns:
            List of TaskResult objects.
        """
        results = []
        for task in tasks:
            if self._cancel_requested:
                break

            result = self._execute_task(task)
            results.append(result)

            # Update progress
            self.completed_task_ids.add(task.id)
            self.progress.completed_tasks += 1
            if result.success:
                self.progress.successful_tasks += 1
            else:
                self.progress.failed_tasks += 1

            self._notify_progress(
                current_task=str(task.id)[:8],
                track=task.track if isinstance(task.track, str) else task.track.value,
            )

        return results

    async def _execute_batch_async(
        self,
        tasks: list[Task],
    ) -> list[TaskResult]:
        """Execute a batch of tasks asynchronously.

        Args:
            tasks: Tasks to execute.

        Returns:
            List of TaskResult objects.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def execute_with_progress(task: Task) -> TaskResult:
            result = await self._execute_task_async(task, semaphore)

            self.completed_task_ids.add(task.id)
            self.progress.completed_tasks += 1
            if result.success:
                self.progress.successful_tasks += 1
            else:
                self.progress.failed_tasks += 1

            self._notify_progress(
                current_task=str(task.id)[:8],
                track=task.track if isinstance(task.track, str) else task.track.value,
            )

            return result

        coroutines = [execute_with_progress(task) for task in tasks]
        return await asyncio.gather(*coroutines)

    def _save_checkpoint(self) -> None:
        """Save current execution state to checkpoint."""
        checkpoint_path = self.config.output_dir / f"{self.config.name}_checkpoint.json"

        # Calculate track progress
        track_progress = {}
        for track in self.config.get_enabled_tracks():
            track_tasks = [r for r in self.results if r.track == track]
            track_progress[track.value] = len(track_tasks)

        checkpoint = Checkpoint(
            run_name=self.config.name,
            config=self.config.to_dict(),
            state=self.state,
            progress=self.progress,
            completed_task_ids=[str(tid) for tid in self.completed_task_ids],
            track_progress=track_progress,
        )

        checkpoint.save(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load execution state from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = Checkpoint.load(path)

        self.progress = checkpoint.progress
        self.completed_task_ids = {UUID(tid) for tid in checkpoint.completed_task_ids}

        logger.info(
            f"Resumed from checkpoint: {checkpoint.progress.completed_tasks} "
            f"tasks already completed"
        )

    def run(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark synchronously.

        Args:
            progress_callback: Optional callback for progress updates.

        Returns:
            BenchmarkReport with all results.
        """
        from chimera.runner.aggregator import ResultsAggregator
        from chimera.runner.report import BenchmarkReport

        if progress_callback:
            self.add_progress_callback(progress_callback)

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        # Calculate total tasks
        all_tasks: list[Task] = []
        for track in self.config.get_enabled_tracks():
            tasks = self._get_tasks_for_track(track)
            all_tasks.extend(tasks)

        if self.config.total_max_tasks:
            all_tasks = all_tasks[: self.config.total_max_tasks]

        self.progress.total_tasks = len(all_tasks) + len(self.completed_task_ids)
        self.progress.start_time = datetime.now()
        self.state = ExecutionState.RUNNING

        if self.config.dry_run:
            logger.info(f"Dry run: would execute {len(all_tasks)} tasks")
            self.state = ExecutionState.COMPLETED
            return BenchmarkReport.empty(self.config)

        logger.info(f"Starting benchmark: {len(all_tasks)} tasks")

        try:
            # Process in batches
            for i in range(0, len(all_tasks), self.config.batch_size):
                if self._cancel_requested:
                    self.state = ExecutionState.CANCELLED
                    break

                batch = all_tasks[i : i + self.config.batch_size]
                self.progress.current_track = batch[0].track if batch else None

                batch_results = self._execute_batch(batch)
                self.results.extend(batch_results)

                # Save checkpoint periodically
                if (
                    self.config.save_intermediate
                    and self.progress.completed_tasks % self.config.checkpoint_interval == 0
                ):
                    self._save_checkpoint()

            if self.state != ExecutionState.CANCELLED:
                self.state = ExecutionState.COMPLETED

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            self.state = ExecutionState.FAILED
            self._save_checkpoint()
            raise

        # Aggregate results
        aggregator = ResultsAggregator(
            results=self.results,
            task_sets=self.task_sets,
            config=self.config,
        )

        return aggregator.generate_report()

    async def run_async(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark asynchronously.

        Args:
            progress_callback: Optional callback for progress updates.

        Returns:
            BenchmarkReport with all results.
        """
        from chimera.runner.aggregator import ResultsAggregator
        from chimera.runner.report import BenchmarkReport

        if progress_callback:
            self.add_progress_callback(progress_callback)

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        # Calculate total tasks
        all_tasks: list[Task] = []
        for track in self.config.get_enabled_tracks():
            tasks = self._get_tasks_for_track(track)
            all_tasks.extend(tasks)

        if self.config.total_max_tasks:
            all_tasks = all_tasks[: self.config.total_max_tasks]

        self.progress.total_tasks = len(all_tasks) + len(self.completed_task_ids)
        self.progress.start_time = datetime.now()
        self.state = ExecutionState.RUNNING

        if self.config.dry_run:
            logger.info(f"Dry run: would execute {len(all_tasks)} tasks")
            self.state = ExecutionState.COMPLETED
            return BenchmarkReport.empty(self.config)

        logger.info(f"Starting async benchmark: {len(all_tasks)} tasks")

        try:
            # Process in batches
            for i in range(0, len(all_tasks), self.config.batch_size):
                if self._cancel_requested:
                    self.state = ExecutionState.CANCELLED
                    break

                batch = all_tasks[i : i + self.config.batch_size]
                self.progress.current_track = batch[0].track if batch else None

                batch_results = await self._execute_batch_async(batch)
                self.results.extend(batch_results)

                # Save checkpoint periodically
                if (
                    self.config.save_intermediate
                    and self.progress.completed_tasks % self.config.checkpoint_interval == 0
                ):
                    self._save_checkpoint()

            if self.state != ExecutionState.CANCELLED:
                self.state = ExecutionState.COMPLETED

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            self.state = ExecutionState.FAILED
            self._save_checkpoint()
            raise

        # Aggregate results
        aggregator = ResultsAggregator(
            results=self.results,
            task_sets=self.task_sets,
            config=self.config,
        )

        return aggregator.generate_report()

    def cancel(self) -> None:
        """Request cancellation of the current run."""
        self._cancel_requested = True
        logger.info("Cancellation requested")

    def pause(self) -> None:
        """Pause execution and save checkpoint."""
        self.state = ExecutionState.PAUSED
        self._save_checkpoint()
        logger.info("Execution paused")
