"""Results aggregation for CHIMERA benchmark.

This module provides the ResultsAggregator class that combines individual
task results into track-level summaries and overall benchmark metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from chimera.metrics.calibration import (
    CalibrationMetricsComputer,
    CalibrationSummary,
)
from chimera.models.task import Task, TaskSet, TrackType
from chimera.runner.config import RunConfig
from chimera.runner.executor import TaskResult

if TYPE_CHECKING:
    from chimera.runner.report import BenchmarkReport


@dataclass
class TrackSummary:
    """Summary statistics for a single evaluation track.

    Attributes:
        track: Track type.
        total_tasks: Number of tasks in track.
        completed_tasks: Number of successfully completed tasks.
        failed_tasks: Number of failed tasks.
        accuracy: Overall accuracy on the track.
        mean_confidence: Mean model confidence.
        mean_latency_ms: Mean response latency.
        calibration: Calibration metrics (for calibration track).
        task_results: Individual task results.
    """

    track: TrackType
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    accuracy: float = 0.0
    mean_confidence: float = 0.0
    mean_latency_ms: float = 0.0
    calibration: CalibrationSummary | None = None
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Percentage of successfully completed tasks."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def ece(self) -> float | None:
        """Expected Calibration Error (if computed)."""
        if self.calibration:
            return self.calibration.ece.value
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        track_str = self.track.value if hasattr(self.track, "value") else str(self.track)
        result: dict[str, Any] = {
            "track": track_str,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
            "mean_latency_ms": self.mean_latency_ms,
        }

        if self.calibration:
            result["calibration"] = {
                "ece": self.calibration.ece.value,
                "mce": self.calibration.mce.value,
                "brier": self.calibration.brier.value,
                "bias": self.calibration.overconfidence.calibration_bias,
            }

        return result


class ResultsAggregator:
    """Aggregates task results into track and overall summaries.

    Computes metrics, generates summaries, and produces the final
    benchmark report.

    Example:
        >>> aggregator = ResultsAggregator(
        ...     results=task_results,
        ...     task_sets=task_sets,
        ...     config=run_config,
        ... )
        >>> report = aggregator.generate_report()
        >>> print(report.summary())
    """

    def __init__(
        self,
        results: list[TaskResult],
        task_sets: dict[TrackType, TaskSet],
        config: RunConfig,
    ) -> None:
        """Initialize aggregator.

        Args:
            results: List of task results from execution.
            task_sets: Task sets used in benchmark.
            config: Run configuration.
        """
        self.results = results
        self.task_sets = task_sets
        self.config = config

        # Group results by track
        self.results_by_track: dict[TrackType, list[TaskResult]] = {}
        for result in results:
            if result.track not in self.results_by_track:
                self.results_by_track[result.track] = []
            self.results_by_track[result.track].append(result)

        # Create task lookup
        self.tasks_by_id: dict[UUID, Task] = {}
        for task_set in task_sets.values():
            for task in task_set.tasks:
                self.tasks_by_id[task.id] = task

    def compute_accuracy(
        self,
        results: list[TaskResult],
    ) -> tuple[float, list[bool]]:
        """Compute accuracy for a set of results.

        Args:
            results: Task results to evaluate.

        Returns:
            Tuple of (accuracy, list of correctness values).
        """
        correct = []

        for result in results:
            if not result.success:
                correct.append(False)
                continue

            task = self.tasks_by_id.get(result.task_id)
            if not task:
                correct.append(False)
                continue

            response = result.response
            if not response or not response.parsed_answer:
                correct.append(False)
                continue

            # Compare normalized answers
            model_answer = response.parsed_answer.normalized.lower().strip()
            correct_answer = str(task.correct_answer).lower().strip()

            # Check for match
            is_correct = (
                model_answer == correct_answer
                or correct_answer in model_answer
                or model_answer in correct_answer
            )
            correct.append(is_correct)

        if not correct:
            return 0.0, []

        return sum(correct) / len(correct), correct

    def compute_track_summary(
        self,
        track: TrackType,
    ) -> TrackSummary:
        """Compute summary for a single track.

        Args:
            track: Track to summarize.

        Returns:
            TrackSummary with all metrics.
        """
        results = self.results_by_track.get(track, [])

        completed = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Compute accuracy
        accuracy, correctness = self.compute_accuracy(results)

        # Compute mean confidence
        confidences = []
        for result in completed:
            if result.response and result.response.confidence:
                confidences.append(result.response.confidence.numeric)

        mean_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Compute mean latency
        latencies = [r.latency_ms for r in results]
        mean_latency = float(np.mean(latencies)) if latencies else 0.0

        # Compute calibration metrics for calibration track
        calibration = None
        if track == TrackType.CALIBRATION and confidences and correctness:
            computer = CalibrationMetricsComputer(n_bins=10)
            calibration = computer.compute_all(
                np.array(confidences),
                np.array([float(c) for c in correctness[: len(confidences)]]),
            )

        return TrackSummary(
            track=track,
            total_tasks=len(results),
            completed_tasks=len(completed),
            failed_tasks=len(failed),
            accuracy=accuracy,
            mean_confidence=mean_confidence,
            mean_latency_ms=mean_latency,
            calibration=calibration,
            task_results=results,
        )

    def compute_overall_metrics(
        self,
        track_summaries: dict[TrackType, TrackSummary],
    ) -> dict[str, Any]:
        """Compute overall benchmark metrics.

        Args:
            track_summaries: Per-track summaries.

        Returns:
            Dictionary of overall metrics.
        """
        # Aggregate across tracks
        total_tasks = sum(s.total_tasks for s in track_summaries.values())
        total_completed = sum(s.completed_tasks for s in track_summaries.values())
        total_failed = sum(s.failed_tasks for s in track_summaries.values())

        # Weighted accuracy
        accuracies = []
        weights = []
        for summary in track_summaries.values():
            if summary.completed_tasks > 0:
                accuracies.append(summary.accuracy)
                weights.append(summary.completed_tasks)

        overall_accuracy = float(np.average(accuracies, weights=weights)) if accuracies else 0.0

        # Mean confidence across tracks
        confidences = []
        for summary in track_summaries.values():
            if summary.mean_confidence > 0:
                confidences.append(summary.mean_confidence)

        overall_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Mean latency
        latencies = [s.mean_latency_ms for s in track_summaries.values() if s.mean_latency_ms > 0]
        overall_latency = float(np.mean(latencies)) if latencies else 0.0

        # ECE from calibration track
        ece = None
        if TrackType.CALIBRATION in track_summaries:
            cal_summary = track_summaries[TrackType.CALIBRATION]
            if cal_summary.calibration:
                ece = cal_summary.calibration.ece.value

        return {
            "total_tasks": total_tasks,
            "completed_tasks": total_completed,
            "failed_tasks": total_failed,
            "success_rate": (total_completed / total_tasks * 100) if total_tasks > 0 else 0.0,
            "overall_accuracy": overall_accuracy,
            "overall_confidence": overall_confidence,
            "overall_latency_ms": overall_latency,
            "calibration_ece": ece,
            "tracks_evaluated": len(track_summaries),
        }

    def generate_report(self) -> BenchmarkReport:
        """Generate the full benchmark report.

        Returns:
            BenchmarkReport with all results and analysis.
        """
        from chimera.runner.report import BenchmarkReport, ReportSection

        # Compute per-track summaries
        track_summaries = {}
        for track in self.results_by_track:
            track_summaries[track] = self.compute_track_summary(track)

        # Compute overall metrics
        overall = self.compute_overall_metrics(track_summaries)

        # Build report sections
        sections = []

        # Overview section
        sections.append(
            ReportSection(
                title="Overview",
                content={
                    "run_name": self.config.name,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.config.metadata.get("model_name", "unknown"),
                    **overall,
                },
            )
        )

        # Per-track sections
        for track, summary in track_summaries.items():
            track_str = track.value if hasattr(track, "value") else str(track)
            sections.append(
                ReportSection(
                    title=f"Track: {track_str.replace('_', ' ').title()}",
                    content=summary.to_dict(),
                )
            )

        return BenchmarkReport(
            name=self.config.name,
            config=self.config,
            track_summaries=track_summaries,
            overall_metrics=overall,
            sections=sections,
            created_at=datetime.now(),
        )
