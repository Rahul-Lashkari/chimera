"""Cross-track result aggregation for CHIMERA benchmarks.

This module provides utilities for aggregating and analyzing results
across multiple evaluation tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class TrackCorrelation:
    """Correlation between two tracks.

    Attributes:
        track_a: First track name.
        track_b: Second track name.
        correlation: Correlation coefficient (-1.0 to 1.0).
        sample_size: Number of samples used for correlation.
        p_value: Statistical p-value (if computed).
    """

    track_a: str
    track_b: str
    correlation: float
    sample_size: int
    p_value: float | None = None

    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant."""
        if self.p_value is None:
            return False
        return self.p_value < 0.05

    @property
    def strength(self) -> str:
        """Categorize correlation strength."""
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"


@dataclass
class SimpleTrackSummary:
    """Lightweight track summary for cross-track aggregation.

    This is a standalone summary structure that doesn't depend on
    the full runner infrastructure.

    Attributes:
        track: Track name.
        total_tasks: Number of tasks in track.
        correct_tasks: Number of correct tasks.
        accuracy: Accuracy score (0.0 to 1.0).
        metrics: Additional track-specific metrics.
    """

    track: str
    total_tasks: int = 0
    correct_tasks: int = 0
    accuracy: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute accuracy if not provided."""
        if self.total_tasks > 0 and self.accuracy == 0.0:
            self.accuracy = self.correct_tasks / self.total_tasks


class CrossTrackSummary(BaseModel):
    """Summary of results across all tracks.

    Attributes:
        track_summaries: Summary for each individual track.
        overall_accuracy: Weighted average accuracy across tracks.
        total_tasks: Total number of tasks across all tracks.
        total_correct: Total number of correct tasks.
        track_weights: Weights used for each track.
        correlations: Correlations between track performances.
        best_track: Track with highest accuracy.
        worst_track: Track with lowest accuracy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    track_summaries: dict[str, SimpleTrackSummary] = Field(default_factory=dict)
    overall_accuracy: float = Field(default=0.0)
    total_tasks: int = Field(default=0)
    total_correct: int = Field(default=0)
    track_weights: dict[str, float] = Field(default_factory=dict)
    correlations: list[TrackCorrelation] = Field(default_factory=list)
    best_track: str | None = Field(default=None)
    worst_track: str | None = Field(default=None)

    def get_accuracy_ranking(self) -> list[tuple[str, float]]:
        """Get tracks ranked by accuracy (descending)."""
        return sorted(
            [(track, summary.accuracy) for track, summary in self.track_summaries.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_track_contribution(self) -> dict[str, float]:
        """Get each track's contribution to overall score."""
        if not self.track_summaries:
            return {}

        contributions = {}
        total_weighted = sum(
            summary.accuracy * self.track_weights.get(track, 1.0)
            for track, summary in self.track_summaries.items()
        )

        for track, summary in self.track_summaries.items():
            weight = self.track_weights.get(track, 1.0)
            if total_weighted > 0:
                contributions[track] = (summary.accuracy * weight) / total_weighted
            else:
                contributions[track] = 0.0

        return contributions


class CrossTrackAggregator:
    """Aggregates results across multiple evaluation tracks.

    This class provides cross-track analysis, correlations, and
    unified summaries without depending on the full runner infrastructure.

    Example:
        >>> aggregator = CrossTrackAggregator()
        >>> aggregator.add_track_summary("calibration", SimpleTrackSummary(...))
        >>> aggregator.add_track_summary("error_detection", SimpleTrackSummary(...))
        >>> summary = aggregator.compute_summary()
        >>> print(f"Overall: {summary.overall_accuracy:.1%}")
    """

    def __init__(
        self,
        track_weights: dict[str, float] | None = None,
    ):
        """Initialize the cross-track aggregator.

        Args:
            track_weights: Optional weights for each track.
                          Defaults to equal weighting.
        """
        self._track_weights = track_weights or {}
        self._track_summaries: dict[str, SimpleTrackSummary] = {}
        self._track_results: dict[str, list[Any]] = {}

    def add_track_summary(self, track: str, summary: SimpleTrackSummary) -> None:
        """Add a summary for a specific track.

        Args:
            track: Name of the track.
            summary: Track summary.
        """
        self._track_summaries[track] = summary

    def add_track_results(self, track: str, results: list[Any]) -> None:
        """Add raw results for a specific track.

        Args:
            track: Name of the track.
            results: List of task results.
        """
        self._track_results[track] = results

    def compute_summary(self) -> CrossTrackSummary:
        """Compute cross-track summary.

        Returns:
            CrossTrackSummary with aggregated metrics.
        """
        total_tasks = 0
        total_correct = 0

        for summary in self._track_summaries.values():
            total_tasks += summary.total_tasks
            total_correct += summary.correct_tasks

        # Compute weighted average accuracy
        if self._track_summaries:
            total_weight = sum(
                self._track_weights.get(track, 1.0) for track in self._track_summaries
            )
            weighted_sum = sum(
                summary.accuracy * self._track_weights.get(track, 1.0)
                for track, summary in self._track_summaries.items()
            )
            overall_accuracy = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall_accuracy = 0.0

        # Find best and worst tracks
        if self._track_summaries:
            accuracies = {track: s.accuracy for track, s in self._track_summaries.items()}
            best_track = max(accuracies, key=accuracies.get)  # type: ignore[arg-type]
            worst_track = min(accuracies, key=accuracies.get)  # type: ignore[arg-type]
        else:
            best_track = None
            worst_track = None

        # Compute correlations
        correlations = self._compute_correlations()

        return CrossTrackSummary(
            track_summaries=dict(self._track_summaries),
            overall_accuracy=overall_accuracy,
            total_tasks=total_tasks,
            total_correct=total_correct,
            track_weights=self._get_effective_weights(),
            correlations=correlations,
            best_track=best_track,
            worst_track=worst_track,
        )

    def _get_effective_weights(self) -> dict[str, float]:
        """Get effective weights for all tracks."""
        weights = {}
        for track in self._track_summaries:
            weights[track] = self._track_weights.get(track, 1.0)
        return weights

    def _compute_correlations(self) -> list[TrackCorrelation]:
        """Compute correlations between track performances.

        Returns:
            List of TrackCorrelation objects.
        """
        correlations = []
        tracks = list(self._track_summaries.keys())

        # For now, return placeholder correlations
        # Full implementation would compute task-level correlations
        # when tasks overlap across tracks

        for i, track_a in enumerate(tracks):
            for track_b in tracks[i + 1 :]:
                # Placeholder correlation
                correlation = TrackCorrelation(
                    track_a=track_a,
                    track_b=track_b,
                    correlation=0.0,
                    sample_size=0,
                    p_value=None,
                )
                correlations.append(correlation)

        return correlations

    def get_track_summary(self, track: str) -> SimpleTrackSummary | None:
        """Get summary for a specific track.

        Args:
            track: Name of the track.

        Returns:
            SimpleTrackSummary or None if track not found.
        """
        return self._track_summaries.get(track)

    def get_all_results(self) -> dict[str, list[Any]]:
        """Get all results organized by track."""
        return dict(self._track_results)

    def clear(self) -> None:
        """Clear all stored results."""
        self._track_summaries.clear()
        self._track_results.clear()

    def set_track_weight(self, track: str, weight: float) -> None:
        """Set the weight for a specific track.

        Args:
            track: Name of the track.
            weight: Weight value (> 0).

        Raises:
            ValueError: If weight is not positive.
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")
        self._track_weights[track] = weight

    def get_improvement_suggestions(self) -> list[str]:
        """Generate suggestions for improving model performance.

        Returns:
            List of improvement suggestions.
        """
        suggestions = []
        summary = self.compute_summary()

        if summary.worst_track:
            worst_summary = summary.track_summaries.get(summary.worst_track)
            if worst_summary and worst_summary.accuracy < 0.5:
                suggestions.append(
                    f"Focus on improving {summary.worst_track} performance "
                    f"(currently {worst_summary.accuracy:.1%})"
                )

        # Check for large accuracy gaps
        if len(summary.track_summaries) >= 2:
            accuracies = [s.accuracy for s in summary.track_summaries.values()]
            gap = max(accuracies) - min(accuracies)
            if gap > 0.2:
                suggestions.append(
                    f"Large performance gap ({gap:.1%}) between tracks - "
                    "consider specialized training"
                )

        # Overall accuracy suggestions
        if summary.overall_accuracy < 0.6:
            suggestions.append("Overall accuracy below 60% - model may need additional training")
        elif summary.overall_accuracy < 0.8:
            suggestions.append("Good baseline performance - fine-tuning may help reach 80%+")

        return suggestions
