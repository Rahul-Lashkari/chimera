"""Model comparison utilities for CHIMERA benchmarks.

This module provides tools for comparing multiple models on
benchmark results, computing performance deltas, and generating
rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from chimera.evaluation.aggregation import CrossTrackAggregator, CrossTrackSummary


class ComparisonMetric(str, Enum):
    """Metrics available for comparison."""

    ACCURACY = "accuracy"
    CORRECT_TASKS = "correct_tasks"
    TOTAL_TASKS = "total_tasks"
    WEIGHTED_SCORE = "weighted_score"


@dataclass
class PerformanceDelta:
    """Performance difference between two models.

    Attributes:
        model_a: First model identifier.
        model_b: Second model identifier.
        track: Track being compared (or "overall").
        metric: Metric being compared.
        value_a: Value for model A.
        value_b: Value for model B.
        delta: Absolute difference (B - A).
        delta_percent: Percentage difference.
    """

    model_a: str
    model_b: str
    track: str
    metric: ComparisonMetric
    value_a: float
    value_b: float
    delta: float = field(init=False)
    delta_percent: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute delta and percentage."""
        self.delta = self.value_b - self.value_a
        if self.value_a != 0:
            self.delta_percent = (self.delta / self.value_a) * 100
        else:
            self.delta_percent = 0.0 if self.delta == 0 else float("inf")

    @property
    def winner(self) -> str | None:
        """Get the model with better performance."""
        if self.delta > 0:
            return self.model_b
        elif self.delta < 0:
            return self.model_a
        return None  # Tie

    @property
    def is_significant(self) -> bool:
        """Check if the difference is significant (>5%)."""
        return abs(self.delta_percent) >= 5.0


class ModelRanking(BaseModel):
    """Ranking of a model within a comparison.

    Attributes:
        model_id: Identifier for the model.
        rank: Numerical rank (1 = best).
        overall_score: Overall weighted score.
        track_scores: Scores for each track.
        track_ranks: Rank for each track.
        strengths: Tracks where model performs best.
        weaknesses: Tracks where model performs worst.
    """

    model_id: str = Field(...)
    rank: int = Field(ge=1)
    overall_score: float = Field(ge=0.0, le=1.0)
    track_scores: dict[str, float] = Field(default_factory=dict)
    track_ranks: dict[str, int] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    def to_summary(self) -> str:
        """Generate a text summary of the ranking."""
        lines = [
            f"Model: {self.model_id}",
            f"Overall Rank: #{self.rank}",
            f"Overall Score: {self.overall_score:.1%}",
        ]

        if self.track_scores:
            lines.append("\nTrack Scores:")
            for track, score in sorted(self.track_scores.items(), key=lambda x: x[1], reverse=True):
                rank = self.track_ranks.get(track, 0)
                lines.append(f"  - {track}: {score:.1%} (#{rank})")

        if self.strengths:
            lines.append(f"\nStrengths: {', '.join(self.strengths)}")

        if self.weaknesses:
            lines.append(f"Weaknesses: {', '.join(self.weaknesses)}")

        return "\n".join(lines)


class ModelComparison:
    """Compare multiple models on CHIMERA benchmarks.

    This class manages results from multiple models and provides
    utilities for computing deltas, rankings, and comparisons.

    Example:
        >>> comparison = ModelComparison()
        >>> comparison.add_model_results("gpt-4", gpt4_results)
        >>> comparison.add_model_results("gemini-2.0", gemini_results)
        >>> rankings = comparison.compute_rankings()
        >>> deltas = comparison.compute_deltas("gpt-4", "gemini-2.0")
    """

    def __init__(
        self,
        track_weights: dict[str, float] | None = None,
    ):
        """Initialize model comparison.

        Args:
            track_weights: Optional weights for each track.
        """
        self._track_weights = track_weights or {}
        self._model_summaries: dict[str, CrossTrackSummary] = {}
        self._model_metadata: dict[str, dict[str, Any]] = {}

    def add_model_results(
        self,
        model_id: str,
        results: dict[str, list[Any]],
        metadata: dict[str, Any] | None = None,
    ) -> CrossTrackSummary:
        """Add results for a model.

        Args:
            model_id: Identifier for the model.
            results: Dictionary mapping track names to result lists.
            metadata: Optional metadata about the model.

        Returns:
            CrossTrackSummary for the model.
        """
        aggregator = CrossTrackAggregator(track_weights=self._track_weights)

        for track, track_results in results.items():
            aggregator.add_track_results(track, track_results)

        summary = aggregator.compute_summary()
        self._model_summaries[model_id] = summary
        self._model_metadata[model_id] = metadata or {}

        return summary

    def add_model_summary(
        self,
        model_id: str,
        summary: CrossTrackSummary,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a pre-computed summary for a model.

        Args:
            model_id: Identifier for the model.
            summary: Pre-computed CrossTrackSummary.
            metadata: Optional metadata about the model.
        """
        self._model_summaries[model_id] = summary
        self._model_metadata[model_id] = metadata or {}

    def get_model_summary(self, model_id: str) -> CrossTrackSummary | None:
        """Get summary for a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            CrossTrackSummary or None if not found.
        """
        return self._model_summaries.get(model_id)

    def compute_rankings(
        self,
        metric: ComparisonMetric = ComparisonMetric.ACCURACY,
    ) -> list[ModelRanking]:
        """Compute rankings for all models.

        Args:
            metric: Metric to use for ranking.

        Returns:
            List of ModelRanking objects sorted by rank.
        """
        if not self._model_summaries:
            return []

        # Compute scores based on metric
        scores: list[tuple[str, float]] = []
        for model_id, summary in self._model_summaries.items():
            if metric == ComparisonMetric.ACCURACY:
                score = summary.overall_accuracy
            elif metric == ComparisonMetric.CORRECT_TASKS:
                score = float(summary.total_correct)
            elif metric == ComparisonMetric.TOTAL_TASKS:
                score = float(summary.total_tasks)
            else:
                score = summary.overall_accuracy
            scores.append((model_id, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Compute per-track rankings
        track_rankings = self._compute_track_rankings()

        # Build ModelRanking objects
        rankings = []
        for rank, (model_id, overall_score) in enumerate(scores, 1):
            summary = self._model_summaries[model_id]

            track_scores = {track: ts.accuracy for track, ts in summary.track_summaries.items()}

            track_ranks = {}
            for track in track_scores:
                if track in track_rankings:
                    for r, (m, _) in enumerate(track_rankings[track], 1):
                        if m == model_id:
                            track_ranks[track] = r
                            break

            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            for track, r in track_ranks.items():
                if r == 1:
                    strengths.append(track)
                elif r == len(self._model_summaries):
                    weaknesses.append(track)

            ranking = ModelRanking(
                model_id=model_id,
                rank=rank,
                overall_score=overall_score,
                track_scores=track_scores,
                track_ranks=track_ranks,
                strengths=strengths,
                weaknesses=weaknesses,
            )
            rankings.append(ranking)

        return rankings

    def _compute_track_rankings(self) -> dict[str, list[tuple[str, float]]]:
        """Compute rankings for each track.

        Returns:
            Dictionary mapping track names to sorted (model, score) tuples.
        """
        # Collect all tracks
        all_tracks: set[str] = set()
        for summary in self._model_summaries.values():
            all_tracks.update(summary.track_summaries.keys())

        # Rank models for each track
        track_rankings = {}
        for track in all_tracks:
            track_scores = []
            for model_id, summary in self._model_summaries.items():
                if track in summary.track_summaries:
                    score = summary.track_summaries[track].accuracy
                    track_scores.append((model_id, score))
            track_scores.sort(key=lambda x: x[1], reverse=True)
            track_rankings[track] = track_scores

        return track_rankings

    def compute_deltas(
        self,
        model_a: str,
        model_b: str,
        metric: ComparisonMetric = ComparisonMetric.ACCURACY,
    ) -> list[PerformanceDelta]:
        """Compute performance deltas between two models.

        Args:
            model_a: First model identifier.
            model_b: Second model identifier.
            metric: Metric to compare.

        Returns:
            List of PerformanceDelta objects for each track and overall.

        Raises:
            ValueError: If either model is not found.
        """
        if model_a not in self._model_summaries:
            raise ValueError(f"Model not found: {model_a}")
        if model_b not in self._model_summaries:
            raise ValueError(f"Model not found: {model_b}")

        summary_a = self._model_summaries[model_a]
        summary_b = self._model_summaries[model_b]

        deltas = []

        # Overall delta
        overall_delta = PerformanceDelta(
            model_a=model_a,
            model_b=model_b,
            track="overall",
            metric=metric,
            value_a=summary_a.overall_accuracy,
            value_b=summary_b.overall_accuracy,
        )
        deltas.append(overall_delta)

        # Per-track deltas
        all_tracks = set(summary_a.track_summaries.keys()) | set(summary_b.track_summaries.keys())

        for track in all_tracks:
            value_a = 0.0
            value_b = 0.0

            if track in summary_a.track_summaries:
                if metric == ComparisonMetric.ACCURACY:
                    value_a = summary_a.track_summaries[track].accuracy
                elif metric == ComparisonMetric.CORRECT_TASKS:
                    value_a = float(summary_a.track_summaries[track].correct_tasks)

            if track in summary_b.track_summaries:
                if metric == ComparisonMetric.ACCURACY:
                    value_b = summary_b.track_summaries[track].accuracy
                elif metric == ComparisonMetric.CORRECT_TASKS:
                    value_b = float(summary_b.track_summaries[track].correct_tasks)

            delta = PerformanceDelta(
                model_a=model_a,
                model_b=model_b,
                track=track,
                metric=metric,
                value_a=value_a,
                value_b=value_b,
            )
            deltas.append(delta)

        return deltas

    def compute_all_deltas(
        self,
        metric: ComparisonMetric = ComparisonMetric.ACCURACY,
    ) -> list[PerformanceDelta]:
        """Compute deltas for all model pairs.

        Args:
            metric: Metric to compare.

        Returns:
            List of all PerformanceDelta objects.
        """
        deltas = []
        models = list(self._model_summaries.keys())

        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                deltas.extend(self.compute_deltas(model_a, model_b, metric))

        return deltas

    def get_best_model(
        self,
        track: str | None = None,
        metric: ComparisonMetric = ComparisonMetric.ACCURACY,
    ) -> str | None:
        """Get the best performing model.

        Args:
            track: Optional track to consider. None for overall.
            metric: Metric to use for comparison.

        Returns:
            Model identifier or None if no models.
        """
        if not self._model_summaries:
            return None

        if track is None:
            # Overall best
            rankings = self.compute_rankings(metric)
            return rankings[0].model_id if rankings else None
        else:
            # Best for specific track
            best_model = None
            best_score = -1.0

            for model_id, summary in self._model_summaries.items():
                if track in summary.track_summaries:
                    if metric == ComparisonMetric.ACCURACY:
                        score = summary.track_summaries[track].accuracy
                    else:
                        score = float(summary.track_summaries[track].correct_tasks)

                    if score > best_score:
                        best_score = score
                        best_model = model_id

            return best_model

    def generate_comparison_report(self, format: str = "markdown") -> str:
        """Generate a comparison report.

        Args:
            format: Output format ("markdown" or "html").

        Returns:
            Report content as a string.
        """
        rankings = self.compute_rankings()

        if format == "markdown":
            return self._generate_markdown_report(rankings)
        elif format == "html":
            return self._generate_html_report(rankings)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown_report(self, rankings: list[ModelRanking]) -> str:
        """Generate Markdown comparison report."""
        lines = [
            "# CHIMERA Model Comparison Report",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "## Overall Rankings",
            "",
            "| Rank | Model | Overall Score |",
            "|------|-------|---------------|",
        ]

        for ranking in rankings:
            lines.append(
                f"| #{ranking.rank} | {ranking.model_id} | " f"{ranking.overall_score:.1%} |"
            )

        lines.extend(["", "## Per-Track Performance", ""])

        # Collect all tracks
        all_tracks: set[str] = set()
        for ranking in rankings:
            all_tracks.update(ranking.track_scores.keys())

        for track in sorted(all_tracks):
            lines.extend([f"### {track.replace('_', ' ').title()}", ""])

            track_scores = []
            for ranking in rankings:
                if track in ranking.track_scores:
                    track_scores.append(
                        (
                            ranking.model_id,
                            ranking.track_scores[track],
                            ranking.track_ranks.get(track, 0),
                        )
                    )

            track_scores.sort(key=lambda x: x[1], reverse=True)

            for model, score, rank in track_scores:
                lines.append(f"- **#{rank} {model}**: {score:.1%}")

            lines.append("")

        # Significant deltas
        lines.extend(["## Significant Performance Differences", ""])

        deltas = self.compute_all_deltas()
        significant = [d for d in deltas if d.is_significant and d.track != "overall"]

        if significant:
            for delta in significant[:10]:  # Top 10
                direction = "↑" if delta.delta > 0 else "↓"
                lines.append(
                    f"- **{delta.track}**: {delta.model_b} vs {delta.model_a}: "
                    f"{direction} {abs(delta.delta_percent):.1f}%"
                )
        else:
            lines.append("No significant differences found.")

        return "\n".join(lines)

    def _generate_html_report(self, rankings: list[ModelRanking]) -> str:
        """Generate HTML comparison report."""
        # Generate table rows for rankings
        ranking_rows = ""
        for ranking in rankings:
            ranking_rows += f"""
            <tr>
                <td>#{ranking.rank}</td>
                <td><strong>{ranking.model_id}</strong></td>
                <td>{ranking.overall_score:.1%}</td>
                <td>{', '.join(ranking.strengths) if ranking.strengths else '-'}</td>
                <td>{', '.join(ranking.weaknesses) if ranking.weaknesses else '-'}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CHIMERA Model Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f4f8;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2d3748;
            border-bottom: 3px solid #4299e1;
            padding-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{ background: #f7fafc; }}
        .rank-1 {{ background: linear-gradient(90deg, #fef5e7 0%, white 100%); }}
        .rank-2 {{ background: linear-gradient(90deg, #f5f5f5 0%, white 100%); }}
        .rank-3 {{ background: linear-gradient(90deg, #fdf5ef 0%, white 100%); }}
        .medal {{
            font-size: 24px;
            margin-right: 8px;
        }}
        .footer {{
            text-align: center;
            color: #718096;
            margin-top: 30px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 CHIMERA Model Comparison</h1>

        <h2>Overall Rankings</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Overall Score</th>
                    <th>Strengths</th>
                    <th>Weaknesses</th>
                </tr>
            </thead>
            <tbody>
                {ranking_rows}
            </tbody>
        </table>

        <div class="footer">
            <p>Generated: {datetime.now().isoformat()}</p>
            <p>CHIMERA Benchmark Suite</p>
        </div>
    </div>
</body>
</html>"""
        return html

    @property
    def model_ids(self) -> list[str]:
        """Get list of all model IDs."""
        return list(self._model_summaries.keys())

    @property
    def num_models(self) -> int:
        """Get number of models being compared."""
        return len(self._model_summaries)
