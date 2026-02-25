"""Unified evaluation pipeline for CHIMERA benchmarks.

This module provides the core pipeline for orchestrating multi-track
evaluation, managing execution flow, and collecting results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from chimera.evaluation.aggregation import CrossTrackAggregator, SimpleTrackSummary


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    INITIALIZATION = "initialization"
    CALIBRATION = "calibration"
    ERROR_DETECTION = "error_detection"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    SELF_CORRECTION = "self_correction"
    AGGREGATION = "aggregation"
    REPORTING = "reporting"
    COMPLETE = "complete"
    FAILED = "failed"


class TrackType(str, Enum):
    """Available evaluation tracks."""

    CALIBRATION = "calibration"
    ERROR_DETECTION = "error_detection"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    SELF_CORRECTION = "self_correction"


class PipelineConfig(BaseModel):
    """Configuration for the evaluation pipeline.

    Attributes:
        tracks: List of tracks to evaluate.
        model_provider: Provider for the model (e.g., "gemini", "openai").
        model_name: Name of the model to evaluate.
        samples_per_track: Number of samples to evaluate per track.
        timeout_seconds: Timeout for each track evaluation.
        parallel: Whether to run tracks in parallel.
        output_dir: Directory for output files.
        cache_results: Whether to cache intermediate results.
        verbose: Enable verbose logging.
    """

    tracks: list[str] = Field(
        default_factory=lambda: [
            "calibration",
            "error_detection",
            "knowledge_boundary",
            "self_correction",
        ]
    )
    model_provider: str = Field(default="gemini")
    model_name: str = Field(default="gemini-2.0-flash")
    samples_per_track: int | None = Field(default=None)
    timeout_seconds: float = Field(default=3600.0)
    parallel: bool = Field(default=False)
    output_dir: str | None = Field(default=None)
    cache_results: bool = Field(default=True)
    verbose: bool = Field(default=False)

    @field_validator("tracks")
    @classmethod
    def validate_tracks(cls, v: list[str]) -> list[str]:
        """Validate that all tracks are valid."""
        valid_tracks = {t.value for t in TrackType}
        for track in v:
            if track not in valid_tracks:
                raise ValueError(f"Invalid track: {track}. Valid tracks: {valid_tracks}")
        return v


@dataclass
class TrackEvaluation:
    """Results from evaluating a single track.

    Attributes:
        track: The track that was evaluated.
        summary: Aggregated summary for the track.
        raw_results: List of individual task results.
        start_time: When evaluation started.
        end_time: When evaluation completed.
        error: Error message if evaluation failed.
    """

    track: str
    summary: SimpleTrackSummary | None = None
    raw_results: list[Any] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate evaluation duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        """Check if evaluation was successful."""
        return self.error is None and self.summary is not None


@dataclass
class EvaluationResult:
    """Results from a complete pipeline evaluation.

    Attributes:
        config: Pipeline configuration used.
        track_evaluations: Results for each track.
        overall_score: Weighted overall score across tracks.
        start_time: When pipeline started.
        end_time: When pipeline completed.
        stage: Final stage of the pipeline.
        metadata: Additional metadata.
    """

    config: PipelineConfig
    track_evaluations: dict[str, TrackEvaluation] = field(default_factory=dict)
    overall_score: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None
    stage: PipelineStage = PipelineStage.INITIALIZATION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Calculate total pipeline duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.stage == PipelineStage.COMPLETE

    @property
    def failed_tracks(self) -> list[str]:
        """Get list of tracks that failed."""
        return [
            track for track, evaluation in self.track_evaluations.items() if not evaluation.success
        ]

    def get_track_scores(self) -> dict[str, float]:
        """Get accuracy scores for each track."""
        return {
            track: evaluation.summary.accuracy
            for track, evaluation in self.track_evaluations.items()
            if evaluation.summary is not None
        }

    def generate_report(self, format: str = "markdown") -> str:
        """Generate a report in the specified format.

        Args:
            format: Output format - "markdown", "html", or "json".

        Returns:
            Report content as a string.
        """
        if format == "markdown":
            return self._generate_markdown_report()
        elif format == "json":
            return self._generate_json_report()
        elif format == "html":
            return self._generate_html_report()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown_report(self) -> str:
        """Generate a Markdown report."""
        lines = [
            "# CHIMERA Evaluation Report",
            "",
            f"**Model**: {self.config.model_provider}/{self.config.model_name}",
            f"**Overall Score**: {self.overall_score:.1%}",
            f"**Duration**: {self.duration_seconds:.1f}s",
            f"**Status**: {'✓ Complete' if self.success else '✗ Failed'}",
            "",
            "## Track Results",
            "",
            "| Track | Accuracy | Tasks | Correct | Duration | Status |",
            "|-------|----------|-------|---------|----------|--------|",
        ]

        for track, evaluation in self.track_evaluations.items():
            if evaluation.summary:
                lines.append(
                    f"| {track.replace('_', ' ').title()} | "
                    f"{evaluation.summary.accuracy:.1%} | "
                    f"{evaluation.summary.total_tasks} | "
                    f"{evaluation.summary.correct_tasks} | "
                    f"{evaluation.duration_seconds:.1f}s | "
                    f"✓ Complete |"
                )
            else:
                lines.append(
                    f"| {track.replace('_', ' ').title()} | "
                    f"- | - | - | "
                    f"{evaluation.duration_seconds:.1f}s | "
                    f"✗ {evaluation.error or 'Failed'} |"
                )

        lines.extend(
            [
                "",
                "## Configuration",
                "",
                f"- Tracks: {', '.join(self.config.tracks)}",
                f"- Timeout: {self.config.timeout_seconds}s",
                f"- Parallel: {self.config.parallel}",
                "",
                f"*Generated: {datetime.now().isoformat()}*",
            ]
        )

        return "\n".join(lines)

    def _generate_json_report(self) -> str:
        """Generate a JSON report."""
        import json

        tracks_data: dict[str, Any] = {}
        report_data: dict[str, Any] = {
            "model": {
                "provider": self.config.model_provider,
                "name": self.config.model_name,
            },
            "overall_score": self.overall_score,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "stage": self.stage.value,
            "tracks": tracks_data,
            "config": self.config.model_dump(),
            "generated_at": datetime.now().isoformat(),
        }

        for track, evaluation in self.track_evaluations.items():
            track_data: dict[str, Any] = {
                "duration_seconds": evaluation.duration_seconds,
                "success": evaluation.success,
                "error": evaluation.error,
            }
            if evaluation.summary:
                track_data.update(
                    {
                        "accuracy": evaluation.summary.accuracy,
                        "total_tasks": evaluation.summary.total_tasks,
                        "correct_tasks": evaluation.summary.correct_tasks,
                    }
                )
            tracks_data[track] = track_data

        return json.dumps(report_data, indent=2, default=str)

    def _generate_html_report(self) -> str:
        """Generate an HTML report."""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CHIMERA Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a73e8;
            border-bottom: 2px solid #1a73e8;
            padding-bottom: 10px;
        }}
        h2 {{ color: #333; margin-top: 30px; }}
        h3 {{ color: #666; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background: #1a73e8;
            color: white;
        }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .score {{
            font-size: 24px;
            font-weight: bold;
            color: #1a73e8;
        }}
        .metric-card {{
            display: inline-block;
            padding: 20px;
            margin: 10px;
            background: #e8f0fe;
            border-radius: 8px;
            min-width: 150px;
        }}
        .metric-card h4 {{
            margin: 0;
            color: #666;
            font-size: 14px;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #1a73e8;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .success {{ color: #0d904f; }}
        .error {{ color: #d93025; }}
        .metadata {{
            font-size: 12px;
            color: #888;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 CHIMERA Evaluation Report</h1>

        <div class="metrics">
            <div class="metric-card">
                <h4>Overall Score</h4>
                <div class="value">{self.overall_score:.1%}</div>
            </div>
            <div class="metric-card">
                <h4>Tracks Evaluated</h4>
                <div class="value">{len(self.track_evaluations)}</div>
            </div>
            <div class="metric-card">
                <h4>Duration</h4>
                <div class="value">{self.duration_seconds:.1f}s</div>
            </div>
            <div class="metric-card">
                <h4>Status</h4>
                <div class="value {'success' if self.success else 'error'}">{
                    '✓ Complete' if self.success else '✗ Failed'
                }</div>
            </div>
        </div>

        <h2>Track Results</h2>
        <table>
            <tr>
                <th>Track</th>
                <th>Accuracy</th>
                <th>Total Tasks</th>
                <th>Correct</th>
                <th>Duration</th>
                <th>Status</th>
            </tr>
            {''.join(self._generate_track_row(track, eval)
                for track, eval in self.track_evaluations.items())}
        </table>

        <h2>Configuration</h2>
        <pre>{self.config.model_dump_json(indent=2)}</pre>

        <div class="metadata">
            <p>Model: {self.config.model_provider}/{self.config.model_name}</p>
            <p>Generated: {datetime.now().isoformat()}</p>
            <p>Pipeline: {self.stage.value}</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_track_row(self, track: str, evaluation: TrackEvaluation) -> str:
        """Generate an HTML table row for a track."""
        if evaluation.summary:
            return f"""
            <tr>
                <td><strong>{track.replace('_', ' ').title()}</strong></td>
                <td>{evaluation.summary.accuracy:.1%}</td>
                <td>{evaluation.summary.total_tasks}</td>
                <td>{evaluation.summary.correct_tasks}</td>
                <td>{evaluation.duration_seconds:.1f}s</td>
                <td class="success">✓ Complete</td>
            </tr>"""
        else:
            return f"""
            <tr>
                <td><strong>{track.replace('_', ' ').title()}</strong></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>{evaluation.duration_seconds:.1f}s</td>
                <td class="error">✗ {evaluation.error or 'Failed'}</td>
            </tr>"""


class EvaluationPipeline:
    """Unified evaluation pipeline for CHIMERA benchmarks.

    This class orchestrates the evaluation of models across multiple
    tracks, handling execution, aggregation, and reporting.

    Example:
        >>> config = PipelineConfig(
        ...     tracks=["calibration", "error_detection"],
        ...     model_name="gemini-2.0-flash"
        ... )
        >>> pipeline = EvaluationPipeline(config)
        >>> results = pipeline.run()
        >>> print(f"Overall score: {results.overall_score:.1%}")
    """

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Callable[[PipelineStage, str], None] | None = None,
    ):
        """Initialize the evaluation pipeline.

        Args:
            config: Pipeline configuration.
            progress_callback: Optional callback for progress updates.
        """
        self.config = config
        self.progress_callback = progress_callback
        self._logger = logging.getLogger(__name__)
        self._aggregator = CrossTrackAggregator()

        if config.verbose:
            logging.basicConfig(level=logging.DEBUG)

    def run(self) -> EvaluationResult:
        """Run the complete evaluation pipeline.

        Returns:
            EvaluationResult with all track evaluations and overall score.
        """
        result = EvaluationResult(
            config=self.config,
            start_time=datetime.now(),
            stage=PipelineStage.INITIALIZATION,
        )

        self._notify_progress(PipelineStage.INITIALIZATION, "Starting pipeline")

        try:
            # Run each track
            for track in self.config.tracks:
                stage = self._track_to_stage(track)
                self._notify_progress(stage, f"Evaluating {track}")

                evaluation = self._evaluate_track(track)
                result.track_evaluations[track] = evaluation
                result.stage = stage

            # Aggregate results
            self._notify_progress(PipelineStage.AGGREGATION, "Aggregating results")
            result.stage = PipelineStage.AGGREGATION
            result.overall_score = self._compute_overall_score(result)

            # Complete
            result.stage = PipelineStage.COMPLETE
            result.end_time = datetime.now()
            self._notify_progress(PipelineStage.COMPLETE, "Pipeline complete")

        except Exception as e:
            self._logger.exception("Pipeline failed")
            result.stage = PipelineStage.FAILED
            result.end_time = datetime.now()
            result.metadata["error"] = str(e)

        return result

    def _evaluate_track(self, track: str) -> TrackEvaluation:
        """Evaluate a single track.

        Args:
            track: Name of the track to evaluate.

        Returns:
            TrackEvaluation with results.
        """
        evaluation = TrackEvaluation(track=track, start_time=datetime.now())

        try:
            # Import and run the appropriate track evaluator
            results = self._run_track_evaluator(track)
            evaluation.raw_results = results

            # Aggregate results for this track
            if results:
                self._aggregator.add_track_results(track, results)
                evaluation.summary = self._aggregator.get_track_summary(track)

            evaluation.end_time = datetime.now()

        except Exception as e:
            self._logger.exception(f"Track {track} failed")
            evaluation.error = str(e)
            evaluation.end_time = datetime.now()

        return evaluation

    def _run_track_evaluator(self, track: str) -> list[Any]:
        """Run the evaluator for a specific track.

        This method would import and execute the appropriate track module.
        Currently returns empty results - actual implementation would
        instantiate track evaluators and run them.

        Args:
            track: Name of the track.

        Returns:
            List of task results.
        """
        # This is a placeholder - actual implementation would:
        # 1. Import the track module (e.g., chimera.tracks.calibration)
        # 2. Instantiate the track evaluator with config
        # 3. Run the evaluation
        # 4. Return results

        self._logger.info(f"Running evaluator for track: {track}")

        # For now, return empty results
        # In production, this would call into track-specific evaluators
        return []

    def _compute_overall_score(self, result: EvaluationResult) -> float:
        """Compute the weighted overall score across tracks.

        Args:
            result: The evaluation result with track evaluations.

        Returns:
            Weighted average score (0.0 to 1.0).
        """
        scores = result.get_track_scores()
        if not scores:
            return 0.0

        # Equal weighting for now
        return sum(scores.values()) / len(scores)

    def _track_to_stage(self, track: str) -> PipelineStage:
        """Convert track name to pipeline stage."""
        stage_map = {
            "calibration": PipelineStage.CALIBRATION,
            "error_detection": PipelineStage.ERROR_DETECTION,
            "knowledge_boundary": PipelineStage.KNOWLEDGE_BOUNDARY,
            "self_correction": PipelineStage.SELF_CORRECTION,
        }
        return stage_map.get(track, PipelineStage.INITIALIZATION)

    def _notify_progress(self, stage: PipelineStage, message: str) -> None:
        """Notify progress callback if registered."""
        if self.progress_callback:
            self.progress_callback(stage, message)
        if self.config.verbose:
            self._logger.info(f"[{stage.value}] {message}")

    def run_track(self, track: str) -> TrackEvaluation:
        """Run evaluation for a single track.

        Args:
            track: Name of the track to evaluate.

        Returns:
            TrackEvaluation with results.
        """
        if track not in [t.value for t in TrackType]:
            raise ValueError(f"Invalid track: {track}")

        return self._evaluate_track(track)

    def get_supported_tracks(self) -> list[str]:
        """Get list of supported evaluation tracks."""
        return [t.value for t in TrackType]
