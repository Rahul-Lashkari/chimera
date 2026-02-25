"""CHIMERA evaluation package.

This module provides the unified evaluation pipeline for running
multi-track benchmarks, aggregating cross-track results, and
comparing model performance.

Example:
    >>> from chimera.evaluation import EvaluationPipeline, PipelineConfig
    >>>
    >>> # Configure pipeline
    >>> config = PipelineConfig(
    ...     tracks=["calibration", "error_detection", "knowledge_boundary", "self_correction"],
    ...     model_provider="gemini",
    ...     model_name="gemini-2.0-flash",
    ... )
    >>>
    >>> # Run evaluation
    >>> pipeline = EvaluationPipeline(config)
    >>> results = pipeline.run()
    >>>
    >>> # Generate report
    >>> report = results.generate_report(format="html")
"""

from chimera.evaluation.aggregation import (
    CrossTrackAggregator,
    CrossTrackSummary,
    SimpleTrackSummary,
    TrackCorrelation,
)
from chimera.evaluation.comparison import (
    ModelComparison,
    ModelRanking,
    PerformanceDelta,
)
from chimera.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    PipelineConfig,
    PipelineStage,
    TrackEvaluation,
)

__all__ = [
    # Pipeline
    "EvaluationPipeline",
    "EvaluationResult",
    "PipelineConfig",
    "PipelineStage",
    "TrackEvaluation",
    # Aggregation
    "CrossTrackAggregator",
    "CrossTrackSummary",
    "SimpleTrackSummary",
    "TrackCorrelation",
    # Comparison
    "ModelComparison",
    "ModelRanking",
    "PerformanceDelta",
]
