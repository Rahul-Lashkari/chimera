"""CHIMERA benchmark runner package.

This module provides the orchestration layer for running CHIMERA benchmarks,
managing execution, tracking progress, and aggregating results.

Core Components:
    - BenchmarkRunner: Main orchestrator for benchmark execution
    - RunConfig: Configuration for benchmark runs
    - ResultsAggregator: Combines results across tracks
    - BenchmarkReport: Structured output with analysis

Example:
    >>> from chimera.runner import BenchmarkRunner, RunConfig
    >>> from chimera import GeminiModel, GeminiConfig
    >>>
    >>> # Configure and run benchmark
    >>> model = GeminiModel(GeminiConfig())
    >>> config = RunConfig(tracks=["calibration"], max_tasks=100)
    >>> runner = BenchmarkRunner(model, config)
    >>>
    >>> # Run with progress tracking
    >>> report = runner.run()
    >>> print(report.summary())
    >>> report.save("results/gemini_benchmark.json")
"""

from chimera.runner.aggregator import (
    ResultsAggregator,
    TrackSummary,
)
from chimera.runner.config import (
    OutputFormat,
    ProgressCallback,
    RunConfig,
    TrackConfig,
)
from chimera.runner.executor import (
    BenchmarkRunner,
    ExecutionState,
    TaskResult,
)
from chimera.runner.report import (
    BenchmarkReport,
    ReportSection,
    export_report,
)

__all__ = [
    # Configuration
    "OutputFormat",
    "ProgressCallback",
    "RunConfig",
    "TrackConfig",
    # Execution
    "BenchmarkRunner",
    "ExecutionState",
    "TaskResult",
    # Aggregation
    "ResultsAggregator",
    "TrackSummary",
    # Reporting
    "BenchmarkReport",
    "ReportSection",
    "export_report",
]
