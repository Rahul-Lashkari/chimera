# Evaluation API Reference

This document describes the evaluation pipeline and related APIs.

## Overview

The evaluation module provides:

- `EvaluationPipeline` - Main orchestrator for running evaluations
- `CrossTrackAggregator` - Aggregates results across tracks
- `ModelComparison` - Compare multiple models

## EvaluationPipeline

The main entry point for running CHIMERA evaluations.

### Basic Usage

```python
from chimera.evaluation import EvaluationPipeline, PipelineConfig

# Configure the pipeline
config = PipelineConfig(
    tracks=["calibration", "error_detection"],
    model_provider="gemini",
    model_name="gemini-2.0-flash",
    n_tasks=100,
)

# Run evaluation
pipeline = EvaluationPipeline(config)
results = pipeline.run()

# Access results
print(f"Overall Score: {results.overall_score:.2%}")
print(f"Success: {results.success}")
```

### PipelineConfig

Configuration for the evaluation pipeline:

```python
from chimera.evaluation import PipelineConfig

config = PipelineConfig(
    # Tracks to evaluate
    tracks=["calibration", "error_detection", "knowledge_boundary", "self_correction"],
    
    # Model settings
    model_provider="gemini",
    model_name="gemini-2.0-flash",
    
    # Generation settings
    n_tasks=100,
    seed=42,
    
    # Execution settings
    batch_size=10,
    max_retries=3,
    timeout=60,
    
    # Output settings
    output_dir="results/",
    verbose=True,
)
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tracks` | `list[str]` | All tracks | Tracks to evaluate |
| `model_provider` | `str` | "gemini" | Model provider |
| `model_name` | `str` | Provider default | Specific model |
| `n_tasks` | `int` | 100 | Tasks per track |
| `seed` | `int \| None` | None | Random seed |
| `batch_size` | `int` | 10 | Batch size for API calls |
| `max_retries` | `int` | 3 | Max retries on failure |
| `timeout` | `int` | 60 | Request timeout (seconds) |
| `output_dir` | `str \| None` | None | Output directory |
| `verbose` | `bool` | False | Verbose logging |

### EvaluationResult

The result of running the pipeline:

```python
results = pipeline.run()

# Overall metrics
print(results.overall_score)  # 0.0-1.0
print(results.success)  # True if all tracks completed
print(results.stage)  # Current/final stage

# Track-specific results
for track, evaluation in results.track_evaluations.items():
    print(f"{track}: {evaluation.summary.accuracy:.2%}")
    print(f"  Duration: {evaluation.duration_seconds:.1f}s")
    if evaluation.error:
        print(f"  Error: {evaluation.error}")

# Get scores as dict
scores = results.get_track_scores()
# {"calibration": 0.85, "error_detection": 0.72, ...}
```

### Report Generation

Generate reports in multiple formats:

```python
# Markdown report
md_report = results.generate_report(format="markdown")
with open("report.md", "w") as f:
    f.write(md_report)

# HTML report
html_report = results.generate_report(format="html")
with open("report.html", "w") as f:
    f.write(html_report)

# JSON report
json_report = results.generate_report(format="json")
with open("report.json", "w") as f:
    f.write(json_report)
```

### Progress Callbacks

Monitor progress during evaluation:

```python
def progress_callback(stage: PipelineStage, message: str):
    print(f"[{stage.value}] {message}")

pipeline = EvaluationPipeline(config, progress_callback=progress_callback)
results = pipeline.run()

# Output:
# [initialization] Starting pipeline
# [calibration] Evaluating calibration
# [error_detection] Evaluating error_detection
# [aggregation] Aggregating results
# [complete] Pipeline complete
```

### Running Individual Tracks

Run specific tracks independently:

```python
pipeline = EvaluationPipeline(config)

# Run single track
cal_result = pipeline.run_track("calibration")
print(f"Calibration ECE: {cal_result.summary.ece:.4f}")

# Get supported tracks
tracks = pipeline.get_supported_tracks()
# ["calibration", "error_detection", "knowledge_boundary", "self_correction"]
```

## CrossTrackAggregator

Aggregates results across multiple tracks:

```python
from chimera.evaluation import CrossTrackAggregator, SimpleTrackSummary

aggregator = CrossTrackAggregator()

# Add track results
aggregator.add_track_results("calibration", calibration_results)
aggregator.add_track_results("error_detection", error_detection_results)

# Get aggregated summary
summary = aggregator.compute_summary()
print(f"Overall Score: {summary.overall_score:.2%}")
print(f"Tracks: {summary.tracks}")

# Get per-track summary
cal_summary = aggregator.get_track_summary("calibration")
print(f"Calibration Accuracy: {cal_summary.accuracy:.2%}")
```

### CrossTrackSummary

Aggregated summary across tracks:

```python
from chimera.evaluation import CrossTrackSummary

summary = CrossTrackSummary(
    overall_score=0.78,
    tracks=["calibration", "error_detection"],
    track_summaries={
        "calibration": cal_summary,
        "error_detection": ed_summary,
    },
    correlations=[
        TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.65,
        )
    ],
)
```

## ModelComparison

Compare multiple models:

```python
from chimera.evaluation import ModelComparison, PerformanceDelta

comparison = ModelComparison()

# Add model results
comparison.add_model_results("gemini-2.0-flash", gemini_results)
comparison.add_model_results("gpt-4o", gpt4_results)

# Compute rankings
rankings = comparison.compute_rankings()
for rank in rankings:
    print(f"{rank.rank}. {rank.model_name}: {rank.score:.2%}")

# Get performance deltas
delta = comparison.compute_deltas("gemini-2.0-flash", "gpt-4o")
print(f"Score difference: {delta.overall_delta:+.2%}")
print(f"Per-track deltas: {delta.track_deltas}")

# Find best model
best_overall = comparison.get_best_model_overall()
best_per_track = comparison.get_best_model_per_track()
```

### Comparison Reports

Generate comparison reports:

```python
# Markdown comparison
md_report = comparison.generate_comparison_report(format="markdown")

# HTML comparison with charts
html_report = comparison.generate_comparison_report(format="html")
```

## Pipeline Stages

The pipeline progresses through stages:

```python
from chimera.evaluation import PipelineStage

class PipelineStage(str, Enum):
    INITIALIZATION = "initialization"
    CALIBRATION = "calibration"
    ERROR_DETECTION = "error_detection"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    SELF_CORRECTION = "self_correction"
    AGGREGATION = "aggregation"
    COMPLETE = "complete"
    ERROR = "error"
```

## Track Types

Available evaluation tracks:

```python
from chimera.evaluation import TrackType

class TrackType(str, Enum):
    CALIBRATION = "calibration"
    ERROR_DETECTION = "error_detection"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    SELF_CORRECTION = "self_correction"
```

## Error Handling

The pipeline handles errors gracefully:

```python
try:
    results = pipeline.run()
except Exception as e:
    print(f"Pipeline failed: {e}")
    
# Check for partial failures
if not results.success:
    for track, evaluation in results.track_evaluations.items():
        if evaluation.error:
            print(f"{track} failed: {evaluation.error}")
```

## Async Evaluation

For asynchronous evaluation:

```python
import asyncio
from chimera.evaluation import AsyncEvaluationPipeline

async def run_async_evaluation():
    pipeline = AsyncEvaluationPipeline(config)
    results = await pipeline.run_async()
    return results

results = asyncio.run(run_async_evaluation())
```

## Custom Evaluation

Extend the pipeline for custom evaluation:

```python
from chimera.evaluation import EvaluationPipeline, TrackEvaluation

class CustomPipeline(EvaluationPipeline):
    """Pipeline with custom track."""
    
    def _evaluate_custom_track(self) -> TrackEvaluation:
        # Custom evaluation logic
        ...
        return TrackEvaluation(
            track="custom",
            summary=custom_summary,
            raw_results=raw_results,
        )
```

## Integration with Metrics

Access detailed metrics:

```python
from chimera.metrics.calibration import (
    expected_calibration_error,
    reliability_diagram,
)

# Get raw predictions for analysis
cal_results = results.track_evaluations["calibration"].raw_results

# Compute detailed metrics
confidences = [r.confidence for r in cal_results]
correctness = [r.is_correct for r in cal_results]

ece = expected_calibration_error(confidences, correctness)
diagram = reliability_diagram(confidences, correctness)
```

## See Also

- [Models API](models.md) - Data models
- [Generators API](generators.md) - Task generation
- [Metrics Reference](../concepts/metrics.md) - Evaluation metrics
- [Configuration Guide](../configuration.md) - Configuration options
