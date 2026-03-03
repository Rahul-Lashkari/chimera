# Tutorial: Model Comparison

This tutorial shows how to compare multiple models using CHIMERA.

## Overview

CHIMERA provides tools to:

1. Run the same evaluation on multiple models
2. Compute performance differences
3. Generate comparison reports
4. Visualize results

## Quick Start

```python
from chimera.evaluation import (
    EvaluationPipeline,
    PipelineConfig,
    ModelComparison,
)

# Define models to compare
models = [
    ("gemini", "gemini-2.0-flash"),
    ("gemini", "gemini-2.0-pro"),
    ("openai", "gpt-4o-mini"),
]

# Run evaluations
comparison = ModelComparison()

for provider, model_name in models:
    config = PipelineConfig(
        tracks=["calibration", "error_detection"],
        model_provider=provider,
        model_name=model_name,
        n_tasks=100,
        seed=42,  # Same seed for fair comparison
    )
    
    pipeline = EvaluationPipeline(config)
    results = pipeline.run()
    
    comparison.add_model_results(model_name, results)

# Generate comparison
rankings = comparison.compute_rankings()
for rank in rankings:
    print(f"{rank.rank}. {rank.model_name}: {rank.score:.2%}")
```

## Detailed Comparison

### Computing Performance Deltas

```python
# Compare two specific models
delta = comparison.compute_deltas("gemini-2.0-flash", "gpt-4o-mini")

print(f"Overall difference: {delta.overall_delta:+.2%}")
print(f"Per-track differences:")
for track, track_delta in delta.track_deltas.items():
    print(f"  {track}: {track_delta:+.2%}")

# Statistical significance
print(f"Statistically significant: {delta.is_significant}")
print(f"P-value: {delta.p_value:.4f}")
```

### All-vs-All Comparison

```python
# Compute all pairwise comparisons
all_deltas = comparison.compute_all_deltas()

for model_a, model_b, delta in all_deltas:
    if delta.overall_delta > 0:
        print(f"{model_a} beats {model_b} by {delta.overall_delta:.2%}")
```

### Finding Best Models

```python
# Best overall
best = comparison.get_best_model_overall()
print(f"Best overall: {best}")

# Best per track
best_per_track = comparison.get_best_model_per_track()
for track, model in best_per_track.items():
    print(f"Best at {track}: {model}")
```

## Generating Reports

### Markdown Report

```python
report = comparison.generate_comparison_report(format="markdown")
with open("comparison.md", "w") as f:
    f.write(report)
```

Example output:

```markdown
# Model Comparison Report

## Rankings

| Rank | Model | Overall Score | Calibration | Error Detection |
|------|-------|---------------|-------------|-----------------|
| 1 | gemini-2.0-pro | 82.5% | 85.0% | 80.0% |
| 2 | gpt-4o-mini | 78.3% | 75.0% | 81.5% |
| 3 | gemini-2.0-flash | 76.1% | 78.0% | 74.2% |

## Performance Deltas

| Comparison | Delta | Significant? |
|------------|-------|--------------|
| gemini-2.0-pro vs gpt-4o-mini | +4.2% | Yes |
| gpt-4o-mini vs gemini-2.0-flash | +2.2% | No |
```

### HTML Report with Charts

```python
report = comparison.generate_comparison_report(format="html")
with open("comparison.html", "w") as f:
    f.write(report)
```

The HTML report includes:
- Interactive ranking table
- Bar charts comparing models
- Radar charts for multi-track comparison
- Reliability diagrams for each model

## Visualization

### Radar Chart

```python
import matplotlib.pyplot as plt
from chimera.visualization import create_radar_chart

# Create radar chart comparing tracks
fig = create_radar_chart(
    models=["gemini-2.0-pro", "gpt-4o-mini", "gemini-2.0-flash"],
    tracks=["calibration", "error_detection", "knowledge_boundary", "self_correction"],
    scores={
        "gemini-2.0-pro": [0.85, 0.80, 0.82, 0.78],
        "gpt-4o-mini": [0.75, 0.81, 0.79, 0.76],
        "gemini-2.0-flash": [0.78, 0.74, 0.75, 0.73],
    },
)
fig.savefig("radar_comparison.png")
```

### Bar Chart Comparison

```python
from chimera.visualization import create_comparison_bar_chart

fig = create_comparison_bar_chart(
    comparison=comparison,
    metric="overall_score",
)
fig.savefig("bar_comparison.png")
```

### Reliability Diagram Overlay

```python
from chimera.visualization import overlay_reliability_diagrams

# Compare calibration across models
fig = overlay_reliability_diagrams(
    model_results={
        "gemini-2.0-pro": gemini_results,
        "gpt-4o-mini": gpt_results,
    },
    track="calibration",
)
fig.savefig("calibration_comparison.png")
```

## Statistical Analysis

### Significance Testing

```python
from chimera.analysis import compare_models_statistically

results = compare_models_statistically(
    comparison,
    method="paired_ttest",  # or "wilcoxon", "bootstrap"
    confidence_level=0.95,
)

for model_pair, stats in results.items():
    print(f"{model_pair}:")
    print(f"  t-statistic: {stats.t_stat:.4f}")
    print(f"  p-value: {stats.p_value:.4f}")
    print(f"  Significant: {stats.is_significant}")
```

### Effect Size

```python
from chimera.analysis import compute_effect_size

effect = compute_effect_size(
    model_a_scores=gemini_scores,
    model_b_scores=gpt_scores,
    method="cohens_d",
)

print(f"Cohen's d: {effect:.2f}")
# Interpretation:
# 0.2 = small, 0.5 = medium, 0.8 = large
```

### Confidence Intervals

```python
from chimera.analysis import compute_score_with_ci

for model_name in comparison.model_names:
    score, ci_low, ci_high = compute_score_with_ci(
        comparison.get_model_results(model_name),
        confidence_level=0.95,
    )
    print(f"{model_name}: {score:.2%} [{ci_low:.2%}, {ci_high:.2%}]")
```

## Batch Comparison Script

Full script for comparing multiple models:

```python
#!/usr/bin/env python3
"""Compare multiple models on CHIMERA benchmark."""

import json
from pathlib import Path
from chimera.evaluation import (
    EvaluationPipeline,
    PipelineConfig,
    ModelComparison,
)

# Configuration
MODELS = [
    ("gemini", "gemini-2.0-flash"),
    ("gemini", "gemini-2.0-pro"),
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4o"),
]

TRACKS = ["calibration", "error_detection", "knowledge_boundary", "self_correction"]
N_TASKS = 100
SEED = 42
OUTPUT_DIR = Path("results/comparison")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = ModelComparison()
    
    # Run evaluations
    for provider, model_name in MODELS:
        print(f"\nEvaluating {model_name}...")
        
        config = PipelineConfig(
            tracks=TRACKS,
            model_provider=provider,
            model_name=model_name,
            n_tasks=N_TASKS,
            seed=SEED,
            output_dir=str(OUTPUT_DIR / model_name),
        )
        
        pipeline = EvaluationPipeline(config)
        results = pipeline.run()
        
        comparison.add_model_results(model_name, results)
        print(f"  Overall score: {results.overall_score:.2%}")
    
    # Generate rankings
    print("\n" + "="*50)
    print("RANKINGS")
    print("="*50)
    
    rankings = comparison.compute_rankings()
    for rank in rankings:
        print(f"{rank.rank}. {rank.model_name}: {rank.score:.2%}")
    
    # Save reports
    md_report = comparison.generate_comparison_report(format="markdown")
    (OUTPUT_DIR / "comparison.md").write_text(md_report)
    
    html_report = comparison.generate_comparison_report(format="html")
    (OUTPUT_DIR / "comparison.html").write_text(html_report)
    
    # Save raw data
    with open(OUTPUT_DIR / "comparison_data.json", "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)
    
    print(f"\nReports saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Use Same Seed

Always use the same random seed for fair comparison:

```python
SEED = 42
for model in models:
    config = PipelineConfig(seed=SEED, ...)
```

### 2. Sufficient Sample Size

Use enough tasks for statistical power:

- Quick comparison: 50-100 tasks
- Standard evaluation: 200-500 tasks
- Publication-quality: 1000+ tasks

### 3. Multiple Runs

For high-stakes comparisons, run multiple times:

```python
all_scores = []
for run in range(5):
    config = PipelineConfig(seed=run, ...)
    results = pipeline.run()
    all_scores.append(results.overall_score)

mean_score = sum(all_scores) / len(all_scores)
```

### 4. Control for Cost

Different models have different costs:

```python
# Track API costs
for model_name, results in comparison.items():
    print(f"{model_name}:")
    print(f"  Score: {results.overall_score:.2%}")
    print(f"  Tokens: {results.total_tokens:,}")
    print(f"  Est. Cost: ${results.estimated_cost:.2f}")
```

## See Also

- [Evaluation API](../api/evaluation.md) - Pipeline reference
- [Metrics](../concepts/metrics.md) - Metric definitions
- [Examples](../../examples/) - More examples
