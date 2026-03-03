# CHIMERA Documentation

**CHIMERA** (Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment) is a comprehensive benchmark for evaluating the meta-cognitive calibration of Large Language Models.

## Documentation Overview

### Getting Started

| Guide | Description |
|-------|-------------|
| [Quick Start](quickstart.md) | Get running in 5 minutes |
| [Configuration](configuration.md) | YAML and environment configuration |
| [CLI Reference](cli_reference.md) | Complete command-line documentation |

### Core Concepts

| Concept | Description |
|---------|-------------|
| [Calibration](concepts/calibration.md) | Understanding confidence calibration |
| [Introspection](concepts/introspection.md) | Meta-cognitive evaluation theory |
| [Metrics](concepts/metrics.md) | All evaluation metrics explained |

### API Reference

| Module | Description |
|--------|-------------|
| [Models](api/models.md) | Data models (Task, Response, Evaluation) |
| [Generators](api/generators.md) | Task generators for each track |
| [Evaluation](api/evaluation.md) | Evaluation pipeline and comparison |

### Tutorials

| Tutorial | Description |
|----------|-------------|
| [Custom Tasks](tutorials/custom_tasks.md) | Creating custom evaluation tasks |
| [Model Comparison](tutorials/model_comparison.md) | Comparing multiple models |

### Examples

| Example | Description |
|---------|-------------|
| [Basic Evaluation](../examples/basic_evaluation.py) | Simple evaluation script |
| [Custom Dataset](../examples/custom_dataset.py) | Working with custom data |
| [Calibration Analysis](../examples/notebooks/calibration_analysis.ipynb) | Interactive Jupyter notebook |
| [Model Comparison](../examples/notebooks/model_comparison.ipynb) | Compare models visually |

## The Four Tracks

CHIMERA evaluates meta-cognitive capabilities through four complementary tracks:

### Track 1: Calibration Probing

Tests whether a model's expressed confidence correlates with actual accuracy. A well-calibrated model should be correct 80% of the time when it says it's 80% confident.

### Track 2: Error Detection

Presents statements with deliberate errors and tests whether the model can identify them. Error types include factual, logical, computational, temporal, and magnitude errors.

### Track 3: Knowledge Boundary Recognition

Tests whether models appropriately abstain from unanswerable questions while confidently answering what they know.

### Track 4: Self-Correction Under Perturbation

Introduces corruptions to reasoning chains and tests whether the model can detect and correct the errors.

## Quick Example

```python
from chimera.evaluation import EvaluationPipeline, PipelineConfig

config = PipelineConfig(
    tracks=["calibration", "error_detection"],
    model_provider="gemini",
    model_name="gemini-2.0-flash",
    n_tasks=100,
)

pipeline = EvaluationPipeline(config)
results = pipeline.run()

print(f"Overall Score: {results.overall_score:.2%}")
```

## Key Metrics

| Track | Primary Metric | Description |
|-------|---------------|-------------|
| Calibration | ECE | Expected Calibration Error (lower is better) |
| Error Detection | F1 | Harmonic mean of precision and recall |
| Knowledge Boundary | Abstention F1 | Appropriate abstention accuracy |
| Self-Correction | E2E Success | Detection × Correction rate |

## Building the Docs

This documentation is built with MkDocs Material. To build locally:

```bash
pip install mkdocs-material mkdocstrings[python]

cd docs
mkdocs serve

mkdocs build
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../LICENSE) file for details.
