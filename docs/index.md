# CHIMERA Documentation

**Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment**

Welcome to the CHIMERA documentation! CHIMERA is a comprehensive benchmark for evaluating the meta-cognitive calibration capabilities of large language models (LLMs).

## What is CHIMERA?

CHIMERA assesses whether AI systems "know what they know" through four complementary evaluation tracks:

| Track | What It Measures |
|-------|-----------------|
| **Calibration Probing** | Does the model's confidence match its actual accuracy? |
| **Error Detection** | Can the model identify errors in its own outputs? |
| **Knowledge Boundary** | Does the model recognize the limits of its knowledge? |
| **Self-Correction** | Can the model fix corrupted or flawed reasoning? |

## Why Meta-Cognitive Calibration Matters

Well-calibrated AI systems are essential for:

- **Safety**: Models that accurately assess uncertainty can abstain from high-stakes decisions when unsure
- **Trust**: Users can rely on confidence signals to gauge reliability
- **Collaboration**: Human-AI teams work better when the AI knows its limitations
- **Alignment**: Meta-cognitive awareness is a prerequisite for more advanced alignment techniques

## Quick Navigation

### Getting Started

- [Quick Start Guide](quickstart.md) - Get running in 5 minutes
- [Configuration Guide](configuration.md) - Customize your evaluations
- [CLI Reference](cli_reference.md) - Command-line interface documentation

### Core Concepts

- [Calibration](concepts/calibration.md) - Understanding confidence calibration
- [Introspection](concepts/introspection.md) - Meta-cognitive evaluation principles
- [Metrics](concepts/metrics.md) - Evaluation metrics explained

### API Reference

- [Models](api/models.md) - Data models and schemas
- [Generators](api/generators.md) - Task generation
- [Evaluation](api/evaluation.md) - Evaluation pipeline and metrics

### Tutorials

- [Custom Tasks](tutorials/custom_tasks.md) - Create your own evaluation tasks
- [Model Comparison](tutorials/model_comparison.md) - Compare multiple models

### Examples

- [Basic Evaluation](../examples/basic_evaluation.py) - Simple evaluation script
- [Custom Dataset](../examples/custom_dataset.py) - Working with custom data
- [Notebooks](../examples/notebooks/) - Interactive Jupyter notebooks

## Installation

```bash
# Clone the repository
git clone https://github.com/Rahul-Lashkari/chimera.git
cd chimera

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Example

```python
from chimera.evaluation import EvaluationPipeline, PipelineConfig

# Configure evaluation
config = PipelineConfig(
    tracks=["calibration", "error_detection"],
    model_provider="gemini",
    model_name="gemini-2.0-flash",
)

# Run evaluation
pipeline = EvaluationPipeline(config)
results = pipeline.run()

# Generate report
report = results.generate_report(format="html")
print(f"Overall Score: {results.overall_score:.2%}")
```

## Command Line Usage

```bash
# Run full benchmark
chimera run --track all --model gemini

# Generate tasks only (dry run)
chimera run --track calibration --dry-run

# Analyze existing results
chimera analyze results/run_20250129/summary.json
```

## Project Structure

```
chimera/
├── src/chimera/
│   ├── cli/          # Command-line interface
│   ├── evaluation/   # Evaluation pipeline
│   ├── generators/   # Task generators
│   ├── interfaces/   # Model interfaces
│   ├── metrics/      # Evaluation metrics
│   ├── models/       # Data models
│   └── runner/       # Benchmark runner
├── tests/            # Test suite
├── configs/          # Configuration files
├── docs/             # Documentation
└── examples/         # Example scripts
```

## Evaluation Tracks Overview

### Track 1: Calibration Probing

Tests whether model confidence predictions match actual accuracy. A well-calibrated model should be correct 70% of the time when it reports 70% confidence.

**Key Metrics**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score

### Track 2: Error Detection

Presents the model with statements containing deliberate errors and tests whether it can identify them.

**Error Types**: Factual, Computational, Logical, Temporal, Magnitude, Hallucination

### Track 3: Knowledge Boundary Recognition

Tests whether the model appropriately abstains or expresses uncertainty for unanswerable questions.

**Question Types**: Answerable, Unanswerable (Impossible), Unanswerable (Specific), Obscure Facts, Fictional

### Track 4: Self-Correction Under Perturbation

Presents corrupted reasoning chains and tests whether the model can identify and correct the errors.

**Perturbation Types**: Value Corruption, Step Removal, Logic Inversion, Premise Change

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## License

CHIMERA is released under the MIT License. See [LICENSE](../LICENSE) for details.

## Citation

If you use CHIMERA in your research, please cite:

```bibtex
@software{chimera2025,
  title = {CHIMERA: Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment},
  author = {Rahul Lashkari},
  year = {2025},
  url = {https://github.com/Rahul-Lashkari/chimera}
}
```
