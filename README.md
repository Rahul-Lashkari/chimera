# CHIMERA

**Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linting-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

## Overview

CHIMERA is a comprehensive benchmark for evaluating the **meta-cognitive calibration** of Large Language Models — their ability to accurately recognize, express, and respond to their own uncertainty, errors, and knowledge boundaries.

Traditional benchmarks measure *what* models produce. CHIMERA measures *whether models know when they're wrong*.

## Evaluation Tracks

CHIMERA evaluates meta-cognitive capabilities through four complementary tracks:

| Track | Description | Primary Metric |
|-------|-------------|----------------|
| **Calibration Probing** | Tests whether stated confidence matches actual accuracy | ECE |
| **Error Detection** | Tests ability to identify errors in presented statements | F1 Score |
| **Knowledge Boundary** | Tests appropriate abstention on unanswerable questions | Abstention F1 |
| **Self-Correction** | Tests detection and correction of corrupted reasoning | E2E Success |

### Track 1: Calibration Probing

Tests whether a model's expressed confidence actually predicts correctness. A well-calibrated model should be correct 80% of the time when expressing 80% confidence.

**Metrics:** ECE, MCE, Brier Score, Reliability Diagrams

### Track 2: Error Detection

Presents statements with deliberate errors and tests whether the model can identify them.

**Error Types:** Factual, Logical, Computational, Temporal, Magnitude, Hallucination

### Track 3: Knowledge Boundary Recognition

Tests whether models appropriately abstain from unanswerable questions while confidently answering what they know.

**Question Types:** Answerable, Impossible, Too Specific, Obscure Facts, Future Events

### Track 4: Self-Correction Under Perturbation

Introduces corruptions to reasoning chains and tests whether the model can detect and correct the errors.

**Perturbation Types:** Value Corruption, Step Removal, Logic Inversion, Premise Change

## Quick Start

### Installation

```bash
git clone https://github.com/Rahul-Lashkari/chimera.git
cd chimera

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Configuration

Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

Or use YAML configuration:

```yaml
# configs/gemini_eval.yaml
model:
  provider: gemini
  name: gemini-2.0-flash
  
tracks:
  - calibration
  - error_detection
  - knowledge_boundary
  - self_correction

evaluation:
  n_tasks: 100
  seed: 42
```

## Usage

### Command Line Interface

```bash
# Check environment and dependencies
chimera check

# Run full benchmark
chimera run --model gemini --track all

# Run specific track
chimera run --model gemini --track calibration --n-tasks 50

# Dry run (generate tasks without API calls)
chimera run --track calibration --dry-run

# Use custom configuration
chimera run --config configs/gemini_eval.yaml

# Analyze existing results
chimera analyze results/run_20260130/

# Generate report
chimera report results/run_20260130/ --format html
```

### Python API

```python
from chimera.evaluation import EvaluationPipeline, PipelineConfig

config = PipelineConfig(
    tracks=["calibration", "error_detection"],
    model_provider="gemini",
    model_name="gemini-2.0-flash",
    n_tasks=100,
    seed=42,
)

pipeline = EvaluationPipeline(config)
results = pipeline.run()

print(f"Overall Score: {results.overall_score:.2%}")

for track, summary in results.track_summaries.items():
    print(f"  {track}: {summary.score:.2%}")
```

### Model Comparison

```python
from chimera.evaluation import ModelComparison

comparison = ModelComparison()
comparison.add_model_results("gemini-2.0-flash", gemini_results)
comparison.add_model_results("gpt-4o", gpt4_results)

rankings = comparison.compute_rankings()
for rank in rankings:
    print(f"{rank.rank}. {rank.model_name}: {rank.score:.2%}")
```

## Metrics Reference

### Calibration Metrics

| Metric | Description | Optimal |
|--------|-------------|---------|
| ECE | Expected Calibration Error | 0 |
| MCE | Maximum Calibration Error | 0 |
| Brier Score | Mean squared error of probabilistic predictions | 0 |

### Error Detection Metrics

| Metric | Description |
|--------|-------------|
| Precision | Fraction of detected errors that are actual errors |
| Recall | Fraction of actual errors that were detected |
| F1 Score | Harmonic mean of precision and recall |

### Knowledge Boundary Metrics

| Metric | Description |
|--------|-------------|
| Abstention Rate | Frequency of declining to answer |
| Appropriate Abstention F1 | Accuracy of abstention decisions |

### Self-Correction Metrics

| Metric | Description |
|--------|-------------|
| Detection Rate | Correctly identified corruptions |
| Correction Accuracy | Correctly fixed errors |
| E2E Success | Detection × Correction |

## Project Structure

```
chimera/
├── src/chimera/
│   ├── cli/                 # Command-line interface
│   ├── evaluation/          # Evaluation pipeline and aggregation
│   ├── generators/          # Task generators (one per track)
│   ├── interfaces/          # Model API interfaces
│   ├── metrics/             # Metric computation
│   ├── models/              # Pydantic data models
│   └── runner/              # Benchmark execution
├── tests/                   # Test suite (802 tests)
├── configs/                 # YAML configuration files
├── docs/                    # Documentation (MkDocs)
├── examples/                # Example scripts and notebooks
└── results/                 # Evaluation outputs
```

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start Guide](docs/quickstart.md) | Get running in 5 minutes |
| [Configuration Guide](docs/configuration.md) | YAML and environment setup |
| [CLI Reference](docs/cli_reference.md) | Complete CLI documentation |
| [Calibration Concepts](docs/concepts/calibration.md) | Theory of confidence calibration |
| [Introspection Concepts](docs/concepts/introspection.md) | Meta-cognitive evaluation framework |
| [Metrics Reference](docs/concepts/metrics.md) | All metrics explained |
| [API: Models](docs/api/models.md) | Data models reference |
| [API: Generators](docs/api/generators.md) | Task generators reference |
| [API: Evaluation](docs/api/evaluation.md) | Evaluation pipeline reference |

## Testing

```bash
# Run all tests
pytest tests -v

# Run with coverage
pytest tests --cov=src/chimera --cov-report=html

# Run specific test module
pytest tests/test_evaluation/ -v
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src tests
isort src tests

# Lint
ruff check src tests

# Type check
mypy src

# Security check
bandit -r src
```

## Why CHIMERA Matters

**Safety:** A model that knows when it's uncertain is safer than one that confidently hallucinates. CHIMERA directly measures safety-relevant epistemic properties.

**Agents:** Agentic systems must know when to pause and ask for help. CHIMERA tests this prerequisite capability.

**Trust:** Users need to know when to trust model outputs. Calibrated confidence enables appropriate human-AI collaboration.

**Alignment:** Truthfulness requires knowing what is true. CHIMERA measures the meta-cognitive foundations of honest AI.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

CHIMERA draws inspiration from foundational research on calibration and meta-cognition:

- Guo et al. (2017) - *On Calibration of Modern Neural Networks*
- Kadavath et al. (2022) - *Language Models (Mostly) Know What They Know*
- Lin et al. (2022) - *Teaching Models to Express Their Uncertainty in Words*

## Citation

```bibtex
@software{chimera2026,
  title={CHIMERA: Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment},
  author={Rahul Lashkari},
  year={2026},
  url={https://github.com/Rahul-Lashkari/chimera}
}
```
