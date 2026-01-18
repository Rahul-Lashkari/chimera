# CHIMERA: Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

CHIMERA is a comprehensive benchmark for evaluating the **meta-cognitive calibration** of Large Language Models — their ability to accurately recognize, express, and respond to their own uncertainty, errors, and knowledge boundaries.

## What CHIMERA Evaluates

| Capability | Description |
|------------|-------------|
| **Confidence Calibration** | Does stated confidence correlate with actual correctness? |
| **Self-Error Detection** | Can models recognize their own mistakes before feedback? |
| **Knowledge Boundaries** | Do models know what they don't know? |
| **Self-Correction** | Can models fix errors through introspection alone? |
| **Appropriate Deferral** | Do models refuse tasks beyond their competence? |

## Four Evaluation Tracks

### Track 1: Calibration Probing
Tests whether a model's expressed confidence (e.g., "I'm 90% sure") actually predicts correctness.

### Track 2: Error Detection
Challenges models to review their own responses and identify errors without external hints.

### Track 3: Knowledge Boundary Recognition
Presents questions ranging from well-known facts to unanswerable queries, evaluating appropriate abstention.

### Track 4: Self-Correction Under Perturbation
Introduces subtle corruptions to model reasoning and tests detection/correction abilities.

## Key Metrics

- **Expected Calibration Error (ECE)**: Gap between stated confidence and accuracy
- **Self-Error Detection F1**: Precision/recall for self-identified errors
- **Appropriate Abstention Rate**: Correct "I don't know" responses
- **Corruption Detection AUC**: Ability to spot reasoning perturbations
- **Sycophancy Resistance Index**: Maintaining truth under pressure

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rahul-Lashkari/chimera.git
cd chimera

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

### Run Evaluation

```bash
# Run all tracks on Gemini
chimera run --model gemini-2.0-flash --config configs/default.yaml

# Run specific track
chimera run --model gemini-2.0-flash --track calibration

# Generate report from existing results
chimera report --results-dir results/gemini-2.0-flash/
```

## Project Structure

```
chimera/
├── src/
│   └── chimera/
│       ├── __init__.py
│       ├── models/          # LLM interface abstractions
│       ├── datasets/        # Dataset generators
│       ├── metrics/         # Calibration and detection metrics
│       ├── prompts/         # Prompt templates
│       ├── runner/          # Benchmark execution engine
│       ├── evaluation/      # Multi-track evaluation pipeline
│       └── reporting/       # Report generation
├── tests/                   # Unit and integration tests
├── configs/                 # YAML configuration files
├── data/                    # Seed datasets and resources
├── results/                 # Evaluation outputs
└── docs/                    # Documentation
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## Why CHIMERA Matters

### For Safety
A model that knows when it's uncertain is safer than one that confidently hallucinates. CHIMERA directly measures safety-relevant epistemic properties.

### For Agents
Agentic systems must know when to pause and ask for help. CHIMERA tests this prerequisite capability.

### For Trust
Users need to know when to trust model outputs. Calibrated confidence enables appropriate human-AI collaboration.

### For Alignment
Truthfulness requires knowing what is true. CHIMERA measures the meta-cognitive foundations of honest AI.

## Documentation

- [Conceptual Overview](docs/concepts/)
- [API Reference](docs/api/)
- [Configuration Guide](docs/configuration.md)
- [Contributing](CONTRIBUTING.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{chimera2026,
  title={CHIMERA: Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment},
  author={Rahul Lashkari},
  year={2026},
  url={https://github.com/Rahul-Lashkari/chimera}
}
```
