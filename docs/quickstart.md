# CHIMERA Quick Start Guide

Get started with CHIMERA in 5 minutes!

## Prerequisites

- Python 3.10+
- A Gemini API key or OpenAI API key

## Installation

```bash
# Clone the repository
git clone https://github.com/Rahul-Lashkari/chimera.git
cd chimera

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

### API Keys

Set your API key as an environment variable:

```bash
# For Gemini (recommended)
export GOOGLE_API_KEY="your-gemini-api-key"

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

On Windows PowerShell:
```powershell
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

## Running Your First Evaluation

### Quick Test (Dry Run)

Generate tasks without making API calls:

```bash
# Generate 10 calibration tasks
chimera run --track calibration --n-tasks 10 --dry-run
```

### Run a Single Track

```bash
# Run calibration track with Gemini
chimera run --track calibration --model gemini --n-tasks 50

# Run error detection with OpenAI
chimera run --track error_detection --model openai --n-tasks 30
```

### Run All Tracks

```bash
# Full benchmark with all 4 tracks
chimera run --track all --n-tasks 100
```

## Viewing Results

Results are saved to the `results/` directory by default.

### Analyze Results

```bash
# Analyze a completed run
chimera analyze results/run_20250129_120000/summary.json
```

### Generate Reports

```bash
# HTML report
chimera report results/run_20250129_120000/ --format html

# Markdown report
chimera report results/run_20250129_120000/ --format markdown
```

## Using Configuration Files

For complex evaluations, use a config file:

```bash
# Use provided Gemini config
chimera run --config configs/gemini_eval.yaml

# Use provided OpenAI config
chimera run --config configs/openai_eval.yaml

# Full benchmark config
chimera run --config configs/full_benchmark.yaml
```

## CLI Help

```bash
# View all commands
chimera --help

# Command-specific help
chimera run --help
chimera generate --help
chimera analyze --help
```

## Example Workflow

```bash
# 1. Check environment is ready
chimera check

# 2. View available tracks and models
chimera info

# 3. Generate tasks for review (dry run)
chimera run -t calibration --dry-run -o preview/

# 4. Run actual evaluation
chimera run -t calibration -m gemini -n 100 -o results/cal_run/

# 5. Analyze results
chimera analyze results/cal_run/summary.json

# 6. Generate HTML report
chimera report results/cal_run/ --format html
```

## Tracks Overview

| Track | Description | What it Measures |
|-------|-------------|------------------|
| `calibration` | Confidence calibration | Does confidence match accuracy? |
| `error_detection` | Self-error identification | Can the model spot its own errors? |
| `knowledge_boundary` | Knowledge limit recognition | Does it know what it doesn't know? |
| `self_correction` | Reasoning correction | Can it fix corrupted reasoning? |

## Next Steps

- Read the [Configuration Guide](configuration.md) for advanced options
- See the [CLI Reference](cli_reference.md) for all commands
- Check out the [API Documentation](../api/index.md) for programmatic usage

## Troubleshooting

### API Key Not Found

```
Error: GOOGLE_API_KEY environment variable not set
```

Solution: Export your API key before running:
```bash
export GOOGLE_API_KEY="your-key-here"
```

### Rate Limiting

If you hit rate limits, reduce the batch size or add delays:
```bash
chimera run --track calibration --batch-size 5
```

### Memory Issues

For large evaluations, process tracks separately:
```bash
chimera run -t calibration -n 500 -o results/cal/
chimera run -t error_detection -n 300 -o results/ed/
```
