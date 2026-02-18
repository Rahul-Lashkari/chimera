# Command Line Interface

CHIMERA provides a comprehensive command-line interface for running evaluations,
generating tasks, and analyzing results.

## Installation

After installing CHIMERA, the `chimera` command becomes available:

```bash
pip install chimera-benchmark
chimera --version
```

## Commands Overview

### `chimera run`

Run a benchmark evaluation on one or more tracks.

```bash
# Run all tracks with default settings
chimera run --track all

# Run specific track with custom settings
chimera run --track calibration --model gemini --n-tasks 100

# Run with configuration file
chimera run --config config.yaml --output results/

# Dry run (generate tasks only, no API calls)
chimera run --track all --dry-run --n-tasks 50
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--track` | `-t` | Evaluation track(s) to run | `all` |
| `--model` | `-m` | Model provider (gemini, openai) | `gemini` |
| `--model-name` | `-n` | Specific model name | Provider default |
| `--config` | `-c` | Path to configuration file | None |
| `--output` | `-o` | Output directory for results | Auto-generated |
| `--n-tasks` | | Number of tasks per track | 100 |
| `--seed` | | Random seed for reproducibility | None |
| `--dry-run` | | Generate tasks without running | False |

### `chimera generate`

Generate evaluation tasks without running the full benchmark.

```bash
# Generate calibration tasks as JSON
chimera generate --track calibration -n 100 -o calibration_tasks.json

# Generate as JSONL format
chimera generate -t error_detection -n 50 -o tasks.jsonl --format jsonl

# Generate with specific seed
chimera generate -t knowledge_boundary -n 100 -o tasks.json --seed 42
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--track` | `-t` | Track to generate tasks for | Required |
| `--n-tasks` | `-n` | Number of tasks to generate | 100 |
| `--output` | `-o` | Output file path | Required |
| `--seed` | | Random seed | None |
| `--format` | | Output format (json, jsonl) | `json` |

### `chimera analyze`

Analyze and display evaluation results.

```bash
# Display results as table
chimera analyze results/

# Export as markdown report
chimera analyze results/ --format markdown -o report.md

# Output as JSON
chimera analyze results/evaluation_summary.json --format json
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--format` | | Output format (table, json, markdown) | `table` |
| `--output` | `-o` | Save report to file | None |

### `chimera info`

Display information about CHIMERA benchmark.

```bash
chimera info
```

Shows:
- Available evaluation tracks
- Supported models
- Quick start examples

### `chimera check`

Check environment and dependencies.

```bash
# Basic check
chimera check

# Include optional dependencies
chimera check --all
```

Verifies:
- Python version (requires 3.10+)
- Core dependencies (click, rich, pydantic, etc.)
- API clients (google-generativeai, openai)
- API key environment variables

## Global Options

These options are available for all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | | Show version and exit |
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress non-essential output |
| `--help` | | Show help message |

## Configuration File

CHIMERA supports YAML configuration files:

```yaml
# config.yaml
model:
  provider: gemini
  name: gemini-2.0-flash-exp

generation:
  n_tasks: 100
  seed: 42

evaluation:
  batch_size: 10
  max_retries: 3
  timeout: 60
```

Use with: `chimera run --config config.yaml`

## Environment Variables

Configure API keys via environment variables:

```bash
# For Google Gemini
export GOOGLE_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"
```

## Examples

### Quick Evaluation

```bash
# Run calibration track with 50 tasks
chimera run -t calibration -n 50 --seed 42
```

### Generate Tasks for Custom Pipeline

```bash
# Generate tasks in JSONL format for processing
chimera generate -t error_detection -n 200 -o tasks.jsonl --format jsonl
```

### Analyze Multiple Runs

```bash
# Generate markdown reports for all runs
for dir in results/run_*; do
    chimera analyze "$dir" --format markdown -o "${dir}/report.md"
done
```

### Reproducible Benchmarking

```bash
# Use seed for reproducible results
chimera run --track all --seed 42 -o results/reproducible_run

# Verify by regenerating with same seed
chimera generate -t calibration -n 100 --seed 42 -o verify.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (see message for details) |
| 2 | Invalid command or options |

## Troubleshooting

### "No API key found"

Set the appropriate environment variable:

```bash
export GOOGLE_API_KEY="your-key"  # For Gemini
export OPENAI_API_KEY="your-key"  # For OpenAI
```

### Import Errors

Ensure all dependencies are installed:

```bash
pip install chimera-benchmark[all]
```

### Rate Limiting

Reduce batch size in configuration:

```yaml
evaluation:
  batch_size: 5
  timeout: 120
```
