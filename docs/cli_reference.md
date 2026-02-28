# CHIMERA CLI Reference

Complete reference for all CHIMERA command-line interface commands.

## Global Options

These options are available for all commands:

```
--version        Show version and exit
--verbose, -v    Enable verbose output
--quiet, -q      Suppress non-essential output
--help           Show help message and exit
```

## Commands

### chimera run

Run the CHIMERA evaluation benchmark.

```
chimera run [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--track` | `-t` | Choice | `all` | Track to evaluate: `calibration`, `error_detection`, `knowledge_boundary`, `self_correction`, `all` |
| `--model` | `-m` | Choice | `gemini` | Model provider: `gemini`, `openai` |
| `--model-name` | `-n` | String | Provider default | Specific model name |
| `--config` | `-c` | Path | None | Path to configuration file |
| `--output` | `-o` | Path | `results/run_<timestamp>` | Output directory |
| `--n-tasks` | | Int | 100 | Number of tasks per track |
| `--seed` | | Int | None | Random seed for reproducibility |
| `--dry-run` | | Flag | False | Generate tasks without evaluation |

**Examples:**

```bash
# Run all tracks with Gemini
chimera run

# Run specific track
chimera run --track calibration

# Run with OpenAI GPT-4
chimera run -m openai -n gpt-4o

# Use custom configuration
chimera run --config configs/full_benchmark.yaml

# Dry run to preview tasks
chimera run --track error_detection --dry-run

# Reproducible run with seed
chimera run --seed 42 --n-tasks 100

# Custom output directory
chimera run -o results/my_experiment/
```

---

### chimera generate

Generate evaluation tasks without running the full benchmark.

```
chimera generate [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--track` | `-t` | Choice | `calibration` | Track to generate tasks for |
| `--output` | `-o` | Path | `generated/` | Output directory |
| `--n-tasks` | `-n` | Int | 100 | Number of tasks |
| `--format` | `-f` | Choice | `json` | Output format: `json`, `jsonl`, `csv` |
| `--seed` | | Int | None | Random seed |

**Examples:**

```bash
# Generate calibration tasks
chimera generate -t calibration -n 50

# Generate as JSONL
chimera generate -t error_detection -f jsonl

# Generate to specific directory
chimera generate -t knowledge_boundary -o data/tasks/
```

---

### chimera analyze

Analyze results from a completed evaluation run.

```
chimera analyze [OPTIONS] RESULTS_PATH
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `RESULTS_PATH` | Path | Path to results file (summary.json) or directory |

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--metric` | `-m` | Choice | `all` | Specific metric to analyze |
| `--format` | `-f` | Choice | `table` | Output format: `table`, `json`, `markdown` |
| `--detailed` | `-d` | Flag | False | Show detailed breakdown |

**Examples:**

```bash
# Analyze results
chimera analyze results/run_20250129/summary.json

# Detailed analysis
chimera analyze results/run_20250129/ --detailed

# Output as JSON
chimera analyze results/run_20250129/ --format json

# Analyze specific metric
chimera analyze results/run_20250129/ --metric ece
```

---

### chimera report

Generate reports from evaluation results.

```
chimera report [OPTIONS] RESULTS_PATH
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `RESULTS_PATH` | Path | Path to results directory |

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | Choice | `html` | Report format: `html`, `markdown`, `pdf` |
| `--output` | `-o` | Path | Same as input | Output path for report |
| `--template` | | Path | None | Custom template file |
| `--include-plots` | | Flag | True | Include visualization plots |

**Examples:**

```bash
# Generate HTML report
chimera report results/run_20250129/

# Generate Markdown report
chimera report results/run_20250129/ -f markdown

# Custom output location
chimera report results/run_20250129/ -o reports/final_report.html
```

---

### chimera compare

Compare results across multiple evaluation runs.

```
chimera compare [OPTIONS] RESULTS_PATHS...
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `RESULTS_PATHS` | Paths | Two or more result directories to compare |

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output` | `-o` | Path | `comparison/` | Output directory |
| `--format` | `-f` | Choice | `table` | Output format |

**Examples:**

```bash
# Compare two runs
chimera compare results/gemini_run/ results/openai_run/

# Compare multiple models
chimera compare results/model_a/ results/model_b/ results/model_c/
```

---

### chimera info

Display information about CHIMERA tracks and models.

```
chimera info [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--tracks` | Flag | Show track information |
| `--models` | Flag | Show supported models |
| `--metrics` | Flag | Show available metrics |

**Examples:**

```bash
# Show all info
chimera info

# Show only tracks
chimera info --tracks

# Show supported models
chimera info --models
```

---

### chimera check

Check environment and dependencies.

```
chimera check [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--all` | Flag | Run all checks |
| `--api` | Flag | Check API connectivity |
| `--deps` | Flag | Check dependencies |

**Examples:**

```bash
# Run all checks
chimera check

# Check API connectivity only
chimera check --api
```

---

### chimera validate

Validate configuration files.

```
chimera validate [OPTIONS] CONFIG_PATH
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `CONFIG_PATH` | Path | Path to configuration file |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--strict` | Flag | Enable strict validation |

**Examples:**

```bash
# Validate configuration
chimera validate configs/my_eval.yaml

# Strict validation
chimera validate --strict configs/full_benchmark.yaml
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `CHIMERA_CONFIG` | Default config file path |
| `CHIMERA_MODEL_PROVIDER` | Override model provider |
| `CHIMERA_MODEL_NAME` | Override model name |
| `CHIMERA_MODEL_TEMPERATURE` | Override temperature |
| `CHIMERA_OUTPUT_DIR` | Override output directory |
| `CHIMERA_LOG_LEVEL` | Set log level |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | API error |
| 4 | Validation error |

## Output Files

Running `chimera run` produces:

```
results/run_<timestamp>/
├── summary.json           # Overall results summary
├── calibration/
│   ├── tasks.json        # Generated tasks
│   ├── responses.json    # Model responses
│   └── metrics.json      # Track metrics
├── error_detection/
│   └── ...
├── knowledge_boundary/
│   └── ...
├── self_correction/
│   └── ...
├── report.html           # HTML report
├── report.md             # Markdown report
└── plots/
    ├── reliability_diagram.png
    ├── calibration_curve.png
    └── ...
```

## See Also

- [Quick Start Guide](quickstart.md)
- [Configuration Guide](configuration.md)
