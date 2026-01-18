# Quick Start Guide

This guide will help you run your first CHIMERA evaluation in minutes.

## Prerequisites

1. CHIMERA installed (see [Installation](installation.md))
2. Google AI API key configured in `.env`

## Running Your First Evaluation

### 1. Basic Evaluation

Run CHIMERA with default settings:

```bash
chimera run --model gemini-2.0-flash
```

This will:
- Load the default configuration
- Run all four evaluation tracks
- Generate a report in `results/`

### 2. Run a Specific Track

To run only the calibration track:

```bash
chimera run --model gemini-2.0-flash --track calibration
```

### 3. Quick Test Run

For a quick test with limited samples:

```bash
chimera run --model gemini-2.0-flash --max-samples 10
```

## Understanding Results

After evaluation, find your results in the `results/` directory:

```
results/
└── gemini-2.0-flash_20250110_143022/
    ├── summary_report.html      # Visual report
    ├── summary_report.md        # Markdown report
    ├── calibration_results.json # Track 1 details
    ├── error_detection.json     # Track 2 details
    ├── knowledge_boundary.json  # Track 3 details
    ├── self_correction.json     # Track 4 details
    └── plots/
        ├── reliability_diagram.png
        └── calibration_curve.png
```

## Key Metrics

| Metric | Good Value | Description |
|--------|------------|-------------|
| ECE | < 0.10 | Expected Calibration Error |
| Self-Error F1 | > 0.70 | Error detection accuracy |
| Abstention Rate | 0.8+ on unknowable | Appropriate "I don't know" |
| Corruption Detection | > 0.85 AUC | Spotting reasoning errors |

## Next Steps

- [Configuration Guide](../configuration.md) - Customize your evaluations
- [Concepts](../concepts/calibration.md) - Understand what's being measured
- [API Reference](../api/models.md) - Programmatic usage
