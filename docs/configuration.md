# CHIMERA Configuration Guide

This guide covers all configuration options for CHIMERA evaluations.

## Configuration Sources

CHIMERA loads configuration from multiple sources with the following precedence (highest to lowest):

1. **CLI arguments** - Command-line options override everything
2. **Environment variables** - `CHIMERA_*` variables override file settings
3. **User config file** - Specified via `--config`
4. **Default values** - Built-in defaults

## Configuration Files

### Default Configuration

The default configuration is at `configs/default.yaml`. Copy and modify it:

```bash
cp configs/default.yaml configs/my_eval.yaml
chimera run --config configs/my_eval.yaml
```

### Pre-built Configurations

| File | Purpose |
|------|---------|
| `configs/default.yaml` | General-purpose defaults |
| `configs/gemini_eval.yaml` | Optimized for Gemini models |
| `configs/openai_eval.yaml` | Optimized for OpenAI models |
| `configs/full_benchmark.yaml` | Complete benchmark at full scale |

## Environment Variables

### API Keys

```bash
# Required for respective providers
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
```

### Configuration Overrides

| Variable | Description | Example |
|----------|-------------|---------|
| `CHIMERA_CONFIG` | Path to config file | `/path/to/config.yaml` |
| `CHIMERA_MODEL_PROVIDER` | Model provider | `gemini`, `openai` |
| `CHIMERA_MODEL_NAME` | Model name | `gemini-2.0-flash` |
| `CHIMERA_MODEL_TEMPERATURE` | Temperature | `0.0` |
| `CHIMERA_OUTPUT_DIR` | Output directory | `results/my_run` |
| `CHIMERA_LOG_LEVEL` | Log level | `DEBUG`, `INFO`, `WARNING` |

### Environment Variable Interpolation

Config files support environment variable interpolation:

```yaml
model:
  name: "${CHIMERA_MODEL_NAME:gemini-2.0-flash}"  # With default
  api_key: "${GOOGLE_API_KEY}"  # Required - no default
```

Syntax:
- `${VAR_NAME}` - Required variable (error if not set)
- `${VAR_NAME:default}` - Variable with default value

## Configuration Sections

### Benchmark Metadata

```yaml
benchmark:
  name: "my-evaluation"
  version: "1.0.0"
  description: "Custom CHIMERA evaluation"
```

### Model Configuration

```yaml
model:
  # Provider: "gemini", "openai", "anthropic", "local"
  provider: "gemini"
  
  # Model identifier (provider-specific)
  name: "gemini-2.0-flash"
  
  # Generation parameters
  temperature: 0.0      # 0.0 = deterministic, 1.0+ = creative
  max_tokens: 2048      # Maximum output tokens
  top_p: 1.0           # Nucleus sampling
  
  # Request settings
  timeout_seconds: 60   # Per-request timeout
  max_retries: 3        # Retry on failure
  retry_delay_seconds: 1.0
```

### Track Configuration

Each track can be individually configured:

```yaml
tracks:
  calibration:
    enabled: true
    max_samples: 500
    difficulty_levels: ["L1", "L2", "L3", "L4", "L5"]
    categories:
      - "factual"
      - "reasoning"
      - "numerical"
      - "commonsense"

  error_detection:
    enabled: true
    max_samples: 300
    include_self_review: true
    error_types:
      - "factual"
      - "logical"
      - "arithmetic"
      - "hallucination"

  knowledge_boundary:
    enabled: true
    max_samples: 400
    question_types:
      - "answerable"
      - "unanswerable_impossible"
      - "unanswerable_specific"

  self_correction:
    enabled: true
    max_samples: 250
    perturbation_types:
      - "value_corruption"
      - "step_removal"
      - "logic_inversion"
```

### Evaluation Settings

```yaml
evaluation:
  # Calibration bins for ECE calculation
  n_bins: 15
  bin_strategy: "uniform"  # or "quantile"
  
  # Statistical bootstrapping
  bootstrap_samples: 1000
  confidence_level: 0.95
  
  # Thresholds
  abstention_confidence_threshold: 0.3
  error_detection_threshold: 0.5
```

### Prompt Configuration

```yaml
prompts:
  # Ask model for confidence
  elicit_confidence: true
  
  # Format: "numeric" (0-100), "verbal", "both"
  confidence_format: "numeric"
  
  # Request reasoning traces
  include_reasoning: true
  
  # Additional system instructions
  system_additions: |
    When answering:
    1. Be clear and concise
    2. Express confidence as a percentage
    3. Explain your reasoning
```

### Output Configuration

```yaml
output:
  results_dir: "results"
  experiment_name: null      # Auto-generated if null
  timestamp_format: "%Y%m%d_%H%M%S"
  
  # What to save
  save_raw_responses: true
  save_parsed_responses: true
  save_intermediate_results: true
  
  # Report generation
  generate_html_report: true
  generate_markdown_report: true
  generate_plots: true
```

### Logging Configuration

```yaml
logging:
  level: "INFO"           # DEBUG, INFO, WARNING, ERROR
  format: "rich"          # "rich" or "plain"
  log_file: null          # Path to file, or null for stdout
```

### Caching Configuration

```yaml
cache:
  enabled: true
  cache_dir: ".api_cache"
  ttl_hours: 24           # Time-to-live for cached responses
```

### Rate Limiting

```yaml
rate_limiting:
  enabled: true
  requests_per_minute: 60
  tokens_per_minute: 100000
```

## CLI Override Examples

Override any config option from the command line:

```bash
# Override model provider and name
chimera run --model openai --model-name gpt-4o

# Override number of tasks
chimera run --n-tasks 200

# Override output directory
chimera run --output results/experiment_1/

# Combine with config file
chimera run --config configs/gemini_eval.yaml --n-tasks 50 --verbose
```

## Validation

Validate your configuration before running:

```bash
# Check configuration validity
chimera validate --config configs/my_eval.yaml
```

The validator checks:
- Required fields are present
- Values are within valid ranges
- Environment variables are set
- File paths are accessible

## Configuration Merging

When using multiple sources, configurations are deep-merged:

```yaml
# default.yaml
model:
  provider: gemini
  temperature: 0.0
  max_tokens: 2048

# user_override.yaml  
model:
  temperature: 0.5  # Only this is changed
  
# Result after merge:
model:
  provider: gemini      # From default
  temperature: 0.5      # From override
  max_tokens: 2048      # From default
```

## Example Configurations

### Minimal Quick Test

```yaml
model:
  provider: gemini
  name: gemini-2.0-flash
  
tracks:
  calibration:
    enabled: true
    max_samples: 10
```

### Production Evaluation

```yaml
model:
  provider: gemini
  name: gemini-2.0-flash
  temperature: 0.0
  max_retries: 5

tracks:
  calibration:
    enabled: true
    max_samples: 1000
  error_detection:
    enabled: true
    max_samples: 600
  knowledge_boundary:
    enabled: true
    max_samples: 800
  self_correction:
    enabled: true
    max_samples: 500

evaluation:
  bootstrap_samples: 5000
  confidence_level: 0.95

output:
  generate_html_report: true
  generate_markdown_report: true
  generate_plots: true
```

### Multi-Model Comparison

Run the same config with different models:

```bash
# Run with Gemini
CHIMERA_MODEL_PROVIDER=gemini CHIMERA_MODEL_NAME=gemini-2.0-flash \
  chimera run --config configs/full_benchmark.yaml -o results/gemini/

# Run with OpenAI
CHIMERA_MODEL_PROVIDER=openai CHIMERA_MODEL_NAME=gpt-4o \
  chimera run --config configs/full_benchmark.yaml -o results/openai/
```

## See Also

- [Quick Start Guide](quickstart.md)
- [CLI Reference](cli_reference.md)
