"""Configuration loading and validation for CHIMERA CLI.

This module provides utilities for loading, merging, and validating
configuration files with support for environment variable interpolation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration settings."""

    provider: str = Field(default="gemini", description="Model provider name")
    name: str = Field(default="gemini-2.0-flash", description="Model identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=100000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate model provider."""
        allowed = {"gemini", "openai", "anthropic", "local"}
        if v.lower() not in allowed:
            raise ValueError(f"Provider must be one of {allowed}, got '{v}'")
        return v.lower()


class TrackConfig(BaseModel):
    """Configuration for an evaluation track."""

    enabled: bool = Field(default=True)
    max_samples: int = Field(default=100, ge=1)


class CalibrationTrackConfig(TrackConfig):
    """Calibration track specific configuration."""

    difficulty_levels: list[str] = Field(default=["L1", "L2", "L3", "L4", "L5"])
    categories: list[str] = Field(default=["factual", "reasoning", "numerical", "commonsense"])


class ErrorDetectionTrackConfig(TrackConfig):
    """Error detection track specific configuration."""

    include_self_review: bool = Field(default=True)
    error_types: list[str] = Field(default=["factual", "logical", "arithmetic", "incomplete"])


class KnowledgeBoundaryTrackConfig(TrackConfig):
    """Knowledge boundary track specific configuration."""

    question_types: list[str] = Field(
        default=[
            "answerable",
            "unanswerable_impossible",
            "unanswerable_specific",
            "obscure_facts",
            "fictional",
        ]
    )


class SelfCorrectionTrackConfig(TrackConfig):
    """Self-correction track specific configuration."""

    perturbation_types: list[str] = Field(
        default=[
            "value_corruption",
            "step_removal",
            "logic_inversion",
            "premise_change",
        ]
    )


class TracksConfig(BaseModel):
    """Configuration for all evaluation tracks."""

    calibration: CalibrationTrackConfig = Field(default_factory=CalibrationTrackConfig)
    error_detection: ErrorDetectionTrackConfig = Field(default_factory=ErrorDetectionTrackConfig)
    knowledge_boundary: KnowledgeBoundaryTrackConfig = Field(
        default_factory=KnowledgeBoundaryTrackConfig
    )
    self_correction: SelfCorrectionTrackConfig = Field(default_factory=SelfCorrectionTrackConfig)


class EvaluationConfig(BaseModel):
    """Evaluation settings."""

    n_bins: int = Field(default=15, ge=1, le=100)
    bin_strategy: str = Field(default="uniform")
    bootstrap_samples: int = Field(default=1000, ge=100)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    abstention_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    error_detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("bin_strategy")
    @classmethod
    def validate_bin_strategy(cls, v: str) -> str:
        """Validate bin strategy."""
        allowed = {"uniform", "quantile"}
        if v.lower() not in allowed:
            raise ValueError(f"bin_strategy must be one of {allowed}")
        return v.lower()


class OutputConfig(BaseModel):
    """Output settings."""

    results_dir: str = Field(default="results")
    experiment_name: str | None = Field(default=None)
    timestamp_format: str = Field(default="%Y%m%d_%H%M%S")
    save_raw_responses: bool = Field(default=True)
    save_parsed_responses: bool = Field(default=True)
    save_intermediate_results: bool = Field(default=True)
    generate_html_report: bool = Field(default=True)
    generate_markdown_report: bool = Field(default=True)
    generate_plots: bool = Field(default=True)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="rich")
    log_file: str | None = Field(default=None)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"level must be one of {allowed}")
        return v.upper()


class CacheConfig(BaseModel):
    """Caching configuration."""

    enabled: bool = Field(default=True)
    cache_dir: str = Field(default=".api_cache")
    ttl_hours: int = Field(default=24, ge=1)


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=100000, ge=1000)


class BenchmarkConfig(BaseModel):
    """Complete benchmark configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    tracks: TracksConfig = Field(default_factory=TracksConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)


# Environment variable pattern: ${VAR_NAME} or ${VAR_NAME:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in config values.

    Supports:
        ${VAR_NAME} - Required env var
        ${VAR_NAME:default} - Env var with default value

    Args:
        value: Configuration value (string, dict, list, or primitive)

    Returns:
        Value with environment variables interpolated

    Raises:
        ValueError: If required env var is not set
    """
    if isinstance(value, str):
        return _interpolate_string(value)
    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]
    return value


def _interpolate_string(value: str) -> str:
    """Interpolate environment variables in a string."""

    def replace_match(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)

        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        elif default is not None:
            return default
        else:
            raise ValueError(
                f"Environment variable '{var_name}' is not set and no default provided"
            )

    return ENV_VAR_PATTERN.sub(replace_match, value)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Values in override take precedence. Nested dicts are merged recursively.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    return config


def load_config(
    config_path: Path | None = None,
    *,
    interpolate_env: bool = True,
    validate: bool = True,
) -> dict[str, Any] | BenchmarkConfig:
    """Load and optionally validate a configuration file.

    Args:
        config_path: Path to config file. If None, returns default config.
        interpolate_env: Whether to interpolate environment variables
        validate: Whether to validate and return a BenchmarkConfig model

    Returns:
        Configuration dictionary or BenchmarkConfig model if validate=True
    """
    # Start with empty config or loaded config
    config = load_yaml_config(config_path) if config_path is not None else {}

    # Interpolate environment variables
    if interpolate_env:
        config = interpolate_env_vars(config)

    # Validate and return model or raw dict
    if validate:
        return BenchmarkConfig.model_validate(config)
    return config


def get_default_config_path() -> Path | None:
    """Get the default configuration file path.

    Searches for config in order:
        1. CHIMERA_CONFIG environment variable
        2. ./chimera.yaml
        3. ./chimera.yml
        4. ./configs/default.yaml

    Returns:
        Path to config file or None if not found
    """
    # Check environment variable
    env_config = os.environ.get("CHIMERA_CONFIG")
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path

    # Check common locations
    search_paths = [
        Path("chimera.yaml"),
        Path("chimera.yml"),
        Path("configs/default.yaml"),
        Path("configs/chimera.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    try:
        BenchmarkConfig.model_validate(config)
    except Exception as e:
        errors.append(str(e))

    return errors


def get_env_config_overrides() -> dict[str, Any]:
    """Get configuration overrides from environment variables.

    Supports:
        CHIMERA_MODEL_PROVIDER - Model provider
        CHIMERA_MODEL_NAME - Model name
        CHIMERA_MODEL_TEMPERATURE - Temperature
        CHIMERA_OUTPUT_DIR - Output directory
        CHIMERA_LOG_LEVEL - Log level
        CHIMERA_API_KEY_GEMINI - Gemini API key
        CHIMERA_API_KEY_OPENAI - OpenAI API key

    Returns:
        Configuration dictionary with env var overrides
    """
    overrides: dict[str, Any] = {}

    # Model configuration
    if provider := os.environ.get("CHIMERA_MODEL_PROVIDER"):
        overrides.setdefault("model", {})["provider"] = provider

    if name := os.environ.get("CHIMERA_MODEL_NAME"):
        overrides.setdefault("model", {})["name"] = name

    if temp := os.environ.get("CHIMERA_MODEL_TEMPERATURE"):
        import contextlib

        with contextlib.suppress(ValueError):
            overrides.setdefault("model", {})["temperature"] = float(temp)

    # Output configuration
    if output_dir := os.environ.get("CHIMERA_OUTPUT_DIR"):
        overrides.setdefault("output", {})["results_dir"] = output_dir

    # Logging configuration
    if log_level := os.environ.get("CHIMERA_LOG_LEVEL"):
        overrides.setdefault("logging", {})["level"] = log_level

    return overrides


def create_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    *,
    use_env_vars: bool = True,
) -> BenchmarkConfig:
    """Create a complete configuration from multiple sources.

    Configuration precedence (highest to lowest):
        1. CLI overrides
        2. Environment variable overrides
        3. User config file
        4. Default values

    Args:
        config_path: Path to user config file
        cli_overrides: Overrides from CLI options
        use_env_vars: Whether to include environment variable overrides

    Returns:
        Complete BenchmarkConfig
    """
    # Start with defaults
    config: dict[str, Any] = {}

    # Load user config if provided
    if config_path is not None:
        user_config = load_yaml_config(config_path)
        user_config = interpolate_env_vars(user_config)
        config = deep_merge(config, user_config)

    # Apply environment variable overrides
    if use_env_vars:
        env_overrides = get_env_config_overrides()
        config = deep_merge(config, env_overrides)

    # Apply CLI overrides
    if cli_overrides:
        config = deep_merge(config, cli_overrides)

    return BenchmarkConfig.model_validate(config)
