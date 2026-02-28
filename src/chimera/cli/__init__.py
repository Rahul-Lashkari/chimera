"""CHIMERA Command Line Interface.

This module provides the command-line interface for the CHIMERA benchmark,
enabling users to run evaluations, generate tasks, and analyze results.
"""

from chimera.cli.config_loader import (
    BenchmarkConfig,
    CacheConfig,
    CalibrationTrackConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    RateLimitingConfig,
    TracksConfig,
    create_config,
    deep_merge,
    get_default_config_path,
    get_env_config_overrides,
    interpolate_env_vars,
    load_config,
    load_yaml_config,
    validate_config,
)
from chimera.cli.main import cli

__all__ = [
    # CLI entry point
    "cli",
    # Configuration models
    "BenchmarkConfig",
    "ModelConfig",
    "TracksConfig",
    "CalibrationTrackConfig",
    "EvaluationConfig",
    "OutputConfig",
    "LoggingConfig",
    "CacheConfig",
    "RateLimitingConfig",
    # Configuration utilities
    "load_config",
    "load_yaml_config",
    "create_config",
    "validate_config",
    "interpolate_env_vars",
    "get_env_config_overrides",
    "get_default_config_path",
    "deep_merge",
]
