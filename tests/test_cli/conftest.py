"""Pytest configuration for CLI tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create an isolated CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir(parents=True)
    return output


@pytest.fixture
def sample_evaluation_results(tmp_path: Path) -> Path:
    """Create sample evaluation results for testing."""
    results: dict[str, Any] = {
        "timestamp": "2024-01-15T10:30:00",
        "config": {
            "model": {
                "provider": "gemini",
                "name": "gemini-2.0-flash-exp",
            },
            "generation": {
                "n_tasks": 100,
                "seed": 42,
            },
            "evaluation": {
                "batch_size": 10,
                "max_retries": 3,
                "timeout": 60,
            },
        },
        "results": {
            "calibration": {
                "status": "success",
                "n_tasks": 100,
                "message": "Completed successfully",
                "metrics": {
                    "ece": 0.085,
                    "mce": 0.15,
                    "brier_score": 0.12,
                },
            },
            "error_detection": {
                "status": "success",
                "n_tasks": 100,
                "message": "Completed successfully",
                "metrics": {
                    "precision": 0.82,
                    "recall": 0.78,
                    "f1": 0.80,
                },
            },
            "knowledge_boundary": {
                "status": "success",
                "n_tasks": 100,
                "message": "Completed successfully",
                "metrics": {
                    "appropriate_abstention_rate": 0.65,
                    "false_answer_rate": 0.15,
                },
            },
            "self_correction": {
                "status": "skipped",
                "message": "Track not yet implemented",
            },
        },
    }

    results_dir = tmp_path / "evaluation_run"
    results_dir.mkdir(parents=True)

    summary_file = results_dir / "evaluation_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))

    return results_dir


@pytest.fixture
def sample_yaml_config(tmp_path: Path) -> Path:
    """Create a sample YAML configuration file."""
    config_content = """\
# CHIMERA Benchmark Configuration
model:
  provider: gemini
  name: gemini-2.0-flash-exp

generation:
  n_tasks: 50
  seed: 123

evaluation:
  batch_size: 5
  max_retries: 2
  timeout: 45

tracks:
  - calibration
  - error_detection
  - knowledge_boundary
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Create a minimal configuration file."""
    config_content = """\
model:
  provider: openai
generation:
  n_tasks: 10
"""
    config_file = tmp_path / "minimal_config.yaml"
    config_file.write_text(config_content)
    return config_file
