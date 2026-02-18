"""Tests for CHIMERA CLI module.

This module provides comprehensive tests for the command-line interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from chimera.cli.main import (
    _deep_merge,
    _generate_markdown_report,
    _get_default_model_name,
    _load_config,
    cli,
    get_model_choices,
    get_track_choices,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config_content = """
model:
    provider: openai
    name: gpt-4o-mini
generation:
    n_tasks: 50
    seed: 123
evaluation:
    batch_size: 5
    timeout: 30
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_results_file(tmp_path: Path) -> Path:
    """Create sample evaluation results."""
    results = {
        "timestamp": "2024-01-01T12:00:00",
        "config": {
            "model": {"provider": "gemini", "name": "gemini-2.0-flash-exp"},
            "generation": {"n_tasks": 100, "seed": 42},
        },
        "results": {
            "calibration": {"status": "success", "n_tasks": 100, "message": "Completed"},
            "error_detection": {"status": "error", "error": "API error"},
        },
    }
    results_path = tmp_path / "evaluation_summary.json"
    results_path.write_text(json.dumps(results, indent=2))
    return results_path


class TestCLIGroup:
    """Tests for the main CLI group."""

    def test_cli_version(self, cli_runner: CliRunner) -> None:
        """Test --version flag."""
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "chimera" in result.output.lower()

    def test_cli_help(self, cli_runner: CliRunner) -> None:
        """Test --help flag."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CHIMERA" in result.output
        assert "calibration" in result.output.lower()
        assert "meta-cognitive" in result.output.lower()

    def test_cli_verbose_flag(self, cli_runner: CliRunner) -> None:
        """Test --verbose flag sets context."""
        result = cli_runner.invoke(cli, ["--verbose", "info"])
        assert result.exit_code == 0

    def test_cli_quiet_flag(self, cli_runner: CliRunner) -> None:
        """Test --quiet flag sets context."""
        result = cli_runner.invoke(cli, ["--quiet", "info"])
        assert result.exit_code == 0


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self, cli_runner: CliRunner) -> None:
        """Test run command help."""
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--track" in result.output
        assert "--model" in result.output
        assert "--config" in result.output

    def test_run_dry_run_calibration(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test dry-run with calibration track."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "5",
                "--output",
                str(temp_dir),
            ],
        )
        assert result.exit_code == 0
        assert "calibration" in result.output.lower()
        # Check that tasks file was created
        assert (temp_dir / "calibration_tasks.json").exists()

    def test_run_dry_run_error_detection(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test dry-run with error detection track."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "error_detection",
                "--dry-run",
                "--n-tasks",
                "5",
                "--output",
                str(temp_dir),
            ],
        )
        assert result.exit_code == 0
        assert "error_detection" in result.output.lower()

    def test_run_dry_run_knowledge_boundary(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test dry-run with knowledge boundary track."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "knowledge_boundary",
                "--dry-run",
                "--n-tasks",
                "5",
                "--output",
                str(temp_dir),
            ],
        )
        assert result.exit_code == 0
        assert "knowledge_boundary" in result.output.lower()

    def test_run_with_config_file(
        self,
        cli_runner: CliRunner,
        temp_dir: Path,
        sample_config_file: Path,
    ) -> None:
        """Test run with configuration file."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--config",
                str(sample_config_file),
                "--dry-run",
                "--output",
                str(temp_dir),
            ],
        )
        assert result.exit_code == 0

    def test_run_with_seed(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test run with seed for reproducibility."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "5",
                "--seed",
                "42",
                "--output",
                str(temp_dir),
            ],
        )
        assert result.exit_code == 0

    def test_run_all_tracks_dry_run(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test dry-run with all tracks."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "all",
                "--dry-run",
                "--n-tasks",
                "3",
                "--output",
                str(temp_dir),
            ],
        )
        assert result.exit_code == 0
        # Should mention multiple tracks
        assert "calibration" in result.output.lower()
        assert "error_detection" in result.output.lower()


class TestGenerateCommand:
    """Tests for the generate command."""

    def test_generate_help(self, cli_runner: CliRunner) -> None:
        """Test generate command help."""
        result = cli_runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--track" in result.output
        assert "--n-tasks" in result.output
        assert "--output" in result.output

    def test_generate_calibration_json(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test generating calibration tasks as JSON."""
        output_file = temp_dir / "tasks.json"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "calibration",
                "--n-tasks",
                "5",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        assert "tasks" in data
        assert data["n_tasks"] == 5

    def test_generate_calibration_jsonl(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test generating calibration tasks as JSONL."""
        output_file = temp_dir / "tasks.jsonl"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "calibration",
                "--n-tasks",
                "5",
                "--output",
                str(output_file),
                "--format",
                "jsonl",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSONL structure
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            task = json.loads(line)
            assert "id" in task
            assert "question" in task

    def test_generate_error_detection(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test generating error detection tasks."""
        output_file = temp_dir / "ed_tasks.json"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "error_detection",
                "--n-tasks",
                "3",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_knowledge_boundary(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test generating knowledge boundary tasks."""
        output_file = temp_dir / "kb_tasks.json"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "knowledge_boundary",
                "--n-tasks",
                "3",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_with_seed(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test generating with seed produces consistent results."""
        output1 = temp_dir / "tasks1.json"
        output2 = temp_dir / "tasks2.json"

        # Generate twice with same seed
        result1 = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "calibration",
                "--n-tasks",
                "5",
                "--seed",
                "42",
                "--output",
                str(output1),
            ],
        )
        result2 = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "calibration",
                "--n-tasks",
                "5",
                "--seed",
                "42",
                "--output",
                str(output2),
            ],
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Results should be identical
        with open(output1) as f:
            data1 = json.load(f)
        with open(output2) as f:
            data2 = json.load(f)

        # The questions should be identical (deterministic)
        for t1, t2 in zip(data1["tasks"], data2["tasks"], strict=True):
            assert t1["question"] == t2["question"]


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_help(self, cli_runner: CliRunner) -> None:
        """Test analyze command help."""
        result = cli_runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "RESULTS_PATH" in result.output

    def test_analyze_table_format(self, cli_runner: CliRunner, sample_results_file: Path) -> None:
        """Test analyze with table output."""
        result = cli_runner.invoke(cli, ["analyze", str(sample_results_file), "--format", "table"])
        assert result.exit_code == 0
        assert "calibration" in result.output.lower()

    def test_analyze_json_format(self, cli_runner: CliRunner, sample_results_file: Path) -> None:
        """Test analyze with JSON output."""
        result = cli_runner.invoke(cli, ["analyze", str(sample_results_file), "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON in output
        assert "timestamp" in result.output

    def test_analyze_markdown_format(
        self, cli_runner: CliRunner, sample_results_file: Path
    ) -> None:
        """Test analyze with markdown output."""
        result = cli_runner.invoke(
            cli, ["analyze", str(sample_results_file), "--format", "markdown"]
        )
        assert result.exit_code == 0
        assert "# CHIMERA" in result.output

    def test_analyze_markdown_to_file(
        self,
        cli_runner: CliRunner,
        sample_results_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test saving markdown report to file."""
        output_file = temp_dir / "report.md"
        result = cli_runner.invoke(
            cli,
            [
                "analyze",
                str(sample_results_file),
                "--format",
                "markdown",
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        assert "# CHIMERA" in output_file.read_text()

    def test_analyze_directory(self, cli_runner: CliRunner, sample_results_file: Path) -> None:
        """Test analyze with directory containing results."""
        results_dir = sample_results_file.parent
        result = cli_runner.invoke(cli, ["analyze", str(results_dir)])
        assert result.exit_code == 0


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_command(self, cli_runner: CliRunner) -> None:
        """Test info command displays information."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "CHIMERA" in result.output
        assert "calibration" in result.output.lower()
        assert "error_detection" in result.output.lower()
        assert "knowledge_boundary" in result.output.lower()
        assert "self_correction" in result.output.lower()


class TestCheckCommand:
    """Tests for the check command."""

    def test_check_command(self, cli_runner: CliRunner) -> None:
        """Test check command runs without error."""
        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0
        assert "Python" in result.output
        assert "click" in result.output or "CLI" in result.output

    def test_check_all_flag(self, cli_runner: CliRunner) -> None:
        """Test check command with --all flag."""
        result = cli_runner.invoke(cli, ["check", "--all"])
        assert result.exit_code == 0


class TestHelperFunctions:
    """Tests for CLI helper functions."""

    def test_get_track_choices(self) -> None:
        """Test track choices include all tracks."""
        choices = get_track_choices()
        assert "calibration" in choices
        assert "error_detection" in choices
        assert "knowledge_boundary" in choices
        assert "self_correction" in choices
        assert "all" in choices

    def test_get_model_choices(self) -> None:
        """Test model choices include supported providers."""
        choices = get_model_choices()
        assert "gemini" in choices
        assert "openai" in choices

    def test_get_default_model_name(self) -> None:
        """Test default model names."""
        assert "gemini" in _get_default_model_name("gemini").lower()
        assert "gpt" in _get_default_model_name("openai").lower()
        # Unknown provider should return gemini default
        assert _get_default_model_name("unknown") == "gemini-2.0-flash-exp"

    def test_deep_merge_simple(self) -> None:
        """Test deep merge with simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self) -> None:
        """Test deep merge with nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 10, "z": 20}, "c": 4}
        result = _deep_merge(base, override)
        assert result["a"] == {"x": 1, "y": 10, "z": 20}
        assert result["b"] == 3
        assert result["c"] == 4

    def test_load_config_defaults(self, temp_dir: Path) -> None:
        """Test loading config with defaults."""
        config = _load_config(None, "gemini", None, 100, 42)
        assert config["model"]["provider"] == "gemini"
        assert config["generation"]["n_tasks"] == 100
        assert config["generation"]["seed"] == 42

    def test_load_config_with_file(self, sample_config_file: Path) -> None:
        """Test loading config from file."""
        config = _load_config(sample_config_file, "gemini", None, 100, None)
        # File overrides defaults
        assert config["model"]["provider"] == "openai"
        assert config["generation"]["n_tasks"] == 50
        assert config["generation"]["seed"] == 123

    def test_generate_markdown_report(self) -> None:
        """Test markdown report generation."""
        data: dict[str, Any] = {
            "timestamp": "2024-01-01T12:00:00",
            "config": {
                "model": {"provider": "gemini", "name": "gemini-2.0-flash"},
                "generation": {"n_tasks": 100, "seed": 42},
            },
            "results": {
                "calibration": {"status": "success", "n_tasks": 100},
            },
        }
        report = _generate_markdown_report(data)
        assert "# CHIMERA" in report
        assert "gemini" in report
        assert "100" in report
        assert "calibration" in report


class TestCLIEdgeCases:
    """Tests for CLI edge cases and error handling."""

    def test_run_invalid_track(self, cli_runner: CliRunner) -> None:
        """Test run with invalid track name."""
        result = cli_runner.invoke(cli, ["run", "--track", "invalid_track"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    def test_run_invalid_model(self, cli_runner: CliRunner) -> None:
        """Test run with invalid model provider."""
        result = cli_runner.invoke(cli, ["run", "--model", "invalid_model"])
        assert result.exit_code != 0

    def test_analyze_nonexistent_file(self, cli_runner: CliRunner) -> None:
        """Test analyze with nonexistent file."""
        result = cli_runner.invoke(cli, ["analyze", "/nonexistent/path/results.json"])
        assert result.exit_code != 0

    def test_generate_self_correction_not_implemented(
        self, cli_runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test generating self_correction shows appropriate message."""
        output_file = temp_dir / "tasks.json"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "--track",
                "self_correction",
                "--n-tasks",
                "5",
                "--output",
                str(output_file),
            ],
        )
        # Should handle gracefully
        assert "coming soon" in result.output.lower() or result.exit_code == 0


class TestCLIIntegration:
    """Integration tests for the CLI."""

    def test_full_workflow_dry_run(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test complete workflow: run -> analyze."""
        # Step 1: Run evaluation (dry run)
        run_result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "5",
                "--output",
                str(temp_dir),
            ],
        )
        assert run_result.exit_code == 0

        # Verify output
        assert (temp_dir / "calibration_tasks.json").exists()

    def test_generate_then_verify(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test generating tasks and verifying structure."""
        output_file = temp_dir / "tasks.json"

        # Generate tasks
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "-t",
                "calibration",
                "-n",
                "10",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0

        # Verify structure
        with open(output_file) as f:
            data = json.load(f)

        assert data["track"] == "calibration"
        assert data["n_tasks"] == 10
        assert len(data["tasks"]) == 10

        for task in data["tasks"]:
            assert "id" in task
            assert "question" in task
            assert "difficulty" in task
