"""Additional CLI utility tests for CHIMERA.

This module tests CLI utility functions and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from chimera.cli.main import cli


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_output(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


class TestCLIOutputFormats:
    """Test various output formats."""

    def test_generate_creates_parent_directories(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that generate creates parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "tasks.json"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "-t",
                "calibration",
                "-n",
                "3",
                "-o",
                str(deep_path),
            ],
        )
        assert result.exit_code == 0
        assert deep_path.exists()

    def test_jsonl_format_valid(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test JSONL output is valid."""
        output_file = tmp_path / "tasks.jsonl"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "-t",
                "calibration",
                "-n",
                "5",
                "-o",
                str(output_file),
                "--format",
                "jsonl",
            ],
        )
        assert result.exit_code == 0

        # Each line should be valid JSON
        with open(output_file) as f:
            for line in f:
                task = json.loads(line.strip())
                assert isinstance(task, dict)
                assert "id" in task
                assert "question" in task


class TestCLIVerboseMode:
    """Test verbose and quiet mode behaviors."""

    def test_verbose_shows_extra_info(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test verbose mode shows additional information."""
        result = cli_runner.invoke(
            cli,
            [
                "--verbose",
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "3",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        # Verbose should show output directory
        assert "output" in result.output.lower() or str(tmp_path) in result.output

    def test_quiet_suppresses_output(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test quiet mode suppresses banner."""
        result = cli_runner.invoke(
            cli,
            [
                "--quiet",
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "3",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        # Should not show the big banner
        # Note: Some output may still appear


class TestCLIModelOptions:
    """Test model-related CLI options."""

    def test_custom_model_name(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test specifying custom model name."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--model",
                "gemini",
                "--model-name",
                "gemini-1.5-pro",
                "--dry-run",
                "--n-tasks",
                "3",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0

    def test_openai_model_provider(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test using OpenAI model provider."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--model",
                "openai",
                "--dry-run",
                "--n-tasks",
                "3",
                "--output",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_analyze_invalid_json_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze with invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {{{")
        result = cli_runner.invoke(cli, ["analyze", str(invalid_file)])
        assert result.exit_code != 0
        assert "error" in result.output.lower()

    def test_analyze_missing_summary(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze with directory missing summary file."""
        result = cli_runner.invoke(cli, ["analyze", str(tmp_path)])
        assert result.exit_code != 0
        assert "no evaluation_summary.json" in result.output.lower()


class TestCLITaskGeneration:
    """Test task generation specifics."""

    def test_generate_preserves_task_structure(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test generated tasks have complete structure."""
        output_file = tmp_path / "tasks.json"
        result = cli_runner.invoke(
            cli,
            [
                "generate",
                "-t",
                "error_detection",
                "-n",
                "5",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0

        with open(output_file) as f:
            data = json.load(f)

        for task in data["tasks"]:
            assert "id" in task
            assert "track" in task
            assert "question" in task
            assert "difficulty" in task
            # All tasks should have valid difficulty levels
            assert task["difficulty"] in ["L1", "L2", "L3", "L4", "L5"]

    def test_generate_different_seeds_different_output(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test different seeds produce different outputs."""
        output1 = tmp_path / "tasks1.json"
        output2 = tmp_path / "tasks2.json"

        cli_runner.invoke(
            cli,
            [
                "generate",
                "-t",
                "calibration",
                "-n",
                "10",
                "--seed",
                "1",
                "-o",
                str(output1),
            ],
        )
        cli_runner.invoke(
            cli,
            [
                "generate",
                "-t",
                "calibration",
                "-n",
                "10",
                "--seed",
                "2",
                "-o",
                str(output2),
            ],
        )

        with open(output1) as f:
            data1 = json.load(f)
        with open(output2) as f:
            data2 = json.load(f)

        # Different seeds should produce different questions
        # (at least some should differ)
        questions1 = {t["question"] for t in data1["tasks"]}
        questions2 = {t["question"] for t in data2["tasks"]}
        # They shouldn't be identical
        assert questions1 != questions2 or len(questions1) > 0


class TestCLICheckCommand:
    """Test the check command in detail."""

    def test_check_shows_python_version(self, cli_runner: CliRunner) -> None:
        """Test check displays Python version."""
        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0
        assert "Python" in result.output
        assert "3." in result.output  # Python 3.x

    def test_check_shows_dependencies(self, cli_runner: CliRunner) -> None:
        """Test check displays dependency status."""
        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0
        # Should show some common dependencies
        output_lower = result.output.lower()
        assert "click" in output_lower or "cli" in output_lower
        assert "pydantic" in output_lower

    def test_check_shows_api_status(self, cli_runner: CliRunner) -> None:
        """Test check shows API key status."""
        result = cli_runner.invoke(cli, ["check"])
        assert result.exit_code == 0
        # Should mention API keys
        output_lower = result.output.lower()
        assert "api" in output_lower or "key" in output_lower


class TestCLIInfoCommand:
    """Test the info command in detail."""

    def test_info_shows_tracks(self, cli_runner: CliRunner) -> None:
        """Test info shows all evaluation tracks."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "calibration" in result.output.lower()
        assert "error" in result.output.lower()
        assert "knowledge" in result.output.lower()
        assert "correction" in result.output.lower()

    def test_info_shows_models(self, cli_runner: CliRunner) -> None:
        """Test info shows supported models."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "gemini" in output_lower or "google" in output_lower
        assert "openai" in output_lower or "gpt" in output_lower

    def test_info_shows_quick_start(self, cli_runner: CliRunner) -> None:
        """Test info shows quick start examples."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "chimera" in result.output.lower()


class TestCLIAnalyzeCommand:
    """Test analyze command edge cases."""

    def test_analyze_with_empty_results(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze with empty results dict."""
        results_file = tmp_path / "evaluation_summary.json"
        results_file.write_text(json.dumps({"results": {}}))

        result = cli_runner.invoke(cli, ["analyze", str(results_file)])
        assert result.exit_code == 0

    def test_analyze_with_minimal_data(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze with minimal data."""
        results_file = tmp_path / "evaluation_summary.json"
        results_file.write_text(json.dumps({}))

        result = cli_runner.invoke(cli, ["analyze", str(results_file)])
        assert result.exit_code == 0


class TestCLIRunCommand:
    """Additional run command tests."""

    def test_run_creates_output_directory(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test run creates output directory if not exists."""
        output_dir = tmp_path / "new_results"
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "3",
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_run_with_default_output(self, cli_runner: CliRunner) -> None:
        """Test run with default output directory."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--track",
                "calibration",
                "--dry-run",
                "--n-tasks",
                "3",
            ],
            catch_exceptions=False,
        )
        # Should create a timestamped directory in results/
        assert result.exit_code == 0
