"""Integration tests for CHIMERA CLI module.

These tests verify that the CLI integrates properly with the rest
of the CHIMERA package and can be imported without errors.
"""

from __future__ import annotations


class TestCLIImports:
    """Test that CLI can be imported properly."""

    def test_import_cli_main(self) -> None:
        """Test importing the main CLI module."""
        from chimera.cli import main

        assert hasattr(main, "cli")
        assert callable(main.cli)

    def test_import_cli_from_package(self) -> None:
        """Test importing CLI from chimera.cli package."""
        from chimera.cli import cli

        assert callable(cli)

    def test_import_cli_from_main_package(self) -> None:
        """Test importing CLI from main chimera package."""
        from chimera import cli

        assert callable(cli)

    def test_cli_group_name(self) -> None:
        """Test that CLI group has correct name."""
        from chimera.cli.main import cli

        assert cli.name == "cli"

    def test_cli_commands_registered(self) -> None:
        """Test that all commands are registered."""
        from chimera.cli.main import cli

        commands = cli.commands
        expected_commands = {"run", "generate", "analyze", "info", "check"}
        actual_commands = set(commands.keys())
        assert expected_commands.issubset(
            actual_commands
        ), f"Missing commands: {expected_commands - actual_commands}"


class TestCLIHelperFunctions:
    """Test CLI helper function imports."""

    def test_import_helper_functions(self) -> None:
        """Test importing helper functions."""
        from chimera.cli.main import (
            _deep_merge,
            _generate_markdown_report,
            _get_default_model_name,
            _load_config,
            get_model_choices,
            get_track_choices,
        )

        assert callable(_deep_merge)
        assert callable(_get_default_model_name)
        assert callable(_generate_markdown_report)
        assert callable(_load_config)
        assert callable(get_model_choices)
        assert callable(get_track_choices)


class TestCLIEntryPoint:
    """Test CLI entry point configuration."""

    def test_entry_point_callable(self) -> None:
        """Test that the entry point target is callable."""
        # This simulates what happens when `chimera` is called
        # cli should be a click.Group
        import click

        from chimera.cli.main import cli

        assert isinstance(cli, click.Group)

    def test_cli_version_attribute(self) -> None:
        """Test CLI has version info."""
        from chimera.version import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestCLIGeneratorIntegration:
    """Test CLI integration with generators."""

    def test_calibration_generator_available(self) -> None:
        """Test that calibration generator can be imported."""
        from chimera.generators.calibration import (
            CalibrationGeneratorConfig,
            CalibrationTaskGenerator,
        )

        config = CalibrationGeneratorConfig(n_tasks=5, seed=42)
        generator = CalibrationTaskGenerator(config)
        assert generator is not None

    def test_error_detection_generator_available(self) -> None:
        """Test that error detection generator can be imported."""
        from chimera.generators.error_detection import (
            ErrorDetectionGeneratorConfig,
            ErrorDetectionTaskGenerator,
        )

        config = ErrorDetectionGeneratorConfig(n_tasks=5, seed=42)
        generator = ErrorDetectionTaskGenerator(config)
        assert generator is not None

    def test_knowledge_boundary_generator_available(self) -> None:
        """Test that knowledge boundary generator can be imported."""
        from chimera.generators.knowledge_boundary import (
            KnowledgeBoundaryGeneratorConfig,
            KnowledgeBoundaryTaskGenerator,
        )

        config = KnowledgeBoundaryGeneratorConfig(n_tasks=5, seed=42)
        generator = KnowledgeBoundaryTaskGenerator(config)
        assert generator is not None


class TestCLIDependencies:
    """Test that CLI dependencies are available."""

    def test_click_available(self) -> None:
        """Test click is installed."""
        import click

        assert click is not None

    def test_rich_available(self) -> None:
        """Test rich is installed."""
        import rich

        assert rich is not None

    def test_pyyaml_available(self) -> None:
        """Test PyYAML is installed."""
        import yaml

        assert yaml is not None


class TestCLIModelChoices:
    """Test model choice functions."""

    def test_track_choices_content(self) -> None:
        """Test track choices have expected content."""
        from chimera.cli.main import get_track_choices

        choices = get_track_choices()
        assert "calibration" in choices
        assert "error_detection" in choices
        assert "knowledge_boundary" in choices
        assert "self_correction" in choices
        assert "all" in choices

    def test_model_choices_content(self) -> None:
        """Test model choices have expected content."""
        from chimera.cli.main import get_model_choices

        choices = get_model_choices()
        assert "gemini" in choices
        assert "openai" in choices


class TestCLIDeepMerge:
    """Test the _deep_merge function."""

    def test_merge_flat_dicts(self) -> None:
        """Test merging flat dictionaries."""
        from chimera.cli.main import _deep_merge

        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}
        # Ensure original wasn't modified
        assert base == {"a": 1, "b": 2}

    def test_merge_nested_dicts(self) -> None:
        """Test merging nested dictionaries."""
        from chimera.cli.main import _deep_merge

        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 10, "inner3": 3}}
        result = _deep_merge(base, override)

        assert result["outer"]["inner1"] == 1
        assert result["outer"]["inner2"] == 10
        assert result["outer"]["inner3"] == 3

    def test_merge_mixed_types(self) -> None:
        """Test merging with mixed value types."""
        from chimera.cli.main import _deep_merge

        base = {"a": {"nested": 1}, "b": "string"}
        override = {"a": {"extra": 2}, "c": [1, 2, 3]}
        result = _deep_merge(base, override)

        assert result["a"]["nested"] == 1
        assert result["a"]["extra"] == 2
        assert result["b"] == "string"
        assert result["c"] == [1, 2, 3]


class TestCLIDefaultModelName:
    """Test the _get_default_model_name function."""

    def test_gemini_default(self) -> None:
        """Test default model for Gemini."""
        from chimera.cli.main import _get_default_model_name

        name = _get_default_model_name("gemini")
        assert "gemini" in name.lower()

    def test_openai_default(self) -> None:
        """Test default model for OpenAI."""
        from chimera.cli.main import _get_default_model_name

        name = _get_default_model_name("openai")
        assert "gpt" in name.lower()

    def test_unknown_provider(self) -> None:
        """Test default for unknown provider."""
        from chimera.cli.main import _get_default_model_name

        # Should return some default
        name = _get_default_model_name("unknown")
        assert name is not None
        assert len(name) > 0
