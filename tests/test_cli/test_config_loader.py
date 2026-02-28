"""Tests for configuration loader module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chimera.cli.config_loader import (
    BenchmarkConfig,
    CalibrationTrackConfig,
    EvaluationConfig,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
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


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.provider == "gemini"
        assert config.name == "gemini-2.0-flash"
        assert config.temperature == 0.0
        assert config.max_tokens == 2048

    def test_valid_provider(self) -> None:
        """Test valid provider values."""
        for provider in ["gemini", "openai", "anthropic", "local"]:
            config = ModelConfig(provider=provider)
            assert config.provider == provider

    def test_invalid_provider(self) -> None:
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Provider must be one of"):
            ModelConfig(provider="invalid")

    def test_temperature_bounds(self) -> None:
        """Test temperature validation."""
        ModelConfig(temperature=0.0)
        ModelConfig(temperature=2.0)

        with pytest.raises(ValueError):
            ModelConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(temperature=2.1)

    def test_max_tokens_bounds(self) -> None:
        """Test max_tokens validation."""
        ModelConfig(max_tokens=1)
        ModelConfig(max_tokens=100000)

        with pytest.raises(ValueError):
            ModelConfig(max_tokens=0)


class TestEvaluationConfig:
    """Tests for EvaluationConfig validation."""

    def test_default_values(self) -> None:
        """Test default evaluation config."""
        config = EvaluationConfig()
        assert config.n_bins == 15
        assert config.bin_strategy == "uniform"
        assert config.bootstrap_samples == 1000

    def test_valid_bin_strategy(self) -> None:
        """Test valid bin strategies."""
        for strategy in ["uniform", "quantile"]:
            config = EvaluationConfig(bin_strategy=strategy)
            assert config.bin_strategy == strategy

    def test_invalid_bin_strategy(self) -> None:
        """Test invalid bin strategy."""
        with pytest.raises(ValueError, match="bin_strategy must be one of"):
            EvaluationConfig(bin_strategy="invalid")


class TestLoggingConfig:
    """Tests for LoggingConfig validation."""

    def test_valid_log_levels(self) -> None:
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_case_insensitive_log_level(self) -> None:
        """Test log level is case-insensitive."""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"

    def test_invalid_log_level(self) -> None:
        """Test invalid log level."""
        with pytest.raises(ValueError, match="level must be one of"):
            LoggingConfig(level="TRACE")


class TestTracksConfig:
    """Tests for TracksConfig."""

    def test_default_tracks(self) -> None:
        """Test all tracks have defaults."""
        config = TracksConfig()
        assert isinstance(config.calibration, CalibrationTrackConfig)
        assert isinstance(config.error_detection.error_types, list)
        assert config.self_correction.enabled is True

    def test_track_max_samples(self) -> None:
        """Test track max_samples validation."""
        config = TracksConfig(calibration=CalibrationTrackConfig(max_samples=500))
        assert config.calibration.max_samples == 500


class TestBenchmarkConfig:
    """Tests for complete BenchmarkConfig."""

    def test_default_config(self) -> None:
        """Test default benchmark config."""
        config = BenchmarkConfig()
        assert config.model.provider == "gemini"
        assert config.tracks.calibration.enabled is True
        assert config.cache.enabled is True

    def test_nested_config(self) -> None:
        """Test nested configuration."""
        config = BenchmarkConfig(
            model=ModelConfig(provider="openai", name="gpt-4o"),
            output=OutputConfig(results_dir="custom_results"),
        )
        assert config.model.provider == "openai"
        assert config.output.results_dir == "custom_results"


class TestInterpolateEnvVars:
    """Tests for environment variable interpolation."""

    def test_simple_interpolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test simple env var interpolation."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = interpolate_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_interpolation_with_default(self) -> None:
        """Test interpolation with default value."""
        # Ensure var is not set
        os.environ.pop("UNSET_VAR", None)
        result = interpolate_env_vars("${UNSET_VAR:default_value}")
        assert result == "default_value"

    def test_required_var_not_set(self) -> None:
        """Test required var raises error."""
        os.environ.pop("REQUIRED_VAR", None)
        with pytest.raises(ValueError, match="Environment variable 'REQUIRED_VAR'"):
            interpolate_env_vars("${REQUIRED_VAR}")

    def test_interpolate_nested_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test interpolation in nested dict."""
        monkeypatch.setenv("MODEL_NAME", "gemini-pro")
        data = {
            "model": {
                "name": "${MODEL_NAME}",
                "temperature": 0.0,
            },
            "output": "${OUTPUT_DIR:results}",
        }
        result = interpolate_env_vars(data)
        assert result["model"]["name"] == "gemini-pro"
        assert result["output"] == "results"

    def test_interpolate_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test interpolation in list."""
        monkeypatch.setenv("ITEM_A", "value_a")
        data = ["${ITEM_A}", "${ITEM_B:default_b}"]
        result = interpolate_env_vars(data)
        assert result == ["value_a", "default_b"]

    def test_no_interpolation_for_non_strings(self) -> None:
        """Test non-strings are passed through."""
        data = {"number": 42, "flag": True, "none": None}
        result = interpolate_env_vars(data)
        assert result == data


class TestDeepMerge:
    """Tests for deep merge functionality."""

    def test_simple_merge(self) -> None:
        """Test simple dict merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Test nested dict merge."""
        base = {"model": {"provider": "gemini", "temperature": 0.0}}
        override = {"model": {"temperature": 0.5}}
        result = deep_merge(base, override)
        assert result == {"model": {"provider": "gemini", "temperature": 0.5}}

    def test_deep_nested_merge(self) -> None:
        """Test deeply nested merge."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 3}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 3, "d": 2}}}

    def test_override_replaces_non_dict(self) -> None:
        """Test override replaces non-dict values."""
        base = {"a": {"b": 1}}
        override = {"a": "string"}
        result = deep_merge(base, override)
        assert result == {"a": "string"}

    def test_base_not_modified(self) -> None:
        """Test base dict is not modified."""
        base = {"a": 1}
        override = {"b": 2}
        deep_merge(base, override)
        assert base == {"a": 1}


class TestLoadYamlConfig:
    """Tests for YAML loading."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  provider: openai\n  name: gpt-4")

        result = load_yaml_config(config_file)
        assert result["model"]["provider"] == "openai"
        assert result["model"]["name"] == "gpt-4"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading empty YAML returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = load_yaml_config(config_file)
        assert result == {}


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self) -> None:
        """Test loading default config without file."""
        config = load_config(validate=True)
        assert isinstance(config, BenchmarkConfig)
        assert config.model.provider == "gemini"

    def test_load_with_file(self, tmp_path: Path) -> None:
        """Test loading config from file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  provider: openai\n  name: gpt-4o-mini")

        config = load_config(config_file, validate=True)
        assert isinstance(config, BenchmarkConfig)
        assert config.model.provider == "openai"
        assert config.model.name == "gpt-4o-mini"

    def test_load_without_validation(self, tmp_path: Path) -> None:
        """Test loading config as raw dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("custom_field: custom_value")

        config = load_config(config_file, validate=False)
        assert isinstance(config, dict)
        assert config["custom_field"] == "custom_value"


class TestGetDefaultConfigPath:
    """Tests for default config path detection."""

    def test_returns_none_when_no_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns None when no config file found."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CHIMERA_CONFIG", raising=False)
        result = get_default_config_path()
        assert result is None

    def test_finds_chimera_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test finds chimera.yaml in current directory."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CHIMERA_CONFIG", raising=False)
        config_file = tmp_path / "chimera.yaml"
        config_file.write_text("model:\n  provider: gemini")

        result = get_default_config_path()
        # Compare resolved paths since result may be relative
        assert result is not None
        assert result.resolve() == config_file.resolve()

    def test_env_var_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test CHIMERA_CONFIG env var takes precedence."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("model:\n  provider: openai")
        monkeypatch.setenv("CHIMERA_CONFIG", str(config_file))

        result = get_default_config_path()
        assert result == config_file


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_valid_config(self) -> None:
        """Test validating a valid config."""
        config = {"model": {"provider": "gemini"}}
        errors = validate_config(config)
        assert errors == []

    def test_invalid_provider(self) -> None:
        """Test invalid provider produces error."""
        config = {"model": {"provider": "invalid_provider"}}
        errors = validate_config(config)
        assert len(errors) > 0
        assert "provider" in errors[0].lower() or "Provider" in errors[0]


class TestGetEnvConfigOverrides:
    """Tests for environment variable overrides."""

    def test_no_overrides_when_no_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns empty when no env vars set."""
        # Clear relevant env vars
        for var in [
            "CHIMERA_MODEL_PROVIDER",
            "CHIMERA_MODEL_NAME",
            "CHIMERA_MODEL_TEMPERATURE",
            "CHIMERA_OUTPUT_DIR",
            "CHIMERA_LOG_LEVEL",
        ]:
            monkeypatch.delenv(var, raising=False)

        overrides = get_env_config_overrides()
        assert overrides == {}

    def test_model_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test model provider override."""
        monkeypatch.setenv("CHIMERA_MODEL_PROVIDER", "openai")
        overrides = get_env_config_overrides()
        assert overrides["model"]["provider"] == "openai"

    def test_multiple_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test multiple overrides."""
        monkeypatch.setenv("CHIMERA_MODEL_PROVIDER", "openai")
        monkeypatch.setenv("CHIMERA_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("CHIMERA_OUTPUT_DIR", "/tmp/results")

        overrides = get_env_config_overrides()
        assert overrides["model"]["provider"] == "openai"
        assert overrides["model"]["name"] == "gpt-4"
        assert overrides["output"]["results_dir"] == "/tmp/results"

    def test_temperature_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test temperature string is converted to float."""
        monkeypatch.setenv("CHIMERA_MODEL_TEMPERATURE", "0.7")
        overrides = get_env_config_overrides()
        assert overrides["model"]["temperature"] == 0.7

    def test_invalid_temperature_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invalid temperature is ignored."""
        monkeypatch.setenv("CHIMERA_MODEL_TEMPERATURE", "not_a_number")
        overrides = get_env_config_overrides()
        assert "temperature" not in overrides.get("model", {})


class TestCreateConfig:
    """Tests for create_config function."""

    def test_default_config(self) -> None:
        """Test creating default config."""
        config = create_config()
        assert isinstance(config, BenchmarkConfig)
        assert config.model.provider == "gemini"

    def test_with_config_file(self, tmp_path: Path) -> None:
        """Test creating config from file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  provider: openai\n  name: gpt-4o")

        config = create_config(config_path=config_file)
        assert config.model.provider == "openai"
        assert config.model.name == "gpt-4o"

    def test_cli_overrides(self) -> None:
        """Test CLI overrides take precedence."""
        cli_overrides = {"model": {"provider": "openai", "name": "gpt-4o-mini"}}
        config = create_config(cli_overrides=cli_overrides)
        assert config.model.provider == "openai"
        assert config.model.name == "gpt-4o-mini"

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable overrides."""
        monkeypatch.setenv("CHIMERA_MODEL_PROVIDER", "openai")
        config = create_config(use_env_vars=True)
        assert config.model.provider == "openai"

    def test_precedence_order(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration precedence: CLI > env > file > defaults."""
        # File says openai
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  provider: openai")

        # Env says anthropic
        monkeypatch.setenv("CHIMERA_MODEL_PROVIDER", "anthropic")

        # CLI says local
        cli_overrides = {"model": {"provider": "local"}}

        # CLI should win
        config = create_config(
            config_path=config_file,
            cli_overrides=cli_overrides,
            use_env_vars=True,
        )
        assert config.model.provider == "local"

    def test_disabled_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test env vars can be disabled."""
        monkeypatch.setenv("CHIMERA_MODEL_PROVIDER", "openai")
        config = create_config(use_env_vars=False)
        # Should use default, not env var
        assert config.model.provider == "gemini"
