"""Unit tests for CHIMERA runner configuration.

Tests cover:
- RunConfig defaults and customization
- TrackConfig settings
- Configuration serialization
- Factory methods (quick, full)
"""

from pathlib import Path

import pytest

from chimera.models.task import DifficultyLevel, TrackType
from chimera.runner.config import (
    OutputFormat,
    RunConfig,
    TrackConfig,
)


class TestTrackConfig:
    """Tests for TrackConfig."""

    def test_default_config(self) -> None:
        """Test default track configuration."""
        config = TrackConfig(track=TrackType.CALIBRATION)

        assert config.track == TrackType.CALIBRATION
        assert config.enabled is True
        assert config.max_tasks is None
        assert config.difficulty_filter is None

    def test_custom_config(self) -> None:
        """Test custom track configuration."""
        config = TrackConfig(
            track=TrackType.ERROR_DETECTION,
            enabled=True,
            max_tasks=50,
            difficulty_filter=[DifficultyLevel.L3, DifficultyLevel.L4],
            seed=42,
        )

        assert config.max_tasks == 50
        assert len(config.difficulty_filter) == 2

    def test_disabled_track(self) -> None:
        """Test disabling a track."""
        config = TrackConfig(
            track=TrackType.SELF_CORRECTION,
            enabled=False,
        )

        assert config.enabled is False


class TestRunConfig:
    """Tests for RunConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RunConfig()

        assert config.name.startswith("chimera_run_")
        assert config.tracks is None  # All tracks
        assert config.max_tasks_per_track == 100
        assert config.batch_size == 10
        assert config.max_concurrent == 5
        assert config.shuffle_tasks is True
        assert config.verbose is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RunConfig(
            name="test_run",
            tracks=[TrackType.CALIBRATION],
            max_tasks_per_track=50,
            batch_size=5,
            verbose=True,
        )

        assert config.name == "test_run"
        assert len(config.tracks) == 1
        assert config.max_tasks_per_track == 50

    def test_tracks_from_strings(self) -> None:
        """Test tracks can be specified as strings."""
        config = RunConfig(tracks=["calibration", "error_detection"])

        assert TrackType.CALIBRATION in config.tracks
        assert TrackType.ERROR_DETECTION in config.tracks

    def test_output_dir_as_string(self) -> None:
        """Test output_dir accepts string."""
        config = RunConfig(output_dir="my_results")

        assert config.output_dir == Path("my_results")

    def test_get_enabled_tracks_all(self) -> None:
        """Test get_enabled_tracks returns all when not filtered."""
        config = RunConfig()

        enabled = config.get_enabled_tracks()

        assert len(enabled) == len(TrackType)

    def test_get_enabled_tracks_filtered(self) -> None:
        """Test get_enabled_tracks respects tracks list."""
        config = RunConfig(tracks=[TrackType.CALIBRATION])

        enabled = config.get_enabled_tracks()

        assert len(enabled) == 1
        assert TrackType.CALIBRATION in enabled

    def test_get_enabled_tracks_with_disabled(self) -> None:
        """Test get_enabled_tracks respects disabled tracks."""
        config = RunConfig(
            track_configs={
                TrackType.CALIBRATION: TrackConfig(
                    track=TrackType.CALIBRATION,
                    enabled=False,
                ),
            }
        )

        enabled = config.get_enabled_tracks()

        assert TrackType.CALIBRATION not in enabled

    def test_get_track_config(self) -> None:
        """Test getting track-specific config."""
        track_config = TrackConfig(
            track=TrackType.CALIBRATION,
            max_tasks=25,
        )
        config = RunConfig(track_configs={TrackType.CALIBRATION: track_config})

        result = config.get_track_config(TrackType.CALIBRATION)

        assert result.max_tasks == 25

    def test_get_track_config_default(self) -> None:
        """Test getting default track config."""
        config = RunConfig()

        result = config.get_track_config(TrackType.CALIBRATION)

        assert result.track == TrackType.CALIBRATION
        assert result.enabled is True

    def test_get_max_tasks(self) -> None:
        """Test getting max tasks for track."""
        config = RunConfig(
            max_tasks_per_track=100,
            track_configs={
                TrackType.CALIBRATION: TrackConfig(
                    track=TrackType.CALIBRATION,
                    max_tasks=50,
                ),
            },
        )

        # Track with override
        assert config.get_max_tasks(TrackType.CALIBRATION) == 50

        # Track without override
        assert config.get_max_tasks(TrackType.ERROR_DETECTION) == 100


class TestRunConfigSerialization:
    """Tests for RunConfig serialization."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        config = RunConfig(
            name="test",
            tracks=[TrackType.CALIBRATION],
            max_tasks_per_track=50,
        )

        d = config.to_dict()

        assert d["name"] == "test"
        assert d["tracks"] == ["calibration"]
        assert d["max_tasks_per_track"] == 50
        assert isinstance(d["output_dir"], str)
        assert isinstance(d["created_at"], str)

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        d = {
            "name": "restored_run",
            "tracks": ["calibration"],
            "max_tasks_per_track": 75,
            "created_at": "2025-01-15T10:30:00",
        }

        config = RunConfig.from_dict(d)

        assert config.name == "restored_run"
        assert config.max_tasks_per_track == 75

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        original = RunConfig(
            name="roundtrip_test",
            tracks=[TrackType.CALIBRATION, TrackType.ERROR_DETECTION],
            max_tasks_per_track=100,
            batch_size=20,
            verbose=True,
        )

        d = original.to_dict()
        restored = RunConfig.from_dict(d)

        assert restored.name == original.name
        assert restored.max_tasks_per_track == original.max_tasks_per_track
        assert restored.batch_size == original.batch_size


class TestRunConfigFactories:
    """Tests for RunConfig factory methods."""

    def test_quick_config(self) -> None:
        """Test quick configuration factory."""
        config = RunConfig.quick(
            tracks=["calibration"],
            max_tasks=25,
        )

        assert config.max_tasks_per_track == 25
        assert config.batch_size == 5
        assert config.verbose is True

    def test_quick_config_defaults(self) -> None:
        """Test quick config with defaults."""
        config = RunConfig.quick()

        assert config.max_tasks_per_track == 50

    def test_full_config(self) -> None:
        """Test full benchmark configuration."""
        config = RunConfig.full(name="full_test")

        assert config.name == "full_test"
        assert config.max_tasks_per_track == 500
        assert config.batch_size == 20
        assert config.max_concurrent == 10
        assert config.tracks is None  # All tracks


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_format_values(self) -> None:
        """Test output format values."""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.JSONL.value == "jsonl"
        assert OutputFormat.CSV.value == "csv"
        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_default_formats(self) -> None:
        """Test default output formats in config."""
        config = RunConfig()

        assert OutputFormat.JSON in config.output_formats
        assert OutputFormat.MARKDOWN in config.output_formats


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_batch_size_bounds(self) -> None:
        """Test batch_size validation."""
        # Valid
        config = RunConfig(batch_size=50)
        assert config.batch_size == 50

        # Invalid (too high)
        with pytest.raises(ValueError):
            RunConfig(batch_size=200)

        # Invalid (too low)
        with pytest.raises(ValueError):
            RunConfig(batch_size=0)

    def test_max_concurrent_bounds(self) -> None:
        """Test max_concurrent validation."""
        config = RunConfig(max_concurrent=25)
        assert config.max_concurrent == 25

        with pytest.raises(ValueError):
            RunConfig(max_concurrent=100)

    def test_timeout_positive(self) -> None:
        """Test timeout must be positive."""
        with pytest.raises(ValueError):
            RunConfig(timeout_seconds=0)

    def test_max_tasks_positive(self) -> None:
        """Test max_tasks_per_track must be positive."""
        with pytest.raises(ValueError):
            RunConfig(max_tasks_per_track=0)
