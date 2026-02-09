"""Configuration classes for CHIMERA benchmark runner.

This module defines configuration options for controlling benchmark execution,
including track selection, sampling, concurrency, and output settings.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from chimera.models.task import DifficultyLevel, TrackType


class OutputFormat(str, Enum):
    """Output format options for benchmark results.

    Attributes:
        JSON: JSON format (default, machine-readable).
        JSONL: JSON Lines format (streaming-friendly).
        CSV: CSV format (spreadsheet-compatible).
        MARKDOWN: Markdown format (human-readable reports).
    """

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    MARKDOWN = "markdown"


class ProgressCallback(Protocol):
    """Protocol for progress callback functions.

    Implement this protocol to receive progress updates during
    benchmark execution.
    """

    def __call__(
        self,
        completed: int,
        total: int,
        current_task: str | None = None,
        track: str | None = None,
    ) -> None:
        """Called on progress update.

        Args:
            completed: Number of completed tasks.
            total: Total number of tasks.
            current_task: Description of current task (optional).
            track: Current track name (optional).
        """
        ...


class TrackConfig(BaseModel):
    """Configuration for a specific evaluation track.

    Allows fine-grained control over individual tracks.

    Attributes:
        track: The track type to configure.
        enabled: Whether this track is enabled.
        max_tasks: Maximum tasks for this track (overrides global).
        difficulty_filter: Filter tasks by difficulty levels.
        categories: Filter tasks by categories.
        seed: Random seed for task sampling.
    """

    track: TrackType
    enabled: bool = True
    max_tasks: int | None = None
    difficulty_filter: list[DifficultyLevel] | None = None
    categories: list[str] | None = None
    seed: int | None = None

    model_config = {"extra": "allow"}


class RunConfig(BaseModel):
    """Configuration for benchmark execution.

    Controls all aspects of benchmark running including track selection,
    task sampling, concurrency, and output settings.

    Attributes:
        name: Name for this benchmark run.
        description: Optional description of the run.
        tracks: List of tracks to evaluate (None = all tracks).
        track_configs: Per-track configuration overrides.
        max_tasks_per_track: Global limit on tasks per track.
        total_max_tasks: Absolute maximum tasks across all tracks.
        batch_size: Number of tasks to process in each batch.
        max_concurrent: Maximum concurrent API requests.
        timeout_seconds: Timeout for individual API calls.
        max_retries: Maximum retries for failed API calls.
        retry_delay: Delay between retries in seconds.
        shuffle_tasks: Whether to shuffle task order.
        seed: Random seed for reproducibility.
        save_responses: Whether to save raw model responses.
        save_intermediate: Whether to save intermediate results.
        output_dir: Directory for output files.
        output_formats: Formats for output files.
        verbose: Enable verbose logging.
        dry_run: Validate configuration without running.
        resume_from: Path to checkpoint for resuming runs.
        checkpoint_interval: Save checkpoint every N tasks.
        created_at: Timestamp of configuration creation.
        metadata: Additional metadata for the run.
    """

    # Run identification
    name: str = Field(default_factory=lambda: f"chimera_run_{datetime.now():%Y%m%d_%H%M%S}")
    description: str | None = None

    # Track selection
    tracks: list[TrackType] | None = None  # None = all tracks
    track_configs: dict[TrackType, TrackConfig] = Field(default_factory=dict)

    # Task limits
    max_tasks_per_track: int = Field(default=100, ge=1)
    total_max_tasks: int | None = None

    # Execution settings
    batch_size: int = Field(default=10, ge=1, le=100)
    max_concurrent: int = Field(default=5, ge=1, le=50)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0)

    # Task ordering
    shuffle_tasks: bool = True
    seed: int | None = None

    # Output settings
    save_responses: bool = True
    save_intermediate: bool = True
    output_dir: Path = Field(default=Path("results"))
    output_formats: list[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.JSON, OutputFormat.MARKDOWN]
    )

    # Logging
    verbose: bool = False

    # Checkpointing
    dry_run: bool = False
    resume_from: Path | None = None
    checkpoint_interval: int = Field(default=50, ge=1)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("tracks", mode="before")
    @classmethod
    def validate_tracks(cls, v: Any) -> list[TrackType] | None:
        """Validate and convert track specifications."""
        if v is None:
            return None
        if isinstance(v, str):
            return [TrackType(v)]
        if isinstance(v, list):
            return [TrackType(t) if isinstance(t, str) else t for t in v]
        # Handle case where v is already a list of TrackType
        return list(v) if hasattr(v, "__iter__") else None

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v: Any) -> Path:
        """Convert output_dir to Path."""
        return Path(v) if isinstance(v, str) else v

    def get_enabled_tracks(self) -> list[TrackType]:
        """Get list of enabled tracks.

        Returns:
            List of TrackType values that are enabled for this run.
        """
        enabled = list(self.tracks) if self.tracks is not None else list(TrackType)

        # Apply per-track config
        for track, config in self.track_configs.items():
            if not config.enabled and track in enabled:
                enabled.remove(track)

        return enabled

    def get_track_config(self, track: TrackType) -> TrackConfig:
        """Get configuration for a specific track.

        Args:
            track: The track to get configuration for.

        Returns:
            TrackConfig for the specified track.
        """
        if track in self.track_configs:
            return self.track_configs[track]
        return TrackConfig(track=track)

    def get_max_tasks(self, track: TrackType) -> int:
        """Get maximum tasks for a specific track.

        Args:
            track: The track to get limit for.

        Returns:
            Maximum number of tasks for the track.
        """
        track_config = self.get_track_config(track)
        if track_config.max_tasks is not None:
            return track_config.max_tasks
        return self.max_tasks_per_track

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        data = self.model_dump()
        # Convert enums and paths
        data["tracks"] = [t.value for t in self.tracks] if self.tracks else None
        data["output_dir"] = str(self.output_dir)
        data["output_formats"] = [f.value for f in self.output_formats]
        data["created_at"] = self.created_at.isoformat()
        if self.resume_from:
            data["resume_from"] = str(self.resume_from)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            RunConfig instance.
        """
        # Handle datetime
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    @classmethod
    def quick(
        cls,
        tracks: list[str] | None = None,
        max_tasks: int = 50,
        **kwargs: Any,
    ) -> "RunConfig":
        """Create a quick configuration for testing.

        Args:
            tracks: Track names to evaluate.
            max_tasks: Maximum tasks per track.
            **kwargs: Additional configuration options.

        Returns:
            RunConfig for quick testing.
        """
        track_types = None
        if tracks:
            track_types = [TrackType(t) for t in tracks]

        return cls(
            name=f"quick_run_{datetime.now():%H%M%S}",
            tracks=track_types,
            max_tasks_per_track=max_tasks,
            batch_size=5,
            verbose=True,
            **kwargs,
        )

    @classmethod
    def full(
        cls,
        name: str | None = None,
        **kwargs: Any,
    ) -> "RunConfig":
        """Create a full benchmark configuration.

        Args:
            name: Name for the benchmark run.
            **kwargs: Additional configuration options.

        Returns:
            RunConfig for full benchmark.
        """
        return cls(
            name=name or f"full_benchmark_{datetime.now():%Y%m%d}",
            tracks=None,  # All tracks
            max_tasks_per_track=500,
            batch_size=20,
            max_concurrent=10,
            save_responses=True,
            save_intermediate=True,
            **kwargs,
        )
