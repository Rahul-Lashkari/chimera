"""Base classes for CHIMERA task generators.

This module defines the abstract base class and configuration
for all task generators in the CHIMERA benchmark.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chimera.models.task import DifficultyLevel, Task, TaskCategory, TaskSet, TrackType


class GeneratorConfig(BaseModel):
    """Base configuration for task generators.

    Attributes:
        n_tasks: Total number of tasks to generate
        seed: Random seed for reproducibility
        difficulty_distribution: Distribution across difficulty levels
        category_distribution: Distribution across task categories
        output_dir: Directory to save generated tasks
        validate_tasks: Whether to validate generated tasks
    """

    model_config = ConfigDict(extra="allow")

    n_tasks: int = Field(
        default=100,
        ge=1,
        description="Total number of tasks to generate",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    difficulty_distribution: dict[str, float] = Field(
        default_factory=lambda: {
            "L1": 0.15,  # 15% easy
            "L2": 0.25,  # 25% medium-easy
            "L3": 0.30,  # 30% medium
            "L4": 0.20,  # 20% medium-hard
            "L5": 0.10,  # 10% hard
        },
        description="Distribution across difficulty levels (must sum to 1.0)",
    )
    category_distribution: dict[str, float] = Field(
        default_factory=lambda: {
            "factual": 0.30,
            "reasoning": 0.30,
            "numerical": 0.20,
            "commonsense": 0.10,
            "scientific": 0.10,
        },
        description="Distribution across task categories (must sum to 1.0)",
    )
    output_dir: Path | None = Field(
        default=None,
        description="Directory to save generated tasks",
    )
    validate_tasks: bool = Field(
        default=True,
        description="Whether to validate generated tasks",
    )

    def get_difficulty_counts(self) -> dict[DifficultyLevel, int]:
        """Calculate number of tasks per difficulty level.

        Returns:
            Dictionary mapping difficulty levels to task counts.
        """
        counts = {}
        remaining = self.n_tasks

        # Calculate counts, ensuring they sum to n_tasks
        sorted_levels = sorted(
            self.difficulty_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (level_str, proportion) in enumerate(sorted_levels):
            level = DifficultyLevel(level_str)
            if i == len(sorted_levels) - 1:
                # Last level gets remaining tasks
                counts[level] = remaining
            else:
                count = int(self.n_tasks * proportion)
                counts[level] = count
                remaining -= count

        return counts

    def get_category_counts(self) -> dict[TaskCategory, int]:
        """Calculate number of tasks per category.

        Returns:
            Dictionary mapping categories to task counts.
        """
        counts = {}
        remaining = self.n_tasks

        sorted_categories = sorted(
            self.category_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (cat_str, proportion) in enumerate(sorted_categories):
            category = TaskCategory(cat_str)
            if i == len(sorted_categories) - 1:
                counts[category] = remaining
            else:
                count = int(self.n_tasks * proportion)
                counts[category] = count
                remaining -= count

        return counts


class BaseTaskGenerator(ABC):
    """Abstract base class for all CHIMERA task generators.

    This class defines the interface that all task generators must implement.
    It provides common functionality for task generation, validation, and export.

    Attributes:
        config: Generator configuration
        track: The evaluation track this generator creates tasks for
        _rng: Random number generator for reproducibility
    """

    def __init__(self, config: GeneratorConfig, track: TrackType) -> None:
        """Initialize the generator.

        Args:
            config: Generator configuration.
            track: The evaluation track for generated tasks.
        """
        self.config = config
        self.track = track
        self._rng: Any = None
        self._setup_rng()

    def _setup_rng(self) -> None:
        """Set up the random number generator."""
        import random

        # Using standard random for reproducible task generation, not for security
        if self.config.seed is not None:
            self._rng = random.Random(self.config.seed)  # nosec B311
        else:
            self._rng = random.Random()  # nosec B311

    @abstractmethod
    def generate_task(
        self,
        difficulty: DifficultyLevel,
        category: TaskCategory,
    ) -> Task:
        """Generate a single task.

        Args:
            difficulty: The difficulty level for the task.
            category: The category for the task.

        Returns:
            A generated Task.
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        n_tasks: int,
        difficulty: DifficultyLevel | None = None,
        category: TaskCategory | None = None,
    ) -> list[Task]:
        """Generate a batch of tasks.

        Args:
            n_tasks: Number of tasks to generate.
            difficulty: Optional specific difficulty level.
            category: Optional specific category.

        Returns:
            List of generated Tasks.
        """
        pass

    def generate_all(self) -> TaskSet:
        """Generate all tasks according to configuration.

        Returns:
            TaskSet containing all generated tasks.
        """
        tasks: list[Task] = []

        difficulty_counts = self.config.get_difficulty_counts()
        # category_counts used for validation/logging in subclasses
        _ = self.config.get_category_counts()

        # Create a matrix of (difficulty, category) -> count
        # Distribute proportionally
        for difficulty, diff_count in difficulty_counts.items():
            for category, cat_proportion in self.config.category_distribution.items():
                cat_enum = TaskCategory(category)
                count = max(1, int(diff_count * cat_proportion))

                batch = self.generate_batch(
                    n_tasks=count,
                    difficulty=difficulty,
                    category=cat_enum,
                )
                tasks.extend(batch)

        # Trim to exact count if we generated too many
        if len(tasks) > self.config.n_tasks:
            self._rng.shuffle(tasks)
            tasks = tasks[: self.config.n_tasks]

        # Pad if we generated too few
        while len(tasks) < self.config.n_tasks:
            difficulty = self._rng.choice(list(difficulty_counts.keys()))
            category = self._rng.choice(list(TaskCategory))
            tasks.append(self.generate_task(difficulty, category))

        return TaskSet(
            name=f"{self.track.value}_generated",
            description=f"Generated {self.track.value} tasks",
            tasks=tasks,
            track=self.track,
        )

    def validate_task(self, task: Task) -> list[str]:
        """Validate a generated task.

        Args:
            task: The task to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check required fields
        if not task.question or not task.question.strip():
            errors.append("Task question is empty")

        if task.track != self.track:
            errors.append(f"Task track {task.track} does not match generator track {self.track}")

        # Track-specific validation
        errors.extend(self._validate_track_specific(task))

        return errors

    def _validate_track_specific(self, task: Task) -> list[str]:  # noqa: ARG002
        """Perform track-specific validation.

        Override in subclasses for specific validation logic.

        Args:
            task: The task to validate.

        Returns:
            List of validation error messages.
        """
        return []

    def save(self, tasks: TaskSet, output_path: Path | None = None) -> Path:
        """Save generated tasks to a JSONL file.

        Args:
            tasks: The TaskSet to save.
            output_path: Optional output path. Uses config.output_dir if not provided.

        Returns:
            Path to the saved file.
        """
        if output_path is None:
            if self.config.output_dir is None:
                raise ValueError("No output path specified and config.output_dir is None")
            output_path = self.config.output_dir / f"{self.track.value}_tasks.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_content = tasks.to_jsonl()
        output_path.write_text(jsonl_content, encoding="utf-8")

        return output_path

    def get_statistics(self, tasks: TaskSet) -> dict[str, Any]:
        """Get statistics about generated tasks.

        Args:
            tasks: The TaskSet to analyze.

        Returns:
            Dictionary of statistics.
        """
        difficulty_dist = tasks.get_difficulty_distribution()
        category_dist = tasks.get_category_distribution()

        return {
            "total_tasks": len(tasks),
            "track": self.track.value,
            "difficulty_distribution": difficulty_dist,
            "category_distribution": category_dist,
            "config": {
                "seed": self.config.seed,
                "target_n_tasks": self.config.n_tasks,
            },
        }
