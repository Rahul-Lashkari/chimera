"""Difficulty stratification for CHIMERA task generation.

This module provides utilities for stratifying tasks across
difficulty levels and ensuring balanced dataset generation.
"""

from dataclasses import dataclass, field
from typing import Any

from chimera.models.task import DifficultyLevel, Task


@dataclass
class StratificationConfig:
    """Configuration for difficulty stratification.

    Attributes:
        target_distribution: Target distribution across difficulty levels
        allow_rebalancing: Whether to allow rebalancing when targets can't be met
        min_per_level: Minimum tasks per difficulty level
        tolerance: Tolerance for distribution matching (0.0 to 1.0)
    """

    target_distribution: dict[DifficultyLevel, float] = field(
        default_factory=lambda: {
            DifficultyLevel.L1: 0.15,
            DifficultyLevel.L2: 0.25,
            DifficultyLevel.L3: 0.30,
            DifficultyLevel.L4: 0.20,
            DifficultyLevel.L5: 0.10,
        }
    )
    allow_rebalancing: bool = True
    min_per_level: int = 1
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        """Validate the configuration."""
        total = sum(self.target_distribution.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Target distribution must sum to 1.0, got {total}")


class DifficultyStratifier:
    """Stratifies tasks across difficulty levels.

    This class ensures that generated tasks are distributed according
    to a target difficulty distribution, with options for rebalancing
    and validation.
    """

    def __init__(self, config: StratificationConfig | None = None) -> None:
        """Initialize the stratifier.

        Args:
            config: Stratification configuration. Uses defaults if None.
        """
        self.config = config or StratificationConfig()

    def get_target_counts(self, total_tasks: int) -> dict[DifficultyLevel, int]:
        """Calculate target task counts per difficulty level.

        Args:
            total_tasks: Total number of tasks to distribute.

        Returns:
            Dictionary mapping difficulty levels to target counts.
        """
        counts: dict[DifficultyLevel, int] = {}
        remaining = total_tasks

        # Sort by proportion descending to handle rounding better
        sorted_levels = sorted(
            self.config.target_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for i, (level, proportion) in enumerate(sorted_levels):
            if i == len(sorted_levels) - 1:
                # Last level gets remaining tasks
                counts[level] = max(self.config.min_per_level, remaining)
            else:
                count = max(
                    self.config.min_per_level,
                    int(total_tasks * proportion),
                )
                counts[level] = count
                remaining -= count

        return counts

    def validate_distribution(
        self,
        tasks: list[Task],
    ) -> tuple[bool, dict[str, Any]]:
        """Validate that tasks match the target distribution.

        Args:
            tasks: List of tasks to validate.

        Returns:
            Tuple of (is_valid, details) where details contains
            the actual vs target distribution and any issues.
        """
        if not tasks:
            return False, {"error": "No tasks provided"}

        # Count tasks per level
        actual_counts: dict[DifficultyLevel, int] = dict.fromkeys(DifficultyLevel, 0)
        for task in tasks:
            if task.difficulty:
                actual_counts[task.difficulty] += 1

        total = len(tasks)
        actual_distribution = {level: count / total for level, count in actual_counts.items()}

        # Check against target
        issues = []
        for level, target_prop in self.config.target_distribution.items():
            actual_prop = actual_distribution.get(level, 0.0)
            diff = abs(actual_prop - target_prop)

            if diff > self.config.tolerance:
                issues.append(
                    {
                        "level": level.value,
                        "target": target_prop,
                        "actual": actual_prop,
                        "difference": diff,
                    }
                )

        is_valid = len(issues) == 0

        return is_valid, {
            "is_valid": is_valid,
            "total_tasks": total,
            "target_distribution": {k.value: v for k, v in self.config.target_distribution.items()},
            "actual_distribution": {k.value: v for k, v in actual_distribution.items()},
            "actual_counts": {k.value: v for k, v in actual_counts.items()},
            "issues": issues,
        }

    def rebalance(
        self,
        tasks: list[Task],
        rng: Any,
    ) -> list[Task]:
        """Rebalance tasks to match target distribution.

        This method samples from the provided tasks to create a
        balanced distribution. It may duplicate or remove tasks.

        Args:
            tasks: List of tasks to rebalance.
            rng: Random number generator.

        Returns:
            Rebalanced list of tasks.
        """
        if not self.config.allow_rebalancing:
            return tasks

        total = len(tasks)
        target_counts = self.get_target_counts(total)

        # Group tasks by difficulty
        by_difficulty: dict[DifficultyLevel, list[Task]] = {level: [] for level in DifficultyLevel}
        for task in tasks:
            if task.difficulty:
                by_difficulty[task.difficulty].append(task)

        # Rebalance
        result: list[Task] = []
        for level, target_count in target_counts.items():
            available = by_difficulty[level]

            if len(available) >= target_count:
                # Sample without replacement
                sampled = rng.sample(available, target_count)
            elif len(available) > 0:
                # Sample with replacement (duplication)
                sampled = rng.choices(available, k=target_count)
            else:
                # No tasks at this level - skip or borrow from adjacent
                continue

            result.extend(sampled)

        return result

    def suggest_adjustments(
        self,
        current_counts: dict[DifficultyLevel, int],
        target_total: int,
    ) -> dict[DifficultyLevel, int]:
        """Suggest adjustments to reach target distribution.

        Args:
            current_counts: Current task counts per level.
            target_total: Target total number of tasks.

        Returns:
            Dictionary mapping levels to adjustment amounts
            (positive = add, negative = remove).
        """
        target_counts = self.get_target_counts(target_total)

        adjustments = {}
        for level in DifficultyLevel:
            current = current_counts.get(level, 0)
            target = target_counts.get(level, 0)
            adjustments[level] = target - current

        return adjustments

    def get_difficulty_for_next_task(
        self,
        current_counts: dict[DifficultyLevel, int],
        target_total: int,
    ) -> DifficultyLevel:
        """Determine the best difficulty level for the next task.

        This method looks at current counts and target distribution
        to suggest which difficulty level needs more tasks.

        Args:
            current_counts: Current task counts per level.
            target_total: Target total number of tasks.

        Returns:
            Recommended difficulty level for next task.
        """
        target_counts = self.get_target_counts(target_total)

        # Find the level with the largest deficit
        max_deficit = -float("inf")
        best_level = DifficultyLevel.L3  # Default to medium

        for level in DifficultyLevel:
            current = current_counts.get(level, 0)
            target = target_counts.get(level, 0)
            deficit = target - current

            if deficit > max_deficit:
                max_deficit = deficit
                best_level = level

        return best_level


def create_balanced_sample(
    tasks: list[Task],
    n_samples: int,
    rng: Any,
    stratifier: DifficultyStratifier | None = None,
) -> list[Task]:
    """Create a balanced sample of tasks.

    This utility function creates a sample that maintains
    the target difficulty distribution.

    Args:
        tasks: Pool of tasks to sample from.
        n_samples: Number of tasks to sample.
        rng: Random number generator.
        stratifier: Optional stratifier to use. Uses default if None.

    Returns:
        Balanced sample of tasks.
    """
    if stratifier is None:
        stratifier = DifficultyStratifier()

    target_counts = stratifier.get_target_counts(n_samples)

    # Group by difficulty
    by_difficulty: dict[DifficultyLevel, list[Task]] = {level: [] for level in DifficultyLevel}
    for task in tasks:
        if task.difficulty:
            by_difficulty[task.difficulty].append(task)

    result: list[Task] = []
    for level, target_count in target_counts.items():
        available = by_difficulty[level]

        if len(available) == 0:
            continue
        elif len(available) >= target_count:
            sampled = rng.sample(available, target_count)
        else:
            # Not enough tasks - take all available
            sampled = available.copy()

        result.extend(sampled)

    # If we don't have enough, pad with random tasks
    while len(result) < n_samples and tasks:
        result.append(rng.choice(tasks))

    # Shuffle to mix difficulty levels
    rng.shuffle(result)

    return result[:n_samples]
