"""Base classes and utilities for CHIMERA metrics.

This module provides the foundation for all calibration and evaluation metrics:
- BaseMetric: Abstract base class for all metrics
- MetricResult: Container for metric computation results
- CalibrationBin: Data structure for binned calibration data
- Utility functions for data validation and processing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator


class BinningStrategy(str, Enum):
    """Strategy for binning confidence values.

    Attributes:
        UNIFORM: Equal-width bins (standard ECE).
        ADAPTIVE: Equal-mass bins (adaptive calibration error).
        ISOTONIC: Isotonic regression-based binning.
    """

    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    ISOTONIC = "isotonic"


class MetricConfig(BaseModel):
    """Configuration for metric computation.

    Attributes:
        n_bins: Number of bins for calibration metrics (default: 10).
        binning_strategy: Strategy for creating bins (default: uniform).
        min_samples_per_bin: Minimum samples required per bin (default: 1).
        confidence_threshold: Threshold for binary confidence (default: 0.5).
    """

    n_bins: int = Field(default=10, ge=2, le=100)
    binning_strategy: BinningStrategy = Field(default=BinningStrategy.UNIFORM)
    min_samples_per_bin: int = Field(default=1, ge=1)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("n_bins")
    @classmethod
    def validate_n_bins(cls, v: int) -> int:
        """Validate number of bins."""
        if v < 2:
            raise ValueError("n_bins must be at least 2")
        return v


@dataclass
class CalibrationBin:
    """Data for a single calibration bin.

    Represents aggregated statistics for predictions falling within
    a specific confidence range.

    Attributes:
        bin_lower: Lower bound of the bin (inclusive).
        bin_upper: Upper bound of the bin (exclusive, except for last bin).
        bin_mid: Midpoint of the bin.
        count: Number of samples in the bin.
        accuracy: Mean accuracy of samples in the bin.
        confidence: Mean confidence of samples in the bin.
        gap: Absolute difference between confidence and accuracy.
        samples: Indices of samples in this bin (optional).
    """

    bin_lower: float
    bin_upper: float
    bin_mid: float
    count: int
    accuracy: float
    confidence: float
    gap: float
    samples: list[int] = field(default_factory=list)

    @property
    def is_overconfident(self) -> bool:
        """Check if bin shows overconfidence (confidence > accuracy)."""
        return self.confidence > self.accuracy

    @property
    def is_underconfident(self) -> bool:
        """Check if bin shows underconfidence (confidence < accuracy)."""
        return self.confidence < self.accuracy

    @property
    def weight(self) -> float:
        """Weight of this bin (proportion of total samples)."""
        # This is set externally when computing metrics
        return getattr(self, "_weight", 0.0)

    @weight.setter
    def weight(self, value: float) -> None:
        """Set the weight of this bin."""
        self._weight = value


@dataclass
class MetricResult:
    """Container for metric computation results.

    Provides a standardized format for returning metric values along
    with supporting information for analysis and visualization.

    Attributes:
        name: Name of the metric.
        value: Primary metric value.
        bins: List of calibration bins (for binned metrics).
        metadata: Additional metric-specific information.
        confidence_interval: Optional (lower, upper) confidence bounds.
    """

    name: str
    value: float
    bins: list[CalibrationBin] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None

    def __repr__(self) -> str:
        """String representation."""
        if self.confidence_interval:
            return (
                f"{self.name}: {self.value:.4f} "
                f"[{self.confidence_interval[0]:.4f}, "
                f"{self.confidence_interval[1]:.4f}]"
            )
        return f"{self.name}: {self.value:.4f}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata,
        }
        if self.confidence_interval:
            result["confidence_interval"] = {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1],
            }
        if self.bins:
            result["bins"] = [
                {
                    "lower": b.bin_lower,
                    "upper": b.bin_upper,
                    "count": b.count,
                    "accuracy": b.accuracy,
                    "confidence": b.confidence,
                    "gap": b.gap,
                }
                for b in self.bins
            ]
        return result


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    Provides a consistent interface for computing metrics from
    confidence and accuracy data.

    Subclasses must implement:
        - compute(): Main computation method
        - name: Property returning the metric name
    """

    def __init__(self, config: MetricConfig | None = None) -> None:
        """Initialize metric with configuration.

        Args:
            config: Metric configuration. Uses defaults if not provided.
        """
        self.config = config or MetricConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        ...

    @abstractmethod
    def compute(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> MetricResult:
        """Compute the metric from confidence and accuracy data.

        Args:
            confidences: Array of confidence values in [0, 1].
            accuracies: Array of accuracy values (0 or 1 for binary).

        Returns:
            MetricResult containing the computed value and metadata.
        """
        ...

    def validate_inputs(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Validate and preprocess input arrays.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values.

        Returns:
            Tuple of validated (confidences, accuracies) arrays.

        Raises:
            ValueError: If inputs are invalid.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        accuracies = np.asarray(accuracies, dtype=np.float64)

        if confidences.shape != accuracies.shape:
            raise ValueError(
                f"Shape mismatch: confidences {confidences.shape} "
                f"vs accuracies {accuracies.shape}"
            )

        if len(confidences) == 0:
            raise ValueError("Empty input arrays")

        if np.any(confidences < 0) or np.any(confidences > 1):
            raise ValueError("Confidence values must be in [0, 1]")

        # Allow soft accuracies in [0, 1]
        if not np.all(np.isin(accuracies, [0, 1])) and (
            np.any(accuracies < 0) or np.any(accuracies > 1)
        ):
            raise ValueError("Accuracy values must be in [0, 1]")

        return confidences, accuracies


def create_uniform_bins(
    n_bins: int,
) -> list[tuple[float, float]]:
    """Create uniform-width bin boundaries.

    Args:
        n_bins: Number of bins.

    Returns:
        List of (lower, upper) tuples for each bin.
    """
    boundaries = np.linspace(0, 1, n_bins + 1)
    return [(boundaries[i], boundaries[i + 1]) for i in range(n_bins)]


def create_adaptive_bins(
    confidences: NDArray[np.floating[Any]],
    n_bins: int,
) -> list[tuple[float, float]]:
    """Create adaptive (equal-mass) bin boundaries.

    Each bin contains approximately the same number of samples.

    Args:
        confidences: Confidence values to bin.
        n_bins: Number of bins.

    Returns:
        List of (lower, upper) tuples for each bin.
    """
    sorted_conf = np.sort(confidences)
    n = len(sorted_conf)

    # Calculate quantile boundaries
    boundaries = [0.0]
    for i in range(1, n_bins):
        idx = int(i * n / n_bins)
        # Use midpoint between adjacent values to avoid boundary issues
        if idx < n:
            boundaries.append(float(sorted_conf[idx]))
    boundaries.append(1.0)

    # Remove duplicates while preserving order
    unique_boundaries: list[float] = []
    for b in boundaries:
        if not unique_boundaries or b > unique_boundaries[-1]:
            unique_boundaries.append(b)

    # Create bins from unique boundaries
    return [
        (unique_boundaries[i], unique_boundaries[i + 1]) for i in range(len(unique_boundaries) - 1)
    ]


def bin_predictions(
    confidences: NDArray[np.floating[Any]],
    accuracies: NDArray[np.floating[Any]],
    bins: list[tuple[float, float]],
    min_samples: int = 1,
) -> list[CalibrationBin]:
    """Bin predictions by confidence value.

    Args:
        confidences: Confidence values.
        accuracies: Accuracy values.
        bins: List of (lower, upper) bin boundaries.
        min_samples: Minimum samples required per bin.

    Returns:
        List of CalibrationBin objects.
    """
    result = []
    n_total = len(confidences)

    for i, (lower, upper) in enumerate(bins):
        # Include upper boundary for last bin
        if i == len(bins) - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        indices = np.where(mask)[0].tolist()
        count = len(indices)

        if count >= min_samples:
            bin_conf = float(np.mean(confidences[mask]))
            bin_acc = float(np.mean(accuracies[mask]))
            gap = abs(bin_conf - bin_acc)
        else:
            bin_conf = (lower + upper) / 2
            bin_acc = 0.0
            gap = 0.0

        calibration_bin = CalibrationBin(
            bin_lower=lower,
            bin_upper=upper,
            bin_mid=(lower + upper) / 2,
            count=count,
            accuracy=bin_acc,
            confidence=bin_conf,
            gap=gap,
            samples=indices,
        )

        # Set weight
        calibration_bin.weight = count / n_total if n_total > 0 else 0.0

        result.append(calibration_bin)

    return result


def bootstrap_confidence_interval(
    confidences: NDArray[np.floating[Any]],
    accuracies: NDArray[np.floating[Any]],
    metric_fn: Any,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        confidences: Confidence values.
        accuracies: Accuracy values.
        metric_fn: Function that computes the metric value.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower, upper) confidence bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(confidences)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        value = metric_fn(confidences[indices], accuracies[indices])
        bootstrap_values.append(value)

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

    return (lower, upper)
