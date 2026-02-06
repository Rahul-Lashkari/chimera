"""Calibration metrics for CHIMERA benchmark.

This module implements core calibration metrics for evaluating how well
a model's stated confidence predicts its actual accuracy.

Metrics:
    - ECE (Expected Calibration Error): Primary calibration metric
    - MCE (Maximum Calibration Error): Worst-case calibration
    - ACE (Adaptive Calibration Error): Equal-mass binning variant
    - Brier Score: Proper scoring rule for probabilistic predictions
    - Overconfidence metrics: Analysis of systematic bias

References:
    - Guo et al. "On Calibration of Modern Neural Networks" (2017)
    - Naeini et al. "Obtaining Well Calibrated Probabilities Using
      Bayesian Binning into Quantiles" (2015)
    - Nixon et al. "Measuring Calibration in Deep Learning" (2019)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chimera.metrics.base import (
    BaseMetric,
    BinningStrategy,
    CalibrationBin,
    MetricConfig,
    MetricResult,
    bin_predictions,
    bootstrap_confidence_interval,
    create_adaptive_bins,
    create_uniform_bins,
)


class ExpectedCalibrationError(BaseMetric):
    """Expected Calibration Error (ECE) metric.

    ECE measures the weighted average absolute difference between
    confidence and accuracy across confidence bins.

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    where B_m is the set of samples in bin m, acc is accuracy,
    and conf is mean confidence.

    Lower ECE indicates better calibration (0 = perfect calibration).

    Example:
        >>> ece = ExpectedCalibrationError()
        >>> result = ece.compute(confidences, accuracies)
        >>> print(f"ECE: {result.value:.4f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "ECE"

    def compute(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
        compute_ci: bool = False,
    ) -> MetricResult:
        """Compute Expected Calibration Error.

        Args:
            confidences: Array of confidence values in [0, 1].
            accuracies: Array of accuracy values (0 or 1).
            compute_ci: Whether to compute bootstrap confidence interval.

        Returns:
            MetricResult with ECE value and calibration bins.
        """
        confidences, accuracies = self.validate_inputs(confidences, accuracies)

        # Create bins based on strategy
        if self.config.binning_strategy == BinningStrategy.ADAPTIVE:
            bin_boundaries = create_adaptive_bins(confidences, self.config.n_bins)
        else:
            bin_boundaries = create_uniform_bins(self.config.n_bins)

        # Bin predictions
        bins = bin_predictions(
            confidences,
            accuracies,
            bin_boundaries,
            self.config.min_samples_per_bin,
        )

        # Compute ECE
        ece = self._compute_ece(bins)

        # Compute confidence interval if requested
        ci = None
        if compute_ci:
            ci = bootstrap_confidence_interval(
                confidences,
                accuracies,
                lambda c, a: self._compute_ece_from_arrays(c, a),
            )

        return MetricResult(
            name=self.name,
            value=ece,
            bins=bins,
            metadata={
                "n_bins": len(bins),
                "n_samples": len(confidences),
                "binning_strategy": self.config.binning_strategy.value,
            },
            confidence_interval=ci,
        )

    def _compute_ece(self, bins: list[CalibrationBin]) -> float:
        """Compute ECE from calibration bins."""
        total_samples = sum(b.count for b in bins)
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for bin_data in bins:
            if bin_data.count > 0:
                weight = bin_data.count / total_samples
                ece += weight * bin_data.gap

        return float(ece)

    def _compute_ece_from_arrays(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> float:
        """Compute ECE directly from arrays (for bootstrapping)."""
        bin_boundaries = create_uniform_bins(self.config.n_bins)
        bins = bin_predictions(confidences, accuracies, bin_boundaries, 1)
        return self._compute_ece(bins)


class MaximumCalibrationError(BaseMetric):
    """Maximum Calibration Error (MCE) metric.

    MCE measures the maximum absolute difference between confidence
    and accuracy across all bins - the worst-case calibration error.

    MCE = max_m |acc(B_m) - conf(B_m)|

    Useful for safety-critical applications where worst-case
    calibration matters.

    Example:
        >>> mce = MaximumCalibrationError()
        >>> result = mce.compute(confidences, accuracies)
        >>> print(f"MCE: {result.value:.4f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "MCE"

    def compute(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> MetricResult:
        """Compute Maximum Calibration Error.

        Args:
            confidences: Array of confidence values in [0, 1].
            accuracies: Array of accuracy values (0 or 1).

        Returns:
            MetricResult with MCE value and worst bin info.
        """
        confidences, accuracies = self.validate_inputs(confidences, accuracies)

        # Create bins
        if self.config.binning_strategy == BinningStrategy.ADAPTIVE:
            bin_boundaries = create_adaptive_bins(confidences, self.config.n_bins)
        else:
            bin_boundaries = create_uniform_bins(self.config.n_bins)

        bins = bin_predictions(
            confidences,
            accuracies,
            bin_boundaries,
            self.config.min_samples_per_bin,
        )

        # Find maximum gap
        mce = 0.0
        worst_bin_idx = 0

        for i, bin_data in enumerate(bins):
            if bin_data.count >= self.config.min_samples_per_bin and bin_data.gap > mce:
                mce = bin_data.gap
                worst_bin_idx = i

        return MetricResult(
            name=self.name,
            value=mce,
            bins=bins,
            metadata={
                "worst_bin_idx": worst_bin_idx,
                "worst_bin_range": (
                    (
                        bins[worst_bin_idx].bin_lower,
                        bins[worst_bin_idx].bin_upper,
                    )
                    if bins
                    else (0, 0)
                ),
                "n_samples": len(confidences),
            },
        )


class AdaptiveCalibrationError(BaseMetric):
    """Adaptive Calibration Error (ACE) metric.

    ACE uses equal-mass (adaptive) binning instead of equal-width bins,
    ensuring each bin has approximately the same number of samples.

    This addresses the issue of sparsely populated bins in standard ECE.

    Reference:
        Nixon et al. "Measuring Calibration in Deep Learning" (2019)

    Example:
        >>> ace = AdaptiveCalibrationError(n_bins=15)
        >>> result = ace.compute(confidences, accuracies)
        >>> print(f"ACE: {result.value:.4f}")
    """

    def __init__(
        self,
        config: MetricConfig | None = None,
        n_bins: int = 15,
    ) -> None:
        """Initialize ACE metric.

        Args:
            config: Metric configuration.
            n_bins: Number of bins (default: 15).
        """
        if config is None:
            config = MetricConfig(
                n_bins=n_bins,
                binning_strategy=BinningStrategy.ADAPTIVE,
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "ACE"

    def compute(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> MetricResult:
        """Compute Adaptive Calibration Error.

        Args:
            confidences: Array of confidence values in [0, 1].
            accuracies: Array of accuracy values (0 or 1).

        Returns:
            MetricResult with ACE value and calibration bins.
        """
        confidences, accuracies = self.validate_inputs(confidences, accuracies)

        # Create adaptive bins
        bin_boundaries = create_adaptive_bins(confidences, self.config.n_bins)

        bins = bin_predictions(
            confidences,
            accuracies,
            bin_boundaries,
            self.config.min_samples_per_bin,
        )

        # Compute ACE (same formula as ECE, different binning)
        total_samples = sum(b.count for b in bins)
        ace = 0.0

        for bin_data in bins:
            if bin_data.count > 0:
                weight = bin_data.count / total_samples
                ace += weight * bin_data.gap

        return MetricResult(
            name=self.name,
            value=ace,
            bins=bins,
            metadata={
                "n_bins": len(bins),
                "n_samples": len(confidences),
                "samples_per_bin": [b.count for b in bins],
            },
        )


class BrierScore(BaseMetric):
    """Brier Score metric.

    The Brier score is a proper scoring rule that measures the mean
    squared error between predicted probabilities and outcomes.

    Brier = (1/n) Σ (confidence_i - accuracy_i)²

    Properties:
        - Proper scoring rule (optimized by true probabilities)
        - Range: [0, 1] where 0 is perfect
        - Can be decomposed into reliability + resolution - uncertainty

    Example:
        >>> brier = BrierScore()
        >>> result = brier.compute(confidences, accuracies)
        >>> print(f"Brier: {result.value:.4f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "Brier"

    def compute(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
        decompose: bool = True,
    ) -> MetricResult:
        """Compute Brier Score.

        Args:
            confidences: Array of confidence values in [0, 1].
            accuracies: Array of accuracy values (0 or 1).
            decompose: Whether to compute decomposition.

        Returns:
            MetricResult with Brier score and optional decomposition.
        """
        confidences, accuracies = self.validate_inputs(confidences, accuracies)

        # Compute Brier score
        brier = float(np.mean((confidences - accuracies) ** 2))

        metadata: dict[str, Any] = {
            "n_samples": len(confidences),
        }

        # Compute decomposition if requested
        if decompose:
            decomposition = self._compute_decomposition(confidences, accuracies)
            metadata.update(decomposition)

        return MetricResult(
            name=self.name,
            value=brier,
            metadata=metadata,
        )

    def _compute_decomposition(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> dict[str, float]:
        """Compute Brier score decomposition.

        Decomposes into:
            - Reliability: Calibration component
            - Resolution: Discriminative ability
            - Uncertainty: Inherent uncertainty in outcomes

        Brier = Reliability - Resolution + Uncertainty
        """
        # Base rate (overall accuracy)
        base_rate = float(np.mean(accuracies))

        # Uncertainty component
        uncertainty = base_rate * (1 - base_rate)

        # Bin predictions for decomposition
        bin_boundaries = create_uniform_bins(self.config.n_bins)
        bins = bin_predictions(confidences, accuracies, bin_boundaries, 1)

        n_total = len(confidences)
        reliability = 0.0
        resolution = 0.0

        for bin_data in bins:
            if bin_data.count > 0:
                weight = bin_data.count / n_total

                # Reliability: weighted average of (conf - acc)²
                reliability += weight * (bin_data.confidence - bin_data.accuracy) ** 2

                # Resolution: weighted average of (acc - base_rate)²
                resolution += weight * (bin_data.accuracy - base_rate) ** 2

        return {
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "base_rate": base_rate,
        }


@dataclass
class OverconfidenceMetrics:
    """Analysis of systematic over/underconfidence.

    Provides detailed metrics for understanding directional
    calibration errors.

    Attributes:
        overconfidence_ratio: Proportion of samples where conf > acc.
        underconfidence_ratio: Proportion of samples where conf < acc.
        mean_overconfidence: Mean (conf - acc) for overconfident predictions.
        mean_underconfidence: Mean (acc - conf) for underconfident predictions.
        overconfidence_ece: ECE contribution from overconfident bins.
        underconfidence_ece: ECE contribution from underconfident bins.
    """

    overconfidence_ratio: float = 0.0
    underconfidence_ratio: float = 0.0
    mean_overconfidence: float = 0.0
    mean_underconfidence: float = 0.0
    overconfidence_ece: float = 0.0
    underconfidence_ece: float = 0.0
    bins: list[CalibrationBin] = field(default_factory=list)

    @classmethod
    def compute(
        cls,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
        n_bins: int = 10,
    ) -> "OverconfidenceMetrics":
        """Compute overconfidence metrics.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values.
            n_bins: Number of bins for analysis.

        Returns:
            OverconfidenceMetrics instance.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        accuracies = np.asarray(accuracies, dtype=np.float64)

        n_samples = len(confidences)

        # Sample-level analysis
        overconf_mask = confidences > accuracies
        underconf_mask = confidences < accuracies

        n_overconf = np.sum(overconf_mask)
        n_underconf = np.sum(underconf_mask)

        overconf_ratio = float(n_overconf / n_samples) if n_samples > 0 else 0.0
        underconf_ratio = float(n_underconf / n_samples) if n_samples > 0 else 0.0

        # Mean over/underconfidence
        mean_overconf = 0.0
        mean_underconf = 0.0

        if n_overconf > 0:
            mean_overconf = float(np.mean(confidences[overconf_mask] - accuracies[overconf_mask]))

        if n_underconf > 0:
            mean_underconf = float(
                np.mean(accuracies[underconf_mask] - confidences[underconf_mask])
            )

        # Bin-level analysis
        bin_boundaries = create_uniform_bins(n_bins)
        bins = bin_predictions(confidences, accuracies, bin_boundaries, 1)

        overconf_ece = 0.0
        underconf_ece = 0.0

        for bin_data in bins:
            if bin_data.count > 0:
                weight = bin_data.count / n_samples
                if bin_data.is_overconfident:
                    overconf_ece += weight * bin_data.gap
                elif bin_data.is_underconfident:
                    underconf_ece += weight * bin_data.gap

        return cls(
            overconfidence_ratio=overconf_ratio,
            underconfidence_ratio=underconf_ratio,
            mean_overconfidence=mean_overconf,
            mean_underconfidence=mean_underconf,
            overconfidence_ece=overconf_ece,
            underconfidence_ece=underconf_ece,
            bins=bins,
        )

    @property
    def calibration_bias(self) -> str:
        """Determine overall calibration bias direction."""
        if self.overconfidence_ece > self.underconfidence_ece * 1.2:
            return "overconfident"
        elif self.underconfidence_ece > self.overconfidence_ece * 1.2:
            return "underconfident"
        return "balanced"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overconfidence_ratio": self.overconfidence_ratio,
            "underconfidence_ratio": self.underconfidence_ratio,
            "mean_overconfidence": self.mean_overconfidence,
            "mean_underconfidence": self.mean_underconfidence,
            "overconfidence_ece": self.overconfidence_ece,
            "underconfidence_ece": self.underconfidence_ece,
            "calibration_bias": self.calibration_bias,
        }


@dataclass
class CalibrationSummary:
    """Complete calibration analysis summary.

    Aggregates all calibration metrics for comprehensive reporting.
    """

    ece: MetricResult
    mce: MetricResult
    ace: MetricResult
    brier: MetricResult
    overconfidence: OverconfidenceMetrics
    n_samples: int = 0
    accuracy: float = 0.0
    mean_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ece": self.ece.value,
            "mce": self.mce.value,
            "ace": self.ace.value,
            "brier": self.brier.value,
            "overconfidence": self.overconfidence.to_dict(),
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CalibrationSummary(\n"
            f"  ECE={self.ece.value:.4f}, MCE={self.mce.value:.4f}, "
            f"ACE={self.ace.value:.4f}, Brier={self.brier.value:.4f}\n"
            f"  Accuracy={self.accuracy:.2%}, "
            f"Mean Confidence={self.mean_confidence:.2%}\n"
            f"  Bias: {self.overconfidence.calibration_bias}\n"
            f")"
        )


class CalibrationMetricsComputer:
    """Comprehensive calibration metrics computation.

    Computes all standard calibration metrics in a single pass
    for efficiency.

    Example:
        >>> computer = CalibrationMetricsComputer(n_bins=10)
        >>> summary = computer.compute_all(confidences, accuracies)
        >>> print(summary)
        >>> print(f"ECE: {summary.ece.value:.4f}")
    """

    def __init__(
        self,
        n_bins: int = 10,
        compute_ci: bool = False,
    ) -> None:
        """Initialize calibration metrics computer.

        Args:
            n_bins: Number of bins for binned metrics.
            compute_ci: Whether to compute confidence intervals.
        """
        self.n_bins = n_bins
        self.compute_ci = compute_ci

        # Initialize individual metrics
        config = MetricConfig(n_bins=n_bins)
        self.ece_metric = ExpectedCalibrationError(config)
        self.mce_metric = MaximumCalibrationError(config)
        self.ace_metric = AdaptiveCalibrationError(n_bins=n_bins)
        self.brier_metric = BrierScore(config)

    def compute(
        self,
        confidences: NDArray[np.floating[Any]] | list[float],
        accuracies: NDArray[np.floating[Any]] | list[float],
    ) -> MetricResult:
        """Compute primary ECE metric.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values.

        Returns:
            ECE MetricResult.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        accuracies = np.asarray(accuracies, dtype=np.float64)
        return self.ece_metric.compute(confidences, accuracies, compute_ci=self.compute_ci)

    def compute_all(
        self,
        confidences: NDArray[np.floating[Any]] | list[float],
        accuracies: NDArray[np.floating[Any]] | list[float],
    ) -> CalibrationSummary:
        """Compute all calibration metrics.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values.

        Returns:
            CalibrationSummary with all metrics.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        accuracies = np.asarray(accuracies, dtype=np.float64)

        # Compute all metrics
        ece = self.ece_metric.compute(confidences, accuracies, compute_ci=self.compute_ci)
        mce = self.mce_metric.compute(confidences, accuracies)
        ace = self.ace_metric.compute(confidences, accuracies)
        brier = self.brier_metric.compute(confidences, accuracies)
        overconfidence = OverconfidenceMetrics.compute(confidences, accuracies, self.n_bins)

        return CalibrationSummary(
            ece=ece,
            mce=mce,
            ace=ace,
            brier=brier,
            overconfidence=overconfidence,
            n_samples=len(confidences),
            accuracy=float(np.mean(accuracies)),
            mean_confidence=float(np.mean(confidences)),
        )
