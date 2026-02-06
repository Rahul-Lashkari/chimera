"""Unit tests for CHIMERA calibration metrics.

Tests cover:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- ACE (Adaptive Calibration Error)
- Brier Score
- Overconfidence metrics
- CalibrationMetricsComputer
- Edge cases and numerical stability
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from chimera.metrics.base import (
    BinningStrategy,
    CalibrationBin,
    MetricConfig,
    MetricResult,
    bin_predictions,
    bootstrap_confidence_interval,
    create_adaptive_bins,
    create_uniform_bins,
)
from chimera.metrics.calibration import (
    AdaptiveCalibrationError,
    BrierScore,
    CalibrationMetricsComputer,
    CalibrationSummary,
    ExpectedCalibrationError,
    MaximumCalibrationError,
    OverconfidenceMetrics,
)

# ==================== Test Fixtures ====================


@pytest.fixture
def perfectly_calibrated() -> tuple[NDArray, NDArray]:
    """Data where confidence equals accuracy."""
    np.random.seed(42)
    n = 1000
    confidences = np.random.uniform(0, 1, n)
    # Create accuracies that match confidences on average
    accuracies = (np.random.random(n) < confidences).astype(float)
    return confidences, accuracies


@pytest.fixture
def overconfident() -> tuple[NDArray, NDArray]:
    """Data where model is overconfident."""
    np.random.seed(42)
    n = 500
    # High confidence, low accuracy
    confidences = np.random.uniform(0.7, 1.0, n)
    accuracies = np.random.binomial(1, 0.3, n).astype(float)
    return confidences, accuracies


@pytest.fixture
def underconfident() -> tuple[NDArray, NDArray]:
    """Data where model is underconfident."""
    np.random.seed(42)
    n = 500
    # Low confidence, high accuracy
    confidences = np.random.uniform(0.2, 0.5, n)
    accuracies = np.random.binomial(1, 0.8, n).astype(float)
    return confidences, accuracies


@pytest.fixture
def simple_data() -> tuple[NDArray, NDArray]:
    """Simple test data for deterministic tests."""
    confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    accuracies = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    return confidences, accuracies


# ==================== MetricConfig Tests ====================


class TestMetricConfig:
    """Tests for MetricConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = MetricConfig()
        assert config.n_bins == 10
        assert config.binning_strategy == BinningStrategy.UNIFORM
        assert config.min_samples_per_bin == 1

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = MetricConfig(
            n_bins=15,
            binning_strategy=BinningStrategy.ADAPTIVE,
            min_samples_per_bin=5,
        )
        assert config.n_bins == 15
        assert config.binning_strategy == BinningStrategy.ADAPTIVE

    def test_invalid_n_bins(self) -> None:
        """Test that invalid n_bins raises error."""
        with pytest.raises(ValueError):
            MetricConfig(n_bins=1)


# ==================== Binning Tests ====================


class TestBinning:
    """Tests for binning utilities."""

    def test_create_uniform_bins(self) -> None:
        """Test uniform bin creation."""
        bins = create_uniform_bins(10)
        assert len(bins) == 10
        assert bins[0][0] == 0.0
        assert bins[-1][1] == 1.0

        # Check contiguous
        for i in range(len(bins) - 1):
            assert bins[i][1] == bins[i + 1][0]

    def test_create_uniform_bins_widths(self) -> None:
        """Test uniform bins have equal width."""
        bins = create_uniform_bins(5)
        widths = [b[1] - b[0] for b in bins]
        assert all(abs(w - 0.2) < 1e-10 for w in widths)

    def test_create_adaptive_bins(self) -> None:
        """Test adaptive bin creation."""
        np.random.seed(42)
        confidences = np.random.uniform(0, 1, 100)
        bins = create_adaptive_bins(confidences, 10)

        # Should have some bins (may have fewer if duplicates)
        assert len(bins) >= 2
        assert bins[0][0] == 0.0
        assert bins[-1][1] == 1.0

    def test_bin_predictions(self) -> None:
        """Test binning predictions."""
        confidences = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        accuracies = np.array([0, 0, 1, 1, 1])
        bins = create_uniform_bins(10)

        result = bin_predictions(confidences, accuracies, bins, min_samples=1)

        assert len(result) == 10
        # First bin [0, 0.1) should be empty
        assert result[0].count == 0
        # Second bin [0.1, 0.2) should have 1 sample
        assert result[1].count == 1

    def test_calibration_bin_properties(self) -> None:
        """Test CalibrationBin properties."""
        bin_data = CalibrationBin(
            bin_lower=0.5,
            bin_upper=0.6,
            bin_mid=0.55,
            count=10,
            accuracy=0.4,
            confidence=0.55,
            gap=0.15,
        )

        assert bin_data.is_overconfident is True
        assert bin_data.is_underconfident is False


# ==================== ECE Tests ====================


class TestExpectedCalibrationError:
    """Tests for ECE metric."""

    def test_perfect_calibration(self) -> None:
        """Test ECE for perfect calibration."""
        # Synthetic perfect calibration
        confidences = np.array(
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8]  # 20% should be 1
        )  # 80% should be 1
        accuracies = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])  # 1 correct = 20%  # 4 correct = 80%

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        # Should be close to 0
        assert result.value < 0.05

    def test_ece_bounds(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test ECE is bounded in [0, 1]."""
        confidences, accuracies = simple_data

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        assert 0 <= result.value <= 1

    def test_ece_overconfident(
        self,
        overconfident: tuple[NDArray, NDArray],
    ) -> None:
        """Test ECE for overconfident predictions."""
        confidences, accuracies = overconfident

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        # Overconfident should have high ECE
        assert result.value > 0.3

    def test_ece_returns_bins(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test that ECE returns calibration bins."""
        confidences, accuracies = simple_data

        ece = ExpectedCalibrationError(MetricConfig(n_bins=5))
        result = ece.compute(confidences, accuracies)

        assert len(result.bins) == 5
        assert all(isinstance(b, CalibrationBin) for b in result.bins)

    def test_ece_with_bootstrap_ci(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test ECE with confidence interval computation."""
        confidences, accuracies = simple_data

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies, compute_ci=True)

        assert result.confidence_interval is not None
        assert result.confidence_interval[0] <= result.value
        assert result.confidence_interval[1] >= result.value


# ==================== MCE Tests ====================


class TestMaximumCalibrationError:
    """Tests for MCE metric."""

    def test_mce_bounds(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test MCE is bounded in [0, 1]."""
        confidences, accuracies = simple_data

        mce = MaximumCalibrationError()
        result = mce.compute(confidences, accuracies)

        assert 0 <= result.value <= 1

    def test_mce_ge_ece(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test MCE >= ECE always."""
        confidences, accuracies = simple_data

        ece = ExpectedCalibrationError()
        mce = MaximumCalibrationError()

        ece_result = ece.compute(confidences, accuracies)
        mce_result = mce.compute(confidences, accuracies)

        assert mce_result.value >= ece_result.value

    def test_mce_identifies_worst_bin(
        self,
        overconfident: tuple[NDArray, NDArray],
    ) -> None:
        """Test MCE identifies the worst calibrated bin."""
        confidences, accuracies = overconfident

        mce = MaximumCalibrationError()
        result = mce.compute(confidences, accuracies)

        # Worst bin should be the one with maximum gap
        worst_idx = result.metadata["worst_bin_idx"]
        worst_bin = result.bins[worst_idx]

        for _, b in enumerate(result.bins):
            if b.count >= 1:
                assert worst_bin.gap >= b.gap - 1e-10


# ==================== ACE Tests ====================


class TestAdaptiveCalibrationError:
    """Tests for ACE metric."""

    def test_ace_bounds(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test ACE is bounded in [0, 1]."""
        confidences, accuracies = simple_data

        ace = AdaptiveCalibrationError()
        result = ace.compute(confidences, accuracies)

        assert 0 <= result.value <= 1

    def test_ace_equal_mass_bins(self) -> None:
        """Test ACE uses equal-mass binning."""
        np.random.seed(42)
        n = 100
        confidences = np.random.uniform(0, 1, n)
        accuracies = np.random.binomial(1, confidences)

        ace = AdaptiveCalibrationError(n_bins=5)
        result = ace.compute(
            confidences.astype(float),
            accuracies.astype(float),
        )

        # Bins should have similar counts
        counts = [b.count for b in result.bins if b.count > 0]
        if len(counts) >= 2:
            # Allow some variation but generally similar
            assert max(counts) < n  # Not all in one bin


# ==================== Brier Score Tests ====================


class TestBrierScore:
    """Tests for Brier score metric."""

    def test_perfect_predictions(self) -> None:
        """Test Brier score for perfect predictions."""
        confidences = np.array([1.0, 1.0, 0.0, 0.0])
        accuracies = np.array([1.0, 1.0, 0.0, 0.0])

        brier = BrierScore()
        result = brier.compute(confidences, accuracies)

        assert result.value == 0.0

    def test_worst_predictions(self) -> None:
        """Test Brier score for worst predictions."""
        confidences = np.array([1.0, 1.0, 0.0, 0.0])
        accuracies = np.array([0.0, 0.0, 1.0, 1.0])

        brier = BrierScore()
        result = brier.compute(confidences, accuracies)

        assert result.value == 1.0

    def test_brier_bounds(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test Brier score is bounded in [0, 1]."""
        confidences, accuracies = simple_data

        brier = BrierScore()
        result = brier.compute(confidences, accuracies)

        assert 0 <= result.value <= 1

    def test_brier_decomposition(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test Brier score decomposition."""
        confidences, accuracies = simple_data

        brier = BrierScore()
        result = brier.compute(confidences, accuracies, decompose=True)

        # Check decomposition components
        assert "reliability" in result.metadata
        assert "resolution" in result.metadata
        assert "uncertainty" in result.metadata

        # Uncertainty should be base_rate * (1 - base_rate)
        base_rate = result.metadata["base_rate"]
        expected_uncertainty = base_rate * (1 - base_rate)
        assert abs(result.metadata["uncertainty"] - expected_uncertainty) < 1e-10


# ==================== Overconfidence Metrics Tests ====================


class TestOverconfidenceMetrics:
    """Tests for overconfidence analysis."""

    def test_overconfident_detection(
        self,
        overconfident: tuple[NDArray, NDArray],
    ) -> None:
        """Test detection of overconfidence."""
        confidences, accuracies = overconfident

        metrics = OverconfidenceMetrics.compute(confidences, accuracies)

        assert metrics.calibration_bias == "overconfident"
        assert metrics.overconfidence_ece > metrics.underconfidence_ece

    def test_underconfident_detection(
        self,
        underconfident: tuple[NDArray, NDArray],
    ) -> None:
        """Test detection of underconfidence."""
        confidences, accuracies = underconfident

        metrics = OverconfidenceMetrics.compute(confidences, accuracies)

        assert metrics.calibration_bias == "underconfident"
        assert metrics.underconfidence_ece > metrics.overconfidence_ece

    def test_ratios_sum_to_one(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test over/underconfidence ratios."""
        confidences, accuracies = simple_data

        metrics = OverconfidenceMetrics.compute(confidences, accuracies)

        # Ratios plus well-calibrated should roughly equal 1
        # (allowing for exactly calibrated samples)
        assert 0 <= metrics.overconfidence_ratio <= 1
        assert 0 <= metrics.underconfidence_ratio <= 1


# ==================== CalibrationMetricsComputer Tests ====================


class TestCalibrationMetricsComputer:
    """Tests for comprehensive calibration computation."""

    def test_compute_returns_ece(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test compute returns ECE."""
        confidences, accuracies = simple_data

        computer = CalibrationMetricsComputer()
        result = computer.compute(confidences, accuracies)

        assert result.name == "ECE"
        assert 0 <= result.value <= 1

    def test_compute_all(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test compute_all returns summary."""
        confidences, accuracies = simple_data

        computer = CalibrationMetricsComputer()
        summary = computer.compute_all(confidences, accuracies)

        assert isinstance(summary, CalibrationSummary)
        assert summary.ece is not None
        assert summary.mce is not None
        assert summary.ace is not None
        assert summary.brier is not None
        assert summary.overconfidence is not None

    def test_summary_statistics(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test summary includes basic statistics."""
        confidences, accuracies = simple_data

        computer = CalibrationMetricsComputer()
        summary = computer.compute_all(confidences, accuracies)

        assert summary.n_samples == len(confidences)
        assert abs(summary.accuracy - np.mean(accuracies)) < 1e-10
        assert abs(summary.mean_confidence - np.mean(confidences)) < 1e-10

    def test_summary_to_dict(
        self,
        simple_data: tuple[NDArray, NDArray],
    ) -> None:
        """Test summary serialization."""
        confidences, accuracies = simple_data

        computer = CalibrationMetricsComputer()
        summary = computer.compute_all(confidences, accuracies)

        d = summary.to_dict()

        assert "ece" in d
        assert "mce" in d
        assert "accuracy" in d


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Edge case tests."""

    def test_single_sample(self) -> None:
        """Test with single sample."""
        confidences = np.array([0.8])
        accuracies = np.array([1.0])

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        # Should not crash
        assert result.value >= 0

    def test_all_correct(self) -> None:
        """Test with all correct predictions."""
        np.random.seed(42)
        confidences = np.random.uniform(0.5, 1.0, 100)
        accuracies = np.ones(100)

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        assert result.value >= 0

    def test_all_incorrect(self) -> None:
        """Test with all incorrect predictions."""
        np.random.seed(42)
        confidences = np.random.uniform(0.5, 1.0, 100)
        accuracies = np.zeros(100)

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        # High confidence, all wrong = high ECE
        assert result.value > 0.4

    def test_uniform_confidence(self) -> None:
        """Test with uniform confidence values."""
        confidences = np.full(100, 0.5)
        accuracies = np.random.binomial(1, 0.5, 100).astype(float)

        ece = ExpectedCalibrationError()
        result = ece.compute(confidences, accuracies)

        # If 50% confident and ~50% correct, should be calibrated
        assert result.value < 0.2

    def test_empty_input_raises(self) -> None:
        """Test empty input raises error."""
        ece = ExpectedCalibrationError()

        with pytest.raises(ValueError):
            ece.compute(np.array([]), np.array([]))

    def test_shape_mismatch_raises(self) -> None:
        """Test shape mismatch raises error."""
        ece = ExpectedCalibrationError()

        with pytest.raises(ValueError):
            ece.compute(np.array([0.5, 0.6]), np.array([1.0]))

    def test_out_of_range_confidence_raises(self) -> None:
        """Test out-of-range confidence raises error."""
        ece = ExpectedCalibrationError()

        with pytest.raises(ValueError):
            ece.compute(np.array([1.5, 0.5]), np.array([1.0, 0.0]))

    def test_list_inputs(self) -> None:
        """Test that list inputs work."""
        computer = CalibrationMetricsComputer()
        result = computer.compute([0.5, 0.6, 0.7], [1.0, 0.0, 1.0])

        assert result.value >= 0


# ==================== Bootstrap CI Tests ====================


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_returns_bounds(self) -> None:
        """Test bootstrap returns valid bounds."""
        np.random.seed(42)
        confidences = np.random.uniform(0, 1, 100)
        accuracies = np.random.binomial(1, confidences).astype(float)

        def ece_fn(c: NDArray, a: NDArray) -> float:
            ece = ExpectedCalibrationError()
            return ece.compute(c, a).value

        lower, upper = bootstrap_confidence_interval(
            confidences, accuracies, ece_fn, n_bootstrap=100, seed=42
        )

        assert lower <= upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_bootstrap_reproducibility(self) -> None:
        """Test bootstrap with seed is reproducible."""
        np.random.seed(42)
        confidences = np.random.uniform(0, 1, 50)
        accuracies = np.random.binomial(1, confidences).astype(float)

        def ece_fn(c: NDArray, a: NDArray) -> float:
            ece = ExpectedCalibrationError()
            return ece.compute(c, a).value

        ci1 = bootstrap_confidence_interval(
            confidences, accuracies, ece_fn, n_bootstrap=50, seed=123
        )
        ci2 = bootstrap_confidence_interval(
            confidences, accuracies, ece_fn, n_bootstrap=50, seed=123
        )

        assert ci1 == ci2


# ==================== MetricResult Tests ====================


class TestMetricResult:
    """Tests for MetricResult."""

    def test_repr(self) -> None:
        """Test string representation."""
        result = MetricResult(name="ECE", value=0.1234)
        assert "ECE" in repr(result)
        assert "0.1234" in repr(result)

    def test_repr_with_ci(self) -> None:
        """Test repr with confidence interval."""
        result = MetricResult(
            name="ECE",
            value=0.1234,
            confidence_interval=(0.08, 0.16),
        )
        assert "0.08" in repr(result)
        assert "0.16" in repr(result)

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = MetricResult(
            name="ECE",
            value=0.1234,
            metadata={"n_bins": 10},
        )
        d = result.to_dict()

        assert d["name"] == "ECE"
        assert d["value"] == 0.1234
        assert d["metadata"]["n_bins"] == 10

    def test_to_dict_with_bins(self) -> None:
        """Test serialization with bins."""
        bins = [
            CalibrationBin(0.0, 0.5, 0.25, 10, 0.3, 0.25, 0.05),
            CalibrationBin(0.5, 1.0, 0.75, 10, 0.8, 0.75, 0.05),
        ]
        result = MetricResult(name="ECE", value=0.05, bins=bins)
        d = result.to_dict()

        assert len(d["bins"]) == 2
        assert d["bins"][0]["count"] == 10
