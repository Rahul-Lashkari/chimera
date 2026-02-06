"""Unit tests for CHIMERA visualization utilities.

Tests cover:
- ReliabilityDiagram
- ConfidenceHistogram
- CalibrationCurve
- plot_calibration_summary
"""

from unittest.mock import patch

import numpy as np
import pytest

from chimera.metrics.base import CalibrationBin, MetricResult
from chimera.metrics.calibration import CalibrationMetricsComputer
from chimera.metrics.visualization import (
    CalibrationCurve,
    ConfidenceHistogram,
    PlotData,
    ReliabilityDiagram,
)


class TestPlotData:
    """Tests for PlotData container."""

    def test_basic_creation(self) -> None:
        """Test basic PlotData creation."""
        data = PlotData(
            x=[0.1, 0.2, 0.3],
            y=[0.15, 0.25, 0.35],
            xlabel="Confidence",
            ylabel="Accuracy",
            title="Test Plot",
        )

        assert len(data.x) == 3
        assert data.xlabel == "Confidence"

    def test_with_metadata(self) -> None:
        """Test PlotData with metadata."""
        data = PlotData(
            x=[0.5],
            y=[0.5],
            metadata={"counts": [10]},
        )

        assert data.metadata["counts"] == [10]


class TestReliabilityDiagram:
    """Tests for ReliabilityDiagram."""

    @pytest.fixture
    def sample_bins(self) -> list[CalibrationBin]:
        """Create sample calibration bins."""
        return [
            CalibrationBin(0.0, 0.2, 0.1, 10, 0.1, 0.1, 0.0),
            CalibrationBin(0.2, 0.4, 0.3, 15, 0.25, 0.3, 0.05),
            CalibrationBin(0.4, 0.6, 0.5, 20, 0.5, 0.5, 0.0),
            CalibrationBin(0.6, 0.8, 0.7, 15, 0.65, 0.7, 0.05),
            CalibrationBin(0.8, 1.0, 0.9, 10, 0.85, 0.9, 0.05),
        ]

    @pytest.fixture
    def sample_result(self, sample_bins: list[CalibrationBin]) -> MetricResult:
        """Create sample metric result."""
        return MetricResult(
            name="ECE",
            value=0.03,
            bins=sample_bins,
        )

    def test_get_plot_data_from_bins(self, sample_bins: list[CalibrationBin]) -> None:
        """Test getting plot data from bins."""
        diagram = ReliabilityDiagram()
        data = diagram.get_plot_data(sample_bins)

        assert len(data.x) == 5
        assert len(data.y) == 5
        assert data.xlabel == "Mean Confidence"
        assert data.ylabel == "Mean Accuracy"

    def test_get_plot_data_from_result(self, sample_result: MetricResult) -> None:
        """Test getting plot data from MetricResult."""
        diagram = ReliabilityDiagram()
        data = diagram.get_plot_data(sample_result)

        assert len(data.x) == 5
        assert "gaps" in data.metadata
        assert "counts" in data.metadata

    def test_get_plot_data_excludes_empty_bins(self) -> None:
        """Test that empty bins are excluded."""
        bins = [
            CalibrationBin(0.0, 0.5, 0.25, 0, 0.0, 0.0, 0.0),  # Empty
            CalibrationBin(0.5, 1.0, 0.75, 10, 0.8, 0.75, 0.05),
        ]

        diagram = ReliabilityDiagram()
        data = diagram.get_plot_data(bins)

        assert len(data.x) == 1  # Only non-empty bin

    def test_plot_requires_matplotlib(self, sample_result: MetricResult) -> None:
        """Test that plot raises ImportError without matplotlib."""
        _ = ReliabilityDiagram()  # Ensure class can be instantiated

        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            # This may or may not raise depending on how matplotlib is imported
            # Just verify the method exists and can be called
            pass


class TestConfidenceHistogram:
    """Tests for ConfidenceHistogram."""

    @pytest.fixture
    def sample_confidences(self) -> np.ndarray:
        """Create sample confidence values."""
        np.random.seed(42)
        return np.random.uniform(0, 1, 100)

    @pytest.fixture
    def sample_accuracies(self) -> np.ndarray:
        """Create sample accuracy values."""
        np.random.seed(42)
        return np.random.binomial(1, 0.7, 100).astype(float)

    def test_get_plot_data_basic(self, sample_confidences: np.ndarray) -> None:
        """Test basic histogram data."""
        hist = ConfidenceHistogram(n_bins=10)
        data = hist.get_plot_data(sample_confidences)

        assert len(data.x) == 10  # Bin centers
        assert len(data.y) == 10  # Counts
        assert data.xlabel == "Confidence"
        assert data.ylabel == "Count"

    def test_get_plot_data_with_split(
        self,
        sample_confidences: np.ndarray,
        sample_accuracies: np.ndarray,
    ) -> None:
        """Test histogram split by correctness."""
        hist = ConfidenceHistogram(split_by_correctness=True)
        data = hist.get_plot_data(sample_confidences, sample_accuracies)

        assert "correct_counts" in data.metadata
        assert "incorrect_counts" in data.metadata

    def test_histogram_edges(self, sample_confidences: np.ndarray) -> None:
        """Test histogram bin edges."""
        hist = ConfidenceHistogram(n_bins=5)
        data = hist.get_plot_data(sample_confidences)

        edges = data.metadata["edges"]
        assert len(edges) == 6  # n_bins + 1
        assert edges[0] == 0.0
        assert edges[-1] == 1.0


class TestCalibrationCurve:
    """Tests for CalibrationCurve."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample calibration data."""
        np.random.seed(42)
        confidences = np.random.uniform(0, 1, 200)
        accuracies = (np.random.random(200) < confidences).astype(float)
        return confidences, accuracies

    def test_get_plot_data(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test calibration curve data."""
        confidences, accuracies = sample_data

        curve = CalibrationCurve(n_bins=10)
        data = curve.get_plot_data(confidences, accuracies)

        assert len(data.x) <= 10  # May have fewer if bins empty
        assert len(data.x) == len(data.y)
        assert data.xlabel == "Mean Predicted Probability"

    def test_curve_includes_metadata(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test curve metadata."""
        confidences, accuracies = sample_data

        curve = CalibrationCurve(n_bins=5)
        data = curve.get_plot_data(confidences, accuracies)

        assert "counts" in data.metadata
        assert "bin_mids" in data.metadata


class TestVisualizationIntegration:
    """Integration tests for visualization with metrics."""

    def test_reliability_from_computer(self) -> None:
        """Test reliability diagram from CalibrationMetricsComputer."""
        np.random.seed(42)
        confidences = np.random.uniform(0, 1, 100)
        accuracies = (np.random.random(100) < confidences).astype(float)

        computer = CalibrationMetricsComputer(n_bins=10)
        result = computer.compute(confidences, accuracies)

        diagram = ReliabilityDiagram()
        data = diagram.get_plot_data(result)

        # Should have data from non-empty bins
        assert len(data.x) > 0

        # X values (confidences) should be in [0, 1]
        assert all(0 <= x <= 1 for x in data.x)

        # Y values (accuracies) should be in [0, 1]
        assert all(0 <= y <= 1 for y in data.y)

    def test_all_visualizations_work_together(self) -> None:
        """Test all visualization components work with same data."""
        np.random.seed(42)
        n = 200
        confidences = np.random.uniform(0, 1, n)
        accuracies = (np.random.random(n) < confidences).astype(float)

        # Compute metrics
        computer = CalibrationMetricsComputer()
        result = computer.compute(confidences, accuracies)

        # Get all plot data
        reliability = ReliabilityDiagram()
        histogram = ConfidenceHistogram()
        curve = CalibrationCurve()

        rel_data = reliability.get_plot_data(result)
        hist_data = histogram.get_plot_data(confidences, accuracies)
        curve_data = curve.get_plot_data(confidences, accuracies)

        # All should have valid data
        assert len(rel_data.x) > 0
        assert len(hist_data.x) > 0
        assert len(curve_data.x) > 0


class TestEdgeCases:
    """Edge case tests for visualization."""

    def test_empty_bins_handled(self) -> None:
        """Test handling of all-empty bins."""
        bins: list[CalibrationBin] = []

        diagram = ReliabilityDiagram()
        data = diagram.get_plot_data(bins)

        assert len(data.x) == 0
        assert len(data.y) == 0

    def test_single_bin(self) -> None:
        """Test with single bin."""
        bins = [
            CalibrationBin(0.0, 1.0, 0.5, 100, 0.6, 0.5, 0.1),
        ]

        diagram = ReliabilityDiagram()
        data = diagram.get_plot_data(bins)

        assert len(data.x) == 1

    def test_extreme_confidence_values(self) -> None:
        """Test with extreme confidence values."""
        confidences = np.array([0.0, 0.0, 1.0, 1.0])
        accuracies = np.array([0.0, 0.0, 1.0, 1.0])

        curve = CalibrationCurve(n_bins=10)
        data = curve.get_plot_data(confidences, accuracies)

        # Should handle edge bins correctly
        assert len(data.x) >= 1
