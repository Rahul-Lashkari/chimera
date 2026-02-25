"""Tests for the cross-track aggregation module."""

import pytest

from chimera.evaluation.aggregation import (
    CrossTrackAggregator,
    CrossTrackSummary,
    SimpleTrackSummary,
    TrackCorrelation,
)


class TestTrackCorrelation:
    """Tests for TrackCorrelation dataclass."""

    def test_basic_correlation(self) -> None:
        """Test basic correlation creation."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.75,
            sample_size=100,
        )

        assert correlation.track_a == "calibration"
        assert correlation.track_b == "error_detection"
        assert correlation.correlation == 0.75
        assert correlation.sample_size == 100
        assert correlation.p_value is None

    def test_correlation_with_p_value(self) -> None:
        """Test correlation with p-value."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.75,
            sample_size=100,
            p_value=0.01,
        )

        assert correlation.p_value == 0.01
        assert correlation.is_significant is True

    def test_not_significant(self) -> None:
        """Test non-significant correlation."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.15,
            sample_size=10,
            p_value=0.10,
        )

        assert correlation.is_significant is False

    def test_significance_without_p_value(self) -> None:
        """Test is_significant when no p-value."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.90,
            sample_size=100,
        )

        assert correlation.is_significant is False

    def test_strength_strong(self) -> None:
        """Test strong correlation strength."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.85,
            sample_size=100,
        )

        assert correlation.strength == "strong"

    def test_strength_moderate(self) -> None:
        """Test moderate correlation strength."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.55,
            sample_size=100,
        )

        assert correlation.strength == "moderate"

    def test_strength_weak(self) -> None:
        """Test weak correlation strength."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.25,
            sample_size=100,
        )

        assert correlation.strength == "weak"

    def test_strength_negligible(self) -> None:
        """Test negligible correlation strength."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=0.10,
            sample_size=100,
        )

        assert correlation.strength == "negligible"

    def test_negative_correlation(self) -> None:
        """Test negative correlation strength."""
        correlation = TrackCorrelation(
            track_a="calibration",
            track_b="error_detection",
            correlation=-0.75,
            sample_size=100,
        )

        assert correlation.strength == "strong"


class TestCrossTrackSummary:
    """Tests for CrossTrackSummary model."""

    def test_empty_summary(self) -> None:
        """Test empty summary creation."""
        summary = CrossTrackSummary()

        assert summary.track_summaries == {}
        assert summary.overall_accuracy == 0.0
        assert summary.total_tasks == 0
        assert summary.total_correct == 0
        assert summary.best_track is None
        assert summary.worst_track is None

    def test_summary_with_tracks(self) -> None:
        """Test summary with track data."""
        summary = CrossTrackSummary(
            track_summaries={
                "calibration": SimpleTrackSummary(
                    track="calibration",
                    total_tasks=10,
                    correct_tasks=8,
                    accuracy=0.8,
                    metrics={},
                ),
                "error_detection": SimpleTrackSummary(
                    track="error_detection",
                    total_tasks=10,
                    correct_tasks=7,
                    accuracy=0.7,
                    metrics={},
                ),
            },
            overall_accuracy=0.75,
            total_tasks=20,
            total_correct=15,
            best_track="calibration",
            worst_track="error_detection",
        )

        assert len(summary.track_summaries) == 2
        assert summary.overall_accuracy == 0.75
        assert summary.total_tasks == 20
        assert summary.total_correct == 15
        assert summary.best_track == "calibration"
        assert summary.worst_track == "error_detection"

    def test_get_accuracy_ranking(self) -> None:
        """Test get_accuracy_ranking method."""
        summary = CrossTrackSummary(
            track_summaries={
                "calibration": SimpleTrackSummary(
                    track="calibration",
                    total_tasks=10,
                    correct_tasks=8,
                    accuracy=0.8,
                    metrics={},
                ),
                "error_detection": SimpleTrackSummary(
                    track="error_detection",
                    total_tasks=10,
                    correct_tasks=9,
                    accuracy=0.9,
                    metrics={},
                ),
                "knowledge_boundary": SimpleTrackSummary(
                    track="knowledge_boundary",
                    total_tasks=10,
                    correct_tasks=6,
                    accuracy=0.6,
                    metrics={},
                ),
            },
        )

        ranking = summary.get_accuracy_ranking()

        assert ranking[0] == ("error_detection", 0.9)
        assert ranking[1] == ("calibration", 0.8)
        assert ranking[2] == ("knowledge_boundary", 0.6)

    def test_get_track_contribution_empty(self) -> None:
        """Test track contribution with empty summaries."""
        summary = CrossTrackSummary()
        contributions = summary.get_track_contribution()

        assert contributions == {}

    def test_get_track_contribution(self) -> None:
        """Test track contribution calculation."""
        summary = CrossTrackSummary(
            track_summaries={
                "calibration": SimpleTrackSummary(
                    track="calibration",
                    total_tasks=10,
                    correct_tasks=8,
                    accuracy=0.8,
                    metrics={},
                ),
                "error_detection": SimpleTrackSummary(
                    track="error_detection",
                    total_tasks=10,
                    correct_tasks=6,
                    accuracy=0.6,
                    metrics={},
                ),
            },
            track_weights={"calibration": 1.0, "error_detection": 1.0},
        )

        contributions = summary.get_track_contribution()

        # Total weighted = 0.8 + 0.6 = 1.4
        # Calibration contribution = 0.8 / 1.4 ≈ 0.571
        assert 0.55 < contributions["calibration"] < 0.60
        assert 0.40 < contributions["error_detection"] < 0.45


class TestCrossTrackAggregator:
    """Tests for CrossTrackAggregator class."""

    def test_initialization(self) -> None:
        """Test aggregator initialization."""
        aggregator = CrossTrackAggregator()
        summary = aggregator.compute_summary()

        assert summary.track_summaries == {}
        assert summary.overall_accuracy == 0.0

    def test_initialization_with_weights(self) -> None:
        """Test aggregator initialization with weights."""
        weights = {"calibration": 2.0, "error_detection": 1.0}
        aggregator = CrossTrackAggregator(track_weights=weights)

        assert aggregator._track_weights == weights

    def test_add_track_results(self) -> None:
        """Test adding track results."""
        aggregator = CrossTrackAggregator()
        results: list = []  # Empty results for testing

        aggregator.add_track_results("calibration", results)
        all_results = aggregator.get_all_results()

        assert "calibration" in all_results
        assert all_results["calibration"] == results

    def test_compute_summary_empty(self) -> None:
        """Test computing summary with no results."""
        aggregator = CrossTrackAggregator()
        summary = aggregator.compute_summary()

        assert summary.overall_accuracy == 0.0
        assert summary.total_tasks == 0
        assert summary.best_track is None
        assert summary.worst_track is None

    def test_get_track_summary_not_found(self) -> None:
        """Test getting summary for non-existent track."""
        aggregator = CrossTrackAggregator()
        summary = aggregator.get_track_summary("unknown_track")

        assert summary is None

    def test_clear(self) -> None:
        """Test clearing all results."""
        aggregator = CrossTrackAggregator()
        aggregator.add_track_results("calibration", [])
        aggregator.clear()

        all_results = aggregator.get_all_results()
        assert all_results == {}

    def test_set_track_weight(self) -> None:
        """Test setting track weight."""
        aggregator = CrossTrackAggregator()
        aggregator.set_track_weight("calibration", 2.0)

        assert aggregator._track_weights["calibration"] == 2.0

    def test_set_track_weight_invalid(self) -> None:
        """Test setting invalid track weight."""
        aggregator = CrossTrackAggregator()

        with pytest.raises(ValueError, match="Weight must be positive"):
            aggregator.set_track_weight("calibration", 0.0)

        with pytest.raises(ValueError, match="Weight must be positive"):
            aggregator.set_track_weight("calibration", -1.0)

    def test_get_improvement_suggestions_empty(self) -> None:
        """Test improvement suggestions with no data."""
        aggregator = CrossTrackAggregator()
        suggestions = aggregator.get_improvement_suggestions()

        # Should still return list, possibly with generic suggestions
        assert isinstance(suggestions, list)

    def test_correlations_computed(self) -> None:
        """Test that correlations are computed."""
        aggregator = CrossTrackAggregator()
        aggregator.add_track_results("calibration", [])
        aggregator.add_track_results("error_detection", [])

        summary = aggregator.compute_summary()

        # Should have at least one correlation for two tracks
        assert isinstance(summary.correlations, list)
