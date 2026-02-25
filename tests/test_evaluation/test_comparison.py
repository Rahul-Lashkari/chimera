"""Tests for the model comparison module."""

import pytest

from chimera.evaluation.aggregation import CrossTrackSummary, SimpleTrackSummary
from chimera.evaluation.comparison import (
    ComparisonMetric,
    ModelComparison,
    ModelRanking,
    PerformanceDelta,
)


class TestPerformanceDelta:
    """Tests for PerformanceDelta dataclass."""

    def test_basic_delta(self) -> None:
        """Test basic performance delta creation."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.70,
            value_b=0.80,
        )

        assert delta.model_a == "gpt-4"
        assert delta.model_b == "gemini"
        assert delta.track == "calibration"
        assert abs(delta.delta - 0.10) < 0.001
        assert abs(delta.delta_percent - 14.29) < 0.1

    def test_negative_delta(self) -> None:
        """Test negative delta (model B worse)."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.80,
            value_b=0.70,
        )

        assert abs(delta.delta - (-0.10)) < 0.001
        assert delta.delta_percent < 0

    def test_winner_model_b(self) -> None:
        """Test winner when model B is better."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.70,
            value_b=0.80,
        )

        assert delta.winner == "gemini"

    def test_winner_model_a(self) -> None:
        """Test winner when model A is better."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.80,
            value_b=0.70,
        )

        assert delta.winner == "gpt-4"

    def test_winner_tie(self) -> None:
        """Test winner when scores are equal."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.75,
            value_b=0.75,
        )

        assert delta.winner is None

    def test_is_significant_large_delta(self) -> None:
        """Test significance with large delta."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.60,
            value_b=0.80,
        )

        assert delta.is_significant is True

    def test_is_significant_small_delta(self) -> None:
        """Test significance with small delta."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.80,
            value_b=0.82,
        )

        assert delta.is_significant is False

    def test_zero_value_a(self) -> None:
        """Test delta percent when value_a is zero."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.0,
            value_b=0.80,
        )

        assert delta.delta_percent == float("inf")

    def test_both_zero(self) -> None:
        """Test delta percent when both are zero."""
        delta = PerformanceDelta(
            model_a="gpt-4",
            model_b="gemini",
            track="calibration",
            metric=ComparisonMetric.ACCURACY,
            value_a=0.0,
            value_b=0.0,
        )

        assert delta.delta_percent == 0.0


class TestModelRanking:
    """Tests for ModelRanking model."""

    def test_basic_ranking(self) -> None:
        """Test basic ranking creation."""
        ranking = ModelRanking(
            model_id="gpt-4",
            rank=1,
            overall_score=0.85,
        )

        assert ranking.model_id == "gpt-4"
        assert ranking.rank == 1
        assert ranking.overall_score == 0.85
        assert ranking.track_scores == {}
        assert ranking.strengths == []
        assert ranking.weaknesses == []

    def test_ranking_with_tracks(self) -> None:
        """Test ranking with track details."""
        ranking = ModelRanking(
            model_id="gpt-4",
            rank=2,
            overall_score=0.75,
            track_scores={
                "calibration": 0.80,
                "error_detection": 0.70,
            },
            track_ranks={
                "calibration": 1,
                "error_detection": 2,
            },
            strengths=["calibration"],
            weaknesses=["error_detection"],
        )

        assert ranking.track_scores["calibration"] == 0.80
        assert ranking.track_ranks["calibration"] == 1
        assert "calibration" in ranking.strengths
        assert "error_detection" in ranking.weaknesses

    def test_to_summary(self) -> None:
        """Test to_summary method."""
        ranking = ModelRanking(
            model_id="gpt-4",
            rank=1,
            overall_score=0.85,
            track_scores={
                "calibration": 0.90,
            },
            track_ranks={
                "calibration": 1,
            },
            strengths=["calibration"],
        )

        summary = ranking.to_summary()

        assert "gpt-4" in summary
        assert "#1" in summary
        assert "85" in summary  # 0.85 as percentage

    def test_rank_validation(self) -> None:
        """Test that rank must be >= 1."""
        with pytest.raises(ValueError):
            ModelRanking(
                model_id="gpt-4",
                rank=0,
                overall_score=0.85,
            )

    def test_score_validation(self) -> None:
        """Test that score must be between 0 and 1."""
        with pytest.raises(ValueError):
            ModelRanking(
                model_id="gpt-4",
                rank=1,
                overall_score=1.5,
            )


class TestComparisonMetric:
    """Tests for ComparisonMetric enum."""

    def test_all_metrics_exist(self) -> None:
        """Test all expected metrics exist."""
        assert ComparisonMetric.ACCURACY.value == "accuracy"
        assert ComparisonMetric.CORRECT_TASKS.value == "correct_tasks"
        assert ComparisonMetric.TOTAL_TASKS.value == "total_tasks"
        assert ComparisonMetric.WEIGHTED_SCORE.value == "weighted_score"


class TestModelComparison:
    """Tests for ModelComparison class."""

    def test_initialization(self) -> None:
        """Test comparison initialization."""
        comparison = ModelComparison()

        assert comparison.num_models == 0
        assert comparison.model_ids == []

    def test_initialization_with_weights(self) -> None:
        """Test initialization with track weights."""
        weights = {"calibration": 2.0}
        comparison = ModelComparison(track_weights=weights)

        assert comparison._track_weights == weights

    def test_add_model_summary(self) -> None:
        """Test adding a model summary."""
        comparison = ModelComparison()
        summary = CrossTrackSummary(
            overall_accuracy=0.80,
            total_tasks=100,
            total_correct=80,
        )

        comparison.add_model_summary("gpt-4", summary)

        assert comparison.num_models == 1
        assert "gpt-4" in comparison.model_ids

    def test_get_model_summary(self) -> None:
        """Test getting a model summary."""
        comparison = ModelComparison()
        summary = CrossTrackSummary(overall_accuracy=0.80)
        comparison.add_model_summary("gpt-4", summary)

        retrieved = comparison.get_model_summary("gpt-4")

        assert retrieved is not None
        assert retrieved.overall_accuracy == 0.80

    def test_get_model_summary_not_found(self) -> None:
        """Test getting summary for unknown model."""
        comparison = ModelComparison()
        summary = comparison.get_model_summary("unknown")

        assert summary is None

    def test_compute_rankings_empty(self) -> None:
        """Test rankings with no models."""
        comparison = ModelComparison()
        rankings = comparison.compute_rankings()

        assert rankings == []

    def test_compute_rankings_single_model(self) -> None:
        """Test rankings with single model."""
        comparison = ModelComparison()
        summary = CrossTrackSummary(overall_accuracy=0.80)
        comparison.add_model_summary("gpt-4", summary)

        rankings = comparison.compute_rankings()

        assert len(rankings) == 1
        assert rankings[0].model_id == "gpt-4"
        assert rankings[0].rank == 1

    def test_compute_rankings_multiple_models(self) -> None:
        """Test rankings with multiple models."""
        comparison = ModelComparison()

        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.70),
        )
        comparison.add_model_summary(
            "model-b",
            CrossTrackSummary(overall_accuracy=0.90),
        )
        comparison.add_model_summary(
            "model-c",
            CrossTrackSummary(overall_accuracy=0.80),
        )

        rankings = comparison.compute_rankings()

        assert len(rankings) == 3
        assert rankings[0].model_id == "model-b"
        assert rankings[0].rank == 1
        assert rankings[1].model_id == "model-c"
        assert rankings[1].rank == 2
        assert rankings[2].model_id == "model-a"
        assert rankings[2].rank == 3

    def test_compute_deltas(self) -> None:
        """Test computing deltas between two models."""
        comparison = ModelComparison()

        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.70),
        )
        comparison.add_model_summary(
            "model-b",
            CrossTrackSummary(overall_accuracy=0.80),
        )

        deltas = comparison.compute_deltas("model-a", "model-b")

        # Should have at least overall delta
        assert len(deltas) >= 1
        overall_delta = next(d for d in deltas if d.track == "overall")
        assert abs(overall_delta.delta - 0.10) < 0.001

    def test_compute_deltas_unknown_model(self) -> None:
        """Test computing deltas with unknown model."""
        comparison = ModelComparison()
        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.70),
        )

        with pytest.raises(ValueError, match="Model not found"):
            comparison.compute_deltas("model-a", "model-unknown")

    def test_compute_all_deltas(self) -> None:
        """Test computing all pairwise deltas."""
        comparison = ModelComparison()

        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.70),
        )
        comparison.add_model_summary(
            "model-b",
            CrossTrackSummary(overall_accuracy=0.80),
        )
        comparison.add_model_summary(
            "model-c",
            CrossTrackSummary(overall_accuracy=0.90),
        )

        deltas = comparison.compute_all_deltas()

        # Should have deltas for 3 pairs: (a,b), (a,c), (b,c)
        # Each pair has at least 1 (overall) delta
        assert len(deltas) >= 3

    def test_get_best_model_overall(self) -> None:
        """Test getting best model overall."""
        comparison = ModelComparison()

        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.70),
        )
        comparison.add_model_summary(
            "model-b",
            CrossTrackSummary(overall_accuracy=0.90),
        )

        best = comparison.get_best_model()

        assert best == "model-b"

    def test_get_best_model_per_track(self) -> None:
        """Test getting best model for specific track."""
        comparison = ModelComparison()

        # Model A is better at calibration
        summary_a = CrossTrackSummary(
            overall_accuracy=0.70,
            track_summaries={
                "calibration": SimpleTrackSummary(
                    track="calibration",
                    total_tasks=10,
                    correct_tasks=9,
                    accuracy=0.90,
                ),
            },
        )
        comparison.add_model_summary("model-a", summary_a)

        # Model B is worse at calibration
        summary_b = CrossTrackSummary(
            overall_accuracy=0.80,
            track_summaries={
                "calibration": SimpleTrackSummary(
                    track="calibration",
                    total_tasks=10,
                    correct_tasks=7,
                    accuracy=0.70,
                ),
            },
        )
        comparison.add_model_summary("model-b", summary_b)

        best = comparison.get_best_model(track="calibration")

        assert best == "model-a"

    def test_get_best_model_empty(self) -> None:
        """Test getting best model with no models."""
        comparison = ModelComparison()
        best = comparison.get_best_model()

        assert best is None

    def test_generate_comparison_report_markdown(self) -> None:
        """Test generating markdown comparison report."""
        comparison = ModelComparison()
        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.80),
        )

        report = comparison.generate_comparison_report(format="markdown")

        assert "# CHIMERA Model Comparison Report" in report
        assert "model-a" in report

    def test_generate_comparison_report_html(self) -> None:
        """Test generating HTML comparison report."""
        comparison = ModelComparison()
        comparison.add_model_summary(
            "model-a",
            CrossTrackSummary(overall_accuracy=0.80),
        )

        report = comparison.generate_comparison_report(format="html")

        assert "<!DOCTYPE html>" in report
        assert "model-a" in report

    def test_generate_comparison_report_invalid_format(self) -> None:
        """Test invalid report format raises error."""
        comparison = ModelComparison()

        with pytest.raises(ValueError, match="Unsupported format"):
            comparison.generate_comparison_report(format="pdf")

    def test_add_model_results(self) -> None:
        """Test adding raw model results."""
        comparison = ModelComparison()
        results: dict[str, list] = {
            "calibration": [],
            "error_detection": [],
        }

        summary = comparison.add_model_results("model-a", results)

        assert summary is not None
        assert comparison.num_models == 1
