"""Tests for the evaluation pipeline module."""

from datetime import datetime

import pytest

from chimera.evaluation.aggregation import SimpleTrackSummary
from chimera.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    PipelineConfig,
    PipelineStage,
    TrackEvaluation,
    TrackType,
)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()

        assert len(config.tracks) == 4
        assert "calibration" in config.tracks
        assert "error_detection" in config.tracks
        assert "knowledge_boundary" in config.tracks
        assert "self_correction" in config.tracks
        assert config.model_provider == "gemini"
        assert config.model_name == "gemini-2.0-flash"
        assert config.timeout_seconds == 3600.0
        assert config.parallel is False
        assert config.cache_results is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PipelineConfig(
            tracks=["calibration", "error_detection"],
            model_provider="openai",
            model_name="gpt-4",
            samples_per_track=100,
            timeout_seconds=1800.0,
            parallel=True,
        )

        assert len(config.tracks) == 2
        assert config.model_provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.samples_per_track == 100
        assert config.timeout_seconds == 1800.0
        assert config.parallel is True

    def test_invalid_track_raises_error(self) -> None:
        """Test that invalid track names raise validation error."""
        with pytest.raises(ValueError, match="Invalid track"):
            PipelineConfig(tracks=["calibration", "invalid_track"])

    def test_single_track(self) -> None:
        """Test single track configuration."""
        config = PipelineConfig(tracks=["calibration"])
        assert len(config.tracks) == 1
        assert config.tracks[0] == "calibration"

    def test_empty_tracks(self) -> None:
        """Test empty tracks list."""
        config = PipelineConfig(tracks=[])
        assert len(config.tracks) == 0


class TestTrackEvaluation:
    """Tests for TrackEvaluation dataclass."""

    def test_basic_evaluation(self) -> None:
        """Test basic track evaluation creation."""
        evaluation = TrackEvaluation(track="calibration")

        assert evaluation.track == "calibration"
        assert evaluation.summary is None
        assert evaluation.raw_results == []
        assert evaluation.start_time is None
        assert evaluation.end_time is None
        assert evaluation.error is None

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 5, 30)

        evaluation = TrackEvaluation(
            track="calibration",
            start_time=start,
            end_time=end,
        )

        assert evaluation.duration_seconds == 330.0  # 5 minutes 30 seconds

    def test_duration_without_times(self) -> None:
        """Test duration returns 0 when times not set."""
        evaluation = TrackEvaluation(track="calibration")
        assert evaluation.duration_seconds == 0.0

    def test_success_with_summary(self) -> None:
        """Test success property with summary."""
        summary = SimpleTrackSummary(
            track="calibration",
            total_tasks=10,
            correct_tasks=8,
            accuracy=0.8,
        )

        evaluation = TrackEvaluation(
            track="calibration",
            summary=summary,
        )

        assert evaluation.success is True

    def test_success_with_error(self) -> None:
        """Test success property with error."""
        evaluation = TrackEvaluation(
            track="calibration",
            error="API timeout",
        )

        assert evaluation.success is False

    def test_success_without_summary(self) -> None:
        """Test success property without summary."""
        evaluation = TrackEvaluation(track="calibration")
        assert evaluation.success is False


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic evaluation result creation."""
        config = PipelineConfig()
        result = EvaluationResult(config=config)

        assert result.config == config
        assert result.track_evaluations == {}
        assert result.overall_score == 0.0
        assert result.stage == PipelineStage.INITIALIZATION
        assert result.metadata == {}

    def test_success_complete(self) -> None:
        """Test success property when complete."""
        config = PipelineConfig()
        result = EvaluationResult(
            config=config,
            stage=PipelineStage.COMPLETE,
        )

        assert result.success is True

    def test_success_failed(self) -> None:
        """Test success property when failed."""
        config = PipelineConfig()
        result = EvaluationResult(
            config=config,
            stage=PipelineStage.FAILED,
        )

        assert result.success is False

    def test_failed_tracks(self) -> None:
        """Test failed_tracks property."""
        config = PipelineConfig()
        result = EvaluationResult(config=config)

        result.track_evaluations["calibration"] = TrackEvaluation(
            track="calibration",
            error="Timeout",
        )
        result.track_evaluations["error_detection"] = TrackEvaluation(
            track="error_detection",
        )

        failed = result.failed_tracks
        assert "calibration" in failed
        assert "error_detection" in failed

    def test_get_track_scores(self) -> None:
        """Test get_track_scores method."""
        config = PipelineConfig()
        result = EvaluationResult(config=config)

        result.track_evaluations["calibration"] = TrackEvaluation(
            track="calibration",
            summary=SimpleTrackSummary(
                track="calibration",
                total_tasks=10,
                correct_tasks=8,
                accuracy=0.8,
            ),
        )
        result.track_evaluations["error_detection"] = TrackEvaluation(
            track="error_detection",
            summary=SimpleTrackSummary(
                track="error_detection",
                total_tasks=10,
                correct_tasks=7,
                accuracy=0.7,
            ),
        )

        scores = result.get_track_scores()
        assert scores["calibration"] == 0.8
        assert scores["error_detection"] == 0.7

    def test_generate_report_markdown(self) -> None:
        """Test markdown report generation."""
        config = PipelineConfig()
        result = EvaluationResult(config=config)

        report = result.generate_report(format="markdown")
        assert isinstance(report, str)

    def test_generate_report_json(self) -> None:
        """Test JSON report generation."""
        import json

        config = PipelineConfig()
        result = EvaluationResult(config=config)

        report = result.generate_report(format="json")
        parsed = json.loads(report)
        assert isinstance(parsed, dict)

    def test_generate_report_html(self) -> None:
        """Test HTML report generation."""
        config = PipelineConfig()
        result = EvaluationResult(
            config=config,
            overall_score=0.75,
        )

        report = result.generate_report(format="html")
        assert "<!DOCTYPE html>" in report
        assert "CHIMERA" in report

    def test_generate_report_invalid_format(self) -> None:
        """Test invalid format raises error."""
        config = PipelineConfig()
        result = EvaluationResult(config=config)

        with pytest.raises(ValueError, match="Unsupported format"):
            result.generate_report(format="pdf")


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_exist(self) -> None:
        """Test all expected stages exist."""
        stages = [
            PipelineStage.INITIALIZATION,
            PipelineStage.CALIBRATION,
            PipelineStage.ERROR_DETECTION,
            PipelineStage.KNOWLEDGE_BOUNDARY,
            PipelineStage.SELF_CORRECTION,
            PipelineStage.AGGREGATION,
            PipelineStage.REPORTING,
            PipelineStage.COMPLETE,
            PipelineStage.FAILED,
        ]

        assert len(stages) == 9

    def test_stage_values(self) -> None:
        """Test stage values are strings."""
        assert PipelineStage.CALIBRATION.value == "calibration"
        assert PipelineStage.ERROR_DETECTION.value == "error_detection"


class TestTrackType:
    """Tests for TrackType enum."""

    def test_all_tracks_exist(self) -> None:
        """Test all expected tracks exist."""
        assert TrackType.CALIBRATION.value == "calibration"
        assert TrackType.ERROR_DETECTION.value == "error_detection"
        assert TrackType.KNOWLEDGE_BOUNDARY.value == "knowledge_boundary"
        assert TrackType.SELF_CORRECTION.value == "self_correction"


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline class."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        config = PipelineConfig()
        pipeline = EvaluationPipeline(config)

        assert pipeline.config == config

    def test_pipeline_with_callback(self) -> None:
        """Test pipeline with progress callback."""
        stages_seen: list[PipelineStage] = []

        def callback(stage: PipelineStage, message: str) -> None:
            stages_seen.append(stage)

        config = PipelineConfig(tracks=[])
        pipeline = EvaluationPipeline(config, progress_callback=callback)
        _result = pipeline.run()  # Run to trigger callbacks

        assert PipelineStage.INITIALIZATION in stages_seen
        assert PipelineStage.COMPLETE in stages_seen

    def test_pipeline_run_empty_tracks(self) -> None:
        """Test running pipeline with no tracks."""
        config = PipelineConfig(tracks=[])
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()

        assert result.success is True
        assert result.stage == PipelineStage.COMPLETE
        assert result.overall_score == 0.0

    def test_pipeline_run_returns_result(self) -> None:
        """Test that run returns EvaluationResult."""
        config = PipelineConfig()
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()

        assert isinstance(result, EvaluationResult)
        assert result.start_time is not None
        assert result.end_time is not None

    def test_get_supported_tracks(self) -> None:
        """Test get_supported_tracks method."""
        config = PipelineConfig()
        pipeline = EvaluationPipeline(config)
        tracks = pipeline.get_supported_tracks()

        assert "calibration" in tracks
        assert "error_detection" in tracks
        assert "knowledge_boundary" in tracks
        assert "self_correction" in tracks

    def test_run_track_invalid(self) -> None:
        """Test running invalid track raises error."""
        config = PipelineConfig()
        pipeline = EvaluationPipeline(config)

        with pytest.raises(ValueError, match="Invalid track"):
            pipeline.run_track("invalid_track")

    def test_run_track_valid(self) -> None:
        """Test running a single valid track."""
        config = PipelineConfig()
        pipeline = EvaluationPipeline(config)
        evaluation = pipeline.run_track("calibration")

        assert isinstance(evaluation, TrackEvaluation)
        assert evaluation.track == "calibration"

    def test_pipeline_duration(self) -> None:
        """Test pipeline duration calculation."""
        config = PipelineConfig(tracks=[])
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()

        assert result.duration_seconds >= 0

    def test_verbose_mode(self) -> None:
        """Test verbose mode configuration."""
        config = PipelineConfig(verbose=True, tracks=[])
        pipeline = EvaluationPipeline(config)
        result = pipeline.run()

        assert result.success is True
