"""Unit tests for CHIMERA results aggregation and reporting.

Tests cover:
- ResultsAggregator
- TrackSummary
- BenchmarkReport
- Report export formats
"""

import json
from pathlib import Path

import pytest

from chimera.models.response import (
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ResponseMetadata,
)
from chimera.models.task import Task, TaskSet, TrackType
from chimera.runner.aggregator import ResultsAggregator, TrackSummary
from chimera.runner.config import OutputFormat, RunConfig
from chimera.runner.executor import TaskResult
from chimera.runner.report import (
    BenchmarkReport,
    ReportSection,
    export_report,
    load_report,
)


class TestTrackSummary:
    """Tests for TrackSummary."""

    def test_basic_summary(self) -> None:
        """Test basic summary creation."""
        summary = TrackSummary(
            track=TrackType.CALIBRATION,
            total_tasks=100,
            completed_tasks=95,
            failed_tasks=5,
            accuracy=0.85,
            mean_confidence=0.80,
        )

        assert summary.track == TrackType.CALIBRATION
        assert summary.success_rate == 95.0
        assert summary.accuracy == 0.85

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        summary = TrackSummary(
            track=TrackType.CALIBRATION,
            total_tasks=50,
            completed_tasks=45,
        )

        assert summary.success_rate == 90.0

    def test_ece_property(self) -> None:
        """Test ECE property without calibration."""
        summary = TrackSummary(track=TrackType.CALIBRATION)

        assert summary.ece is None

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        summary = TrackSummary(
            track=TrackType.CALIBRATION,
            total_tasks=10,
            completed_tasks=10,
            accuracy=0.9,
        )

        d = summary.to_dict()

        assert d["track"] == "calibration"
        assert d["accuracy"] == 0.9


class TestResultsAggregator:
    """Tests for ResultsAggregator."""

    @pytest.fixture
    def sample_tasks(self) -> TaskSet:
        """Create sample tasks."""
        tasks = [
            Task(
                track=TrackType.CALIBRATION,
                question=f"Question {i}",
                correct_answer=f"answer{i}",
            )
            for i in range(5)
        ]
        return TaskSet(
            name="test",
            track=TrackType.CALIBRATION,
            tasks=tasks,
        )

    @pytest.fixture
    def sample_results(self, sample_tasks: TaskSet) -> list[TaskResult]:
        """Create sample results matching tasks."""
        results = []
        for i, task in enumerate(sample_tasks.tasks):
            # Alternate between correct and incorrect
            is_correct = i % 2 == 0
            answer = task.correct_answer if is_correct else "wrong"

            response = ModelResponse(
                task_id=task.id,
                raw_text=f"Answer: {answer}",
                parsed_answer=ParsedAnswer(
                    raw_answer=answer,
                    normalized=answer.lower(),
                ),
                confidence=ConfidenceScore(numeric=0.8 if is_correct else 0.6),
                metadata=ResponseMetadata(
                    model_name="test-model",
                    latency_ms=100.0,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )

            results.append(
                TaskResult(
                    task_id=task.id,
                    track=TrackType.CALIBRATION,
                    response=response,
                    latency_ms=100.0,
                )
            )

        return results

    def test_aggregator_initialization(
        self,
        sample_tasks: TaskSet,
        sample_results: list[TaskResult],
    ) -> None:
        """Test aggregator initialization."""
        config = RunConfig()

        aggregator = ResultsAggregator(
            results=sample_results,
            task_sets={TrackType.CALIBRATION: sample_tasks},
            config=config,
        )

        assert len(aggregator.results) == 5
        assert TrackType.CALIBRATION in aggregator.results_by_track

    def test_compute_accuracy(
        self,
        sample_tasks: TaskSet,
        sample_results: list[TaskResult],
    ) -> None:
        """Test accuracy computation."""
        config = RunConfig()

        aggregator = ResultsAggregator(
            results=sample_results,
            task_sets={TrackType.CALIBRATION: sample_tasks},
            config=config,
        )

        accuracy, correctness = aggregator.compute_accuracy(sample_results)

        # Should be ~60% (3 correct out of 5)
        assert 0.5 <= accuracy <= 0.7

    def test_compute_track_summary(
        self,
        sample_tasks: TaskSet,
        sample_results: list[TaskResult],
    ) -> None:
        """Test track summary computation."""
        config = RunConfig()

        aggregator = ResultsAggregator(
            results=sample_results,
            task_sets={TrackType.CALIBRATION: sample_tasks},
            config=config,
        )

        summary = aggregator.compute_track_summary(TrackType.CALIBRATION)

        assert summary.track == TrackType.CALIBRATION
        assert summary.total_tasks == 5
        assert summary.completed_tasks == 5
        assert summary.mean_confidence > 0

    def test_generate_report(
        self,
        sample_tasks: TaskSet,
        sample_results: list[TaskResult],
    ) -> None:
        """Test report generation."""
        config = RunConfig(name="test_report")

        aggregator = ResultsAggregator(
            results=sample_results,
            task_sets={TrackType.CALIBRATION: sample_tasks},
            config=config,
        )

        report = aggregator.generate_report()

        assert report.name == "test_report"
        assert TrackType.CALIBRATION in report.track_summaries
        assert "total_tasks" in report.overall_metrics


class TestReportSection:
    """Tests for ReportSection."""

    def test_basic_section(self) -> None:
        """Test basic section creation."""
        section = ReportSection(
            title="Overview",
            content={"accuracy": 0.85, "tasks": 100},
        )

        assert section.title == "Overview"
        assert section.content["accuracy"] == 0.85

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        section = ReportSection(
            title="Test",
            content={"key": "value"},
        )

        d = section.to_dict()

        assert d["title"] == "Test"
        assert d["content"]["key"] == "value"

    def test_to_markdown(self) -> None:
        """Test markdown conversion."""
        section = ReportSection(
            title="Results",
            content={"accuracy": 0.85, "count": 100},
        )

        md = section.to_markdown()

        assert "## Results" in md
        assert "Accuracy" in md
        assert "0.85" in md


class TestBenchmarkReport:
    """Tests for BenchmarkReport."""

    @pytest.fixture
    def sample_report(self) -> BenchmarkReport:
        """Create sample report."""
        return BenchmarkReport(
            name="test_benchmark",
            config=RunConfig(name="test"),
            track_summaries={
                TrackType.CALIBRATION: TrackSummary(
                    track=TrackType.CALIBRATION,
                    total_tasks=100,
                    completed_tasks=95,
                    accuracy=0.85,
                    mean_confidence=0.80,
                ),
            },
            overall_metrics={
                "total_tasks": 100,
                "completed_tasks": 95,
                "success_rate": 95.0,
                "overall_accuracy": 0.85,
                "overall_confidence": 0.80,
                "overall_latency_ms": 150.0,
            },
            sections=[
                ReportSection(title="Overview", content={"test": True}),
            ],
        )

    def test_empty_report(self) -> None:
        """Test creating empty report."""
        config = RunConfig(name="empty")
        report = BenchmarkReport.empty(config)

        assert report.name == "empty"
        assert report.metadata.get("dry_run") is True

    def test_summary(self, sample_report: BenchmarkReport) -> None:
        """Test summary generation."""
        summary = sample_report.summary()

        assert "CHIMERA Benchmark Report" in summary
        assert "test_benchmark" in summary
        assert "85" in summary or "0.85" in summary

    def test_to_dict(self, sample_report: BenchmarkReport) -> None:
        """Test dictionary conversion."""
        d = sample_report.to_dict()

        assert d["name"] == "test_benchmark"
        assert "overall_metrics" in d
        assert "track_summaries" in d

    def test_to_json(self, sample_report: BenchmarkReport) -> None:
        """Test JSON conversion."""
        json_str = sample_report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "test_benchmark"

    def test_to_markdown(self, sample_report: BenchmarkReport) -> None:
        """Test markdown conversion."""
        md = sample_report.to_markdown()

        assert "# CHIMERA Benchmark Report" in md
        assert "test_benchmark" in md

    def test_to_csv_rows(self, sample_report: BenchmarkReport) -> None:
        """Test CSV row generation."""
        rows = sample_report.to_csv_rows()

        assert len(rows) == 1
        assert rows[0]["track"] == "calibration"
        assert rows[0]["accuracy"] == 0.85


class TestReportExport:
    """Tests for report export functionality."""

    @pytest.fixture
    def sample_report(self) -> BenchmarkReport:
        """Create sample report for export tests."""
        return BenchmarkReport(
            name="export_test",
            config=RunConfig(),
            track_summaries={
                TrackType.CALIBRATION: TrackSummary(
                    track=TrackType.CALIBRATION,
                    total_tasks=50,
                    completed_tasks=50,
                    accuracy=0.9,
                ),
            },
            overall_metrics={"total_tasks": 50, "overall_accuracy": 0.9},
        )

    def test_save_json(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test saving as JSON."""
        path = tmp_path / "report.json"

        result = sample_report.save(path, OutputFormat.JSON)

        assert result.exists()
        with open(result) as f:
            data = json.load(f)
        assert data["name"] == "export_test"

    def test_save_markdown(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test saving as Markdown."""
        path = tmp_path / "report.md"

        result = sample_report.save(path, OutputFormat.MARKDOWN)

        assert result.exists()
        content = result.read_text()
        assert "CHIMERA" in content

    def test_save_csv(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test saving as CSV."""
        path = tmp_path / "report.csv"

        result = sample_report.save(path, OutputFormat.CSV)

        assert result.exists()
        content = result.read_text()
        assert "track" in content
        assert "calibration" in content

    def test_save_jsonl(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test saving as JSONL."""
        path = tmp_path / "report.jsonl"

        result = sample_report.save(path, OutputFormat.JSONL)

        assert result.exists()
        lines = result.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["track"] == "calibration"

    def test_save_infers_format(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test format inference from extension."""
        json_path = tmp_path / "report.json"
        md_path = tmp_path / "report.md"

        sample_report.save(json_path)
        sample_report.save(md_path)

        # JSON should be valid JSON
        with open(json_path) as f:
            json.load(f)

        # MD should have markdown header
        assert "# CHIMERA" in md_path.read_text()

    def test_save_all(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test saving in all configured formats."""
        sample_report.config.output_formats = [
            OutputFormat.JSON,
            OutputFormat.MARKDOWN,
        ]

        paths = sample_report.save_all(tmp_path)

        assert len(paths) == 2
        assert all(p.exists() for p in paths)

    def test_export_report_function(
        self,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """Test export_report convenience function."""
        path = tmp_path / "exported.json"

        result = export_report(sample_report, path)

        assert result.exists()


class TestReportLoading:
    """Tests for loading reports from files."""

    def test_load_report(self, tmp_path: Path) -> None:
        """Test loading report from JSON."""
        # Create and save a report
        report = BenchmarkReport(
            name="loadable_report",
            config=RunConfig(name="test"),
            track_summaries={
                TrackType.CALIBRATION: TrackSummary(
                    track=TrackType.CALIBRATION,
                    total_tasks=25,
                    accuracy=0.88,
                ).to_dict(),
            },
            overall_metrics={"total_tasks": 25, "overall_accuracy": 0.88},
        )

        path = tmp_path / "report.json"
        report.save(path)

        # Load it back
        loaded = load_report(path)

        assert loaded.name == "loadable_report"
        assert loaded.overall_metrics["total_tasks"] == 25
