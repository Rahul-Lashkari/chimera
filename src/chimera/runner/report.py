"""Report generation for CHIMERA benchmark.

This module provides the BenchmarkReport class and utilities for
generating and exporting benchmark results in various formats.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from chimera.models.task import TrackType
from chimera.runner.config import OutputFormat, RunConfig


@dataclass
class ReportSection:
    """A section within the benchmark report.

    Attributes:
        title: Section title.
        content: Section content (dict or string).
        subsections: Nested subsections.
    """

    title: str
    content: dict[str, Any] | str = field(default_factory=dict)
    subsections: list["ReportSection"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to markdown format.

        Args:
            level: Heading level (2 = ##, 3 = ###, etc.).

        Returns:
            Markdown string.
        """
        lines = []
        prefix = "#" * level
        lines.append(f"{prefix} {self.title}\n")

        if isinstance(self.content, str):
            lines.append(self.content)
        elif isinstance(self.content, dict):
            for key, value in self.content.items():
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    lines.append(f"- **{formatted_key}**: {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"- **{formatted_key}**:")
                    for k, v in value.items():
                        formatted_k = k.replace("_", " ").title()
                        if isinstance(v, float):
                            lines.append(f"  - {formatted_k}: {v:.4f}")
                        else:
                            lines.append(f"  - {formatted_k}: {v}")
                else:
                    lines.append(f"- **{formatted_key}**: {value}")

        lines.append("")

        for subsection in self.subsections:
            lines.append(subsection.to_markdown(level + 1))

        return "\n".join(lines)


@dataclass
class BenchmarkReport:
    """Complete benchmark report with results and analysis.

    Contains all information from a benchmark run including
    configuration, per-track results, and overall metrics.

    Attributes:
        name: Report/run name.
        config: Run configuration used.
        track_summaries: Per-track summary data.
        overall_metrics: Aggregated overall metrics.
        sections: Report sections for output.
        created_at: When report was created.
        metadata: Additional report metadata.
    """

    name: str
    config: RunConfig
    track_summaries: dict[TrackType, Any]
    overall_metrics: dict[str, Any]
    sections: list[ReportSection] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, config: RunConfig) -> "BenchmarkReport":
        """Create an empty report (for dry runs).

        Args:
            config: Run configuration.

        Returns:
            Empty BenchmarkReport.
        """
        return cls(
            name=config.name,
            config=config,
            track_summaries={},
            overall_metrics={
                "total_tasks": 0,
                "completed_tasks": 0,
                "overall_accuracy": 0.0,
            },
            metadata={"dry_run": True},
        )

    def summary(self) -> str:
        """Generate a concise text summary.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "=" * 60,
            f"CHIMERA Benchmark Report: {self.name}",
            "=" * 60,
            "",
            "Overall Results:",
            f"  Total Tasks:     {self.overall_metrics.get('total_tasks', 0):,}",
            f"  Completed:       {self.overall_metrics.get('completed_tasks', 0):,}",
            f"  Success Rate:    {self.overall_metrics.get('success_rate', 0):.1f}%",
            f"  Overall Accuracy: {self.overall_metrics.get('overall_accuracy', 0):.2%}",
            f"  Mean Confidence: {self.overall_metrics.get('overall_confidence', 0):.2%}",
            f"  Mean Latency:    {self.overall_metrics.get('overall_latency_ms', 0):.0f}ms",
        ]

        if self.overall_metrics.get("calibration_ece") is not None:
            lines.append(f"  Calibration ECE: {self.overall_metrics['calibration_ece']:.4f}")

        lines.append("")
        lines.append("Per-Track Results:")

        for track, summary in self.track_summaries.items():
            track_name = track.value.replace("_", " ").title()
            data = summary.to_dict() if hasattr(summary, "to_dict") else summary

            lines.append(f"\n  {track_name}:")
            lines.append(
                f"    Tasks: {data.get('completed_tasks', 0)}/{data.get('total_tasks', 0)}"
            )
            lines.append(f"    Accuracy: {data.get('accuracy', 0):.2%}")

            if "calibration" in data:
                cal = data["calibration"]
                lines.append(f"    ECE: {cal.get('ece', 0):.4f}")
                lines.append(f"    Bias: {cal.get('bias', 'unknown')}")

        lines.append("")
        lines.append(f"Report generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of the report.
        """
        track_data = {}
        for track, summary in self.track_summaries.items():
            if hasattr(summary, "to_dict"):
                track_data[track.value] = summary.to_dict()
            else:
                track_data[track.value] = summary

        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "config": self.config.to_dict(),
            "overall_metrics": self.overall_metrics,
            "track_summaries": track_data,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string.

        Args:
            indent: Indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Convert report to markdown format.

        Returns:
            Markdown string.
        """
        lines = [
            f"# CHIMERA Benchmark Report: {self.name}",
            "",
            f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]

        for section in self.sections:
            lines.append(section.to_markdown())

        return "\n".join(lines)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Convert report to CSV-compatible rows.

        Returns:
            List of dictionaries for CSV export.
        """
        rows = []

        for track, summary in self.track_summaries.items():
            data = summary.to_dict() if hasattr(summary, "to_dict") else summary

            row = {
                "run_name": self.name,
                "track": track.value,
                "total_tasks": data.get("total_tasks", 0),
                "completed_tasks": data.get("completed_tasks", 0),
                "accuracy": data.get("accuracy", 0),
                "mean_confidence": data.get("mean_confidence", 0),
                "mean_latency_ms": data.get("mean_latency_ms", 0),
            }

            if "calibration" in data:
                row["ece"] = data["calibration"].get("ece", "")
                row["mce"] = data["calibration"].get("mce", "")
                row["brier"] = data["calibration"].get("brier", "")

            rows.append(row)

        return rows

    def save(
        self,
        path: str | Path,
        format: OutputFormat | None = None,
    ) -> Path:
        """Save report to file.

        Args:
            path: Output path.
            format: Output format (inferred from extension if None).

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Infer format from extension
        if format is None:
            ext = path.suffix.lower()
            format_map = {
                ".json": OutputFormat.JSON,
                ".jsonl": OutputFormat.JSONL,
                ".csv": OutputFormat.CSV,
                ".md": OutputFormat.MARKDOWN,
                ".markdown": OutputFormat.MARKDOWN,
            }
            format = format_map.get(ext, OutputFormat.JSON)

        if format == OutputFormat.JSON:
            with open(path, "w") as f:
                f.write(self.to_json())

        elif format == OutputFormat.JSONL:
            with open(path, "w") as f:
                # Write one line per track
                for track, summary in self.track_summaries.items():
                    data = summary.to_dict() if hasattr(summary, "to_dict") else summary
                    data["track"] = track.value
                    data["run_name"] = self.name
                    f.write(json.dumps(data, default=str) + "\n")

        elif format == OutputFormat.CSV:
            rows = self.to_csv_rows()
            if rows:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

        elif format == OutputFormat.MARKDOWN:
            with open(path, "w") as f:
                f.write(self.to_markdown())

        return path

    def save_all(
        self,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """Save report in all configured formats.

        Args:
            output_dir: Output directory (uses config if None).

        Returns:
            List of paths to saved files.
        """
        output_dir = output_dir or self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        for fmt in self.config.output_formats:
            ext_map = {
                OutputFormat.JSON: ".json",
                OutputFormat.JSONL: ".jsonl",
                OutputFormat.CSV: ".csv",
                OutputFormat.MARKDOWN: ".md",
            }
            ext = ext_map[fmt]
            path = output_dir / f"{self.name}{ext}"
            self.save(path, fmt)
            saved.append(path)

        return saved


def export_report(
    report: BenchmarkReport,
    path: str | Path,
    format: OutputFormat = OutputFormat.JSON,
) -> Path:
    """Export a benchmark report to file.

    Convenience function for saving reports.

    Args:
        report: Report to export.
        path: Output path.
        format: Output format.

    Returns:
        Path to exported file.
    """
    return report.save(path, format)


def load_report(path: str | Path) -> BenchmarkReport:
    """Load a benchmark report from JSON file.

    Args:
        path: Path to JSON report file.

    Returns:
        BenchmarkReport instance.
    """
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    # Reconstruct config
    config = RunConfig.from_dict(data.get("config", {}))

    # Reconstruct track summaries
    track_summaries = {}
    for track_str, summary_data in data.get("track_summaries", {}).items():
        track = TrackType(track_str)
        track_summaries[track] = summary_data

    # Reconstruct sections
    sections = []
    for section_data in data.get("sections", []):
        sections.append(
            ReportSection(
                title=section_data.get("title", ""),
                content=section_data.get("content", {}),
            )
        )

    return BenchmarkReport(
        name=data.get("name", "unknown"),
        config=config,
        track_summaries=track_summaries,
        overall_metrics=data.get("overall_metrics", {}),
        sections=sections,
        created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
        metadata=data.get("metadata", {}),
    )
