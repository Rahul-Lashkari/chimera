"""Visualization utilities for CHIMERA calibration metrics.

This module provides plotting functions for visualizing calibration
analysis results, including reliability diagrams, calibration curves,
and confidence histograms.

All plotting functions are designed to work without requiring matplotlib
to be installed, returning data structures that can be plotted with
any visualization library.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from chimera.metrics.base import (
    CalibrationBin,
    MetricResult,
    bin_predictions,
    create_uniform_bins,
)
from chimera.metrics.calibration import CalibrationSummary

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class PlotData:
    """Container for plot data.

    Provides data in a format that can be used with any plotting library.

    Attributes:
        x: X-axis values.
        y: Y-axis values.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        metadata: Additional plot-specific data.
    """

    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ReliabilityDiagram:
    """Reliability diagram visualization.

    A reliability diagram plots mean confidence vs mean accuracy for
    each bin, along with a diagonal perfect calibration line.

    Visual interpretation:
        - Points on diagonal: Perfect calibration
        - Points above diagonal: Underconfident
        - Points below diagonal: Overconfident

    Example:
        >>> diagram = ReliabilityDiagram()
        >>> plot_data = diagram.get_plot_data(metrics)
        >>> # Use with matplotlib:
        >>> fig = diagram.plot(metrics)
        >>> fig.savefig("reliability.png")
    """

    def __init__(
        self,
        n_bins: int = 10,
        show_gap: bool = True,
        show_counts: bool = True,
    ) -> None:
        """Initialize reliability diagram.

        Args:
            n_bins: Number of confidence bins.
            show_gap: Whether to show gap bars.
            show_counts: Whether to show sample counts.
        """
        self.n_bins = n_bins
        self.show_gap = show_gap
        self.show_counts = show_counts

    def get_plot_data(
        self,
        result: MetricResult | list[CalibrationBin],
    ) -> PlotData:
        """Get plot data from metric result.

        Args:
            result: MetricResult or list of CalibrationBins.

        Returns:
            PlotData for plotting.
        """
        bins = result.bins if isinstance(result, MetricResult) else result

        # Extract bin data
        confidences = [b.confidence for b in bins if b.count > 0]
        accuracies = [b.accuracy for b in bins if b.count > 0]
        gaps = [b.gap for b in bins if b.count > 0]
        counts = [b.count for b in bins if b.count > 0]

        return PlotData(
            x=confidences,
            y=accuracies,
            xlabel="Mean Confidence",
            ylabel="Mean Accuracy",
            title="Reliability Diagram",
            metadata={
                "gaps": gaps,
                "counts": counts,
                "bin_edges": [(b.bin_lower, b.bin_upper) for b in bins],
            },
        )

    def plot(
        self,
        result: MetricResult | list[CalibrationBin],
        ax: Any = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> "Figure":
        """Plot reliability diagram using matplotlib.

        Args:
            result: MetricResult or list of CalibrationBins.
            ax: Matplotlib axes (creates new figure if None).
            figsize: Figure size if creating new figure.

        Returns:
            Matplotlib Figure object.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. " "Install with: pip install matplotlib"
            ) from e

        bins = result.bins if isinstance(result, MetricResult) else result

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get non-empty bins
        valid_bins = [b for b in bins if b.count > 0]

        if not valid_bins:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return fig

        # Extract data
        confidences = np.array([b.confidence for b in valid_bins])
        accuracies = np.array([b.accuracy for b in valid_bins])
        counts = np.array([b.count for b in valid_bins])

        # Bin width for bar plot
        width = 1.0 / self.n_bins * 0.8

        # Plot bars for accuracy
        bars = ax.bar(
            confidences,
            accuracies,
            width=width,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
            label="Accuracy",
        )

        # Plot gap (difference from diagonal)
        if self.show_gap:
            for conf, acc in zip(confidences, accuracies, strict=True):
                if conf > acc:  # Overconfident
                    ax.bar(
                        conf,
                        conf - acc,
                        bottom=acc,
                        width=width,
                        alpha=0.4,
                        color="red",
                        edgecolor="darkred",
                        linewidth=0.5,
                    )

        # Plot perfect calibration line
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.5,
            label="Perfect Calibration",
        )

        # Show counts on bars
        if self.show_counts:
            for bar, count in zip(bars, counts, strict=True):
                height = bar.get_height()
                ax.annotate(
                    f"n={count}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Confidence", fontsize=12)
        ax.set_ylabel("Mean Accuracy", fontsize=12)
        ax.set_title("Reliability Diagram", fontsize=14)
        ax.legend(loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Add ECE value if available
        if isinstance(result, MetricResult):
            ax.text(
                0.95,
                0.05,
                f"ECE = {result.value:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        plt.tight_layout()
        return fig


class ConfidenceHistogram:
    """Confidence distribution histogram.

    Visualizes the distribution of model confidence values,
    optionally split by correctness.

    Example:
        >>> hist = ConfidenceHistogram()
        >>> fig = hist.plot(confidences, accuracies)
    """

    def __init__(
        self,
        n_bins: int = 20,
        split_by_correctness: bool = True,
    ) -> None:
        """Initialize confidence histogram.

        Args:
            n_bins: Number of histogram bins.
            split_by_correctness: Whether to split by correct/incorrect.
        """
        self.n_bins = n_bins
        self.split_by_correctness = split_by_correctness

    def get_plot_data(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]] | None = None,
    ) -> PlotData:
        """Get histogram data.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values (optional).

        Returns:
            PlotData for plotting.
        """
        confidences = np.asarray(confidences)

        # Compute histogram
        counts, edges = np.histogram(confidences, bins=self.n_bins, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2

        metadata: dict[str, Any] = {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
        }

        if accuracies is not None and self.split_by_correctness:
            accuracies = np.asarray(accuracies)
            correct_mask = accuracies >= 0.5

            correct_counts, _ = np.histogram(
                confidences[correct_mask], bins=self.n_bins, range=(0, 1)
            )
            incorrect_counts, _ = np.histogram(
                confidences[~correct_mask], bins=self.n_bins, range=(0, 1)
            )

            metadata["correct_counts"] = correct_counts.tolist()
            metadata["incorrect_counts"] = incorrect_counts.tolist()

        return PlotData(
            x=centers.tolist(),
            y=counts.tolist(),
            xlabel="Confidence",
            ylabel="Count",
            title="Confidence Distribution",
            metadata=metadata,
        )

    def plot(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]] | None = None,
        ax: Any = None,
        figsize: tuple[int, int] = (8, 5),
    ) -> "Figure":
        """Plot confidence histogram.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values (optional).
            ax: Matplotlib axes.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. " "Install with: pip install matplotlib"
            ) from e

        confidences = np.asarray(confidences)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if accuracies is not None and self.split_by_correctness:
            accuracies = np.asarray(accuracies)
            correct_mask = accuracies >= 0.5

            ax.hist(
                confidences[correct_mask],
                bins=self.n_bins,
                range=(0, 1),
                alpha=0.7,
                color="green",
                label="Correct",
                edgecolor="black",
            )
            ax.hist(
                confidences[~correct_mask],
                bins=self.n_bins,
                range=(0, 1),
                alpha=0.7,
                color="red",
                label="Incorrect",
                edgecolor="black",
            )
            ax.legend()
        else:
            ax.hist(
                confidences,
                bins=self.n_bins,
                range=(0, 1),
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
            )

        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Confidence Distribution", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add mean confidence line
        mean_conf = float(np.mean(confidences))
        ax.axvline(
            mean_conf,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_conf:.2f}",
        )
        ax.legend()

        plt.tight_layout()
        return fig


class CalibrationCurve:
    """Calibration curve plotting.

    Plots predicted probability vs observed frequency,
    similar to reliability diagram but as a line plot.
    """

    def __init__(self, n_bins: int = 10) -> None:
        """Initialize calibration curve.

        Args:
            n_bins: Number of bins.
        """
        self.n_bins = n_bins

    def get_plot_data(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
    ) -> PlotData:
        """Get calibration curve data.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values.

        Returns:
            PlotData for plotting.
        """
        bin_boundaries = create_uniform_bins(self.n_bins)
        bins = bin_predictions(
            np.asarray(confidences),
            np.asarray(accuracies),
            bin_boundaries,
            min_samples=1,
        )

        valid_bins = [b for b in bins if b.count > 0]

        return PlotData(
            x=[b.confidence for b in valid_bins],
            y=[b.accuracy for b in valid_bins],
            xlabel="Mean Predicted Probability",
            ylabel="Fraction of Positives",
            title="Calibration Curve",
            metadata={
                "counts": [b.count for b in valid_bins],
                "bin_mids": [b.bin_mid for b in valid_bins],
            },
        )

    def plot(
        self,
        confidences: NDArray[np.floating[Any]],
        accuracies: NDArray[np.floating[Any]],
        ax: Any = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> "Figure":
        """Plot calibration curve.

        Args:
            confidences: Confidence values.
            accuracies: Accuracy values.
            ax: Matplotlib axes.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib is required for plotting.") from e

        plot_data = self.get_plot_data(confidences, accuracies)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot calibration curve
        ax.plot(
            plot_data.x,
            plot_data.y,
            "o-",
            color="steelblue",
            linewidth=2,
            markersize=8,
            label="Model",
        )

        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(plot_data.xlabel, fontsize=12)
        ax.set_ylabel(plot_data.ylabel, fontsize=12)
        ax.set_title(plot_data.title, fontsize=14)
        ax.legend(loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def plot_calibration_summary(
    summary: CalibrationSummary,
    figsize: tuple[int, int] = (14, 10),
) -> "Figure":
    """Create a comprehensive calibration visualization.

    Creates a 2x2 grid with:
        - Reliability diagram
        - Confidence histogram
        - Calibration metrics table
        - Over/underconfidence analysis

    Args:
        summary: CalibrationSummary from CalibrationMetricsComputer.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with all visualizations.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Reliability diagram (top-left)
    reliability = ReliabilityDiagram()
    reliability.plot(summary.ece, ax=axes[0, 0])

    # 2. Metrics table (top-right)
    ax_table = axes[0, 1]
    ax_table.axis("off")

    metrics_data = [
        ["Metric", "Value"],
        ["ECE", f"{summary.ece.value:.4f}"],
        ["MCE", f"{summary.mce.value:.4f}"],
        ["ACE", f"{summary.ace.value:.4f}"],
        ["Brier Score", f"{summary.brier.value:.4f}"],
        ["", ""],
        ["Accuracy", f"{summary.accuracy:.2%}"],
        ["Mean Confidence", f"{summary.mean_confidence:.2%}"],
        ["N Samples", f"{summary.n_samples:,}"],
    ]

    table = ax_table.table(
        cellText=metrics_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax_table.set_title("Calibration Metrics", fontsize=14, pad=20)

    # 3. Confidence histogram by correctness (bottom-left)
    ax_hist = axes[1, 0]

    # Get bin data for histogram
    bins = summary.ece.bins
    confidences = []
    accuracies = []

    for b in bins:
        if b.count > 0:
            # Approximate individual samples from bin statistics
            n_correct = int(b.accuracy * b.count)
            n_incorrect = b.count - n_correct

            confidences.extend([b.confidence] * b.count)
            accuracies.extend([1] * n_correct + [0] * n_incorrect)

    if confidences:
        conf_arr = np.array(confidences)
        acc_arr = np.array(accuracies)

        hist = ConfidenceHistogram()
        hist.plot(conf_arr, acc_arr, ax=ax_hist)
    else:
        ax_hist.text(0.5, 0.5, "No data", ha="center", va="center")

    # 4. Over/underconfidence analysis (bottom-right)
    ax_bias = axes[1, 1]

    oc = summary.overconfidence

    # Bar chart for over/underconfidence
    categories = ["Overconfident", "Underconfident"]
    ece_contributions = [oc.overconfidence_ece, oc.underconfidence_ece]
    ratios = [oc.overconfidence_ratio, oc.underconfidence_ratio]

    x = np.arange(len(categories))
    width = 0.35

    ax_bias.bar(
        x - width / 2,
        ece_contributions,
        width,
        label="ECE Contribution",
        color=["red", "blue"],
        alpha=0.7,
    )

    ax_bias2 = ax_bias.twinx()
    ax_bias2.bar(
        x + width / 2,
        ratios,
        width,
        label="Sample Ratio",
        color=["salmon", "lightblue"],
        alpha=0.7,
    )

    ax_bias.set_ylabel("ECE Contribution", fontsize=11)
    ax_bias2.set_ylabel("Sample Ratio", fontsize=11)
    ax_bias.set_xticks(x)
    ax_bias.set_xticklabels(categories)
    ax_bias.set_title(
        f"Calibration Bias: {oc.calibration_bias.title()}",
        fontsize=14,
    )

    # Combined legend
    handles = [
        Patch(color="red", alpha=0.7, label="Overconf ECE"),
        Patch(color="blue", alpha=0.7, label="Underconf ECE"),
        Patch(color="salmon", alpha=0.7, label="Overconf Ratio"),
        Patch(color="lightblue", alpha=0.7, label="Underconf Ratio"),
    ]
    ax_bias.legend(handles=handles, loc="upper right", fontsize=9)

    ax_bias.grid(True, alpha=0.3)

    plt.suptitle(
        "CHIMERA Calibration Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    return fig
