"""CHIMERA metrics package.

This module provides comprehensive calibration and evaluation metrics for
assessing meta-cognitive capabilities of language models.

Calibration Metrics:
    - ECE (Expected Calibration Error): Weighted average of calibration gaps
    - MCE (Maximum Calibration Error): Worst-case calibration gap
    - Brier Score: Proper scoring rule for probabilistic predictions
    - ACE (Adaptive Calibration Error): Equal-mass binning variant

Visualization:
    - Reliability diagrams: Confidence vs accuracy plots
    - Calibration curves: Per-bin calibration analysis
    - Confidence histograms: Distribution of model confidence

Example:
    >>> from chimera.metrics import (
    ...     CalibrationMetricsComputer,
    ...     ReliabilityDiagram,
    ... )
    >>>
    >>> # Compute calibration metrics
    >>> computer = CalibrationMetricsComputer(n_bins=10)
    >>> metrics = computer.compute(confidences, accuracies)
    >>> print(f"ECE: {metrics.ece:.4f}")
    >>>
    >>> # Generate reliability diagram
    >>> diagram = ReliabilityDiagram()
    >>> fig = diagram.plot(metrics)
    >>> fig.savefig("reliability.png")
"""

from chimera.metrics.base import (
    BaseMetric,
    BinningStrategy,
    CalibrationBin,
    MetricConfig,
    MetricResult,
)
from chimera.metrics.calibration import (
    AdaptiveCalibrationError,
    BrierScore,
    CalibrationMetricsComputer,
    ExpectedCalibrationError,
    MaximumCalibrationError,
    OverconfidenceMetrics,
)
from chimera.metrics.error_detection import (
    DetectionCalibrationMetrics,
    DetectionOutcome,
    DetectionResult,
    ErrorDetectionMetrics,
    ErrorDetectionMetricsConfig,
    ErrorDetectionSummary,
)
from chimera.metrics.knowledge_boundary import (
    AnswerabilityClassifier,
    BoundaryResult,
    CategoryMetrics,
    KnowledgeBoundaryMetrics,
    KnowledgeBoundaryMetricsConfig,
    KnowledgeBoundarySummary,
    ResponseClassification,
)
from chimera.metrics.self_correction import (
    CorrectionEvaluator,
    CorrectionQuality,
    DetectionEvaluator,
)
from chimera.metrics.self_correction import DetectionResult as SelfCorrectionDetectionResult
from chimera.metrics.self_correction import (
    SelfCorrectionMetricsComputer,
    SelfCorrectionMetricsConfig,
    SelfCorrectionResult,
    SelfCorrectionSummary,
)
from chimera.metrics.visualization import (
    CalibrationCurve,
    ConfidenceHistogram,
    ReliabilityDiagram,
    plot_calibration_summary,
)

__all__ = [
    # Base
    "BaseMetric",
    "BinningStrategy",
    "CalibrationBin",
    "MetricConfig",
    "MetricResult",
    # Calibration metrics
    "AdaptiveCalibrationError",
    "BrierScore",
    "CalibrationMetricsComputer",
    "ExpectedCalibrationError",
    "MaximumCalibrationError",
    "OverconfidenceMetrics",
    # Visualization
    "CalibrationCurve",
    "ConfidenceHistogram",
    "ReliabilityDiagram",
    "plot_calibration_summary",
    # Error detection metrics
    "DetectionCalibrationMetrics",
    "DetectionOutcome",
    "DetectionResult",
    "ErrorDetectionMetrics",
    "ErrorDetectionMetricsConfig",
    "ErrorDetectionSummary",
    # Knowledge boundary metrics
    "AnswerabilityClassifier",
    "BoundaryResult",
    "CategoryMetrics",
    "KnowledgeBoundaryMetrics",
    "KnowledgeBoundaryMetricsConfig",
    "KnowledgeBoundarySummary",
    "ResponseClassification",
    # Self-correction metrics
    "CorrectionEvaluator",
    "CorrectionQuality",
    "DetectionEvaluator",
    "SelfCorrectionDetectionResult",
    "SelfCorrectionMetricsComputer",
    "SelfCorrectionMetricsConfig",
    "SelfCorrectionResult",
    "SelfCorrectionSummary",
]
