"""Evaluation result models for CHIMERA benchmark.

This module defines models for evaluation results and metrics:
- EvaluationResult: Complete evaluation results across all tracks
- CalibrationMetrics: Metrics for Track 1 (calibration probing)
- ErrorDetectionMetrics: Metrics for Track 2 (error detection)
- KnowledgeBoundaryMetrics: Metrics for Track 3 (knowledge boundaries)
- SelfCorrectionMetrics: Metrics for Track 4 (self-correction)
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from chimera.models.task import TrackType


class ConfidenceBin(BaseModel):
    """A single bin in a calibration analysis.

    Attributes:
        bin_lower: Lower bound of the confidence bin
        bin_upper: Upper bound of the confidence bin
        bin_center: Center point of the bin
        count: Number of samples in this bin
        accuracy: Actual accuracy in this bin
        avg_confidence: Average confidence in this bin
        calibration_error: Absolute difference between accuracy and confidence
    """

    bin_lower: float = Field(ge=0.0, le=1.0, description="Lower bound")
    bin_upper: float = Field(ge=0.0, le=1.0, description="Upper bound")
    bin_center: float = Field(ge=0.0, le=1.0, description="Center point")
    count: int = Field(ge=0, description="Number of samples")
    accuracy: float = Field(ge=0.0, le=1.0, description="Actual accuracy")
    avg_confidence: float = Field(ge=0.0, le=1.0, description="Average confidence")
    calibration_error: float = Field(ge=0.0, description="Calibration error")


class CalibrationMetrics(BaseModel):
    """Metrics for Track 1: Calibration Probing.

    Captures how well a model's expressed confidence correlates with
    actual accuracy across different confidence levels.

    Attributes:
        ece: Expected Calibration Error (primary metric)
        mce: Maximum Calibration Error
        ace: Average Calibration Error
        brier_score: Brier score for probabilistic predictions
        overconfidence_rate: Fraction of overconfident predictions
        underconfidence_rate: Fraction of underconfident predictions
        accuracy: Overall accuracy
        avg_confidence: Average confidence expressed
        confidence_bins: Detailed bin-by-bin analysis
        n_samples: Number of samples evaluated
    """

    model_config = ConfigDict(use_enum_values=True)

    ece: float = Field(
        ge=0.0,
        le=1.0,
        description="Expected Calibration Error",
    )
    mce: float = Field(
        ge=0.0,
        le=1.0,
        description="Maximum Calibration Error",
    )
    ace: float = Field(
        ge=0.0,
        le=1.0,
        description="Average Calibration Error",
    )
    brier_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Brier score",
    )
    overconfidence_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of overconfident predictions",
    )
    underconfidence_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of underconfident predictions",
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall accuracy",
    )
    avg_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Average confidence expressed",
    )
    confidence_bins: list[ConfidenceBin] = Field(
        default_factory=list,
        description="Bin-by-bin calibration analysis",
    )
    n_samples: int = Field(
        ge=0,
        description="Number of samples evaluated",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_accuracy_gap(self) -> float:
        """Gap between average confidence and accuracy."""
        return abs(self.avg_confidence - self.accuracy)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_overconfident(self) -> bool:
        """Whether the model is generally overconfident."""
        return self.avg_confidence > self.accuracy

    def get_interpretation(self) -> str:
        """Get human-readable interpretation of calibration quality.

        Returns:
            String describing calibration quality.
        """
        if self.ece < 0.05:
            quality = "Excellent"
        elif self.ece < 0.10:
            quality = "Good"
        elif self.ece < 0.20:
            quality = "Moderate"
        else:
            quality = "Poor"

        direction = "overconfident" if self.is_overconfident else "underconfident"

        return f"{quality} calibration (ECE={self.ece:.3f}), generally {direction}"


class ErrorDetectionMetrics(BaseModel):
    """Metrics for Track 2: Error Detection.

    Captures how well a model can identify errors in its own responses.

    Attributes:
        precision: Fraction of flagged errors that are real errors
        recall: Fraction of real errors that were flagged
        f1_score: Harmonic mean of precision and recall
        false_positive_rate: Rate of incorrectly flagged non-errors
        false_negative_rate: Rate of missed real errors
        false_humility_rate: Rate of flagging correct answers as wrong
        recovery_rate: Fraction of errors corrected after detection
        n_samples: Number of samples evaluated
        n_total_errors: Total number of real errors in samples
        n_detected_errors: Number of errors detected by model
    """

    model_config = ConfigDict(use_enum_values=True)

    precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Error detection precision",
    )
    recall: float = Field(
        ge=0.0,
        le=1.0,
        description="Error detection recall",
    )
    f1_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Error detection F1 score",
    )
    false_positive_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="False positive rate",
    )
    false_negative_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="False negative rate",
    )
    false_humility_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of flagging correct answers as wrong",
    )
    recovery_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of errors corrected",
    )
    n_samples: int = Field(
        ge=0,
        description="Number of samples evaluated",
    )
    n_total_errors: int = Field(
        ge=0,
        description="Total number of real errors",
    )
    n_detected_errors: int = Field(
        ge=0,
        description="Number of errors detected",
    )

    def get_interpretation(self) -> str:
        """Get human-readable interpretation.

        Returns:
            String describing error detection quality.
        """
        if self.f1_score >= 0.8:
            quality = "Excellent"
        elif self.f1_score >= 0.6:
            quality = "Good"
        elif self.f1_score >= 0.4:
            quality = "Moderate"
        else:
            quality = "Poor"

        return f"{quality} error detection (F1={self.f1_score:.3f})"


class KnowledgeBoundaryMetrics(BaseModel):
    """Metrics for Track 3: Knowledge Boundary Recognition.

    Captures how well a model knows what it doesn't know.

    Attributes:
        abstention_accuracy: Accuracy of abstention decisions
        appropriate_abstention_rate: Rate of correct abstentions on unanswerable
        inappropriate_abstention_rate: Rate of abstaining on answerable questions
        overconfidence_on_unknown: Confidence on incorrectly answered unknowables
        auroc: Area under ROC for confidence as correctness predictor
        auprc: Area under precision-recall curve
        n_samples: Number of samples evaluated
        n_answerable: Number of answerable questions
        n_unanswerable: Number of unanswerable questions
    """

    model_config = ConfigDict(use_enum_values=True)

    abstention_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy of abstention decisions",
    )
    appropriate_abstention_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Correct abstention on unanswerable",
    )
    inappropriate_abstention_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Abstaining on answerable questions",
    )
    overconfidence_on_unknown: float = Field(
        ge=0.0,
        le=1.0,
        description="Average confidence on wrong answers to unknowables",
    )
    auroc: float = Field(
        ge=0.0,
        le=1.0,
        description="AUROC for confidence as correctness predictor",
    )
    auprc: float = Field(
        ge=0.0,
        le=1.0,
        description="Area under precision-recall curve",
    )
    n_samples: int = Field(
        ge=0,
        description="Number of samples evaluated",
    )
    n_answerable: int = Field(
        ge=0,
        description="Number of answerable questions",
    )
    n_unanswerable: int = Field(
        ge=0,
        description="Number of unanswerable questions",
    )

    def get_interpretation(self) -> str:
        """Get human-readable interpretation.

        Returns:
            String describing knowledge boundary recognition quality.
        """
        if self.appropriate_abstention_rate >= 0.8:
            quality = "Excellent"
        elif self.appropriate_abstention_rate >= 0.6:
            quality = "Good"
        elif self.appropriate_abstention_rate >= 0.4:
            quality = "Moderate"
        else:
            quality = "Poor"

        return (
            f"{quality} knowledge boundary recognition "
            f"(Abstention Rate={self.appropriate_abstention_rate:.3f})"
        )


class SelfCorrectionMetrics(BaseModel):
    """Metrics for Track 4: Self-Correction Under Perturbation.

    Captures how well a model can detect and correct corrupted reasoning.

    Attributes:
        corruption_detection_rate: Rate of detecting corrupted reasoning
        corruption_detection_auc: AUC for corruption detection
        correction_accuracy: Accuracy of corrections when detected
        sycophancy_index: Rate of changing correct answers incorrectly
        false_alarm_rate: Rate of flagging uncorrupted reasoning
        n_samples: Number of samples evaluated
        n_corrupted: Number of samples with corruption
        n_detected: Number of corruptions detected
        n_corrected: Number of corruptions correctly fixed
    """

    model_config = ConfigDict(use_enum_values=True)

    corruption_detection_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of detecting corrupted reasoning",
    )
    corruption_detection_auc: float = Field(
        ge=0.0,
        le=1.0,
        description="AUC for corruption detection",
    )
    correction_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy of corrections when detected",
    )
    sycophancy_index: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of changing correct to incorrect",
    )
    false_alarm_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of flagging uncorrupted reasoning",
    )
    n_samples: int = Field(
        ge=0,
        description="Number of samples evaluated",
    )
    n_corrupted: int = Field(
        ge=0,
        description="Number of samples with corruption",
    )
    n_detected: int = Field(
        ge=0,
        description="Number of corruptions detected",
    )
    n_corrected: int = Field(
        ge=0,
        description="Number of corruptions correctly fixed",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resilience_score(self) -> float:
        """Combined score of detection and correction ability."""
        return (self.corruption_detection_rate + self.correction_accuracy) / 2

    def get_interpretation(self) -> str:
        """Get human-readable interpretation.

        Returns:
            String describing self-correction quality.
        """
        if self.corruption_detection_auc >= 0.85:
            quality = "Excellent"
        elif self.corruption_detection_auc >= 0.7:
            quality = "Good"
        elif self.corruption_detection_auc >= 0.55:
            quality = "Moderate"
        else:
            quality = "Poor"

        return f"{quality} self-correction " f"(Detection AUC={self.corruption_detection_auc:.3f})"


class TrackResult(BaseModel):
    """Results for a single evaluation track.

    Attributes:
        track: Which track these results are for
        metrics: Track-specific metrics
        n_samples: Number of samples evaluated
        evaluation_time_seconds: Time taken for evaluation
        errors: Any errors encountered during evaluation
    """

    model_config = ConfigDict(use_enum_values=True)

    track: TrackType = Field(description="Which track")
    metrics: (
        CalibrationMetrics
        | ErrorDetectionMetrics
        | KnowledgeBoundaryMetrics
        | SelfCorrectionMetrics
    ) = Field(description="Track-specific metrics")
    n_samples: int = Field(ge=0, description="Number of samples evaluated")
    evaluation_time_seconds: float = Field(
        ge=0,
        description="Time taken for evaluation",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered",
    )


class EvaluationResult(BaseModel):
    """Complete evaluation results across all CHIMERA tracks.

    This is the top-level result structure containing metrics from
    all evaluation tracks plus overall summary statistics.

    Attributes:
        id: Unique evaluation identifier
        model_name: Name of the evaluated model
        model_version: Version of the evaluated model
        config_name: Name of the configuration used
        tracks: Results for each evaluation track
        started_at: When evaluation started
        completed_at: When evaluation completed
        total_samples: Total samples across all tracks
        total_time_seconds: Total evaluation time
        errors: Any errors encountered
        metadata: Additional evaluation metadata
    """

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique evaluation identifier",
    )
    model_name: str = Field(description="Name of the evaluated model")
    model_version: str | None = Field(
        default=None,
        description="Version of the evaluated model",
    )
    config_name: str = Field(
        default="default",
        description="Name of the configuration used",
    )
    tracks: dict[str, TrackResult] = Field(
        default_factory=dict,
        description="Results for each track",
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When evaluation started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When evaluation completed",
    )
    total_samples: int = Field(
        default=0,
        ge=0,
        description="Total samples across all tracks",
    )
    total_time_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Total evaluation time",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata",
    )

    def add_track_result(self, result: TrackResult) -> None:
        """Add results for a track.

        Args:
            result: The track result to add.
        """
        track_name = result.track if isinstance(result.track, str) else result.track.value
        self.tracks[track_name] = result
        self.total_samples += result.n_samples
        self.total_time_seconds += result.evaluation_time_seconds

    def get_track_result(self, track: TrackType) -> TrackResult | None:
        """Get results for a specific track.

        Args:
            track: The track to get results for.

        Returns:
            Track results or None if not available.
        """
        track_name = track if isinstance(track, str) else track.value
        return self.tracks.get(track_name)

    def mark_completed(self) -> None:
        """Mark the evaluation as completed."""
        self.completed_at = datetime.utcnow()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float | None:
        """Total duration of the evaluation."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the evaluation results.

        Returns:
            Dictionary with key metrics from each track.
        """
        summary: dict[str, Any] = {
            "model": self.model_name,
            "total_samples": self.total_samples,
            "duration_seconds": self.duration_seconds,
        }

        # Add key metric from each track
        for track_name, track_result in self.tracks.items():
            metrics = track_result.metrics
            if isinstance(metrics, CalibrationMetrics):
                summary[f"{track_name}_ece"] = metrics.ece
                summary[f"{track_name}_accuracy"] = metrics.accuracy
            elif isinstance(metrics, ErrorDetectionMetrics):
                summary[f"{track_name}_f1"] = metrics.f1_score
            elif isinstance(metrics, KnowledgeBoundaryMetrics):
                summary[f"{track_name}_abstention_rate"] = metrics.appropriate_abstention_rate
            elif isinstance(metrics, SelfCorrectionMetrics):
                summary[f"{track_name}_detection_auc"] = metrics.corruption_detection_auc

        return summary
