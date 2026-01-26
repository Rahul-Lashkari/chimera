"""CHIMERA data models package.

This module contains Pydantic models for all CHIMERA data structures:
- Tasks and task sets
- Model responses with confidence scores
- Evaluation results and metrics

All models support JSON serialization and schema generation for dataset validation.
"""

from chimera.models.evaluation import (
    CalibrationMetrics,
    ErrorDetectionMetrics,
    EvaluationResult,
    KnowledgeBoundaryMetrics,
    SelfCorrectionMetrics,
    TrackResult,
)
from chimera.models.response import (
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ReasoningTrace,
    ResponseMetadata,
)
from chimera.models.task import (
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskMetadata,
    TaskSet,
    TrackType,
)

__all__ = [
    # Task models
    "Task",
    "TaskSet",
    "TaskMetadata",
    "TaskCategory",
    "DifficultyLevel",
    "TrackType",
    # Response models
    "ModelResponse",
    "ConfidenceScore",
    "ReasoningTrace",
    "ParsedAnswer",
    "ResponseMetadata",
    # Evaluation models
    "EvaluationResult",
    "TrackResult",
    "CalibrationMetrics",
    "ErrorDetectionMetrics",
    "KnowledgeBoundaryMetrics",
    "SelfCorrectionMetrics",
]
