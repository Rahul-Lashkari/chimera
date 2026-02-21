"""CHIMERA: Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment.

A comprehensive benchmark for evaluating the meta-cognitive calibration of Large Language Models.

CHIMERA measures:
- Confidence calibration (does stated confidence predict accuracy?)
- Self-error detection (can models identify their own mistakes?)
- Knowledge boundary recognition (do models know what they don't know?)
- Self-correction ability (can models fix errors through introspection?)

Example:
    >>> from chimera import Benchmark, GeminiModel
    >>> model = GeminiModel("gemini-2.0-flash")
    >>> benchmark = Benchmark(model)
    >>> results = benchmark.run()
    >>> print(results.summary())

For more information, see the documentation at:
https://github.com/Rahul-Lashkari/chimera
"""

# isort: skip_file

# Runner
from chimera.runner.config import (
    OutputFormat,
    RunConfig,
    TrackConfig,
)
from chimera.runner.executor import (
    BenchmarkRunner,
    Checkpoint,
    ExecutionProgress,
    ExecutionState,
    TaskResult,
)
from chimera.runner.aggregator import (
    ResultsAggregator,
    TrackSummary,
)
from chimera.runner.report import (
    BenchmarkReport,
    ReportSection,
    export_report,
    load_report,
)

# Generators
from chimera.generators.base import BaseTaskGenerator, GeneratorConfig
from chimera.generators.calibration import (
    CalibrationGeneratorConfig,
    CalibrationTaskGenerator,
)
from chimera.generators.error_detection import (
    ErrorDetectionGeneratorConfig,
    ErrorDetectionTaskGenerator,
    ErrorDetectionTaskType,
    ErrorSeverity,
    SourceResponse,
)
from chimera.generators.error_injection import (
    ErrorInjector,
    ErrorType,
    InjectedError,
    InjectionConfig,
)
from chimera.generators.knowledge_boundary import (
    ExpectedResponse,
    KnowledgeBoundaryGeneratorConfig,
    KnowledgeBoundaryQuestion,
    KnowledgeBoundaryTaskGenerator,
    KnowledgeCategory,
)
from chimera.generators.self_correction import (
    CorrectionExpectation,
    CorruptionType,
    ReasoningTrace as SelfCorrectionReasoningTrace,
    SelfCorrectionGeneratorConfig,
    SelfCorrectionTaskGenerator,
)
from chimera.generators.difficulty import DifficultyStratifier, StratificationConfig
from chimera.generators.templates import QuestionTemplate, TemplateRegistry

# CLI
from chimera.cli import cli

# Model interfaces
from chimera.interfaces.base import (
    BaseModelInterface,
    GenerationResult,
    ModelCapabilities,
    ModelConfig,
)
from chimera.interfaces.gemini import GeminiConfig, GeminiModel
from chimera.interfaces.openai import OpenAIConfig, OpenAIModel
from chimera.interfaces.parsers import (
    ConfidenceParser,
    ParserConfig,
    ResponseParser,
    parse_confidence,
    parse_response,
)

# Metrics
from chimera.metrics.base import (
    BaseMetric,
    BinningStrategy,
    MetricConfig,
    MetricResult,
)
from chimera.metrics.base import (
    CalibrationBin as MetricCalibrationBin,
)
from chimera.metrics.calibration import (
    AdaptiveCalibrationError,
    BrierScore,
    CalibrationMetricsComputer,
    CalibrationSummary,
    ExpectedCalibrationError,
    MaximumCalibrationError,
    OverconfidenceMetrics,
)
from chimera.metrics.visualization import (
    CalibrationCurve,
    ConfidenceHistogram,
    ReliabilityDiagram,
    plot_calibration_summary,
)
from chimera.metrics.error_detection import (
    DetectionCalibrationMetrics,
    DetectionOutcome,
    DetectionResult,
    ErrorDetectionMetrics as ErrorDetectionMetricsComputer,
    ErrorDetectionMetricsConfig,
    ErrorDetectionSummary,
)
from chimera.metrics.knowledge_boundary import (
    AnswerabilityClassifier,
    BoundaryResult,
    CategoryMetrics,
    KnowledgeBoundaryMetrics as KnowledgeBoundaryMetricsComputer,
    KnowledgeBoundaryMetricsConfig,
    KnowledgeBoundarySummary,
    ResponseClassification,
)
from chimera.metrics.self_correction import (
    CorrectionEvaluator,
    CorrectionQuality,
    DetectionEvaluator,
    DetectionResult as SelfCorrectionDetectionResult,
    SelfCorrectionMetricsComputer,
    SelfCorrectionMetricsConfig,
    SelfCorrectionResult,
    SelfCorrectionSummary,
)

# Evaluation models
from chimera.models.evaluation import (
    CalibrationMetrics,
    ConfidenceBin,
    ErrorDetectionMetrics,
    EvaluationResult,
    KnowledgeBoundaryMetrics,
    SelfCorrectionMetrics,
    TrackResult,
)

# Response models
from chimera.models.response import (
    ConfidenceLevel,
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ReasoningStep,
    ReasoningTrace,
    ResponseMetadata,
)

# Task models
from chimera.models.task import (
    AnswerType,
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskMetadata,
    TaskSet,
    TrackType,
)
from chimera.version import __version__

__all__ = [
    "__version__",
    # Task models
    "AnswerType",
    "DifficultyLevel",
    "Task",
    "TaskCategory",
    "TaskMetadata",
    "TaskSet",
    "TrackType",
    # Response models
    "ConfidenceLevel",
    "ConfidenceScore",
    "ModelResponse",
    "ParsedAnswer",
    "ReasoningStep",
    "ReasoningTrace",
    "ResponseMetadata",
    # Evaluation models
    "CalibrationMetrics",
    "ConfidenceBin",
    "ErrorDetectionMetrics",
    "EvaluationResult",
    "KnowledgeBoundaryMetrics",
    "SelfCorrectionMetrics",
    "TrackResult",
    # Generators
    "BaseTaskGenerator",
    "GeneratorConfig",
    "CalibrationGeneratorConfig",
    "CalibrationTaskGenerator",
    "ErrorDetectionGeneratorConfig",
    "ErrorDetectionTaskGenerator",
    "ErrorDetectionTaskType",
    "ErrorSeverity",
    "SourceResponse",
    "ErrorInjector",
    "ErrorType",
    "InjectedError",
    "InjectionConfig",
    "DifficultyStratifier",
    "StratificationConfig",
    "QuestionTemplate",
    "TemplateRegistry",
    # Knowledge boundary generator
    "ExpectedResponse",
    "KnowledgeBoundaryGeneratorConfig",
    "KnowledgeBoundaryQuestion",
    "KnowledgeBoundaryTaskGenerator",
    "KnowledgeCategory",
    # Self-correction generator
    "CorrectionExpectation",
    "CorruptionType",
    "SelfCorrectionReasoningTrace",
    "SelfCorrectionGeneratorConfig",
    "SelfCorrectionTaskGenerator",
    # Model interfaces
    "BaseModelInterface",
    "GenerationResult",
    "ModelCapabilities",
    "ModelConfig",
    "GeminiConfig",
    "GeminiModel",
    "OpenAIConfig",
    "OpenAIModel",
    "ConfidenceParser",
    "ParserConfig",
    "ResponseParser",
    "parse_confidence",
    "parse_response",
    # Metrics
    "BaseMetric",
    "BinningStrategy",
    "MetricCalibrationBin",
    "MetricConfig",
    "MetricResult",
    "AdaptiveCalibrationError",
    "BrierScore",
    "CalibrationMetricsComputer",
    "CalibrationSummary",
    "ExpectedCalibrationError",
    "MaximumCalibrationError",
    "OverconfidenceMetrics",
    "CalibrationCurve",
    "ConfidenceHistogram",
    "ReliabilityDiagram",
    "plot_calibration_summary",
    # Error detection metrics
    "DetectionCalibrationMetrics",
    "DetectionOutcome",
    "DetectionResult",
    "ErrorDetectionMetricsComputer",
    "ErrorDetectionMetricsConfig",
    "ErrorDetectionSummary",
    # Knowledge boundary metrics
    "AnswerabilityClassifier",
    "BoundaryResult",
    "CategoryMetrics",
    "KnowledgeBoundaryMetricsComputer",
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
    # Runner
    "OutputFormat",
    "RunConfig",
    "TrackConfig",
    "BenchmarkRunner",
    "Checkpoint",
    "ExecutionProgress",
    "ExecutionState",
    "TaskResult",
    "ResultsAggregator",
    "TrackSummary",
    "BenchmarkReport",
    "ReportSection",
    "export_report",
    "load_report",
    # CLI
    "cli",
]
