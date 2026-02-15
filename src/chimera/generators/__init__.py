"""Task generators for CHIMERA benchmark.

This module provides generators for creating evaluation tasks
across all four CHIMERA tracks:

- CalibrationTaskGenerator: Generates calibration probing tasks
- ErrorDetectionTaskGenerator: Generates error detection tasks
- KnowledgeBoundaryTaskGenerator: Generates knowledge boundary tasks
- SelfCorrectionTaskGenerator: Generates self-correction tasks (future)
"""

from chimera.generators.base import BaseTaskGenerator, GeneratorConfig
from chimera.generators.calibration import (
    CalibrationGeneratorConfig,
    CalibrationTaskGenerator,
)
from chimera.generators.difficulty import (
    DifficultyStratifier,
    StratificationConfig,
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
from chimera.generators.templates import (
    QuestionTemplate,
    TemplateRegistry,
)

__all__ = [
    # Base classes
    "BaseTaskGenerator",
    "GeneratorConfig",
    # Calibration generator
    "CalibrationTaskGenerator",
    "CalibrationGeneratorConfig",
    # Error detection generator
    "ErrorDetectionTaskGenerator",
    "ErrorDetectionGeneratorConfig",
    "ErrorDetectionTaskType",
    "ErrorSeverity",
    "SourceResponse",
    # Error injection
    "ErrorInjector",
    "ErrorType",
    "InjectedError",
    "InjectionConfig",
    # Knowledge boundary generator
    "ExpectedResponse",
    "KnowledgeBoundaryGeneratorConfig",
    "KnowledgeBoundaryQuestion",
    "KnowledgeBoundaryTaskGenerator",
    "KnowledgeCategory",
    # Templates
    "QuestionTemplate",
    "TemplateRegistry",
    # Difficulty stratification
    "DifficultyStratifier",
    "StratificationConfig",
]
