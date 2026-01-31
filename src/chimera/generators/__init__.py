"""Task generators for CHIMERA benchmark.

This module provides generators for creating evaluation tasks
across all four CHIMERA tracks:

- CalibrationTaskGenerator: Generates calibration probing tasks
- ErrorDetectionTaskGenerator: Generates error detection tasks (future)
- KnowledgeBoundaryTaskGenerator: Generates knowledge boundary tasks (future)
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
    # Templates
    "QuestionTemplate",
    "TemplateRegistry",
    # Difficulty stratification
    "DifficultyStratifier",
    "StratificationConfig",
]
