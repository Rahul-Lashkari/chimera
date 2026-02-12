"""Error detection task generator for CHIMERA benchmark.

This module generates tasks where models must identify errors in previous
responses. The model is shown a question along with a (potentially flawed)
response and must determine whether the response contains errors.

Error Detection Track Design:
1. Present original question + model's previous response
2. Response may contain factual, logical, or computational errors
3. Model must identify: (a) whether errors exist, (b) what the errors are
4. Measures model's ability to recognize its own mistakes

Error Types:
- Factual errors: Incorrect facts or information
- Computational errors: Mathematical mistakes
- Logical errors: Invalid reasoning or contradictions
- Omission errors: Missing critical information
- Hallucination errors: Made-up or fabricated details
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from chimera.generators.base import BaseTaskGenerator, GeneratorConfig
from chimera.generators.error_injection import (
    ErrorInjector,
    ErrorType,
    InjectedError,
    InjectionConfig,
)
from chimera.models.task import (
    AnswerType,
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskMetadata,
    TaskSet,
    TrackType,
)


class ErrorDetectionTaskType(str, Enum):
    """Types of error detection tasks."""

    # Binary detection: Is there an error?
    BINARY_DETECTION = "binary_detection"

    # Error identification: What is the error?
    ERROR_IDENTIFICATION = "error_identification"

    # Error localization: Where is the error?
    ERROR_LOCALIZATION = "error_localization"

    # Error correction: Fix the error
    ERROR_CORRECTION = "error_correction"

    # Multi-error detection: How many errors?
    MULTI_ERROR_DETECTION = "multi_error_detection"


class ErrorSeverity(str, Enum):
    """Severity levels for injected errors."""

    SUBTLE = "subtle"  # Easy to miss
    MODERATE = "moderate"  # Noticeable with attention
    OBVIOUS = "obvious"  # Clearly wrong


@dataclass
class SourceResponse:
    """A source response that may contain errors.

    Attributes:
        question: The original question
        response: The model's response (may contain errors)
        has_errors: Whether the response contains errors
        errors: List of injected errors
        domain: Subject domain
        difficulty: Task difficulty level
    """

    question: str
    response: str
    has_errors: bool
    errors: list[InjectedError] = field(default_factory=list)
    domain: str = "general"
    difficulty: DifficultyLevel = DifficultyLevel.L2
    correct_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ErrorDetectionGeneratorConfig(GeneratorConfig):
    """Configuration for error detection task generator.

    Attributes:
        error_rate: Proportion of tasks with errors (0.0-1.0)
        task_types: Types of detection tasks to generate
        error_types: Types of errors to inject
        severity_distribution: Distribution of error severities
        max_errors_per_task: Maximum errors in multi-error tasks
        include_correct_answer: Whether to include correct response in metadata
        seed_data_path: Path to seed data files
        shuffle: Whether to shuffle generated tasks
    """

    error_rate: float = Field(default=0.7, ge=0.0, le=1.0)

    task_types: list[ErrorDetectionTaskType] = Field(
        default_factory=lambda: [
            ErrorDetectionTaskType.BINARY_DETECTION,
            ErrorDetectionTaskType.ERROR_IDENTIFICATION,
        ]
    )

    error_types: list[ErrorType] = Field(
        default_factory=lambda: [
            ErrorType.FACTUAL,
            ErrorType.COMPUTATIONAL,
            ErrorType.LOGICAL,
        ]
    )

    severity_distribution: dict[ErrorSeverity, float] = Field(
        default_factory=lambda: {
            ErrorSeverity.SUBTLE: 0.3,
            ErrorSeverity.MODERATE: 0.5,
            ErrorSeverity.OBVIOUS: 0.2,
        }
    )

    max_errors_per_task: int = Field(default=3, ge=1, le=5)

    include_correct_answer: bool = True

    seed_data_path: Path | None = None

    shuffle: bool = True

    @property
    def num_tasks(self) -> int:
        """Alias for n_tasks for backward compatibility."""
        return self.n_tasks


class ErrorDetectionTaskGenerator(BaseTaskGenerator):
    """Generator for error detection tasks.

    This generator creates tasks that test a model's ability to identify
    errors in responses. It uses source Q&A pairs and injects various
    types of errors for the model to detect.

    Example:
        >>> config = ErrorDetectionGeneratorConfig(
        ...     num_tasks=100,
        ...     error_rate=0.7,
        ...     task_types=[ErrorDetectionTaskType.BINARY_DETECTION],
        ... )
        >>> generator = ErrorDetectionTaskGenerator(config)
        >>> task_set = generator.generate()
        >>> print(f"Generated {len(task_set.tasks)} tasks")
    """

    config: ErrorDetectionGeneratorConfig

    def __init__(
        self,
        config: ErrorDetectionGeneratorConfig | None = None,
        error_injector: ErrorInjector | None = None,
    ) -> None:
        """Initialize the error detection task generator.

        Args:
            config: Generator configuration
            error_injector: Custom error injector (uses default if None)
        """
        self.config = config or ErrorDetectionGeneratorConfig()
        self.error_injector = error_injector or ErrorInjector()
        self._source_responses: list[SourceResponse] = []
        self._templates = self._load_templates()

    def _load_templates(self) -> dict[ErrorDetectionTaskType, list[str]]:
        """Load prompt templates for each task type."""
        return {
            ErrorDetectionTaskType.BINARY_DETECTION: [
                (
                    "A student answered the following question:\n\n"
                    "Question: {question}\n\n"
                    "Answer: {response}\n\n"
                    "Is this answer correct? Respond with 'Yes' if the answer is "
                    "completely correct, or 'No' if it contains any errors. "
                    "Then explain your reasoning and state your confidence (0-100%)."
                ),
                (
                    "Review the following Q&A for accuracy:\n\n"
                    "Q: {question}\n"
                    "A: {response}\n\n"
                    "Does this response contain any errors? Answer 'Yes' or 'No', "
                    "provide your reasoning, and indicate your confidence level."
                ),
                (
                    "Consider this question and response:\n\n"
                    "Question: {question}\n"
                    "Response: {response}\n\n"
                    "Evaluate whether this response is accurate and complete. "
                    "State whether it contains errors (Yes/No), explain why, "
                    "and give your confidence percentage."
                ),
            ],
            ErrorDetectionTaskType.ERROR_IDENTIFICATION: [
                (
                    "The following response may contain one or more errors:\n\n"
                    "Question: {question}\n"
                    "Response: {response}\n\n"
                    "Identify any errors in this response. For each error:\n"
                    "1. Quote the erroneous part\n"
                    "2. Explain what's wrong\n"
                    "3. Provide the correction\n\n"
                    "If there are no errors, state 'No errors found.'\n"
                    "Rate your confidence in your analysis (0-100%)."
                ),
                (
                    "Analyze this response for mistakes:\n\n"
                    "Q: {question}\n"
                    "A: {response}\n\n"
                    "List all errors you find, explaining each one and how to fix it. "
                    "Include your confidence level."
                ),
            ],
            ErrorDetectionTaskType.ERROR_LOCALIZATION: [
                (
                    "The following response contains an error. Your task is to "
                    "locate it precisely.\n\n"
                    "Question: {question}\n"
                    "Response: {response}\n\n"
                    "Identify the exact location of the error by quoting the "
                    "specific phrase or sentence that is incorrect. "
                    "Explain why it's wrong and state your confidence."
                ),
            ],
            ErrorDetectionTaskType.ERROR_CORRECTION: [
                (
                    "The following response may contain errors. If there are errors, "
                    "provide a corrected version.\n\n"
                    "Question: {question}\n"
                    "Response: {response}\n\n"
                    "If the response is correct, write 'No correction needed.'\n"
                    "If incorrect, provide the corrected response.\n"
                    "Include your confidence level (0-100%)."
                ),
            ],
            ErrorDetectionTaskType.MULTI_ERROR_DETECTION: [
                (
                    "The following response may contain multiple errors:\n\n"
                    "Question: {question}\n"
                    "Response: {response}\n\n"
                    "How many errors does this response contain? "
                    "List each error separately with explanations. "
                    "State your confidence in your count."
                ),
            ],
        }

    def load_seed_data(self, path: Path | None = None) -> None:
        """Load source responses from seed data files.

        Args:
            path: Path to seed data (uses config path if None)
        """
        import json

        data_path = path or self.config.seed_data_path
        if data_path is None:
            # Use default seed data
            self._source_responses = self._get_default_seed_data()
            return

        if data_path.is_file():
            with open(data_path) as f:
                data = json.load(f)
        elif data_path.is_dir():
            data = []
            for file in data_path.glob("*.json"):
                with open(file) as f:
                    data.extend(json.load(f))
        else:
            raise FileNotFoundError(f"Seed data not found: {data_path}")

        self._source_responses = [
            SourceResponse(
                question=item["question"],
                response=item.get("correct_response", item.get("response", "")),
                has_errors=False,  # Original is correct
                correct_response=item.get("correct_response"),
                domain=item.get("domain", "general"),
                difficulty=DifficultyLevel(item.get("difficulty", "medium")),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]

    def _get_default_seed_data(self) -> list[SourceResponse]:
        """Get default seed data for testing."""
        return [
            # Math/Computation
            SourceResponse(
                question="What is 15% of 240?",
                response="15% of 240 is 36.",
                has_errors=False,
                correct_response="15% of 240 is 36.",
                domain="mathematics",
                difficulty=DifficultyLevel.L1,
            ),
            SourceResponse(
                question="Calculate the area of a circle with radius 7 cm.",
                response="The area is π × 7² = 49π ≈ 153.94 square centimeters.",
                has_errors=False,
                correct_response="The area is π × 7² = 49π ≈ 153.94 square centimeters.",
                domain="mathematics",
                difficulty=DifficultyLevel.L2,
            ),
            SourceResponse(
                question="What is the sum of the first 10 positive integers?",
                response="Using the formula n(n+1)/2: 10 × 11 / 2 = 55.",
                has_errors=False,
                correct_response="Using the formula n(n+1)/2: 10 × 11 / 2 = 55.",
                domain="mathematics",
                difficulty=DifficultyLevel.L1,
            ),
            # Science
            SourceResponse(
                question="What is the chemical formula for water?",
                response="The chemical formula for water is H₂O, consisting of two hydrogen atoms and one oxygen atom.",
                has_errors=False,
                correct_response="The chemical formula for water is H₂O, consisting of two hydrogen atoms and one oxygen atom.",
                domain="chemistry",
                difficulty=DifficultyLevel.L1,
            ),
            SourceResponse(
                question="What is the speed of light in a vacuum?",
                response="The speed of light in a vacuum is approximately 299,792,458 meters per second.",
                has_errors=False,
                correct_response="The speed of light in a vacuum is approximately 299,792,458 meters per second.",
                domain="physics",
                difficulty=DifficultyLevel.L2,
            ),
            SourceResponse(
                question="How many planets are in our solar system?",
                response="There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
                has_errors=False,
                correct_response="There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
                domain="astronomy",
                difficulty=DifficultyLevel.L1,
            ),
            # Geography
            SourceResponse(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                has_errors=False,
                correct_response="The capital of France is Paris.",
                domain="geography",
                difficulty=DifficultyLevel.L1,
            ),
            SourceResponse(
                question="What is the longest river in the world?",
                response="The Nile River, at approximately 6,650 kilometers, is generally considered the longest river in the world.",
                has_errors=False,
                correct_response="The Nile River, at approximately 6,650 kilometers, is generally considered the longest river in the world.",
                domain="geography",
                difficulty=DifficultyLevel.L2,
            ),
            # History
            SourceResponse(
                question="In what year did World War II end?",
                response="World War II ended in 1945, with Germany surrendering in May and Japan in September.",
                has_errors=False,
                correct_response="World War II ended in 1945, with Germany surrendering in May and Japan in September.",
                domain="history",
                difficulty=DifficultyLevel.L1,
            ),
            SourceResponse(
                question="Who was the first President of the United States?",
                response="George Washington was the first President of the United States, serving from 1789 to 1797.",
                has_errors=False,
                correct_response="George Washington was the first President of the United States, serving from 1789 to 1797.",
                domain="history",
                difficulty=DifficultyLevel.L1,
            ),
            # Logic/Reasoning
            SourceResponse(
                question="If all roses are flowers and all flowers need water, what can we conclude about roses?",
                response="We can conclude that all roses need water. This follows from the transitive property of the given statements.",
                has_errors=False,
                correct_response="We can conclude that all roses need water. This follows from the transitive property of the given statements.",
                domain="logic",
                difficulty=DifficultyLevel.L2,
            ),
            SourceResponse(
                question="What comes next in the sequence: 2, 4, 8, 16, ?",
                response="The next number is 32. Each number is doubled (multiplied by 2) to get the next.",
                has_errors=False,
                correct_response="The next number is 32. Each number is doubled (multiplied by 2) to get the next.",
                domain="logic",
                difficulty=DifficultyLevel.L1,
            ),
        ]

    def generate_task(
        self,
        difficulty: DifficultyLevel,
        category: TaskCategory,
    ) -> Task:
        """Generate a single error detection task.

        Args:
            difficulty: The difficulty level for the task.
            category: The category for the task.

        Returns:
            A generated Task.
        """
        if not self._source_responses:
            self.load_seed_data()

        # Choose whether to inject an error based on error_rate
        source = random.choice(self._source_responses)  # nosec B311
        task_type = random.choice(self.config.task_types)  # nosec B311

        if random.random() < self.config.error_rate:  # nosec B311
            task = self._create_task_with_error(source, task_type)
        else:
            task = self._create_task_without_error(source, task_type)

        if task:
            return task

        # Fallback if task creation failed
        return self._create_task_without_error(source, task_type) or Task(
            id=uuid4(),
            question="Verify this response for errors.",
            correct_answer="No errors found.",
            difficulty=difficulty,
            category=category,
            track=TrackType.ERROR_DETECTION,
        )

    def generate_batch(
        self,
        n_tasks: int,
        difficulty: DifficultyLevel | None = None,
        category: TaskCategory | None = None,
    ) -> list[Task]:
        """Generate a batch of error detection tasks.

        Args:
            n_tasks: Number of tasks to generate.
            difficulty: Optional specific difficulty level.
            category: Optional specific category.

        Returns:
            List of generated Tasks.
        """
        tasks: list[Task] = []
        for _ in range(n_tasks):
            diff = difficulty or random.choice(list(DifficultyLevel))  # nosec B311
            cat = category or random.choice(list(TaskCategory))  # nosec B311
            tasks.append(self.generate_task(diff, cat))
        return tasks

    def generate(self) -> TaskSet:
        """Generate a set of error detection tasks.

        Returns:
            TaskSet containing error detection tasks
        """
        if not self._source_responses:
            self.load_seed_data()

        tasks: list[Task] = []

        # Calculate number of tasks with errors
        num_with_errors = int(self.config.num_tasks * self.config.error_rate)
        num_without_errors = self.config.num_tasks - num_with_errors

        # Generate tasks with errors
        for _ in range(num_with_errors):
            source = random.choice(self._source_responses)  # nosec B311
            task_type = random.choice(self.config.task_types)  # nosec B311

            task = self._create_task_with_error(source, task_type)
            if task:
                tasks.append(task)

        # Generate tasks without errors (correct responses)
        for _ in range(num_without_errors):
            source = random.choice(self._source_responses)  # nosec B311
            task_type = random.choice(self.config.task_types)  # nosec B311

            task = self._create_task_without_error(source, task_type)
            if task:
                tasks.append(task)

        # Shuffle tasks
        if self.config.shuffle:
            random.shuffle(tasks)  # nosec B311

        return TaskSet(
            name=f"error_detection_{self.config.seed}",
            description=f"Error detection benchmark tasks (error_rate={self.config.error_rate})",
            track=TrackType.ERROR_DETECTION,
            tasks=tasks,
            tags=[
                f"error_rate:{self.config.error_rate}",
            ]
            + [f"task_type:{t.value}" for t in self.config.task_types],
        )

    def _create_task_with_error(
        self,
        source: SourceResponse,
        task_type: ErrorDetectionTaskType,
    ) -> Task | None:
        """Create a task with injected error(s).

        Args:
            source: Source response to inject errors into
            task_type: Type of detection task

        Returns:
            Task with error, or None if injection failed
        """
        # Select error type and severity
        error_type = random.choice(self.config.error_types)  # nosec B311
        severity = self._sample_severity()

        # Determine number of errors for multi-error tasks
        if task_type == ErrorDetectionTaskType.MULTI_ERROR_DETECTION:
            num_errors = random.randint(1, self.config.max_errors_per_task)  # nosec B311
        else:
            num_errors = 1

        # Inject error(s)
        injection_config = InjectionConfig(
            error_type=error_type,
            severity=severity.value,
            num_errors=num_errors,
        )

        try:
            injected_response, errors = self.error_injector.inject(
                question=source.question,
                response=source.response,
                config=injection_config,
            )
        except Exception:
            return None

        if not errors:
            return None

        # Create task
        template = random.choice(self._templates[task_type])  # nosec B311
        question = template.format(
            question=source.question,
            response=injected_response,
        )

        # Determine correct answer based on task type
        correct_answer = self._get_correct_answer(task_type, errors)

        return Task(
            id=uuid4(),
            track=TrackType.ERROR_DETECTION,
            question=question,
            correct_answer=correct_answer,
            answer_type=self._get_answer_type(task_type),
            difficulty=self._adjust_difficulty(source.difficulty, severity),
            category=self._map_domain_to_category(source.domain),
            metadata=TaskMetadata(
                source="error_detection_generator",
                verified=False,
                tags=[
                    f"task_type:{task_type.value}",
                    "has_errors:true",
                    f"domain:{source.domain}",
                ]
                + [f"error_type:{e.error_type.value}" for e in errors],
                notes=f"Error locations: {[e.location for e in errors]}. Original: {source.response[:100]}...",
            ),
        )

    def _create_task_without_error(
        self,
        source: SourceResponse,
        task_type: ErrorDetectionTaskType,
    ) -> Task:
        """Create a task with a correct response (no errors).

        Args:
            source: Source with correct response
            task_type: Type of detection task

        Returns:
            Task with correct response
        """
        template = random.choice(self._templates[task_type])  # nosec B311
        question = template.format(
            question=source.question,
            response=source.response,  # Use original (correct) response
        )

        # Correct answer is "no errors"
        correct_answer = self._get_correct_answer_no_error(task_type)

        return Task(
            id=uuid4(),
            track=TrackType.ERROR_DETECTION,
            question=question,
            correct_answer=correct_answer,
            answer_type=self._get_answer_type(task_type),
            difficulty=source.difficulty,
            category=self._map_domain_to_category(source.domain),
            metadata=TaskMetadata(
                source="error_detection_generator",
                verified=False,
                tags=[
                    f"task_type:{task_type.value}",
                    "has_errors:false",
                    f"domain:{source.domain}",
                ],
                notes=f"Original response: {source.response[:100]}...",
            ),
        )

    def _sample_severity(self) -> ErrorSeverity:
        """Sample error severity based on distribution."""
        severities = list(self.config.severity_distribution.keys())
        weights = list(self.config.severity_distribution.values())
        return random.choices(severities, weights=weights, k=1)[0]  # nosec B311

    def _get_correct_answer(
        self,
        task_type: ErrorDetectionTaskType,
        errors: list[InjectedError],
    ) -> str:
        """Get the correct answer for a task with errors."""
        if task_type == ErrorDetectionTaskType.BINARY_DETECTION:
            return "Yes"  # Yes, there are errors

        elif task_type == ErrorDetectionTaskType.ERROR_IDENTIFICATION:
            error_desc = "; ".join(e.description for e in errors)
            return f"Errors found: {error_desc}"

        elif task_type == ErrorDetectionTaskType.ERROR_LOCALIZATION:
            return errors[0].location if errors else "Unknown"

        elif task_type == ErrorDetectionTaskType.ERROR_CORRECTION:
            return errors[0].correction if errors else "No correction"

        elif task_type == ErrorDetectionTaskType.MULTI_ERROR_DETECTION:
            return str(len(errors))

        return "Yes"

    def _get_correct_answer_no_error(
        self,
        task_type: ErrorDetectionTaskType,
    ) -> str:
        """Get the correct answer for a task without errors."""
        if task_type == ErrorDetectionTaskType.BINARY_DETECTION:
            return "No"  # No errors

        elif task_type == ErrorDetectionTaskType.ERROR_IDENTIFICATION:
            return "No errors found."

        elif task_type == ErrorDetectionTaskType.ERROR_LOCALIZATION:
            return "No errors to locate."

        elif task_type == ErrorDetectionTaskType.ERROR_CORRECTION:
            return "No correction needed."

        elif task_type == ErrorDetectionTaskType.MULTI_ERROR_DETECTION:
            return "0"

        return "No"

    def _get_answer_type(self, task_type: ErrorDetectionTaskType) -> AnswerType:
        """Get answer type based on task type."""
        if task_type == ErrorDetectionTaskType.BINARY_DETECTION:
            return AnswerType.BOOLEAN
        elif task_type == ErrorDetectionTaskType.MULTI_ERROR_DETECTION:
            return AnswerType.NUMERIC
        else:
            return AnswerType.FREE_FORM

    def _map_domain_to_category(self, domain: str) -> TaskCategory:
        """Map domain string to TaskCategory enum.

        Args:
            domain: Domain string like 'mathematics', 'physics', etc.

        Returns:
            Corresponding TaskCategory
        """
        domain_mapping = {
            "mathematics": TaskCategory.NUMERICAL,
            "physics": TaskCategory.SCIENTIFIC,
            "chemistry": TaskCategory.SCIENTIFIC,
            "astronomy": TaskCategory.SCIENTIFIC,
            "biology": TaskCategory.SCIENTIFIC,
            "geography": TaskCategory.FACTUAL,
            "history": TaskCategory.HISTORICAL,
            "logic": TaskCategory.LOGICAL,
            "general": TaskCategory.FACTUAL,
        }
        return domain_mapping.get(domain.lower(), TaskCategory.FACTUAL)

    def _adjust_difficulty(
        self,
        base_difficulty: DifficultyLevel,
        severity: ErrorSeverity,
    ) -> DifficultyLevel:
        """Adjust difficulty based on error severity."""
        difficulty_order = [
            DifficultyLevel.L1,
            DifficultyLevel.L2,
            DifficultyLevel.L3,
            DifficultyLevel.L4,
            DifficultyLevel.L5,
        ]

        base_idx = difficulty_order.index(base_difficulty)

        # Subtle errors are harder to detect
        if severity == ErrorSeverity.SUBTLE:
            new_idx = min(base_idx + 2, len(difficulty_order) - 1)
        elif severity == ErrorSeverity.MODERATE:
            new_idx = min(base_idx + 1, len(difficulty_order) - 1)
        else:  # OBVIOUS
            new_idx = base_idx

        return difficulty_order[new_idx]

    def get_statistics(self, tasks: TaskSet) -> dict[str, Any]:
        """Get generator statistics.

        Args:
            tasks: TaskSet to compute statistics for

        Returns:
            Dictionary of statistics
        """
        return {
            "num_source_responses": len(self._source_responses),
            "num_tasks": len(tasks.tasks),
            "task_types": [t.value for t in self.config.task_types],
            "error_types": [e.value for e in self.config.error_types],
            "error_rate": self.config.error_rate,
            "severity_distribution": {
                k.value: v for k, v in self.config.severity_distribution.items()
            },
        }
