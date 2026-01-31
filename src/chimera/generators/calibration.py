"""Calibration track task generator for CHIMERA.

This module provides the main generator for creating calibration
probing tasks that test a model's confidence calibration.
"""

from pathlib import Path
from typing import Any

from pydantic import Field

from chimera.generators.base import BaseTaskGenerator, GeneratorConfig
from chimera.generators.difficulty import DifficultyStratifier, StratificationConfig
from chimera.generators.templates import (
    TemplateRegistry,
    create_default_calibration_templates,
)
from chimera.models.task import (
    AnswerType,
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskMetadata,
    TrackType,
)


class CalibrationGeneratorConfig(GeneratorConfig):
    """Configuration specific to calibration task generation.

    Attributes:
        use_default_templates: Whether to use built-in templates
        custom_templates_path: Path to custom templates file
        require_confidence_elicitation: Whether tasks should explicitly ask for confidence
        include_trick_questions: Whether to include questions designed to be misleading
        trick_question_ratio: Ratio of trick questions if included
    """

    use_default_templates: bool = Field(
        default=True,
        description="Whether to use built-in templates",
    )
    custom_templates_path: Path | None = Field(
        default=None,
        description="Path to custom templates file",
    )
    require_confidence_elicitation: bool = Field(
        default=True,
        description="Whether tasks should explicitly ask for confidence",
    )
    include_trick_questions: bool = Field(
        default=True,
        description="Whether to include potentially misleading questions",
    )
    trick_question_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Ratio of trick questions",
    )


class CalibrationTaskGenerator(BaseTaskGenerator):
    """Generator for calibration track tasks.

    This generator creates tasks designed to probe a model's
    confidence calibration across various domains and difficulty levels.

    The generated tasks:
    - Cover multiple domains (factual, reasoning, math, etc.)
    - Span all difficulty levels (L1-L5)
    - Include both straightforward and tricky questions
    - Explicitly elicit confidence expressions

    Example:
        >>> config = CalibrationGeneratorConfig(n_tasks=100, seed=42)
        >>> generator = CalibrationTaskGenerator(config)
        >>> tasks = generator.generate_all()
        >>> print(f"Generated {len(tasks)} tasks")
    """

    def __init__(self, config: CalibrationGeneratorConfig | None = None) -> None:
        """Initialize the calibration task generator.

        Args:
            config: Generator configuration. Uses defaults if None.
        """
        if config is None:
            config = CalibrationGeneratorConfig()

        super().__init__(config, TrackType.CALIBRATION)

        self.config: CalibrationGeneratorConfig = config
        self._template_registry = self._setup_templates()
        self._stratifier = DifficultyStratifier(
            StratificationConfig(
                target_distribution={
                    DifficultyLevel(k): v for k, v in config.difficulty_distribution.items()
                }
            )
        )

    def _setup_templates(self) -> TemplateRegistry:
        """Set up the template registry.

        Returns:
            Configured TemplateRegistry.
        """
        registry = TemplateRegistry()

        if self.config.use_default_templates:
            # Get default templates
            default_registry = create_default_calibration_templates()
            for template in default_registry.get_all_templates():
                registry.register(template)

        # TODO: Load custom templates from file if specified

        return registry

    def generate_task(
        self,
        difficulty: DifficultyLevel,
        category: TaskCategory,
    ) -> Task:
        """Generate a single calibration task.

        Args:
            difficulty: The difficulty level for the task.
            category: The category for the task.

        Returns:
            A generated calibration Task.
        """
        # Try to get a template for the exact category and difficulty
        template = self._template_registry.get_template(
            category=category,
            difficulty=difficulty,
            rng=self._rng,
        )

        if template is not None:
            question, answer = template.generate(self._rng)
            tags = template.tags.copy()
        else:
            # Fallback to a generic question
            question, answer, tags = self._generate_fallback(difficulty, category)

        # Add confidence elicitation if configured
        if self.config.require_confidence_elicitation:
            question = self._add_confidence_prompt(question)

        return Task(
            track=TrackType.CALIBRATION,
            question=question,
            correct_answer=answer,
            answer_type=AnswerType.FREE_FORM,
            difficulty=difficulty,
            category=category,
            metadata=TaskMetadata(
                source="calibration_generator",
                verified=True,
                tags=tags,
                notes=f"Expected confidence level: {difficulty.value}",
            ),
        )

    def generate_batch(
        self,
        n_tasks: int,
        difficulty: DifficultyLevel | None = None,
        category: TaskCategory | None = None,
    ) -> list[Task]:
        """Generate a batch of calibration tasks.

        Args:
            n_tasks: Number of tasks to generate.
            difficulty: Optional specific difficulty level.
            category: Optional specific category.

        Returns:
            List of generated Tasks.
        """
        tasks: list[Task] = []

        for _ in range(n_tasks):
            # Determine difficulty
            if difficulty is not None:
                task_difficulty = difficulty
            else:
                # Use stratifier to pick difficulty
                current_counts = self._count_difficulties(tasks)
                task_difficulty = self._stratifier.get_difficulty_for_next_task(
                    current_counts,
                    n_tasks,
                )

            # Determine category
            if category is not None:
                task_category = category
            else:
                task_category = self._rng.choice(list(TaskCategory))

            task = self.generate_task(task_difficulty, task_category)
            tasks.append(task)

        return tasks

    def generate_trick_question(
        self,
        difficulty: DifficultyLevel,
    ) -> Task:
        """Generate a trick question designed to expose overconfidence.

        Trick questions are designed to be misleading or have
        counter-intuitive answers that models often get wrong
        while expressing high confidence.

        Args:
            difficulty: The difficulty level.

        Returns:
            A trick question Task.
        """
        # Pool of trick questions by difficulty
        trick_questions = self._get_trick_questions()

        # Filter by difficulty
        candidates = [q for q in trick_questions if q["difficulty"] == difficulty]

        if not candidates:
            # Fall back to any trick question
            candidates = trick_questions

        if not candidates:
            # No trick questions available, generate regular task
            return self.generate_task(difficulty, TaskCategory.REASONING)

        selected = self._rng.choice(candidates)

        question = selected["question"]
        if self.config.require_confidence_elicitation:
            question = self._add_confidence_prompt(question)

        return Task(
            track=TrackType.CALIBRATION,
            question=question,
            correct_answer=selected["answer"],
            answer_type=AnswerType.FREE_FORM,
            difficulty=difficulty,
            category=TaskCategory(selected.get("category", "reasoning")),
            metadata=TaskMetadata(
                source="calibration_generator_trick",
                verified=True,
                tags=["trick_question", selected.get("trick_type", "misleading")],
                notes=f"Expected confidence level: {difficulty.value}",
            ),
        )

    def _generate_fallback(
        self,
        difficulty: DifficultyLevel,
        category: TaskCategory,  # noqa: ARG002
    ) -> tuple[str, str, list[str]]:
        """Generate a fallback question when no template matches.

        Args:
            difficulty: The difficulty level.
            category: The category.

        Returns:
            Tuple of (question, answer, tags).
        """
        # Simple arithmetic fallback that scales with difficulty
        difficulty_multipliers = {
            DifficultyLevel.L1: (10, 10),
            DifficultyLevel.L2: (50, 50),
            DifficultyLevel.L3: (100, 100),
            DifficultyLevel.L4: (500, 500),
            DifficultyLevel.L5: (1000, 1000),
        }

        max_a, max_b = difficulty_multipliers.get(difficulty, (100, 100))
        a = self._rng.randint(1, max_a)
        b = self._rng.randint(1, max_b)

        question = f"What is {a} + {b}?"
        answer = str(a + b)
        tags = ["arithmetic", "fallback"]

        return question, answer, tags

    def _add_confidence_prompt(self, question: str) -> str:
        """Add a confidence elicitation prompt to a question.

        Args:
            question: The original question.

        Returns:
            Question with confidence prompt added.
        """
        prompts = [
            "\n\nPlease provide your answer and express your confidence level (0-100%).",
            "\n\nAfter answering, indicate how confident you are in your response.",
            "\n\nProvide your answer and rate your confidence from 0% to 100%.",
        ]
        return question + str(self._rng.choice(prompts))

    def _count_difficulties(self, tasks: list[Task]) -> dict[DifficultyLevel, int]:
        """Count tasks by difficulty level.

        Args:
            tasks: List of tasks to count.

        Returns:
            Dictionary mapping difficulty levels to counts.
        """
        counts: dict[DifficultyLevel, int] = dict.fromkeys(DifficultyLevel, 0)
        for task in tasks:
            if task.difficulty:
                counts[task.difficulty] += 1
        return counts

    def _get_trick_questions(self) -> list[dict[str, Any]]:
        """Get the pool of trick questions.

        Returns:
            List of trick question dictionaries.
        """
        return [
            # Classic cognitive biases
            {
                "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "answer": "$0.05 (5 cents)",
                "difficulty": DifficultyLevel.L3,
                "category": "numerical",
                "trick_type": "cognitive_bias",
            },
            {
                "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "answer": "5 minutes",
                "difficulty": DifficultyLevel.L3,
                "category": "reasoning",
                "trick_type": "intuition_trap",
            },
            {
                "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
                "answer": "47 days",
                "difficulty": DifficultyLevel.L3,
                "category": "reasoning",
                "trick_type": "exponential_thinking",
            },
            # Misleading context
            {
                "question": "How many animals of each kind did Moses take on the Ark?",
                "answer": "None - it was Noah, not Moses",
                "difficulty": DifficultyLevel.L2,
                "category": "factual",
                "trick_type": "misleading_premise",
            },
            {
                "question": "Some months have 30 days, some have 31. How many months have 28 days?",
                "answer": "All 12 months have at least 28 days",
                "difficulty": DifficultyLevel.L2,
                "category": "reasoning",
                "trick_type": "assumption_trap",
            },
            # Mathematical tricks
            {
                "question": "What is 0.1 + 0.2? Give an exact decimal answer.",
                "answer": "0.3 (though computers may calculate 0.30000000000000004 due to floating-point representation)",
                "difficulty": DifficultyLevel.L4,
                "category": "numerical",
                "trick_type": "precision_trap",
            },
            # Counter-intuitive facts
            {
                "question": "Is the Great Wall of China visible from space with the naked eye?",
                "answer": "No, it is not visible from space with the naked eye despite popular belief",
                "difficulty": DifficultyLevel.L3,
                "category": "factual",
                "trick_type": "common_misconception",
            },
            {
                "question": "What percentage of the Earth's water is freshwater?",
                "answer": "About 2.5-3% (much less than most people think)",
                "difficulty": DifficultyLevel.L4,
                "category": "scientific",
                "trick_type": "counter_intuitive",
            },
        ]

    def _validate_track_specific(self, task: Task) -> list[str]:
        """Validate calibration-specific requirements.

        Args:
            task: The task to validate.

        Returns:
            List of validation errors.
        """
        errors = []

        # Calibration tasks should have a correct answer
        if task.correct_answer is None:
            errors.append("Calibration tasks must have a correct answer")

        # Should have difficulty set
        if task.difficulty is None:
            errors.append("Calibration tasks should have a difficulty level")

        return errors

    def get_coverage_report(self) -> dict[str, Any]:
        """Get a report on template coverage.

        Returns:
            Dictionary with coverage statistics.
        """
        by_category: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}

        for category in TaskCategory:
            templates = self._template_registry.get_all_templates(category=category)
            by_category[category.value] = len(templates)

        for difficulty in DifficultyLevel:
            templates = self._template_registry.get_all_templates(difficulty=difficulty)
            by_difficulty[difficulty.value] = len(templates)

        report = {
            "total_templates": self._template_registry.count(),
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }

        return report
