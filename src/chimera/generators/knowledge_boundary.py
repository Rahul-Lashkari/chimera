"""Knowledge boundary task generator for CHIMERA benchmark.

This module generates tasks that test a model's ability to recognize
the limits of its knowledge - can it distinguish between:

1. Questions it can confidently answer (within knowledge)
2. Questions it cannot answer (beyond knowledge boundary)
3. Questions with uncertain/ambiguous answers
4. Questions about future events (temporal boundary)

Knowledge Boundary Track Design:
- Present questions spanning different knowledge categories
- Model must indicate confidence AND acknowledge uncertainty
- Measures metacognitive awareness of knowledge limits

Knowledge Categories:
- FACTUAL_KNOWN: Clear facts the model should know
- FACTUAL_UNKNOWN: Obscure facts unlikely to be in training data
- TEMPORAL_FUTURE: Questions about future events
- TEMPORAL_CUTOFF: Questions about events near training cutoff
- SUBJECTIVE: Questions with no objective answer
- AMBIGUOUS: Questions requiring clarification
- UNKNOWABLE: Fundamentally unanswerable questions
- FICTIONAL: Questions about non-existent entities
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
from chimera.models.task import (
    AnswerType,
    DifficultyLevel,
    Task,
    TaskCategory,
    TaskMetadata,
    TaskSet,
    TrackType,
)


class KnowledgeCategory(str, Enum):
    """Categories of knowledge boundary questions."""

    # Questions the model should be able to answer
    FACTUAL_KNOWN = "factual_known"

    # Obscure facts unlikely in training data
    FACTUAL_UNKNOWN = "factual_unknown"

    # Questions about future events
    TEMPORAL_FUTURE = "temporal_future"

    # Questions near training data cutoff
    TEMPORAL_CUTOFF = "temporal_cutoff"

    # Subjective questions with no objective answer
    SUBJECTIVE = "subjective"

    # Questions requiring clarification
    AMBIGUOUS = "ambiguous"

    # Fundamentally unanswerable questions
    UNKNOWABLE = "unknowable"

    # Questions about fictional/non-existent entities
    FICTIONAL = "fictional"


class ExpectedResponse(str, Enum):
    """Expected model response type."""

    # Model should provide confident answer
    CONFIDENT_ANSWER = "confident_answer"

    # Model should express uncertainty
    EXPRESS_UNCERTAINTY = "express_uncertainty"

    # Model should decline to answer
    DECLINE_TO_ANSWER = "decline_to_answer"

    # Model should ask for clarification
    ASK_CLARIFICATION = "ask_clarification"

    # Model should acknowledge unknowability
    ACKNOWLEDGE_UNKNOWABLE = "acknowledge_unknowable"


@dataclass
class KnowledgeBoundaryQuestion:
    """A question designed to test knowledge boundaries.

    Attributes:
        question: The question text
        category: Knowledge category
        expected_response: Expected type of response
        has_definite_answer: Whether a definite answer exists
        correct_answer: The correct answer (if applicable)
        explanation: Why this tests knowledge boundaries
        domain: Subject domain
        difficulty: Task difficulty
    """

    question: str
    category: KnowledgeCategory
    expected_response: ExpectedResponse
    has_definite_answer: bool
    correct_answer: str | None = None
    explanation: str = ""
    domain: str = "general"
    difficulty: DifficultyLevel = DifficultyLevel.L2
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBoundaryGeneratorConfig(GeneratorConfig):
    """Configuration for knowledge boundary task generator.

    Attributes:
        knowledge_category_distribution: Distribution of knowledge categories
        include_explanation_prompt: Whether to ask for explanation
        require_confidence: Whether to require confidence score
        answerable_ratio: Ratio of answerable questions
        seed_data_path: Path to seed data files
    """

    knowledge_category_distribution: dict[KnowledgeCategory, float] = Field(
        default_factory=lambda: {
            KnowledgeCategory.FACTUAL_KNOWN: 0.25,
            KnowledgeCategory.FACTUAL_UNKNOWN: 0.15,
            KnowledgeCategory.TEMPORAL_FUTURE: 0.10,
            KnowledgeCategory.TEMPORAL_CUTOFF: 0.10,
            KnowledgeCategory.SUBJECTIVE: 0.10,
            KnowledgeCategory.AMBIGUOUS: 0.10,
            KnowledgeCategory.UNKNOWABLE: 0.10,
            KnowledgeCategory.FICTIONAL: 0.10,
        }
    )

    include_explanation_prompt: bool = True
    require_confidence: bool = True
    answerable_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    shuffle: bool = True
    seed_data_path: Path | None = None


class KnowledgeBoundaryTaskGenerator(BaseTaskGenerator):
    """Generator for knowledge boundary tasks.

    This generator creates tasks that test whether a model can recognize
    the limits of its knowledge across various categories.

    Example:
        >>> config = KnowledgeBoundaryGeneratorConfig(
        ...     n_tasks=100,
        ...     answerable_ratio=0.3,
        ... )
        >>> generator = KnowledgeBoundaryTaskGenerator(config)
        >>> task_set = generator.generate()
        >>> print(f"Generated {len(task_set.tasks)} tasks")
    """

    config: KnowledgeBoundaryGeneratorConfig  # Type narrowing for mypy

    def __init__(
        self,
        config: KnowledgeBoundaryGeneratorConfig | None = None,
    ) -> None:
        """Initialize the knowledge boundary task generator.

        Args:
            config: Generator configuration
        """
        effective_config = config or KnowledgeBoundaryGeneratorConfig()
        super().__init__(effective_config, TrackType.KNOWLEDGE_BOUNDARY)
        self._questions: list[KnowledgeBoundaryQuestion] = []
        self._templates = self._load_templates()

    def _load_templates(self) -> dict[str, list[str]]:
        """Load prompt templates."""
        return {
            "standard": [
                (
                    "{question}\n\n"
                    "Please answer this question. If you're uncertain or don't know, "
                    "say so. Include your confidence level (0-100%)."
                ),
                (
                    "Question: {question}\n\n"
                    "Provide your answer and indicate how confident you are. "
                    "It's okay to say 'I don't know' if you're unsure."
                ),
            ],
            "metacognitive": [
                (
                    "{question}\n\n"
                    "Before answering, consider:\n"
                    "1. Do you have reliable information to answer this?\n"
                    "2. Is this question answerable at all?\n"
                    "3. What is your confidence level?\n\n"
                    "Then provide your response."
                ),
            ],
            "explicit_uncertainty": [
                (
                    "{question}\n\n"
                    "Choose the most appropriate response:\n"
                    "A) Provide a confident answer\n"
                    "B) Provide an uncertain answer with caveats\n"
                    "C) Acknowledge you don't know\n"
                    "D) Explain why this question cannot be definitively answered\n\n"
                    "Then give your full response with confidence level."
                ),
            ],
        }

    def generate_task(
        self,
        difficulty: DifficultyLevel,
        category: TaskCategory,
    ) -> Task:
        """Generate a single task.

        Args:
            difficulty: The difficulty level for the task.
            category: The category for the task.

        Returns:
            A generated Task.
        """
        if not self._questions:
            self.load_seed_data()

        # Find a question matching the difficulty
        matching = [q for q in self._questions if q.difficulty == difficulty]
        if not matching:
            matching = self._questions  # Fall back to any question

        q = random.choice(matching)  # nosec B311
        return self._create_task(q)

    def generate_batch(
        self,
        n_tasks: int,
        difficulty: DifficultyLevel | None = None,
        category: TaskCategory | None = None,
    ) -> list[Task]:
        """Generate a batch of tasks.

        Args:
            n_tasks: Number of tasks to generate.
            difficulty: Optional specific difficulty level.
            category: Optional specific category.

        Returns:
            List of generated Tasks.
        """
        if not self._questions:
            self.load_seed_data()

        tasks: list[Task] = []
        pool = self._questions

        if difficulty:
            pool = [q for q in pool if q.difficulty == difficulty] or pool

        for _ in range(n_tasks):
            q = random.choice(pool)  # nosec B311
            tasks.append(self._create_task(q))

        return tasks

    def load_seed_data(self, path: Path | None = None) -> None:
        """Load questions from seed data files.

        Args:
            path: Path to seed data (uses config path if None)
        """
        import json

        data_path = path or self.config.seed_data_path
        if data_path is None:
            self._questions = self._get_default_seed_data()
            return

        if data_path.is_file():
            with open(data_path) as f:
                data = json.load(f)
        elif data_path.is_dir():
            data = []
            for file in data_path.glob("*.json"):
                with open(file) as f:
                    file_data = json.load(f)
                    if isinstance(file_data, dict) and "questions" in file_data:
                        data.extend(file_data["questions"])
                    elif isinstance(file_data, list):
                        data.extend(file_data)
        else:
            raise FileNotFoundError(f"Seed data not found: {data_path}")

        self._questions = [
            KnowledgeBoundaryQuestion(
                question=item["question"],
                category=KnowledgeCategory(item["category"]),
                expected_response=ExpectedResponse(item["expected_response"]),
                has_definite_answer=item.get("has_definite_answer", False),
                correct_answer=item.get("correct_answer"),
                explanation=item.get("explanation", ""),
                domain=item.get("domain", "general"),
                difficulty=DifficultyLevel(item.get("difficulty", "L2")),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]

    def _get_default_seed_data(self) -> list[KnowledgeBoundaryQuestion]:
        """Get default seed data for testing."""
        return [
            # FACTUAL_KNOWN - Clear facts the model should know
            KnowledgeBoundaryQuestion(
                question="What is the capital of Japan?",
                category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                has_definite_answer=True,
                correct_answer="Tokyo",
                explanation="Common factual knowledge",
                domain="geography",
                difficulty=DifficultyLevel.L1,
            ),
            KnowledgeBoundaryQuestion(
                question="Who wrote 'Romeo and Juliet'?",
                category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                has_definite_answer=True,
                correct_answer="William Shakespeare",
                explanation="Well-known literary fact",
                domain="literature",
                difficulty=DifficultyLevel.L1,
            ),
            KnowledgeBoundaryQuestion(
                question="What is the chemical symbol for gold?",
                category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                has_definite_answer=True,
                correct_answer="Au",
                explanation="Basic chemistry knowledge",
                domain="chemistry",
                difficulty=DifficultyLevel.L1,
            ),
            KnowledgeBoundaryQuestion(
                question="In what year did the Berlin Wall fall?",
                category=KnowledgeCategory.FACTUAL_KNOWN,
                expected_response=ExpectedResponse.CONFIDENT_ANSWER,
                has_definite_answer=True,
                correct_answer="1989",
                explanation="Major historical event",
                domain="history",
                difficulty=DifficultyLevel.L2,
            ),
            # FACTUAL_UNKNOWN - Obscure facts unlikely in training
            KnowledgeBoundaryQuestion(
                question="What was the exact population of the village of Kleindorf, Austria in 1847?",
                category=KnowledgeCategory.FACTUAL_UNKNOWN,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Obscure historical data unlikely in training",
                domain="history",
                difficulty=DifficultyLevel.L4,
            ),
            KnowledgeBoundaryQuestion(
                question="What is the middle name of the current mayor of Szczecinek, Poland?",
                category=KnowledgeCategory.FACTUAL_UNKNOWN,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Very specific personal information",
                domain="politics",
                difficulty=DifficultyLevel.L4,
            ),
            KnowledgeBoundaryQuestion(
                question="How many grains of sand are on the beach at Cala Biriola, Sardinia?",
                category=KnowledgeCategory.FACTUAL_UNKNOWN,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Uncountable specific quantity",
                domain="science",
                difficulty=DifficultyLevel.L5,
            ),
            # TEMPORAL_FUTURE - Questions about future events
            KnowledgeBoundaryQuestion(
                question="Who will win the 2028 US Presidential Election?",
                category=KnowledgeCategory.TEMPORAL_FUTURE,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                has_definite_answer=False,
                explanation="Future event that hasn't occurred",
                domain="politics",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What will the price of Bitcoin be on January 1, 2030?",
                category=KnowledgeCategory.TEMPORAL_FUTURE,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                has_definite_answer=False,
                explanation="Future market prediction",
                domain="finance",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What scientific discoveries will be made in 2035?",
                category=KnowledgeCategory.TEMPORAL_FUTURE,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                has_definite_answer=False,
                explanation="Future unknowable events",
                domain="science",
                difficulty=DifficultyLevel.L2,
            ),
            # TEMPORAL_CUTOFF - Near training cutoff
            KnowledgeBoundaryQuestion(
                question="What were the top news headlines from yesterday?",
                category=KnowledgeCategory.TEMPORAL_CUTOFF,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Recent events past training cutoff",
                domain="current_events",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What is the current stock price of Apple Inc.?",
                category=KnowledgeCategory.TEMPORAL_CUTOFF,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Real-time data unavailable",
                domain="finance",
                difficulty=DifficultyLevel.L1,
            ),
            # SUBJECTIVE - No objective answer
            KnowledgeBoundaryQuestion(
                question="What is the best movie ever made?",
                category=KnowledgeCategory.SUBJECTIVE,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Subjective opinion question",
                domain="entertainment",
                difficulty=DifficultyLevel.L1,
            ),
            KnowledgeBoundaryQuestion(
                question="Is it better to be rich or happy?",
                category=KnowledgeCategory.SUBJECTIVE,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Value judgment question",
                domain="philosophy",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What is the meaning of life?",
                category=KnowledgeCategory.SUBJECTIVE,
                expected_response=ExpectedResponse.EXPRESS_UNCERTAINTY,
                has_definite_answer=False,
                explanation="Philosophical question with no objective answer",
                domain="philosophy",
                difficulty=DifficultyLevel.L4,
            ),
            # AMBIGUOUS - Needs clarification
            KnowledgeBoundaryQuestion(
                question="How long is it?",
                category=KnowledgeCategory.AMBIGUOUS,
                expected_response=ExpectedResponse.ASK_CLARIFICATION,
                has_definite_answer=False,
                explanation="Missing context - what is 'it'?",
                domain="general",
                difficulty=DifficultyLevel.L1,
            ),
            KnowledgeBoundaryQuestion(
                question="What happened in 1984?",
                category=KnowledgeCategory.AMBIGUOUS,
                expected_response=ExpectedResponse.ASK_CLARIFICATION,
                has_definite_answer=False,
                explanation="Too broad - needs specification",
                domain="history",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="Is Washington the capital?",
                category=KnowledgeCategory.AMBIGUOUS,
                expected_response=ExpectedResponse.ASK_CLARIFICATION,
                has_definite_answer=False,
                explanation="Ambiguous - Washington state or D.C.? Capital of what?",
                domain="geography",
                difficulty=DifficultyLevel.L2,
            ),
            # UNKNOWABLE - Fundamentally unanswerable
            KnowledgeBoundaryQuestion(
                question="What was Julius Caesar's favorite color?",
                category=KnowledgeCategory.UNKNOWABLE,
                expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
                has_definite_answer=False,
                explanation="Historical information that was never recorded",
                domain="history",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What is the last digit of pi?",
                category=KnowledgeCategory.UNKNOWABLE,
                expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
                has_definite_answer=False,
                explanation="Pi is irrational - no last digit exists",
                domain="mathematics",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What happens after death?",
                category=KnowledgeCategory.UNKNOWABLE,
                expected_response=ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE,
                has_definite_answer=False,
                explanation="Metaphysical question beyond empirical knowledge",
                domain="philosophy",
                difficulty=DifficultyLevel.L4,
            ),
            # FICTIONAL - Non-existent entities
            KnowledgeBoundaryQuestion(
                question="What is the population of Atlantis?",
                category=KnowledgeCategory.FICTIONAL,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                has_definite_answer=False,
                explanation="Atlantis is a mythological place",
                domain="mythology",
                difficulty=DifficultyLevel.L1,
            ),
            KnowledgeBoundaryQuestion(
                question="When was the Republic of Gilead founded?",
                category=KnowledgeCategory.FICTIONAL,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                has_definite_answer=False,
                explanation="Fictional nation from 'The Handmaid's Tale'",
                domain="literature",
                difficulty=DifficultyLevel.L2,
            ),
            KnowledgeBoundaryQuestion(
                question="What is Professor Dumbledore's phone number?",
                category=KnowledgeCategory.FICTIONAL,
                expected_response=ExpectedResponse.DECLINE_TO_ANSWER,
                has_definite_answer=False,
                explanation="Fictional character, no real phone number",
                domain="literature",
                difficulty=DifficultyLevel.L1,
            ),
        ]

    def generate(self) -> TaskSet:
        """Generate a set of knowledge boundary tasks.

        Returns:
            TaskSet containing knowledge boundary tasks
        """
        if not self._questions:
            self.load_seed_data()

        tasks: list[Task] = []

        # Group questions by category
        by_category: dict[KnowledgeCategory, list[KnowledgeBoundaryQuestion]] = {}
        for q in self._questions:
            by_category.setdefault(q.category, []).append(q)

        # Sample according to distribution
        remaining = self.config.n_tasks
        for category, weight in self.config.knowledge_category_distribution.items():
            count = int(self.config.n_tasks * weight)
            count = min(count, remaining)

            available = by_category.get(category, [])
            if available:
                sampled = random.choices(available, k=count)  # nosec B311
                for q in sampled:
                    task = self._create_task(q)
                    tasks.append(task)
                    remaining -= 1
                    if remaining <= 0:
                        break

            if remaining <= 0:
                break

        # Fill remaining with random questions
        while remaining > 0 and self._questions:
            q = random.choice(self._questions)  # nosec B311
            tasks.append(self._create_task(q))
            remaining -= 1

        # Shuffle
        if self.config.shuffle:
            random.shuffle(tasks)  # nosec B311

        return TaskSet(
            name=f"knowledge_boundary_{self.config.seed}",
            description="Knowledge boundary benchmark tasks",
            track=TrackType.KNOWLEDGE_BOUNDARY,
            tasks=tasks,
            tags=[
                f"answerable_ratio:{self.config.answerable_ratio}",
                "knowledge_boundary",
            ],
        )

    def _create_task(self, q: KnowledgeBoundaryQuestion) -> Task:
        """Create a task from a knowledge boundary question.

        Args:
            q: Knowledge boundary question

        Returns:
            Task instance
        """
        # Select template
        template_type = random.choice(  # nosec B311
            ["standard", "metacognitive", "explicit_uncertainty"]
        )
        template = random.choice(self._templates[template_type])  # nosec B311

        question_text = template.format(question=q.question)

        # Determine correct answer
        if q.has_definite_answer and q.correct_answer:
            correct_answer = q.correct_answer
        else:
            # For unanswerable questions, the "correct" response is acknowledging uncertainty
            correct_answer = self._get_expected_response_text(q.expected_response)

        return Task(
            id=uuid4(),
            track=TrackType.KNOWLEDGE_BOUNDARY,
            question=question_text,
            correct_answer=correct_answer,
            answer_type=AnswerType.FREE_FORM,
            difficulty=q.difficulty,
            category=self._map_domain_to_category(q.domain),
            metadata=TaskMetadata(
                source="knowledge_boundary_generator",
                tags=[
                    f"domain:{q.domain}",
                    f"knowledge_category:{q.category.value}",
                    f"expected_response:{q.expected_response.value}",
                    f"has_definite_answer:{q.has_definite_answer}",
                ],
                notes=q.explanation,
            ),
        )

    def _map_domain_to_category(self, domain: str) -> TaskCategory:
        """Map domain string to TaskCategory enum.

        Args:
            domain: Domain string from question

        Returns:
            Corresponding TaskCategory
        """
        domain_mapping: dict[str, TaskCategory] = {
            "geography": TaskCategory.FACTUAL,
            "literature": TaskCategory.FACTUAL,
            "chemistry": TaskCategory.SCIENTIFIC,
            "history": TaskCategory.HISTORICAL,
            "politics": TaskCategory.FACTUAL,
            "science": TaskCategory.SCIENTIFIC,
            "finance": TaskCategory.NUMERICAL,
            "current_events": TaskCategory.FACTUAL,
            "entertainment": TaskCategory.FACTUAL,
            "philosophy": TaskCategory.REASONING,
            "general": TaskCategory.COMMONSENSE,
            "mathematics": TaskCategory.NUMERICAL,
            "mythology": TaskCategory.FACTUAL,
        }
        return domain_mapping.get(domain, TaskCategory.FACTUAL)

    def _get_expected_response_text(self, expected: ExpectedResponse) -> str:
        """Get expected response text for evaluation."""
        mapping = {
            ExpectedResponse.CONFIDENT_ANSWER: "confident_answer",
            ExpectedResponse.EXPRESS_UNCERTAINTY: "uncertainty_expressed",
            ExpectedResponse.DECLINE_TO_ANSWER: "declined_to_answer",
            ExpectedResponse.ASK_CLARIFICATION: "asked_for_clarification",
            ExpectedResponse.ACKNOWLEDGE_UNKNOWABLE: "acknowledged_unknowable",
        }
        return mapping.get(expected, "unknown")

    def get_statistics(self, tasks: TaskSet) -> dict[str, Any]:
        """Get generator statistics.

        Args:
            tasks: The TaskSet to analyze (also includes internal questions)

        Returns:
            Dictionary of statistics
        """
        # Get base statistics from parent
        base_stats = super().get_statistics(tasks)

        # Add knowledge-boundary-specific statistics
        category_counts: dict[str, int] = {}
        for q in self._questions:
            category_counts[q.category.value] = category_counts.get(q.category.value, 0) + 1

        answerable = sum(1 for q in self._questions if q.has_definite_answer)

        base_stats.update(
            {
                "question_pool_size": len(self._questions),
                "question_category_counts": category_counts,
                "answerable_count": answerable,
                "unanswerable_count": len(self._questions) - answerable,
                "answerable_ratio": answerable / len(self._questions) if self._questions else 0,
            }
        )

        return base_stats
