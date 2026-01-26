"""Task data models for CHIMERA benchmark.

This module defines the core task structures used across all evaluation tracks:
- Task: Individual evaluation task with question, answer, and metadata
- TaskSet: Collection of tasks for batch processing
- TaskMetadata: Rich metadata for task categorization and analysis
"""

from collections.abc import Iterator
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrackType(str, Enum):
    """Evaluation track types in CHIMERA."""

    CALIBRATION = "calibration"
    ERROR_DETECTION = "error_detection"
    KNOWLEDGE_BOUNDARY = "knowledge_boundary"
    SELF_CORRECTION = "self_correction"


class DifficultyLevel(str, Enum):
    """Task difficulty levels for stratified analysis."""

    L1 = "L1"  # Easy - clear factual questions
    L2 = "L2"  # Medium - 2-3 step reasoning
    L3 = "L3"  # Hard - knowledge boundary edge cases
    L4 = "L4"  # Very Hard - adversarial framing
    L5 = "L5"  # Expert - multi-domain integration


class TaskCategory(str, Enum):
    """Task content categories for analysis."""

    FACTUAL = "factual"
    REASONING = "reasoning"
    NUMERICAL = "numerical"
    COMMONSENSE = "commonsense"
    SCIENTIFIC = "scientific"
    HISTORICAL = "historical"
    LINGUISTIC = "linguistic"
    LOGICAL = "logical"
    UNANSWERABLE = "unanswerable"
    OBSCURE = "obscure"


class AnswerType(str, Enum):
    """Types of expected answers."""

    EXACT_MATCH = "exact_match"  # Must match exactly
    CONTAINS = "contains"  # Must contain the answer
    NUMERIC = "numeric"  # Numeric comparison with tolerance
    MULTIPLE_CHOICE = "multiple_choice"  # One of predefined options
    BOOLEAN = "boolean"  # Yes/No, True/False
    FREE_FORM = "free_form"  # Open-ended, requires semantic evaluation
    ABSTENTION = "abstention"  # Expected to abstain/refuse


class TaskMetadata(BaseModel):
    """Rich metadata for task categorization and analysis.

    Attributes:
        source: Origin of the task (e.g., "synthetic", "wikidata", "curated")
        category: Content category for analysis
        difficulty: Difficulty level for stratification
        tags: Additional tags for filtering
        verified: Whether the task has been human-verified
        created_at: Task creation timestamp
        version: Task version for tracking updates
        notes: Optional notes about the task
    """

    model_config = ConfigDict(use_enum_values=True)

    source: str = Field(
        default="synthetic",
        description="Origin of the task data",
    )
    category: TaskCategory = Field(
        default=TaskCategory.FACTUAL,
        description="Content category for analysis",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.L2,
        description="Difficulty level for stratified analysis",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Additional tags for filtering",
    )
    verified: bool = Field(
        default=False,
        description="Whether the task has been human-verified",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Task creation timestamp",
    )
    version: str = Field(
        default="1.0.0",
        description="Task version for tracking updates",
    )
    notes: str | None = Field(
        default=None,
        description="Optional notes about the task",
    )


class Task(BaseModel):
    """Individual evaluation task for CHIMERA benchmark.

    A Task represents a single evaluation item with a question, correct answer,
    and associated metadata. Tasks are the atomic units of evaluation across
    all CHIMERA tracks.

    Attributes:
        id: Unique task identifier (UUID)
        track: Which evaluation track this task belongs to
        question: The question or prompt to present to the model
        correct_answer: The ground truth answer
        answer_type: Type of answer for evaluation matching
        acceptable_answers: Alternative acceptable answers
        difficulty: Difficulty level for stratification
        category: Content category
        metadata: Rich metadata for analysis
        context: Optional additional context for the question
        is_answerable: Whether the question has a definitive answer
        expected_abstention: Whether the model should abstain

    Example:
        >>> task = Task(
        ...     track=TrackType.CALIBRATION,
        ...     question="What is the capital of France?",
        ...     correct_answer="Paris",
        ...     answer_type=AnswerType.EXACT_MATCH,
        ...     difficulty=DifficultyLevel.L1,
        ... )
    """

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique task identifier",
    )
    track: TrackType = Field(
        description="Evaluation track this task belongs to",
    )
    question: str = Field(
        min_length=1,
        description="The question or prompt to present to the model",
    )
    correct_answer: str | None = Field(
        default=None,
        description="The ground truth answer (None for unanswerable)",
    )
    answer_type: AnswerType = Field(
        default=AnswerType.EXACT_MATCH,
        description="Type of answer for evaluation matching",
    )
    acceptable_answers: list[str] = Field(
        default_factory=list,
        description="Alternative acceptable answers",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.L2,
        description="Difficulty level",
    )
    category: TaskCategory = Field(
        default=TaskCategory.FACTUAL,
        description="Content category",
    )
    metadata: TaskMetadata = Field(
        default_factory=TaskMetadata,
        description="Rich metadata for analysis",
    )
    context: str | None = Field(
        default=None,
        description="Optional additional context for the question",
    )
    is_answerable: bool = Field(
        default=True,
        description="Whether the question has a definitive answer",
    )
    expected_abstention: bool = Field(
        default=False,
        description="Whether the model should abstain from answering",
    )

    # Track-specific fields
    # For error_detection track
    initial_response: str | None = Field(
        default=None,
        description="Initial (potentially incorrect) response for error detection",
    )
    error_locations: list[str] = Field(
        default_factory=list,
        description="Locations of errors in initial_response",
    )

    # For self_correction track
    reasoning_chain: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning chain",
    )
    perturbation_index: int | None = Field(
        default=None,
        description="Index of perturbed step in reasoning_chain",
    )
    perturbation_type: str | None = Field(
        default=None,
        description="Type of perturbation applied",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Ensure question is not just whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()

    def get_all_acceptable_answers(self) -> list[str]:
        """Get all acceptable answers including the primary correct answer.

        Returns:
            List of all acceptable answer strings.
        """
        answers = []
        if self.correct_answer:
            answers.append(self.correct_answer)
        answers.extend(self.acceptable_answers)
        return answers

    def is_track(self, track: TrackType) -> bool:
        """Check if this task belongs to a specific track.

        Args:
            track: The track type to check.

        Returns:
            True if this task belongs to the specified track.
        """
        return self.track == track


class TaskSet(BaseModel):
    """Collection of tasks for batch processing.

    A TaskSet groups related tasks together with shared metadata,
    enabling efficient batch processing and organized evaluation.

    Attributes:
        id: Unique identifier for the task set
        name: Human-readable name for the task set
        description: Description of the task set contents
        track: Primary track for this task set
        tasks: List of tasks in this set
        created_at: Creation timestamp
        version: Version for tracking updates
        tags: Tags for filtering and organization
    """

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the task set",
    )
    name: str = Field(
        min_length=1,
        description="Human-readable name for the task set",
    )
    description: str = Field(
        default="",
        description="Description of the task set contents",
    )
    track: TrackType = Field(
        description="Primary track for this task set",
    )
    tasks: list[Task] = Field(
        default_factory=list,
        description="List of tasks in this set",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    version: str = Field(
        default="1.0.0",
        description="Version for tracking updates",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering and organization",
    )

    def __len__(self) -> int:
        """Return the number of tasks in the set."""
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:  # type: ignore[override]
        """Iterate over tasks in the set."""
        return iter(self.tasks)

    def __getitem__(self, index: int) -> Task:
        """Get a task by index."""
        return self.tasks[index]

    def add_task(self, task: Task) -> None:
        """Add a task to the set.

        Args:
            task: The task to add.
        """
        self.tasks.append(task)

    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> list[Task]:
        """Filter tasks by difficulty level.

        Args:
            difficulty: The difficulty level to filter by.

        Returns:
            List of tasks matching the difficulty level.
        """
        return [t for t in self.tasks if t.difficulty == difficulty]

    def filter_by_category(self, category: TaskCategory) -> list[Task]:
        """Filter tasks by category.

        Args:
            category: The category to filter by.

        Returns:
            List of tasks matching the category.
        """
        return [t for t in self.tasks if t.category == category]

    def get_difficulty_distribution(self) -> dict[str, int]:
        """Get the distribution of tasks across difficulty levels.

        Returns:
            Dictionary mapping difficulty levels to task counts.
        """
        distribution: dict[str, int] = {}
        for task in self.tasks:
            level = task.difficulty if isinstance(task.difficulty, str) else task.difficulty.value
            distribution[level] = distribution.get(level, 0) + 1
        return distribution

    def get_category_distribution(self) -> dict[str, int]:
        """Get the distribution of tasks across categories.

        Returns:
            Dictionary mapping categories to task counts.
        """
        distribution: dict[str, int] = {}
        for task in self.tasks:
            cat = task.category if isinstance(task.category, str) else task.category.value
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution

    def to_jsonl(self) -> str:
        """Serialize task set to JSONL format.

        Returns:
            JSONL string with one task per line.
        """
        lines = [task.model_dump_json() for task in self.tasks]
        return "\n".join(lines)

    @classmethod
    def from_jsonl(
        cls,
        jsonl_content: str,
        name: str,
        track: TrackType,
        description: str = "",
    ) -> "TaskSet":
        """Create a TaskSet from JSONL content.

        Args:
            jsonl_content: JSONL string with one task per line.
            name: Name for the task set.
            track: Primary track for the task set.
            description: Optional description.

        Returns:
            TaskSet containing the parsed tasks.
        """
        import json

        tasks = []
        for line in jsonl_content.strip().split("\n"):
            if line.strip():
                task_data = json.loads(line)
                tasks.append(Task(**task_data))

        return cls(
            name=name,
            track=track,
            description=description,
            tasks=tasks,
        )
