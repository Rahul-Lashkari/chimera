"""Self-correction task generator for CHIMERA benchmark.

This module generates tasks that test a model's ability to detect and correct
corrupted reasoning in its own responses.

Self-Correction Track Design:
1. Present a problem with an initial reasoning trace (correct or corrupted)
2. Model must identify whether the reasoning is valid
3. If corrupted, model must correct the reasoning
4. Measures model's ability to self-correct under perturbation

Corruption Types:
- LOGICAL: Invalid logical steps or conclusions
- COMPUTATIONAL: Mathematical or calculation errors
- FACTUAL: Incorrect facts within reasoning
- PROCEDURAL: Wrong order of steps or missing steps
- PREMISE: Incorrect starting assumptions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

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


class CorruptionType(str, Enum):
    """Types of reasoning corruption."""

    # Invalid logical steps or conclusions
    LOGICAL = "logical"

    # Mathematical or calculation errors
    COMPUTATIONAL = "computational"

    # Incorrect facts within reasoning
    FACTUAL = "factual"

    # Wrong order of steps or missing steps
    PROCEDURAL = "procedural"

    # Incorrect starting assumptions
    PREMISE = "premise"


class CorrectionExpectation(str, Enum):
    """Expected model response for correction tasks."""

    # Model should validate correct reasoning
    VALIDATE_CORRECT = "validate_correct"

    # Model should detect and correct the error
    DETECT_AND_CORRECT = "detect_and_correct"

    # Model should identify uncorrectable reasoning
    IDENTIFY_UNCORRECTABLE = "identify_uncorrectable"


@dataclass
class ReasoningTrace:
    """A reasoning trace that may contain corruption.

    Attributes:
        problem: The original problem statement
        steps: List of reasoning steps
        conclusion: The final conclusion
        is_corrupted: Whether the trace contains corruption
        corruption_type: Type of corruption if any
        corruption_location: Index of corrupted step
        corruption_description: Description of what's wrong
        correct_steps: The correct version of steps
        correct_conclusion: The correct conclusion
    """

    problem: str
    steps: list[str]
    conclusion: str
    is_corrupted: bool = False
    corruption_type: CorruptionType | None = None
    corruption_location: int | None = None
    corruption_description: str = ""
    correct_steps: list[str] | None = None
    correct_conclusion: str | None = None
    difficulty: DifficultyLevel = DifficultyLevel.L3
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


class SelfCorrectionGeneratorConfig(GeneratorConfig):
    """Configuration for self-correction task generator.

    Attributes:
        corruption_rate: Proportion of tasks with corrupted reasoning
        corruption_type_distribution: Distribution of corruption types
        include_correction_prompt: Whether to ask for correction
        require_explanation: Whether to require explanation of error
        seed_data_path: Path to seed data files
        shuffle: Whether to shuffle generated tasks
    """

    corruption_rate: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Proportion of tasks with corrupted reasoning",
    )

    corruption_type_distribution: dict[CorruptionType, float] = Field(
        default_factory=lambda: {
            CorruptionType.LOGICAL: 0.30,
            CorruptionType.COMPUTATIONAL: 0.25,
            CorruptionType.FACTUAL: 0.20,
            CorruptionType.PROCEDURAL: 0.15,
            CorruptionType.PREMISE: 0.10,
        }
    )

    include_correction_prompt: bool = Field(
        default=True,
        description="Whether to ask for correction of errors",
    )

    require_explanation: bool = Field(
        default=True,
        description="Whether to require explanation of identified errors",
    )

    seed_data_path: Path | None = Field(
        default=None,
        description="Path to seed data files",
    )

    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle generated tasks",
    )


class SelfCorrectionTaskGenerator(BaseTaskGenerator):
    """Generator for self-correction tasks.

    This generator creates tasks that test a model's ability to
    identify and correct corrupted reasoning traces.

    Example:
        >>> config = SelfCorrectionGeneratorConfig(n_tasks=100, seed=42)
        >>> generator = SelfCorrectionTaskGenerator(config)
        >>> task_set = generator.generate()
        >>> print(f"Generated {len(task_set.tasks)} tasks")
    """

    config: SelfCorrectionGeneratorConfig  # Type narrowing

    def __init__(
        self,
        config: SelfCorrectionGeneratorConfig | None = None,
    ) -> None:
        """Initialize the self-correction task generator.

        Args:
            config: Generator configuration
        """
        effective_config = config or SelfCorrectionGeneratorConfig()
        super().__init__(effective_config, TrackType.SELF_CORRECTION)
        self._reasoning_traces: list[ReasoningTrace] = []
        self._templates = self._load_templates()

    def _load_templates(self) -> dict[str, list[str]]:
        """Load prompt templates for self-correction tasks."""
        return {
            "validation": [
                (
                    "Problem: {problem}\n\n"
                    "Reasoning:\n{reasoning}\n\n"
                    "Conclusion: {conclusion}\n\n"
                    "Is the reasoning above correct? If not, identify and explain the error."
                ),
                (
                    "Consider the following problem and solution:\n\n"
                    "Problem: {problem}\n\n"
                    "Solution Steps:\n{reasoning}\n\n"
                    "Final Answer: {conclusion}\n\n"
                    "Review this solution. Is it correct? If there are any errors, "
                    "explain what is wrong and provide the correct solution."
                ),
            ],
            "correction": [
                (
                    "Problem: {problem}\n\n"
                    "A previous attempt at solving this produced:\n{reasoning}\n\n"
                    "With conclusion: {conclusion}\n\n"
                    "Please review this reasoning. If you find any errors:\n"
                    "1. Identify exactly where the error occurs\n"
                    "2. Explain why it is wrong\n"
                    "3. Provide the corrected reasoning and answer"
                ),
                (
                    "I tried to solve this problem:\n{problem}\n\n"
                    "My reasoning was:\n{reasoning}\n\n"
                    "And I concluded: {conclusion}\n\n"
                    "Can you check my work? Point out any mistakes and "
                    "show me the correct approach."
                ),
            ],
            "metacognitive": [
                (
                    "Problem: {problem}\n\n"
                    "Here is a reasoning trace:\n{reasoning}\n\n"
                    "Leading to: {conclusion}\n\n"
                    "Before accepting this answer, consider:\n"
                    "1. Is each step logically valid?\n"
                    "2. Are all calculations correct?\n"
                    "3. Are the premises accurate?\n"
                    "4. Is the conclusion properly supported?\n\n"
                    "Provide your assessment and any corrections needed."
                ),
            ],
        }

    def load_seed_data(self, path: Path | None = None) -> None:
        """Load reasoning traces from seed data files.

        Args:
            path: Path to seed data (uses config path if None)
        """
        import json

        data_path = path or self.config.seed_data_path
        if data_path is None:
            self._reasoning_traces = self._get_default_seed_data()
            return

        if data_path.is_file():
            with open(data_path) as f:
                data = json.load(f)
        elif data_path.is_dir():
            data = []
            for file in data_path.glob("*.json"):
                with open(file) as f:
                    file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
        else:
            self._reasoning_traces = self._get_default_seed_data()
            return

        self._reasoning_traces = []
        for item in data:
            trace = ReasoningTrace(
                problem=item.get("problem", ""),
                steps=item.get("steps", []),
                conclusion=item.get("conclusion", ""),
                is_corrupted=item.get("is_corrupted", False),
                corruption_type=(
                    CorruptionType(item["corruption_type"]) if item.get("corruption_type") else None
                ),
                corruption_location=item.get("corruption_location"),
                corruption_description=item.get("corruption_description", ""),
                correct_steps=item.get("correct_steps"),
                correct_conclusion=item.get("correct_conclusion"),
                difficulty=DifficultyLevel(item.get("difficulty", "L3")),
                domain=item.get("domain", "general"),
            )
            self._reasoning_traces.append(trace)

    def _get_default_seed_data(self) -> list[ReasoningTrace]:
        """Generate default seed data for self-correction tasks."""
        traces = []

        # Mathematical reasoning - correct
        traces.append(
            ReasoningTrace(
                problem="Calculate the area of a rectangle with length 8 and width 5.",
                steps=[
                    "The formula for area of a rectangle is: Area = length × width",
                    "Substituting values: Area = 8 × 5",
                    "Calculating: Area = 40",
                ],
                conclusion="The area is 40 square units.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L1,
                domain="mathematics",
            )
        )

        # Mathematical reasoning - corrupted (computational error)
        traces.append(
            ReasoningTrace(
                problem="Calculate the area of a rectangle with length 8 and width 5.",
                steps=[
                    "The formula for area of a rectangle is: Area = length × width",
                    "Substituting values: Area = 8 × 5",
                    "Calculating: Area = 35",  # Error: should be 40
                ],
                conclusion="The area is 35 square units.",
                is_corrupted=True,
                corruption_type=CorruptionType.COMPUTATIONAL,
                corruption_location=2,
                corruption_description="Multiplication error: 8 × 5 = 40, not 35",
                correct_steps=[
                    "The formula for area of a rectangle is: Area = length × width",
                    "Substituting values: Area = 8 × 5",
                    "Calculating: Area = 40",
                ],
                correct_conclusion="The area is 40 square units.",
                difficulty=DifficultyLevel.L2,
                domain="mathematics",
            )
        )

        # Logical reasoning - correct
        traces.append(
            ReasoningTrace(
                problem=(
                    "All mammals are warm-blooded. Whales are mammals. "
                    "What can we conclude about whales?"
                ),
                steps=[
                    "Premise 1: All mammals are warm-blooded.",
                    "Premise 2: Whales are mammals.",
                    "By syllogistic reasoning: Since whales belong to the category 'mammals', "
                    "and all members of 'mammals' are warm-blooded...",
                ],
                conclusion="Therefore, whales are warm-blooded.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L2,
                domain="logic",
            )
        )

        # Logical reasoning - corrupted (logical fallacy)
        traces.append(
            ReasoningTrace(
                problem=(
                    "All birds can fly. Penguins are birds. " "What can we conclude about penguins?"
                ),
                steps=[
                    "Premise 1: All birds can fly.",  # False premise
                    "Premise 2: Penguins are birds.",
                    "Applying the logic: Since penguins are birds, and all birds can fly...",
                ],
                conclusion="Therefore, penguins can fly.",
                is_corrupted=True,
                corruption_type=CorruptionType.PREMISE,
                corruption_location=0,
                corruption_description=(
                    "The first premise is false - not all birds can fly "
                    "(e.g., penguins, ostriches, kiwis)"
                ),
                correct_steps=[
                    "Premise 1: Most birds can fly, but there are exceptions.",
                    "Premise 2: Penguins are birds.",
                    "Penguins are one of the flightless bird species.",
                ],
                correct_conclusion="Penguins are birds but cannot fly.",
                difficulty=DifficultyLevel.L3,
                domain="logic",
            )
        )

        # Procedural reasoning - correct
        traces.append(
            ReasoningTrace(
                problem="Solve for x: 2x + 5 = 13",
                steps=[
                    "Start with equation: 2x + 5 = 13",
                    "Subtract 5 from both sides: 2x + 5 - 5 = 13 - 5",
                    "Simplify: 2x = 8",
                    "Divide both sides by 2: 2x/2 = 8/2",
                    "Simplify: x = 4",
                ],
                conclusion="x = 4",
                is_corrupted=False,
                difficulty=DifficultyLevel.L2,
                domain="algebra",
            )
        )

        # Procedural reasoning - corrupted (wrong order)
        traces.append(
            ReasoningTrace(
                problem="Solve for x: 2x + 5 = 13",
                steps=[
                    "Start with equation: 2x + 5 = 13",
                    "Divide both sides by 2: (2x + 5)/2 = 13/2",  # Wrong order
                    "Simplify: x + 2.5 = 6.5",
                    "Subtract 2.5: x = 4",  # Happens to be correct answer
                ],
                conclusion="x = 4",
                is_corrupted=True,
                corruption_type=CorruptionType.PROCEDURAL,
                corruption_location=1,
                corruption_description=(
                    "The standard procedure is to isolate the variable term first "
                    "(subtract 5), then divide. The method shown is unnecessarily complex."
                ),
                correct_steps=[
                    "Start with equation: 2x + 5 = 13",
                    "Subtract 5 from both sides: 2x = 8",
                    "Divide both sides by 2: x = 4",
                ],
                correct_conclusion="x = 4",
                difficulty=DifficultyLevel.L3,
                domain="algebra",
            )
        )

        # Factual reasoning - corrupted
        traces.append(
            ReasoningTrace(
                problem="What is the capital of Australia and why was it chosen?",
                steps=[
                    "Australia is a country in the Southern Hemisphere.",
                    "The capital of Australia is Sydney.",  # Error: it's Canberra
                    "Sydney was chosen because it is the largest city.",
                ],
                conclusion="Sydney is the capital of Australia, chosen for its size.",
                is_corrupted=True,
                corruption_type=CorruptionType.FACTUAL,
                corruption_location=1,
                corruption_description=(
                    "The capital of Australia is Canberra, not Sydney. "
                    "Canberra was chosen as a compromise between Sydney and Melbourne."
                ),
                correct_steps=[
                    "Australia is a country in the Southern Hemisphere.",
                    "The capital of Australia is Canberra.",
                    "Canberra was chosen as a compromise between rival cities Sydney and Melbourne.",
                ],
                correct_conclusion=(
                    "Canberra is the capital of Australia, chosen as a compromise location."
                ),
                difficulty=DifficultyLevel.L2,
                domain="geography",
            )
        )

        # Complex mathematical reasoning - correct
        traces.append(
            ReasoningTrace(
                problem=(
                    "A store has a 25% off sale. If an item originally costs $80, "
                    "what is the sale price?"
                ),
                steps=[
                    "Original price: $80",
                    "Discount percentage: 25%",
                    "Calculate discount amount: $80 × 0.25 = $20",
                    "Subtract discount from original: $80 - $20 = $60",
                ],
                conclusion="The sale price is $60.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L2,
                domain="mathematics",
            )
        )

        # Complex mathematical reasoning - corrupted
        traces.append(
            ReasoningTrace(
                problem=(
                    "A store has a 25% off sale. If an item originally costs $80, "
                    "what is the sale price?"
                ),
                steps=[
                    "Original price: $80",
                    "Discount percentage: 25%",
                    "Calculate discount amount: $80 × 25 = $2000",  # Error: forgot to convert %
                    "This seems too high, so the sale price must be $80 - $25 = $55",  # Wrong
                ],
                conclusion="The sale price is $55.",
                is_corrupted=True,
                corruption_type=CorruptionType.COMPUTATIONAL,
                corruption_location=2,
                corruption_description=(
                    "The percentage was not properly converted. 25% = 0.25, "
                    "so the discount is $80 × 0.25 = $20, making the sale price $60."
                ),
                correct_steps=[
                    "Original price: $80",
                    "Discount percentage: 25% = 0.25",
                    "Calculate discount amount: $80 × 0.25 = $20",
                    "Sale price: $80 - $20 = $60",
                ],
                correct_conclusion="The sale price is $60.",
                difficulty=DifficultyLevel.L3,
                domain="mathematics",
            )
        )

        # Scientific reasoning - correct
        traces.append(
            ReasoningTrace(
                problem="Why do ice cubes float in water?",
                steps=[
                    "Ice is the solid form of water (H2O).",
                    "When water freezes, its molecules form a crystalline structure.",
                    "This crystalline structure spaces molecules farther apart than in liquid water.",
                    "Because the molecules are more spread out, ice has lower density than liquid water.",
                    "Objects with lower density than a fluid will float in that fluid.",
                ],
                conclusion="Ice floats because it is less dense than liquid water.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L3,
                domain="physics",
            )
        )

        # Scientific reasoning - corrupted (logical error)
        traces.append(
            ReasoningTrace(
                problem="Why does metal feel colder than wood at room temperature?",
                steps=[
                    "Both metal and wood are at room temperature (same temperature).",
                    "Metal has a higher thermal conductivity than wood.",
                    "When you touch metal, it absorbs heat from your hand quickly.",
                    "Since metal is colder, it feels cold to the touch.",  # Logical error
                ],
                conclusion="Metal feels colder because it has a lower temperature than wood.",
                is_corrupted=True,
                corruption_type=CorruptionType.LOGICAL,
                corruption_location=3,
                corruption_description=(
                    "The conclusion contradicts the first premise. Metal is NOT colder - "
                    "it's the same temperature. It FEELS colder because it conducts "
                    "heat away from your hand faster."
                ),
                correct_steps=[
                    "Both metal and wood are at room temperature (same temperature).",
                    "Metal has a higher thermal conductivity than wood.",
                    "When you touch metal, it absorbs heat from your hand quickly.",
                    "This rapid heat transfer makes metal feel colder, even though it's the same temperature.",
                ],
                correct_conclusion=(
                    "Metal feels colder because it conducts heat away from your hand faster, "
                    "not because it's at a lower temperature."
                ),
                difficulty=DifficultyLevel.L4,
                domain="physics",
            )
        )

        # Add more traces to reach reasonable diversity
        traces.extend(self._generate_additional_traces())

        return traces

    def _generate_additional_traces(self) -> list[ReasoningTrace]:
        """Generate additional reasoning traces for variety."""
        additional = []

        # Probability reasoning - correct
        additional.append(
            ReasoningTrace(
                problem="What is the probability of getting heads twice when flipping a fair coin twice?",
                steps=[
                    "Each coin flip is an independent event.",
                    "P(heads on one flip) = 1/2",
                    "P(heads twice) = P(first heads) × P(second heads)",
                    "P(heads twice) = 1/2 × 1/2 = 1/4",
                ],
                conclusion="The probability of getting heads twice is 1/4 or 25%.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L2,
                domain="probability",
            )
        )

        # Probability reasoning - corrupted
        additional.append(
            ReasoningTrace(
                problem="What is the probability of getting heads twice when flipping a fair coin twice?",
                steps=[
                    "Each coin flip is an independent event.",
                    "There are 4 possible outcomes: HH, HT, TH, TT",
                    "Only one outcome has heads both times: HH",
                    "But since we want heads, P = 2/4 = 1/2",  # Error: counting heads, not HH
                ],
                conclusion="The probability is 1/2 or 50%.",
                is_corrupted=True,
                corruption_type=CorruptionType.LOGICAL,
                corruption_location=3,
                corruption_description=(
                    "The reasoning conflates 'getting heads at least once' with "
                    "'getting heads both times'. Only HH satisfies the condition, so P = 1/4."
                ),
                correct_steps=[
                    "There are 4 equally likely outcomes: HH, HT, TH, TT",
                    "Only HH has heads on both flips.",
                    "P(heads twice) = 1/4",
                ],
                correct_conclusion="The probability is 1/4 or 25%.",
                difficulty=DifficultyLevel.L3,
                domain="probability",
            )
        )

        # Geometry - correct
        additional.append(
            ReasoningTrace(
                problem="Find the circumference of a circle with radius 7 cm.",
                steps=[
                    "The formula for circumference is C = 2πr",
                    "Given radius r = 7 cm",
                    "Substituting: C = 2 × π × 7",
                    "C = 14π ≈ 43.98 cm",
                ],
                conclusion="The circumference is approximately 44 cm.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L1,
                domain="geometry",
            )
        )

        # Geometry - corrupted (formula error)
        additional.append(
            ReasoningTrace(
                problem="Find the circumference of a circle with radius 7 cm.",
                steps=[
                    "The formula for circumference is C = πr²",  # Wrong: this is area formula part
                    "Given radius r = 7 cm",
                    "Substituting: C = π × 7²",
                    "C = 49π ≈ 153.94 cm",
                ],
                conclusion="The circumference is approximately 154 cm.",
                is_corrupted=True,
                corruption_type=CorruptionType.FACTUAL,
                corruption_location=0,
                corruption_description=(
                    "Wrong formula used. C = πr² is part of the area formula. "
                    "Circumference formula is C = 2πr."
                ),
                correct_steps=[
                    "The formula for circumference is C = 2πr",
                    "Given radius r = 7 cm",
                    "Substituting: C = 2 × π × 7 = 14π",
                    "C ≈ 43.98 cm",
                ],
                correct_conclusion="The circumference is approximately 44 cm.",
                difficulty=DifficultyLevel.L2,
                domain="geometry",
            )
        )

        # Logic puzzle - correct
        additional.append(
            ReasoningTrace(
                problem=(
                    "If it rains, the ground gets wet. The ground is wet. "
                    "Can we conclude it rained?"
                ),
                steps=[
                    "We have: If rain → wet ground (P → Q)",
                    "We observe: wet ground (Q)",
                    "Can we conclude rain (P)?",
                    "This is the logical fallacy of 'affirming the consequent'.",
                    "The ground could be wet for other reasons (sprinkler, spill, etc.).",
                ],
                conclusion="No, we cannot conclude it rained. This would be a logical fallacy.",
                is_corrupted=False,
                difficulty=DifficultyLevel.L4,
                domain="logic",
            )
        )

        # Logic puzzle - corrupted
        additional.append(
            ReasoningTrace(
                problem=(
                    "If it rains, the ground gets wet. The ground is wet. "
                    "Can we conclude it rained?"
                ),
                steps=[
                    "We have: If rain → wet ground",
                    "We observe: wet ground",
                    "Since rain causes wet ground, and the ground is wet...",
                    "The rain must have caused it.",
                ],
                conclusion="Yes, it must have rained because the ground is wet.",
                is_corrupted=True,
                corruption_type=CorruptionType.LOGICAL,
                corruption_location=3,
                corruption_description=(
                    "This commits the fallacy of affirming the consequent. "
                    "From 'P → Q' and 'Q', we cannot conclude 'P'."
                ),
                correct_steps=[
                    "We have: If rain → wet ground (P → Q)",
                    "We observe: wet ground (Q)",
                    "Affirming the consequent: From Q, we cannot conclude P",
                    "Other causes could explain wet ground.",
                ],
                correct_conclusion=(
                    "No, we cannot conclude it rained. This is the fallacy of "
                    "affirming the consequent."
                ),
                difficulty=DifficultyLevel.L4,
                domain="logic",
            )
        )

        return additional

    def generate_task(
        self,
        difficulty: DifficultyLevel,
        category: TaskCategory,
    ) -> Task:
        """Generate a single self-correction task.

        Args:
            difficulty: The difficulty level for the task.
            category: The category for the task.

        Returns:
            A generated Task.
        """
        if not self._reasoning_traces:
            self.load_seed_data()

        # Find matching traces
        matching = [t for t in self._reasoning_traces if t.difficulty == difficulty]
        if not matching:
            matching = self._reasoning_traces

        trace = random.choice(matching)  # nosec B311
        return self._create_task(trace)

    def generate_batch(
        self,
        n_tasks: int,
        difficulty: DifficultyLevel | None = None,
        category: TaskCategory | None = None,
    ) -> list[Task]:
        """Generate a batch of self-correction tasks.

        Args:
            n_tasks: Number of tasks to generate.
            difficulty: Optional specific difficulty level.
            category: Optional specific category.

        Returns:
            List of generated Tasks.
        """
        if not self._reasoning_traces:
            self.load_seed_data()

        tasks: list[Task] = []
        pool = self._reasoning_traces

        if difficulty:
            filtered = [t for t in pool if t.difficulty == difficulty]
            pool = filtered if filtered else pool

        for _ in range(n_tasks):
            trace = random.choice(pool)  # nosec B311
            tasks.append(self._create_task(trace))

        return tasks

    def generate(self) -> TaskSet:
        """Generate a set of self-correction tasks.

        Returns:
            TaskSet containing self-correction tasks
        """
        if not self._reasoning_traces:
            self.load_seed_data()

        tasks: list[Task] = []

        # Separate traces by corruption status
        corrupted_traces = [t for t in self._reasoning_traces if t.is_corrupted]
        correct_traces = [t for t in self._reasoning_traces if not t.is_corrupted]

        # Calculate target counts
        n_corrupted = int(self.config.n_tasks * self.config.corruption_rate)
        n_correct = self.config.n_tasks - n_corrupted

        # Generate corrupted tasks
        for _ in range(n_corrupted):
            if corrupted_traces:
                trace = random.choice(corrupted_traces)  # nosec B311
                tasks.append(self._create_task(trace))
            elif self._reasoning_traces:
                # Create corruption if no corrupted traces available
                trace = random.choice(self._reasoning_traces)  # nosec B311
                corrupted_trace = self._inject_corruption(trace)
                tasks.append(self._create_task(corrupted_trace))

        # Generate correct tasks
        for _ in range(n_correct):
            if correct_traces:
                trace = random.choice(correct_traces)  # nosec B311
                tasks.append(self._create_task(trace))
            elif self._reasoning_traces:
                trace = random.choice(self._reasoning_traces)  # nosec B311
                # Use trace as-is if no uncorrupted available
                if not trace.is_corrupted:
                    tasks.append(self._create_task(trace))

        # Fill remaining if needed
        while len(tasks) < self.config.n_tasks and self._reasoning_traces:
            trace = random.choice(self._reasoning_traces)  # nosec B311
            tasks.append(self._create_task(trace))

        # Shuffle
        if self.config.shuffle:
            random.shuffle(tasks)  # nosec B311

        return TaskSet(
            name=f"self_correction_{self.config.seed}",
            description="Self-correction benchmark tasks",
            track=TrackType.SELF_CORRECTION,
            tasks=tasks,
            tags=[
                f"corruption_rate:{self.config.corruption_rate}",
                "self_correction",
            ],
        )

    def _create_task(self, trace: ReasoningTrace) -> Task:
        """Create a task from a reasoning trace.

        Args:
            trace: The reasoning trace to convert.

        Returns:
            Task instance
        """
        # Format reasoning steps
        reasoning = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(trace.steps))

        # Select template
        if self.config.include_correction_prompt and trace.is_corrupted:
            template_type = "correction"
        elif self.config.require_explanation:
            template_type = "metacognitive"
        else:
            template_type = "validation"

        templates = self._templates.get(template_type, self._templates["validation"])
        template = random.choice(templates)  # nosec B311

        # Format question
        question = template.format(
            problem=trace.problem,
            reasoning=reasoning,
            conclusion=trace.conclusion,
        )

        # Determine expected response
        if trace.is_corrupted:
            expected = CorrectionExpectation.DETECT_AND_CORRECT
            correct_answer = (
                f"Error found in step {(trace.corruption_location or 0) + 1}: "
                f"{trace.corruption_description}\n"
                f"Correct answer: {trace.correct_conclusion}"
            )
        else:
            expected = CorrectionExpectation.VALIDATE_CORRECT
            correct_answer = f"The reasoning is correct. {trace.conclusion}"

        return Task(
            track=TrackType.SELF_CORRECTION,
            question=question,
            correct_answer=correct_answer,
            answer_type=AnswerType.FREE_FORM,
            difficulty=trace.difficulty,
            category=self._domain_to_category(trace.domain),
            metadata=TaskMetadata(
                source="self_correction_generator",
                verified=True,
                tags=[
                    f"domain:{trace.domain}",
                    f"corrupted:{trace.is_corrupted}",
                    f"expected:{expected.value}",
                ],
                notes=(
                    f"Corruption type: {trace.corruption_type.value if trace.corruption_type else 'none'}"
                ),
                additional_data={
                    "is_corrupted": trace.is_corrupted,
                    "corruption_type": (
                        trace.corruption_type.value if trace.corruption_type else None
                    ),
                    "corruption_location": trace.corruption_location,
                    "expected_response": expected.value,
                    "original_problem": trace.problem,
                },
            ),
        )

    def _domain_to_category(self, domain: str) -> TaskCategory:
        """Convert domain string to TaskCategory.

        Args:
            domain: The domain string.

        Returns:
            Corresponding TaskCategory.
        """
        domain_map = {
            "mathematics": TaskCategory.NUMERICAL,
            "algebra": TaskCategory.NUMERICAL,
            "geometry": TaskCategory.NUMERICAL,
            "probability": TaskCategory.NUMERICAL,
            "logic": TaskCategory.REASONING,
            "physics": TaskCategory.SCIENTIFIC,
            "geography": TaskCategory.FACTUAL,
        }
        return domain_map.get(domain, TaskCategory.REASONING)

    def _inject_corruption(self, trace: ReasoningTrace) -> ReasoningTrace:
        """Inject corruption into an uncorrupted trace.

        Args:
            trace: The original trace.

        Returns:
            A corrupted version of the trace.
        """
        # Select corruption type based on distribution
        corruption_types = list(self.config.corruption_type_distribution.keys())
        weights = list(self.config.corruption_type_distribution.values())
        corruption_type = random.choices(corruption_types, weights=weights, k=1)[0]  # nosec B311

        # Select step to corrupt
        corruption_location = (
            random.randint(0, len(trace.steps) - 1) if len(trace.steps) > 0 else 0  # nosec B311
        )

        # Create corrupted version (simple modification)
        corrupted_steps = trace.steps.copy()
        if corrupted_steps:
            original_step = corrupted_steps[corruption_location]
            corrupted_steps[corruption_location] = f"[ERROR] {original_step}"

        return ReasoningTrace(
            problem=trace.problem,
            steps=corrupted_steps,
            conclusion=trace.conclusion,
            is_corrupted=True,
            corruption_type=corruption_type,
            corruption_location=corruption_location,
            corruption_description="Synthetically injected corruption for testing.",
            correct_steps=trace.steps,
            correct_conclusion=trace.conclusion,
            difficulty=trace.difficulty,
            domain=trace.domain,
        )
