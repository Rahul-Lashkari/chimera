"""Question templates for CHIMERA task generation.

This module provides a template system for generating diverse
questions across different categories and difficulty levels.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from chimera.models.task import DifficultyLevel, TaskCategory


@dataclass
class QuestionTemplate:
    """A template for generating questions.

    Templates use {placeholder} syntax for variable substitution.

    Attributes:
        template: The question template string with {placeholders}
        answer_template: Template for the answer (optional)
        category: The task category this template belongs to
        difficulty: The difficulty level this template is suited for
        variables: Dictionary of variable names to possible values
        answer_func: Optional function to compute the answer from variables
        tags: Optional tags for the template
    """

    template: str
    category: TaskCategory
    difficulty: DifficultyLevel
    answer_template: str | None = None
    variables: dict[str, list[Any]] = field(default_factory=dict)
    answer_func: Callable[..., str] | None = None
    tags: list[str] = field(default_factory=list)

    def generate(self, rng: Any) -> tuple[str, str]:
        """Generate a question and answer from this template.

        Args:
            rng: Random number generator to use.

        Returns:
            Tuple of (question, answer).
        """
        # Select random values for each variable
        selected_vars = {}
        for var_name, values in self.variables.items():
            selected_vars[var_name] = rng.choice(values)

        # Format the question
        question = self.template.format(**selected_vars)

        # Compute the answer
        if self.answer_func is not None:
            answer = self.answer_func(**selected_vars)
        elif self.answer_template is not None:
            answer = self.answer_template.format(**selected_vars)
        else:
            raise ValueError("Template must have either answer_template or answer_func")

        return question, answer


class TemplateRegistry:
    """Registry for question templates organized by category and difficulty.

    This class manages a collection of question templates and provides
    methods to retrieve templates based on category and difficulty.
    """

    def __init__(self) -> None:
        """Initialize an empty template registry."""
        self._templates: dict[TaskCategory, dict[DifficultyLevel, list[QuestionTemplate]]] = {}

    def register(self, template: QuestionTemplate) -> None:
        """Register a template.

        Args:
            template: The template to register.
        """
        if template.category not in self._templates:
            self._templates[template.category] = {}

        if template.difficulty not in self._templates[template.category]:
            self._templates[template.category][template.difficulty] = []

        self._templates[template.category][template.difficulty].append(template)

    def register_many(self, templates: list[QuestionTemplate]) -> None:
        """Register multiple templates.

        Args:
            templates: List of templates to register.
        """
        for template in templates:
            self.register(template)

    def get_template(
        self,
        category: TaskCategory,
        difficulty: DifficultyLevel,
        rng: Any,
    ) -> QuestionTemplate | None:
        """Get a random template for the given category and difficulty.

        Args:
            category: The task category.
            difficulty: The difficulty level.
            rng: Random number generator.

        Returns:
            A random matching template, or None if none available.
        """
        if category not in self._templates:
            return None

        if difficulty not in self._templates[category]:
            # Try to find a template at adjacent difficulty levels
            for adj_diff in [
                self._get_adjacent_difficulty(difficulty, -1),
                self._get_adjacent_difficulty(difficulty, 1),
            ]:
                if adj_diff and adj_diff in self._templates[category]:
                    templates = self._templates[category][adj_diff]
                    selected: QuestionTemplate = rng.choice(templates)
                    return selected
            return None

        templates = self._templates[category][difficulty]
        result: QuestionTemplate = rng.choice(templates)
        return result

    def get_all_templates(
        self,
        category: TaskCategory | None = None,
        difficulty: DifficultyLevel | None = None,
    ) -> list[QuestionTemplate]:
        """Get all templates matching the criteria.

        Args:
            category: Optional category filter.
            difficulty: Optional difficulty filter.

        Returns:
            List of matching templates.
        """
        result = []

        categories = [category] if category else list(self._templates.keys())

        for cat in categories:
            if cat not in self._templates:
                continue

            difficulties = [difficulty] if difficulty else list(self._templates[cat].keys())

            for diff in difficulties:
                if diff in self._templates[cat]:
                    result.extend(self._templates[cat][diff])

        return result

    def count(self) -> int:
        """Get total number of registered templates.

        Returns:
            Total template count.
        """
        total = 0
        for cat_templates in self._templates.values():
            for diff_templates in cat_templates.values():
                total += len(diff_templates)
        return total

    def _get_adjacent_difficulty(
        self,
        difficulty: DifficultyLevel,
        offset: int,
    ) -> DifficultyLevel | None:
        """Get an adjacent difficulty level.

        Args:
            difficulty: Current difficulty level.
            offset: Offset (-1 for easier, +1 for harder).

        Returns:
            Adjacent difficulty level or None if out of bounds.
        """
        levels = list(DifficultyLevel)
        try:
            current_idx = levels.index(difficulty)
            new_idx = current_idx + offset
            if 0 <= new_idx < len(levels):
                return levels[new_idx]
        except ValueError:
            pass
        return None


def create_default_calibration_templates() -> TemplateRegistry:
    """Create the default set of calibration templates.

    Returns:
        A TemplateRegistry populated with default templates.
    """
    registry = TemplateRegistry()

    # ==========================================================================
    # FACTUAL TEMPLATES
    # ==========================================================================

    # L1 (Easy) - Simple factual recall
    registry.register(
        QuestionTemplate(
            template="What is the capital of {country}?",
            answer_template="{capital}",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L1,
            variables={
                "country": [
                    ("France", "Paris"),
                    ("Germany", "Berlin"),
                    ("Japan", "Tokyo"),
                    ("Italy", "Rome"),
                    ("Spain", "Madrid"),
                    ("Australia", "Canberra"),
                    ("Canada", "Ottawa"),
                    ("Brazil", "Brasília"),
                ],
            },
            answer_func=lambda country: country[1],
            tags=["geography", "capitals"],
        )
    )

    registry.register(
        QuestionTemplate(
            template="What color do you get when you mix {color1} and {color2}?",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L1,
            variables={
                "color1": ["red", "blue", "yellow"],
                "color2": ["blue", "yellow", "red"],
            },
            answer_func=lambda color1, color2: {
                frozenset(["red", "blue"]): "purple",
                frozenset(["blue", "yellow"]): "green",
                frozenset(["red", "yellow"]): "orange",
                frozenset(["red", "red"]): "red",
                frozenset(["blue", "blue"]): "blue",
                frozenset(["yellow", "yellow"]): "yellow",
            }.get(frozenset([color1, color2]), "mixed color"),
            tags=["colors", "basic"],
        )
    )

    # L2 (Medium-Easy) - Slightly more obscure facts
    registry.register(
        QuestionTemplate(
            template="In what year did {event} occur?",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L2,
            variables={
                "event": [
                    ("World War I begin", "1914"),
                    ("World War II end", "1945"),
                    ("the first Moon landing happen", "1969"),
                    ("the Berlin Wall fall", "1989"),
                    ("the Titanic sink", "1912"),
                ],
            },
            answer_func=lambda event: event[1],
            tags=["history", "dates"],
        )
    )

    # L3 (Medium) - Requires specific knowledge
    registry.register(
        QuestionTemplate(
            template="What is the chemical symbol for {element}?",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L3,
            variables={
                "element": [
                    ("gold", "Au"),
                    ("silver", "Ag"),
                    ("iron", "Fe"),
                    ("sodium", "Na"),
                    ("potassium", "K"),
                    ("tungsten", "W"),
                    ("mercury", "Hg"),
                    ("lead", "Pb"),
                ],
            },
            answer_func=lambda element: element[1],
            tags=["chemistry", "elements"],
        )
    )

    # L4 (Medium-Hard) - Specialized knowledge
    registry.register(
        QuestionTemplate(
            template="Who was the {ordinal} President of the United States?",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L4,
            variables={
                "ordinal": [
                    ("16th", "Abraham Lincoln"),
                    ("26th", "Theodore Roosevelt"),
                    ("32nd", "Franklin D. Roosevelt"),
                    ("35th", "John F. Kennedy"),
                    ("40th", "Ronald Reagan"),
                ],
            },
            answer_func=lambda ordinal: ordinal[1],
            tags=["history", "presidents", "usa"],
        )
    )

    # L5 (Hard) - Obscure facts
    registry.register(
        QuestionTemplate(
            template="What is the approximate population of {city} as of 2020?",
            category=TaskCategory.FACTUAL,
            difficulty=DifficultyLevel.L5,
            variables={
                "city": [
                    ("Reykjavik, Iceland", "approximately 130,000"),
                    ("Vaduz, Liechtenstein", "approximately 5,500"),
                    ("Luxembourg City", "approximately 125,000"),
                ],
            },
            answer_func=lambda city: city[1],
            tags=["geography", "demographics", "obscure"],
        )
    )

    # ==========================================================================
    # REASONING TEMPLATES
    # ==========================================================================

    # L1 (Easy) - Simple logical deduction
    registry.register(
        QuestionTemplate(
            template="If all {category_a} are {category_b}, and {item} is a {category_a}, is {item} a {category_b}?",
            answer_template="Yes",
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.L1,
            variables={
                "category_a": ["dogs", "cats", "birds"],
                "category_b": ["animals", "mammals", "living things"],
                "item": ["Buddy", "Whiskers", "Tweety"],
            },
            tags=["logic", "syllogism"],
        )
    )

    # L2 (Medium-Easy) - Basic inference
    registry.register(
        QuestionTemplate(
            template="{person1} is taller than {person2}. {person2} is taller than {person3}. Who is the tallest?",
            answer_template="{person1}",
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.L2,
            variables={
                "person1": ["Alice", "Bob", "Carol"],
                "person2": ["David", "Eve", "Frank"],
                "person3": ["Grace", "Henry", "Ivy"],
            },
            tags=["logic", "transitive"],
        )
    )

    # L3 (Medium) - Multi-step reasoning
    registry.register(
        QuestionTemplate(
            template="A {container} contains only {color1} and {color2} balls. If there are {n1} {color1} balls and {n2} {color2} balls, what fraction of the balls are {color1}?",
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.L3,
            variables={
                "container": ["box", "bag", "jar"],
                "color1": ["red", "blue", "green"],
                "color2": ["yellow", "white", "black"],
                "n1": [3, 4, 5, 6],
                "n2": [2, 3, 4, 5],
            },
            answer_func=lambda container, color1, color2, n1, n2: f"{n1}/{n1 + n2}",  # noqa: ARG005
            tags=["fractions", "probability"],
        )
    )

    # L4 (Medium-Hard) - Complex logical reasoning
    registry.register(
        QuestionTemplate(
            template="In a group of {total} people, {n1} like {food1}, {n2} like {food2}, and {n_both} like both. How many people like neither?",
            category=TaskCategory.REASONING,
            difficulty=DifficultyLevel.L4,
            variables={
                "total": [20, 25, 30],
                "n1": [12, 15, 18],
                "n2": [10, 12, 14],
                "n_both": [5, 6, 7],
                "food1": ["pizza", "pasta", "burgers"],
                "food2": ["sushi", "tacos", "salad"],
            },
            answer_func=lambda total, n1, n2, n_both, food1, food2: str(  # noqa: ARG005
                total - (n1 + n2 - n_both)
            ),
            tags=["sets", "inclusion-exclusion"],
        )
    )

    # ==========================================================================
    # MATHEMATICS TEMPLATES
    # ==========================================================================

    # L1 (Easy) - Basic arithmetic
    registry.register(
        QuestionTemplate(
            template="What is {a} + {b}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            variables={
                "a": list(range(1, 20)),
                "b": list(range(1, 20)),
            },
            answer_func=lambda a, b: str(a + b),
            tags=["arithmetic", "addition"],
        )
    )

    registry.register(
        QuestionTemplate(
            template="What is {a} × {b}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L1,
            variables={
                "a": list(range(2, 12)),
                "b": list(range(2, 12)),
            },
            answer_func=lambda a, b: str(a * b),
            tags=["arithmetic", "multiplication"],
        )
    )

    # L2 (Medium-Easy) - Two-step arithmetic
    registry.register(
        QuestionTemplate(
            template="What is ({a} + {b}) × {c}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L2,
            variables={
                "a": list(range(2, 10)),
                "b": list(range(2, 10)),
                "c": list(range(2, 6)),
            },
            answer_func=lambda a, b, c: str((a + b) * c),
            tags=["arithmetic", "order-of-operations"],
        )
    )

    # L3 (Medium) - Percentage calculations
    registry.register(
        QuestionTemplate(
            template="What is {percent}% of {number}?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L3,
            variables={
                "percent": [10, 15, 20, 25, 30, 40, 50],
                "number": [80, 100, 120, 150, 200, 250],
            },
            answer_func=lambda percent, number: str(int(percent * number / 100)),
            tags=["percentages"],
        )
    )

    # L4 (Medium-Hard) - Algebraic problems
    registry.register(
        QuestionTemplate(
            template="If {a}x + {b} = {c}, what is x?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L4,
            variables={
                "a": [2, 3, 4, 5],
                "b": [3, 5, 7, 10],
                "c": [11, 17, 23, 30],
            },
            answer_func=lambda a, b, c: str((c - b) / a) if (c - b) % a == 0 else f"({c - b})/{a}",
            tags=["algebra", "linear-equations"],
        )
    )

    # L5 (Hard) - Complex mathematical reasoning
    registry.register(
        QuestionTemplate(
            template="What is the sum of the first {n} positive integers?",
            category=TaskCategory.NUMERICAL,
            difficulty=DifficultyLevel.L5,
            variables={
                "n": [15, 20, 25, 30, 50],
            },
            answer_func=lambda n: str(n * (n + 1) // 2),
            tags=["series", "formulas"],
        )
    )

    # ==========================================================================
    # COMMON SENSE TEMPLATES
    # ==========================================================================

    # L1 (Easy) - Basic common sense
    registry.register(
        QuestionTemplate(
            template="If you put ice in a warm room, what will happen to it?",
            answer_template="It will melt",
            category=TaskCategory.COMMONSENSE,
            difficulty=DifficultyLevel.L1,
            variables={},
            tags=["physics", "everyday"],
        )
    )

    registry.register(
        QuestionTemplate(
            template="What do you typically use a {tool} for?",
            category=TaskCategory.COMMONSENSE,
            difficulty=DifficultyLevel.L1,
            variables={
                "tool": [
                    ("hammer", "driving nails or breaking things"),
                    ("scissors", "cutting paper or fabric"),
                    ("thermometer", "measuring temperature"),
                    ("umbrella", "protection from rain"),
                ],
            },
            answer_func=lambda tool: tool[1],
            tags=["tools", "everyday"],
        )
    )

    # L2 (Medium-Easy) - Slightly more nuanced
    registry.register(
        QuestionTemplate(
            template="Why would someone typically go to a {place}?",
            category=TaskCategory.COMMONSENSE,
            difficulty=DifficultyLevel.L2,
            variables={
                "place": [
                    ("library", "to borrow books or study"),
                    ("hospital", "to receive medical care"),
                    ("grocery store", "to buy food"),
                    ("post office", "to send or receive mail"),
                ],
            },
            answer_func=lambda place: place[1],
            tags=["places", "purposes"],
        )
    )

    # ==========================================================================
    # SCIENTIFIC TEMPLATES
    # ==========================================================================

    # L2 (Medium-Easy) - Basic science
    registry.register(
        QuestionTemplate(
            template="What is the boiling point of water at sea level in {unit}?",
            category=TaskCategory.SCIENTIFIC,
            difficulty=DifficultyLevel.L2,
            variables={
                "unit": [
                    ("Celsius", "100°C"),
                    ("Fahrenheit", "212°F"),
                    ("Kelvin", "373.15K"),
                ],
            },
            answer_func=lambda unit: unit[1],
            tags=["physics", "temperature"],
        )
    )

    # L3 (Medium) - Science concepts
    registry.register(
        QuestionTemplate(
            template="What is the primary function of the {organ} in the human body?",
            category=TaskCategory.SCIENTIFIC,
            difficulty=DifficultyLevel.L3,
            variables={
                "organ": [
                    ("heart", "to pump blood throughout the body"),
                    ("lungs", "to exchange oxygen and carbon dioxide"),
                    ("liver", "to filter blood and produce bile"),
                    ("kidneys", "to filter waste from blood and produce urine"),
                ],
            },
            answer_func=lambda organ: organ[1],
            tags=["biology", "anatomy"],
        )
    )

    # L4 (Medium-Hard) - More specialized science
    registry.register(
        QuestionTemplate(
            template="What is the speed of light in a vacuum, approximately?",
            answer_template="Approximately 300,000 kilometers per second (or 3 × 10^8 m/s)",
            category=TaskCategory.SCIENTIFIC,
            difficulty=DifficultyLevel.L4,
            variables={},
            tags=["physics", "constants"],
        )
    )

    return registry
