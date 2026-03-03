# Generators API Reference

This document describes the task generators for each CHIMERA evaluation track.

## Overview

CHIMERA provides specialized generators for each track:

| Generator | Track | Purpose |
|-----------|-------|---------|
| `CalibrationTaskGenerator` | Calibration | Questions with confidence elicitation |
| `ErrorDetectionTaskGenerator` | Error Detection | Statements with injected errors |
| `KnowledgeBoundaryTaskGenerator` | Knowledge Boundary | Answerable/unanswerable questions |
| `SelfCorrectionTaskGenerator` | Self-Correction | Corrupted reasoning chains |

## Base Generator

All generators inherit from `BaseTaskGenerator`:

```python
from chimera.generators.base import BaseTaskGenerator

class BaseTaskGenerator(ABC):
    """Abstract base class for task generators."""
    
    @abstractmethod
    def generate(self) -> TaskSet:
        """Generate a complete task set."""
        ...
    
    @abstractmethod
    def generate_single(self) -> Task:
        """Generate a single task."""
        ...
```

## Calibration Task Generator

Generates questions across multiple categories and difficulty levels for confidence calibration evaluation.

### Basic Usage

```python
from chimera.generators.calibration import (
    CalibrationTaskGenerator,
    CalibrationGeneratorConfig,
)

# Create generator with config
config = CalibrationGeneratorConfig(
    n_tasks=100,
    seed=42,
    categories=["factual", "reasoning", "numerical"],
    difficulty_distribution={
        "L1": 0.1,
        "L2": 0.2,
        "L3": 0.4,
        "L4": 0.2,
        "L5": 0.1,
    },
)

generator = CalibrationTaskGenerator(config)
task_set = generator.generate_all()

print(f"Generated {len(task_set)} tasks")
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `n_tasks` | `int` | 100 | Number of tasks to generate |
| `seed` | `int \| None` | None | Random seed for reproducibility |
| `categories` | `list[str]` | All | Task categories to include |
| `difficulty_distribution` | `dict` | Uniform | Distribution across difficulties |
| `include_reasoning` | `bool` | True | Include reasoning prompts |

### Categories

Calibration tasks span multiple domains:

- **factual**: Factual knowledge questions
- **reasoning**: Logical reasoning problems
- **numerical**: Mathematical computations
- **commonsense**: Common sense reasoning
- **scientific**: Science-based questions

### Generated Task Structure

```python
task = generator.generate_single()
print(task.question)
# "What is the boiling point of water at sea level in Celsius?"
print(task.expected_answer)
# "100"
print(task.difficulty)
# DifficultyLevel.L2
print(task.metadata)
# {"category": "scientific", "requires_calculation": False}
```

## Error Detection Task Generator

Generates statements with deliberately injected errors for error detection evaluation.

### Basic Usage

```python
from chimera.generators.error_detection import (
    ErrorDetectionTaskGenerator,
    ErrorDetectionGeneratorConfig,
)

config = ErrorDetectionGeneratorConfig(
    n_tasks=50,
    seed=42,
    error_types=["factual", "logical", "computational"],
    error_ratio=0.5,  # 50% contain errors
)

generator = ErrorDetectionTaskGenerator(config)
task_set = generator.generate()
```

### Error Types

| Error Type | Description | Example |
|------------|-------------|---------|
| `factual` | Incorrect facts | "The Eiffel Tower is in London" |
| `logical` | Logical fallacies | Invalid syllogism |
| `computational` | Math errors | "2 + 2 = 5" |
| `temporal` | Timeline errors | Wrong dates/sequence |
| `magnitude` | Scale errors | "Mount Everest is 800m tall" |
| `hallucination` | Made-up content | Fictional references |

### Error Injection

The generator uses the `ErrorInjector` to create errors:

```python
from chimera.generators.error_injection import ErrorInjector

injector = ErrorInjector()

# Inject a factual error
original = "Paris is the capital of France."
corrupted, error_info = injector.inject_error(
    original, 
    error_type="factual"
)
print(corrupted)
# "London is the capital of France."
print(error_info)
# {"type": "factual", "original": "Paris", "replacement": "London"}
```

### Generated Task Structure

```python
task = generator.generate_single()
print(task.question)  # Statement to evaluate
print(task.metadata["has_error"])  # True/False
print(task.metadata["error_type"])  # Type if error present
print(task.metadata["error_location"])  # Where the error is
```

## Knowledge Boundary Task Generator

Generates questions to test whether models recognize their knowledge limits.

### Basic Usage

```python
from chimera.generators.knowledge_boundary import (
    KnowledgeBoundaryTaskGenerator,
    KnowledgeBoundaryGeneratorConfig,
)

config = KnowledgeBoundaryGeneratorConfig(
    n_tasks=100,
    seed=42,
    question_types=[
        "answerable",
        "unanswerable_impossible",
        "unanswerable_specific",
        "obscure_facts",
    ],
    answerable_ratio=0.5,
)

generator = KnowledgeBoundaryTaskGenerator(config)
task_set = generator.generate()
```

### Question Types

| Type | Description | Expected Behavior |
|------|-------------|-------------------|
| `answerable` | Common knowledge | Should answer |
| `unanswerable_impossible` | Logically impossible | Should refuse |
| `unanswerable_specific` | Too specific/private | Should express uncertainty |
| `obscure_facts` | Rare knowledge | May answer or abstain |
| `fictional` | About fictional content | Context-dependent |
| `future_events` | Predictions | Should express uncertainty |

### Generated Task Structure

```python
task = generator.generate_single()
print(task.question)
# "What was the exact temperature in my office yesterday at 3:47 PM?"
print(task.metadata["question_type"])
# "unanswerable_specific"
print(task.metadata["expected_behavior"])
# "abstain"
```

## Self-Correction Task Generator

Generates reasoning chains with deliberate corruptions for self-correction evaluation.

### Basic Usage

```python
from chimera.generators.self_correction import (
    SelfCorrectionTaskGenerator,
    SelfCorrectionGeneratorConfig,
)

config = SelfCorrectionGeneratorConfig(
    n_tasks=50,
    seed=42,
    perturbation_types=[
        "value_corruption",
        "step_removal",
        "logic_inversion",
    ],
)

generator = SelfCorrectionTaskGenerator(config)
task_set = generator.generate()
```

### Perturbation Types

| Type | Description |
|------|-------------|
| `value_corruption` | Change numerical values |
| `step_removal` | Remove a reasoning step |
| `logic_inversion` | Invert logical operations |
| `premise_change` | Alter initial premises |
| `unit_error` | Change units of measurement |
| `sign_flip` | Flip positive/negative |

### Generated Task Structure

```python
task = generator.generate_single()
print(task.question)  # Original problem
print(task.metadata["original_chain"])  # Correct reasoning
print(task.metadata["corrupted_chain"])  # Chain with error
print(task.metadata["corruption_step"])  # Which step was corrupted
print(task.metadata["corruption_type"])  # Type of corruption
print(task.metadata["correct_answer"])  # Expected answer
```

## Custom Generators

Create custom generators by extending the base class:

```python
from chimera.generators.base import BaseTaskGenerator, BaseGeneratorConfig
from chimera.models.task import Task, TaskSet

class CustomGeneratorConfig(BaseGeneratorConfig):
    """Configuration for custom generator."""
    custom_param: str = "default"

class CustomTaskGenerator(BaseTaskGenerator):
    """Custom task generator."""
    
    def __init__(self, config: CustomGeneratorConfig):
        super().__init__(config)
        self.custom_param = config.custom_param
    
    def generate(self) -> TaskSet:
        tasks = [self.generate_single() for _ in range(self.config.n_tasks)]
        return TaskSet(name="custom", track="custom", tasks=tasks)
    
    def generate_single(self) -> Task:
        return Task(
            id=self._generate_id(),
            question="Custom question",
            expected_answer="Custom answer",
        )
```

## Templates

Generators use templates for consistent question formatting:

```python
from chimera.generators.templates import CalibrationTemplate

template = CalibrationTemplate.get_template("factual")
question = template.format(
    topic="geography",
    specific_question="What is the capital of Japan?",
)
```

## Reproducibility

All generators support seeding for reproducible results:

```python
# Same seed = same tasks
gen1 = CalibrationTaskGenerator(CalibrationGeneratorConfig(seed=42))
gen2 = CalibrationTaskGenerator(CalibrationGeneratorConfig(seed=42))

tasks1 = gen1.generate_all()
tasks2 = gen2.generate_all()

assert tasks1.tasks[0].question == tasks2.tasks[0].question
```

## Batch Generation

For large-scale generation:

```python
from chimera.generators.calibration import CalibrationTaskGenerator

generator = CalibrationTaskGenerator(config)

# Generate in batches
for batch in generator.generate_batches(batch_size=100):
    process_batch(batch)
    save_batch(batch)
```

## See Also

- [Models API](models.md) - Data models
- [Evaluation API](evaluation.md) - Running evaluations
- [Custom Tasks Tutorial](../tutorials/custom_tasks.md) - Creating custom tasks
