# Tutorial: Creating Custom Tasks

This tutorial shows how to create custom evaluation tasks for CHIMERA.

## Overview

CHIMERA's modular design allows you to:

1. Create custom task sets from your own data
2. Extend existing generators with new categories
3. Build entirely new generators for custom tracks
4. Integrate external datasets

## Prerequisites

```python
from chimera.models.task import Task, TaskSet, DifficultyLevel, TaskCategory
from chimera.generators.base import BaseTaskGenerator, BaseGeneratorConfig
```

## Method 1: Creating Tasks Manually

### Single Task

```python
from chimera.models.task import Task, DifficultyLevel

# Create a single task
task = Task(
    id="custom_001",
    question="What programming language was Python named after?",
    expected_answer="Monty Python (the comedy group)",
    difficulty=DifficultyLevel.L2,
    metadata={
        "category": "programming",
        "source": "custom",
    },
)

print(task.model_dump_json(indent=2))
```

### Task Set from List

```python
from chimera.models.task import Task, TaskSet

# Define your questions
questions = [
    ("What is the time complexity of binary search?", "O(log n)", "L3"),
    ("What does HTTP stand for?", "HyperText Transfer Protocol", "L1"),
    ("What is the CAP theorem?", "Consistency, Availability, Partition tolerance", "L4"),
]

# Create tasks
tasks = []
for i, (question, answer, difficulty) in enumerate(questions):
    task = Task(
        id=f"custom_{i:03d}",
        question=question,
        expected_answer=answer,
        difficulty=DifficultyLevel(difficulty),
        metadata={"category": "computer_science"},
    )
    tasks.append(task)

# Create task set
task_set = TaskSet(
    name="cs_fundamentals",
    track="calibration",
    tasks=tasks,
    metadata={"version": "1.0", "author": "custom"},
)

print(f"Created {len(task_set)} tasks")
```

## Method 2: Loading from Files

### From JSON

```python
import json
from chimera.models.task import Task, TaskSet

# Load from JSON file
with open("my_tasks.json") as f:
    data = json.load(f)

tasks = [Task.model_validate(t) for t in data["tasks"]]
task_set = TaskSet(
    name=data["name"],
    track=data["track"],
    tasks=tasks,
)
```

### From CSV

```python
import csv
from chimera.models.task import Task, TaskSet, DifficultyLevel

def load_from_csv(filepath: str) -> TaskSet:
    """Load tasks from CSV file."""
    tasks = []
    
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            task = Task(
                id=f"csv_{i:04d}",
                question=row["question"],
                expected_answer=row.get("answer"),
                difficulty=DifficultyLevel(row.get("difficulty", "L3")),
                metadata={"source": filepath},
            )
            tasks.append(task)
    
    return TaskSet(name="csv_import", track="calibration", tasks=tasks)

# Usage
task_set = load_from_csv("questions.csv")
```

### From JSONL

```python
import json
from chimera.models.task import Task, TaskSet

def load_from_jsonl(filepath: str) -> TaskSet:
    """Load tasks from JSONL file."""
    tasks = []
    
    with open(filepath) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            task = Task(
                id=data.get("id", f"jsonl_{i:04d}"),
                question=data["question"],
                expected_answer=data.get("answer"),
                difficulty=DifficultyLevel(data.get("difficulty", "L3")),
                metadata=data.get("metadata", {}),
            )
            tasks.append(task)
    
    return TaskSet(name="jsonl_import", track="calibration", tasks=tasks)
```

## Method 3: Custom Generator

Create a reusable generator for your custom domain:

```python
from dataclasses import dataclass
from chimera.generators.base import BaseTaskGenerator, BaseGeneratorConfig
from chimera.models.task import Task, TaskSet, DifficultyLevel
import random

@dataclass
class MathGeneratorConfig(BaseGeneratorConfig):
    """Configuration for math task generator."""
    n_tasks: int = 100
    seed: int | None = None
    operations: list[str] = None
    max_number: int = 100
    
    def __post_init__(self):
        if self.operations is None:
            self.operations = ["add", "subtract", "multiply"]

class MathTaskGenerator(BaseTaskGenerator):
    """Generator for arithmetic tasks."""
    
    def __init__(self, config: MathGeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._task_count = 0
    
    def generate(self) -> TaskSet:
        """Generate complete task set."""
        tasks = [self.generate_single() for _ in range(self.config.n_tasks)]
        return TaskSet(
            name="math_tasks",
            track="calibration",
            tasks=tasks,
            metadata={"operations": self.config.operations},
        )
    
    def generate_single(self) -> Task:
        """Generate a single math task."""
        op = self.rng.choice(self.config.operations)
        a = self.rng.randint(1, self.config.max_number)
        b = self.rng.randint(1, self.config.max_number)
        
        if op == "add":
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            difficulty = DifficultyLevel.L1
        elif op == "subtract":
            question = f"What is {a} - {b}?"
            answer = str(a - b)
            difficulty = DifficultyLevel.L2
        elif op == "multiply":
            question = f"What is {a} × {b}?"
            answer = str(a * b)
            difficulty = DifficultyLevel.L3
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        self._task_count += 1
        return Task(
            id=f"math_{self._task_count:04d}",
            question=question,
            expected_answer=answer,
            difficulty=difficulty,
            metadata={"operation": op, "operands": [a, b]},
        )

# Usage
config = MathGeneratorConfig(n_tasks=50, seed=42, max_number=50)
generator = MathTaskGenerator(config)
task_set = generator.generate()

print(f"Generated {len(task_set)} math tasks")
for task in task_set.tasks[:3]:
    print(f"  {task.question} -> {task.expected_answer}")
```

## Method 4: Extending Existing Generators

Add new categories to existing generators:

```python
from chimera.generators.calibration import (
    CalibrationTaskGenerator,
    CalibrationGeneratorConfig,
)
from chimera.generators.templates import CalibrationTemplate

# Register custom template
CalibrationTemplate.register_template(
    "programming",
    """
    Answer the following programming question:
    
    {question}
    
    Provide your answer and confidence level (0-100%).
    """
)

# Create generator with custom category
config = CalibrationGeneratorConfig(
    n_tasks=100,
    categories=["factual", "programming"],  # Include custom category
)

generator = CalibrationTaskGenerator(config)

# Add custom questions to the generator's pool
generator.add_questions("programming", [
    {"question": "What is a closure?", "answer": "A function with its environment", "difficulty": "L3"},
    {"question": "What is Big O notation?", "answer": "Describes algorithm complexity", "difficulty": "L2"},
])

task_set = generator.generate_all()
```

## Method 5: From External Datasets

### Hugging Face Datasets

```python
from datasets import load_dataset
from chimera.models.task import Task, TaskSet, DifficultyLevel

def from_huggingface(dataset_name: str, split: str = "test") -> TaskSet:
    """Load tasks from Hugging Face dataset."""
    dataset = load_dataset(dataset_name, split=split)
    
    tasks = []
    for i, item in enumerate(dataset):
        task = Task(
            id=f"hf_{i:05d}",
            question=item["question"],
            expected_answer=item.get("answer", item.get("label")),
            difficulty=DifficultyLevel.L3,  # Default difficulty
            metadata={
                "source": dataset_name,
                "split": split,
            },
        )
        tasks.append(task)
    
    return TaskSet(
        name=dataset_name.replace("/", "_"),
        track="calibration",
        tasks=tasks,
    )

# Example: Load TruthfulQA
task_set = from_huggingface("truthful_qa", split="validation")
```

## Running Custom Tasks

### With CLI

Save your task set and run via CLI:

```python
# Save to file
task_set.save("custom_tasks.json")

# Run via CLI
# chimera run --tasks custom_tasks.json --model gemini
```

### Programmatically

```python
from chimera.evaluation import EvaluationPipeline, PipelineConfig
from chimera.runner import BenchmarkRunner

# Create pipeline
config = PipelineConfig(
    tracks=["calibration"],
    model_provider="gemini",
    model_name="gemini-2.0-flash",
)

# Run with custom tasks
pipeline = EvaluationPipeline(config)
results = pipeline.run_with_tasks(task_set)

print(f"Accuracy: {results.track_evaluations['calibration'].summary.accuracy:.2%}")
```

## Best Practices

### 1. Consistent IDs

Use a consistent ID scheme:

```python
def generate_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:05d}"
```

### 2. Difficulty Assignment

Use heuristics or manual review:

```python
def estimate_difficulty(question: str, answer: str) -> DifficultyLevel:
    """Estimate difficulty based on complexity."""
    word_count = len(question.split())
    answer_length = len(answer)
    
    if word_count < 10 and answer_length < 20:
        return DifficultyLevel.L1
    elif word_count < 20:
        return DifficultyLevel.L2
    elif word_count < 30:
        return DifficultyLevel.L3
    elif word_count < 50:
        return DifficultyLevel.L4
    else:
        return DifficultyLevel.L5
```

### 3. Validation

Validate tasks before use:

```python
def validate_task_set(task_set: TaskSet) -> list[str]:
    """Validate task set and return errors."""
    errors = []
    ids_seen = set()
    
    for task in task_set.tasks:
        # Check for duplicate IDs
        if task.id in ids_seen:
            errors.append(f"Duplicate ID: {task.id}")
        ids_seen.add(task.id)
        
        # Check for empty questions
        if not task.question.strip():
            errors.append(f"Empty question: {task.id}")
        
        # Check for missing answers (calibration track)
        if task_set.track == "calibration" and not task.expected_answer:
            errors.append(f"Missing answer: {task.id}")
    
    return errors

errors = validate_task_set(task_set)
if errors:
    print("Validation errors:", errors)
```

### 4. Reproducibility

Always use seeds for reproducible generation:

```python
config = MathGeneratorConfig(
    n_tasks=100,
    seed=42,  # Always set seed for reproducibility
)
```

## See Also

- [Generators API](../api/generators.md) - Generator reference
- [Models API](../api/models.md) - Data model reference
- [Examples](../../examples/) - Working examples
