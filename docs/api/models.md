# Data Models API Reference

This document describes the core data models used throughout CHIMERA.

## Overview

CHIMERA uses Pydantic v2 models for all data structures, providing:
- Type validation
- JSON serialization
- Schema generation
- IDE autocompletion

## Task Models

### Task

The base `Task` model represents an evaluation task.

```python
from chimera.models.task import Task, DifficultyLevel, TaskCategory

task = Task(
    id="cal_001",
    question="What is the capital of France?",
    expected_answer="Paris",
    difficulty=DifficultyLevel.L2,
    category=TaskCategory.FACTUAL,
    metadata={"source": "geography", "region": "europe"},
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique task identifier |
| `question` | `str` | The question or prompt |
| `expected_answer` | `str \| None` | Ground truth answer (if applicable) |
| `difficulty` | `DifficultyLevel` | Task difficulty level |
| `category` | `TaskCategory` | Task category |
| `metadata` | `dict[str, Any]` | Additional metadata |
| `created_at` | `datetime` | Creation timestamp |

### DifficultyLevel

Enum representing task difficulty:

```python
from chimera.models.task import DifficultyLevel

class DifficultyLevel(str, Enum):
    L1 = "L1"  # Very Easy
    L2 = "L2"  # Easy
    L3 = "L3"  # Medium
    L4 = "L4"  # Hard
    L5 = "L5"  # Very Hard
```

### TaskCategory

Enum for task categories:

```python
from chimera.models.task import TaskCategory

class TaskCategory(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    NUMERICAL = "numerical"
    COMMONSENSE = "commonsense"
    SCIENTIFIC = "scientific"
```

### TaskSet

A collection of related tasks:

```python
from chimera.models.task import TaskSet

task_set = TaskSet(
    name="calibration_v1",
    track="calibration",
    tasks=[task1, task2, task3],
    metadata={"version": "1.0"},
)

# Access tasks
print(f"Total tasks: {len(task_set)}")
for task in task_set:
    print(task.question)
```

## Response Models

### ModelResponse

Represents a model's response to a task:

```python
from chimera.models.response import ModelResponse

response = ModelResponse(
    task_id="cal_001",
    model_name="gemini-2.0-flash",
    raw_response="The capital of France is Paris.",
    parsed_answer="Paris",
    confidence=0.95,
    reasoning="France is a country in Western Europe...",
    latency_ms=245.5,
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | ID of the task being answered |
| `model_name` | `str` | Name of the responding model |
| `raw_response` | `str` | Complete model output |
| `parsed_answer` | `str \| None` | Extracted answer |
| `confidence` | `float \| None` | Confidence score (0-1) |
| `reasoning` | `str \| None` | Reasoning trace |
| `latency_ms` | `float` | Response latency in milliseconds |
| `token_count` | `int \| None` | Number of tokens generated |
| `timestamp` | `datetime` | When the response was received |

### ConfidenceExtraction

Result of extracting confidence from model output:

```python
from chimera.models.response import ConfidenceExtraction

extraction = ConfidenceExtraction(
    raw_value="I am 85% confident",
    normalized_value=0.85,
    extraction_method="regex",
    success=True,
)
```

## Evaluation Models

### EvaluationResult

Result of evaluating a single response:

```python
from chimera.models.evaluation import EvaluationResult

result = EvaluationResult(
    task_id="cal_001",
    is_correct=True,
    confidence=0.95,
    score=1.0,
    error_type=None,
    feedback="Correct answer with high confidence",
)
```

### TrackSummary

Summary statistics for an evaluation track:

```python
from chimera.models.evaluation import TrackSummary

summary = TrackSummary(
    track="calibration",
    total_tasks=100,
    completed_tasks=100,
    accuracy=0.82,
    avg_confidence=0.75,
    ece=0.08,
    mce=0.15,
)
```

### BenchmarkReport

Complete benchmark results:

```python
from chimera.models.evaluation import BenchmarkReport

report = BenchmarkReport(
    model_name="gemini-2.0-flash",
    tracks=["calibration", "error_detection"],
    summaries={
        "calibration": calibration_summary,
        "error_detection": error_detection_summary,
    },
    overall_score=0.78,
    metadata={"run_id": "run_20250129"},
)

# Generate outputs
report.to_json("results/report.json")
report.to_markdown("results/report.md")
report.to_html("results/report.html")
```

## Schema Utilities

### JSON Schema Generation

Generate JSON schemas for any model:

```python
from chimera.models.task import Task

# Get JSON schema
schema = Task.model_json_schema()
print(json.dumps(schema, indent=2))
```

### Validation

All models validate input on creation:

```python
from chimera.models.task import Task, DifficultyLevel
from pydantic import ValidationError

try:
    # Invalid: confidence must be 0-1
    task = Task(
        id="test",
        question="Test?",
        difficulty="INVALID",  # Will raise ValidationError
    )
except ValidationError as e:
    print(e.errors())
```

### Serialization

Models support various serialization formats:

```python
# To dict
task_dict = task.model_dump()

# To JSON string
task_json = task.model_dump_json()

# From dict
task = Task.model_validate(task_dict)

# From JSON
task = Task.model_validate_json(task_json)
```

## Custom Models

Extend base models for custom use cases:

```python
from chimera.models.task import Task
from pydantic import Field

class CustomTask(Task):
    """Task with additional fields."""
    
    source_url: str = Field(description="Source URL for the question")
    difficulty_score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
```

## Type Hints

All models are fully typed for IDE support:

```python
from chimera.models.task import Task, TaskSet
from chimera.models.response import ModelResponse
from chimera.models.evaluation import EvaluationResult

def process_task(task: Task) -> ModelResponse:
    """Process a single task."""
    ...

def evaluate_responses(responses: list[ModelResponse]) -> list[EvaluationResult]:
    """Evaluate multiple responses."""
    ...
```

## See Also

- [Generators API](generators.md) - Task generation
- [Evaluation API](evaluation.md) - Running evaluations
- [Source Code](https://github.com/Rahul-Lashkari/chimera/tree/main/src/chimera/models)
