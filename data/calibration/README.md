# CHIMERA Calibration Track Data

This directory contains seed data and generated datasets for the CHIMERA calibration track.

## Files

### `seed_tasks.jsonl`

A curated set of 20 seed tasks for the calibration track, covering:

- **Factual questions**: Geography, history, science
- **Mathematics**: Arithmetic, algebra, percentages
- **Reasoning**: Logic, syllogisms, transitive relations
- **Scientific**: Physics, biology, chemistry
- **Common sense**: Everyday knowledge
- **Trick questions**: Cognitive bias traps

### Difficulty Distribution

| Level | Description | Count |
|-------|-------------|-------|
| L1 | Easy - Basic recall | 4 |
| L2 | Medium-Easy - Common knowledge | 5 |
| L3 | Medium - Requires thinking | 5 |
| L4 | Medium-Hard - Specialized knowledge | 5 |
| L5 | Hard - Complex reasoning | 1 |

## Usage

### Loading Seed Data

```python
from chimera.models.task import TaskSet

taskset = TaskSet.from_jsonl("data/calibration/seed_tasks.jsonl")
print(f"Loaded {len(taskset)} tasks")

# Filter by difficulty
easy_tasks = taskset.filter_by_difficulty(DifficultyLevel.L1)

# Filter by category
math_tasks = taskset.filter_by_category(TaskCategory.MATHEMATICS)
```

### Generating More Tasks

```python
from chimera.generators.calibration import (
    CalibrationTaskGenerator,
    CalibrationGeneratorConfig,
)

config = CalibrationGeneratorConfig(
    n_tasks=100,
    seed=42,
    include_trick_questions=True,
)

generator = CalibrationTaskGenerator(config)
taskset = generator.generate_all()

# Save to file
taskset.to_jsonl("data/calibration/generated_100.jsonl")
```

## Task Schema

Each task has the following fields:

```json
{
  "track": "calibration",
  "question": "The question text with confidence elicitation",
  "correct_answer": "The expected correct answer",
  "answer_type": "exact_match | free_form | multiple_choice",
  "difficulty": "l1 | l2 | l3 | l4 | l5",
  "category": "factual | reasoning | mathematics | ...",
  "expected_confidence": "l1 | l2 | l3 | l4 | l5",
  "metadata": {
    "source": "Origin of the task",
    "verified": true,
    "domain": "Specific domain",
    "tags": ["tag1", "tag2"]
  }
}
```

## Adding Custom Tasks

To add custom tasks, create a JSONL file with one JSON object per line:

```jsonl
{"track": "calibration", "question": "Your question?", "correct_answer": "Answer", ...}
{"track": "calibration", "question": "Another question?", "correct_answer": "Answer2", ...}
```

Validate with:

```python
from chimera.models.schemas import validate_jsonl_file
from chimera.models.task import Task

valid_tasks, errors = validate_jsonl_file("your_file.jsonl", Task)
print(f"Valid: {len(valid_tasks)}, Errors: {len(errors)}")
```
