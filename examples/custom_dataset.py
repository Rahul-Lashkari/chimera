#!/usr/bin/env python3
"""Custom dataset evaluation example for CHIMERA.

This script demonstrates how to use CHIMERA with custom datasets and tasks.
It shows multiple approaches for loading and creating custom evaluation tasks.

Usage:
    python custom_dataset.py
"""

import json
from pathlib import Path

from chimera.models import DifficultyLevel, Task, TaskCategory, TaskSet


def create_tasks_manually() -> list[Task]:
    """Create evaluation tasks manually."""
    tasks = [
        Task(
            id="custom_001",
            question="What is the chemical formula for water?",
            expected_answer="H2O",
            difficulty=DifficultyLevel.L1,
            category=TaskCategory.FACTUAL,
            metadata={"domain": "chemistry", "source": "manual"},
        ),
        Task(
            id="custom_002",
            question="Explain the difference between mitosis and meiosis.",
            expected_answer=(
                "Mitosis produces two identical diploid cells for growth and repair, "
                "while meiosis produces four genetically diverse haploid cells for "
                "sexual reproduction."
            ),
            difficulty=DifficultyLevel.L2,
            category=TaskCategory.REASONING,
            metadata={"domain": "biology", "source": "manual"},
        ),
        Task(
            id="custom_003",
            question=(
                "A train leaves Station A at 2:00 PM traveling at 60 mph. "
                "Another train leaves Station B (180 miles away) at 2:30 PM "
                "traveling at 90 mph toward Station A. At what time do they meet?"
            ),
            expected_answer="3:12 PM",
            difficulty=DifficultyLevel.L3,
            category=TaskCategory.REASONING,
            metadata={"domain": "math", "source": "manual"},
        ),
    ]
    return tasks


def load_tasks_from_json(filepath: Path) -> list[Task]:
    """Load tasks from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    tasks = []
    for item in data:
        task = Task(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            difficulty=DifficultyLevel(item.get("difficulty", "L2")),
            category=TaskCategory(item.get("category", "factual")),
            metadata=item.get("metadata", {}),
        )
        tasks.append(task)

    return tasks


def load_tasks_from_csv(filepath: Path) -> list[Task]:
    """Load tasks from a CSV file."""
    import csv

    tasks = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            task = Task(
                id=row.get("id", f"csv_{idx:04d}"),
                question=row["question"],
                expected_answer=row["expected_answer"],
                difficulty=DifficultyLevel(row.get("difficulty", "L2")),
                category=TaskCategory(row.get("category", "factual")),
                metadata={"row_index": idx},
            )
            tasks.append(task)

    return tasks


def create_taskset_from_huggingface(dataset_name: str, split: str = "test") -> TaskSet:
    """Create a TaskSet from a HuggingFace dataset.

    Note: Requires `datasets` library: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: pip install datasets"
        ) from None

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Convert to tasks (adapt field names to your dataset)
    tasks = []
    for idx, item in enumerate(dataset):
        task = Task(
            id=f"hf_{idx:04d}",
            question=item.get("question", item.get("prompt", "")),
            expected_answer=item.get("answer", item.get("expected_answer", "")),
            difficulty=DifficultyLevel.L2,  # Default, adapt as needed
            category=TaskCategory.FACTUAL,  # Default, adapt as needed
            metadata={"source": dataset_name, "split": split, "index": idx},
        )
        tasks.append(task)

    return TaskSet(
        name=dataset_name,
        description=f"Tasks from HuggingFace dataset: {dataset_name}",
        tasks=tasks,
    )


def demonstrate_task_filtering(tasks: list[Task]) -> None:
    """Demonstrate filtering tasks by difficulty and category."""
    print("\nTask Filtering")
    print("-" * 30)

    # Filter by difficulty
    easy_tasks = [t for t in tasks if t.difficulty == DifficultyLevel.L1]
    hard_tasks = [t for t in tasks if t.difficulty == DifficultyLevel.L3]

    print(f"Easy (L1) tasks: {len(easy_tasks)}")
    print(f"Hard (L3) tasks: {len(hard_tasks)}")

    # Filter by category
    factual_tasks = [t for t in tasks if t.category == TaskCategory.FACTUAL]
    reasoning_tasks = [t for t in tasks if t.category == TaskCategory.REASONING]

    print(f"Factual tasks: {len(factual_tasks)}")
    print(f"Reasoning tasks: {len(reasoning_tasks)}")


def demonstrate_taskset_operations(tasks: list[Task]) -> None:
    """Demonstrate TaskSet operations."""
    print("\nTaskSet Operations")
    print("-" * 30)

    # Create a TaskSet
    taskset = TaskSet(
        name="custom_dataset",
        description="A custom dataset for demonstration",
        tasks=tasks,
    )

    print(f"TaskSet: {taskset.name}")
    print(f"Total tasks: {len(taskset)}")

    # Get statistics
    stats = taskset.get_statistics()
    print("\nStatistics:")
    print(f"  Difficulty distribution: {stats['difficulty_distribution']}")
    print(f"  Category distribution: {stats['category_distribution']}")

    # Sample tasks
    sampled = taskset.sample(n=2, seed=42)
    print(f"\nSampled {len(sampled)} tasks:")
    for task in sampled:
        print(f"  - {task.id}: {task.question[:50]}...")


def create_sample_json_file() -> Path:
    """Create a sample JSON file for demonstration."""
    sample_data = [
        {
            "id": "json_001",
            "question": "What is the capital of France?",
            "expected_answer": "Paris",
            "difficulty": "L1",
            "category": "factual",
        },
        {
            "id": "json_002",
            "question": "Explain the concept of recursion in programming.",
            "expected_answer": (
                "Recursion is a programming technique where a function calls "
                "itself to solve smaller instances of the same problem."
            ),
            "difficulty": "L2",
            "category": "reasoning",
        },
    ]

    # Create examples directory if needed
    examples_dir = Path(__file__).parent
    sample_file = examples_dir / "sample_tasks.json"

    with open(sample_file, "w") as f:
        json.dump(sample_data, f, indent=2)

    return sample_file


def main() -> None:
    """Run the custom dataset demonstration."""
    print("CHIMERA Custom Dataset Example")
    print("=" * 50)

    # 1. Create tasks manually
    print("\n1. Creating tasks manually...")
    manual_tasks = create_tasks_manually()
    print(f"   Created {len(manual_tasks)} tasks")
    demonstrate_task_filtering(manual_tasks)

    # 2. Create sample JSON file and load it
    print("\n2. Loading tasks from JSON...")
    sample_json = create_sample_json_file()
    json_tasks = load_tasks_from_json(sample_json)
    print(f"   Loaded {len(json_tasks)} tasks from {sample_json.name}")

    # 3. Demonstrate TaskSet operations
    all_tasks = manual_tasks + json_tasks
    demonstrate_taskset_operations(all_tasks)

    # 4. Show how to run evaluation with custom tasks
    print("\n3. Running evaluation with custom tasks...")
    print("   (Skipping actual API call in example)")

    example_code = """
    from chimera.evaluation import EvaluationPipeline, PipelineConfig

    # Create TaskSet from your custom tasks
    taskset = TaskSet(
        name="custom_dataset",
        description="My custom evaluation dataset",
        tasks=all_tasks,
    )

    # Configure pipeline to use custom tasks
    config = PipelineConfig(
        tracks=["calibration"],
        model_provider="gemini",
        model_name="gemini-2.0-flash",
        custom_taskset=taskset,  # Use custom tasks
    )

    # Run evaluation
    pipeline = EvaluationPipeline(config)
    results = pipeline.run()
    """
    print(example_code)

    # Clean up sample file
    sample_json.unlink()
    print(f"\n   Cleaned up {sample_json.name}")

    print("\nExample complete!")


if __name__ == "__main__":
    main()
