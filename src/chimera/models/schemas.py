"""JSON Schema generation utilities for CHIMERA models.

This module provides utilities for generating JSON schemas from Pydantic models,
enabling dataset validation and documentation generation.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from chimera.models.evaluation import (
    CalibrationMetrics,
    ErrorDetectionMetrics,
    EvaluationResult,
    KnowledgeBoundaryMetrics,
    SelfCorrectionMetrics,
    TrackResult,
)
from chimera.models.response import (
    ConfidenceScore,
    ModelResponse,
    ParsedAnswer,
    ReasoningTrace,
    ResponseMetadata,
)
from chimera.models.task import Task, TaskMetadata, TaskSet


def get_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate JSON schema for a Pydantic model.

    Args:
        model: The Pydantic model class.

    Returns:
        JSON schema as a dictionary.
    """
    return model.model_json_schema()


def get_all_schemas() -> dict[str, dict[str, Any]]:
    """Generate JSON schemas for all CHIMERA models.

    Returns:
        Dictionary mapping model names to their JSON schemas.
    """
    models: list[type[BaseModel]] = [
        # Task models
        Task,
        TaskSet,
        TaskMetadata,
        # Response models
        ModelResponse,
        ConfidenceScore,
        ReasoningTrace,
        ParsedAnswer,
        ResponseMetadata,
        # Evaluation models
        EvaluationResult,
        TrackResult,
        CalibrationMetrics,
        ErrorDetectionMetrics,
        KnowledgeBoundaryMetrics,
        SelfCorrectionMetrics,
    ]

    schemas: dict[str, dict[str, Any]] = {}
    for model in models:
        schemas[model.__name__] = get_json_schema(model)

    return schemas


def save_schemas(output_dir: Path | str) -> None:
    """Save all JSON schemas to a directory.

    Args:
        output_dir: Directory to save schemas to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schemas = get_all_schemas()

    for name, schema in schemas.items():
        schema_file = output_path / f"{name}.schema.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    # Also save a combined schema file
    combined_file = output_path / "chimera.schemas.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)


def validate_jsonl_file(
    file_path: Path | str,
    model: type[BaseModel],
) -> tuple[list[BaseModel], list[dict[str, Any]]]:
    """Validate a JSONL file against a Pydantic model.

    Args:
        file_path: Path to the JSONL file.
        model: The Pydantic model to validate against.

    Returns:
        Tuple of (valid_items, errors) where errors contain line numbers
        and validation error details.
    """
    valid_items: list[BaseModel] = []
    errors: list[dict[str, Any]] = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                item = model.model_validate(data)
                valid_items.append(item)
            except json.JSONDecodeError as e:
                errors.append(
                    {
                        "line": line_num,
                        "error_type": "json_decode",
                        "message": str(e),
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "line": line_num,
                        "error_type": "validation",
                        "message": str(e),
                    }
                )

    return valid_items, errors


def create_example_task() -> Task:
    """Create an example Task for documentation purposes.

    Returns:
        An example Task instance.
    """
    from chimera.models.task import (
        AnswerType,
        DifficultyLevel,
        TaskCategory,
        TrackType,
    )

    return Task(
        track=TrackType.CALIBRATION,
        question="What is the capital of France?",
        correct_answer="Paris",
        answer_type=AnswerType.EXACT_MATCH,
        difficulty=DifficultyLevel.L1,
        category=TaskCategory.FACTUAL,
        metadata=TaskMetadata(
            source="synthetic",
            verified=True,
            tags=["geography", "capitals"],
        ),
    )


def create_example_response(task: Task) -> ModelResponse:
    """Create an example ModelResponse for documentation purposes.

    Args:
        task: The task this response is for.

    Returns:
        An example ModelResponse instance.
    """
    return ModelResponse(
        task_id=task.id,
        raw_text="The capital of France is Paris. I am 95% confident in this answer.",
        parsed_answer=ParsedAnswer(
            raw_answer="Paris",
            normalized="paris",
            answer_type="text",
        ),
        confidence=ConfidenceScore.from_percentage(95, "95% confident"),
        metadata=ResponseMetadata(
            model_name="gemini-2.0-flash",
            latency_ms=245.0,
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            temperature=0.0,
        ),
        is_correct=True,
    )


def generate_documentation_examples() -> dict[str, str]:
    """Generate JSON examples of all models for documentation.

    Returns:
        Dictionary mapping model names to JSON examples.
    """
    task = create_example_task()
    response = create_example_response(task)

    examples = {
        "Task": task.model_dump_json(indent=2),
        "ModelResponse": response.model_dump_json(indent=2),
    }

    return examples
