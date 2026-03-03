#!/usr/bin/env python3
"""Basic evaluation example for CHIMERA.

This script demonstrates how to run a simple evaluation using the CHIMERA
benchmark framework. It evaluates a model on the calibration track and
displays the results.

Usage:
    python basic_evaluation.py
"""

from chimera.evaluation import EvaluationPipeline, PipelineConfig
from chimera.generators import CalibrationTaskGenerator


def main() -> None:
    """Run a basic calibration evaluation."""
    # Configure the evaluation pipeline
    config = PipelineConfig(
        tracks=["calibration"],
        model_provider="gemini",
        model_name="gemini-2.0-flash",
        n_tasks=50,
        seed=42,
        output_dir="results/basic_eval",
    )

    print("CHIMERA Basic Evaluation")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Tracks: {', '.join(config.tracks)}")
    print(f"Tasks: {config.n_tasks}")
    print()

    # Create and run the pipeline
    pipeline = EvaluationPipeline(config)

    print("Running evaluation...")
    results = pipeline.run()

    # Display results
    print()
    print("Results")
    print("-" * 50)
    print(f"Overall Score: {results.overall_score:.2%}")
    print()

    # Per-track results
    for track_name, track_summary in results.track_summaries.items():
        print(f"{track_name.replace('_', ' ').title()}:")
        print(f"  Score: {track_summary.score:.2%}")
        print(f"  Tasks: {track_summary.n_tasks}")

        # Track-specific metrics
        if hasattr(track_summary, "ece") and track_summary.ece is not None:
            print(f"  ECE: {track_summary.ece:.4f}")
        if hasattr(track_summary, "brier_score") and track_summary.brier_score is not None:
            print(f"  Brier Score: {track_summary.brier_score:.4f}")
        print()

    print(f"Results saved to: {config.output_dir}")


def run_custom_generator_example() -> None:
    """Example using a task generator directly."""
    print("\nCustom Generator Example")
    print("=" * 50)

    # Create a generator with specific settings
    generator = CalibrationTaskGenerator(
        n_tasks=10,
        difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        seed=42,
    )

    # Generate tasks
    tasks = generator.generate()

    print(f"Generated {len(tasks)} tasks")
    print()

    # Show task distribution
    difficulty_counts: dict[str, int] = {}
    for task in tasks:
        diff = task.difficulty.value
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print("Difficulty distribution:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count}")


if __name__ == "__main__":
    main()
    run_custom_generator_example()
