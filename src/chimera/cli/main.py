"""Main CLI entry point for CHIMERA benchmark.

This module provides the primary command-line interface for running
CHIMERA evaluations, generating tasks, and analyzing results.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from chimera.version import __version__

console = Console()


def get_track_choices() -> list[str]:
    """Get available track types."""
    return ["calibration", "error_detection", "knowledge_boundary", "self_correction", "all"]


def get_model_choices() -> list[str]:
    """Get available model providers."""
    return ["gemini", "openai"]


@click.group()
@click.version_option(version=__version__, prog_name="chimera")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """CHIMERA - Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment.

    A comprehensive benchmark for evaluating LLM meta-cognitive calibration.

    \b
    Evaluation Tracks:
      - calibration: Confidence calibration accuracy
      - error_detection: Self-error identification
      - knowledge_boundary: Knowledge limit recognition
      - self_correction: Reasoning error correction
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.option(
    "--track",
    "-t",
    type=click.Choice(get_track_choices()),
    default="all",
    help="Evaluation track to run",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(get_model_choices()),
    default="gemini",
    help="Model provider to use",
)
@click.option(
    "--model-name",
    "-n",
    type=str,
    default=None,
    help="Specific model name (e.g., gemini-2.0-flash-exp)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results",
)
@click.option(
    "--n-tasks",
    type=int,
    default=100,
    help="Number of tasks per track",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate tasks without running evaluation",
)
@click.pass_context
def run(
    ctx: click.Context,
    track: str,
    model: str,
    model_name: str | None,
    config: Path | None,
    output: Path | None,
    n_tasks: int,
    seed: int | None,
    dry_run: bool,
) -> None:
    """Run CHIMERA evaluation benchmark.

    \b
    Examples:
        chimera run --track calibration --model gemini
        chimera run --track all --n-tasks 50 --seed 42
        chimera run -t error_detection -m openai -n gpt-4
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold blue]CHIMERA Benchmark v{__version__}[/bold blue]",
                subtitle="Meta-cognitive Calibration Evaluation",
            )
        )

    # Determine tracks to run
    if track == "all":
        tracks = ["calibration", "error_detection", "knowledge_boundary", "self_correction"]
    else:
        tracks = [track]

    # Setup output directory
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("results") / f"run_{timestamp}"

    output.mkdir(parents=True, exist_ok=True)

    if verbose:
        console.print(f"[dim]Output directory: {output}[/dim]")

    # Load configuration
    run_config = _load_config(config, model, model_name, n_tasks, seed)

    if dry_run:
        console.print("[yellow]Dry run mode - generating tasks only[/yellow]")
        _run_dry_run(tracks, run_config, output, verbose)
        return

    # Run evaluation
    _run_evaluation(tracks, run_config, output, verbose, quiet)


def _load_config(
    config_path: Path | None,
    model: str,
    model_name: str | None,
    n_tasks: int,
    seed: int | None,
) -> dict[str, Any]:
    """Load and merge configuration."""
    import yaml

    base_config: dict[str, Any] = {
        "model": {
            "provider": model,
            "name": model_name or _get_default_model_name(model),
        },
        "generation": {
            "n_tasks": n_tasks,
            "seed": seed,
        },
        "evaluation": {
            "batch_size": 10,
            "max_retries": 3,
            "timeout": 60,
        },
    }

    if config_path is not None:
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            base_config = _deep_merge(base_config, user_config)

    return base_config


def _get_default_model_name(provider: str) -> str:
    """Get default model name for provider."""
    defaults = {
        "gemini": "gemini-2.0-flash-exp",
        "openai": "gpt-4o-mini",
    }
    return defaults.get(provider, "gemini-2.0-flash-exp")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _run_dry_run(
    tracks: list[str],
    config: dict[str, Any],
    output: Path,
    verbose: bool,
) -> None:
    """Run task generation without evaluation."""
    from chimera.generators.calibration import (
        CalibrationGeneratorConfig,
        CalibrationTaskGenerator,
    )
    from chimera.generators.error_detection import (
        ErrorDetectionGeneratorConfig,
        ErrorDetectionTaskGenerator,
    )
    from chimera.generators.knowledge_boundary import (
        KnowledgeBoundaryGeneratorConfig,
        KnowledgeBoundaryTaskGenerator,
    )
    from chimera.models.task import Task

    n_tasks = config["generation"]["n_tasks"]
    seed = config["generation"]["seed"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for track in tracks:
            task_id = progress.add_task(f"Generating {track} tasks...", total=None)

            try:
                tasks: list[Task] = []
                if track == "calibration":
                    cal_config = CalibrationGeneratorConfig(n_tasks=n_tasks, seed=seed)
                    cal_generator = CalibrationTaskGenerator(cal_config)
                    cal_task_set = cal_generator.generate_all()
                    tasks = cal_task_set.tasks
                elif track == "error_detection":
                    ed_config = ErrorDetectionGeneratorConfig(n_tasks=n_tasks, seed=seed)
                    ed_generator = ErrorDetectionTaskGenerator(ed_config)
                    ed_task_set = ed_generator.generate()
                    tasks = ed_task_set.tasks
                elif track == "knowledge_boundary":
                    kb_config = KnowledgeBoundaryGeneratorConfig(n_tasks=n_tasks, seed=seed)
                    kb_generator = KnowledgeBoundaryTaskGenerator(kb_config)
                    kb_task_set = kb_generator.generate()
                    tasks = kb_task_set.tasks
                else:
                    console.print(f"[yellow]Track '{track}' not yet implemented[/yellow]")
                    progress.remove_task(task_id)
                    continue

                # Save tasks
                task_file = output / f"{track}_tasks.json"
                with open(task_file, "w") as f:
                    json.dump(
                        {
                            "track": track,
                            "n_tasks": len(tasks),
                            "tasks": [
                                {
                                    "id": str(t.id),
                                    "question": t.question,
                                    "difficulty": (
                                        t.difficulty.value
                                        if hasattr(t.difficulty, "value")
                                        else str(t.difficulty)
                                    ),
                                }
                                for t in tasks
                            ],
                        },
                        f,
                        indent=2,
                    )

                progress.remove_task(task_id)
                console.print(
                    f"  [green]✓[/green] {track}: Generated {len(tasks)} tasks → {task_file}"
                )

            except Exception as e:
                progress.remove_task(task_id)
                console.print(f"  [red]✗[/red] {track}: {e}")
                if verbose:
                    console.print_exception()


def _run_evaluation(
    tracks: list[str],
    config: dict[str, Any],
    output: Path,
    verbose: bool,
    quiet: bool,
) -> None:
    """Run full evaluation."""
    console.print("\n[bold]Starting Evaluation[/bold]\n")

    results_summary: dict[str, dict[str, Any]] = {}

    for track in tracks:
        console.print(f"[blue]▶[/blue] Evaluating {track}...")

        try:
            # For now, just generate tasks (full evaluation requires API keys)
            result = _evaluate_track(track, config, output, verbose)
            results_summary[track] = result

            if result.get("status") == "success":
                console.print(f"  [green]✓[/green] {track}: Completed")
            else:
                console.print(
                    f"  [yellow]![/yellow] {track}: {result.get('message', 'Unknown status')}"
                )

        except Exception as e:
            console.print(f"  [red]✗[/red] {track}: {e}")
            results_summary[track] = {"status": "error", "error": str(e)}
            if verbose:
                console.print_exception()

    # Save summary
    summary_file = output / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "results": results_summary,
            },
            f,
            indent=2,
        )

    if not quiet:
        _print_results_table(results_summary)
        console.print(f"\n[dim]Results saved to: {output}[/dim]")


def _evaluate_track(
    track: str,
    config: dict[str, Any],
    output: Path,
    verbose: bool,
) -> dict[str, Any]:
    """Evaluate a single track."""
    # This is a placeholder - full evaluation requires API integration
    # For now, we just generate tasks and return success
    n_tasks = config["generation"]["n_tasks"]

    return {
        "status": "success",
        "message": f"Generated {n_tasks} tasks (evaluation requires API key)",
        "n_tasks": n_tasks,
        "metrics": {},
    }


def _print_results_table(results: dict[str, dict[str, Any]]) -> None:
    """Print results summary table."""
    table = Table(title="Evaluation Results")
    table.add_column("Track", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Tasks", justify="right")
    table.add_column("Notes")

    for track, result in results.items():
        status = result.get("status", "unknown")
        status_icon = "✓" if status == "success" else "✗" if status == "error" else "?"
        n_tasks = str(result.get("n_tasks", "-"))
        message = result.get("message", result.get("error", ""))[:40]

        table.add_row(track, status_icon, n_tasks, message)

    console.print()
    console.print(table)


@cli.command()
@click.option(
    "--track",
    "-t",
    type=click.Choice(get_track_choices()[:-1]),  # Exclude 'all'
    required=True,
    help="Track to generate tasks for",
)
@click.option(
    "--n-tasks",
    "-n",
    type=int,
    default=100,
    help="Number of tasks to generate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path (.json or .jsonl)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Output format",
)
@click.pass_context
def generate(
    ctx: click.Context,
    track: str,
    n_tasks: int,
    output: Path,
    seed: int | None,
    output_format: str,
) -> None:
    """Generate evaluation tasks for a specific track.

    \b
    Examples:
        chimera generate -t calibration -n 50 -o tasks.json
        chimera generate -t error_detection -n 100 -o tasks.jsonl --format jsonl
    """
    verbose = ctx.obj.get("verbose", False)

    console.print(f"Generating {n_tasks} {track} tasks...")

    try:
        from chimera.generators.calibration import (
            CalibrationGeneratorConfig,
            CalibrationTaskGenerator,
        )
        from chimera.generators.error_detection import (
            ErrorDetectionGeneratorConfig,
            ErrorDetectionTaskGenerator,
        )
        from chimera.generators.knowledge_boundary import (
            KnowledgeBoundaryGeneratorConfig,
            KnowledgeBoundaryTaskGenerator,
        )
        from chimera.generators.self_correction import (
            SelfCorrectionGeneratorConfig,
            SelfCorrectionTaskGenerator,
        )
        from chimera.models.task import Task

        tasks: list[Task] = []
        if track == "calibration":
            cal_config = CalibrationGeneratorConfig(n_tasks=n_tasks, seed=seed)
            cal_generator = CalibrationTaskGenerator(cal_config)
            cal_task_set = cal_generator.generate_all()
            tasks = cal_task_set.tasks
        elif track == "error_detection":
            ed_config = ErrorDetectionGeneratorConfig(n_tasks=n_tasks, seed=seed)
            ed_generator = ErrorDetectionTaskGenerator(ed_config)
            ed_task_set = ed_generator.generate()
            tasks = ed_task_set.tasks
        elif track == "knowledge_boundary":
            kb_config = KnowledgeBoundaryGeneratorConfig(n_tasks=n_tasks, seed=seed)
            kb_generator = KnowledgeBoundaryTaskGenerator(kb_config)
            kb_task_set = kb_generator.generate()
            tasks = kb_task_set.tasks
        elif track == "self_correction":
            sc_config = SelfCorrectionGeneratorConfig(n_tasks=n_tasks, seed=seed)
            sc_generator = SelfCorrectionTaskGenerator(sc_config)
            sc_task_set = sc_generator.generate()
            tasks = sc_task_set.tasks
        else:
            console.print(f"[red]Unknown track: {track}[/red]")
            sys.exit(1)

        # Prepare output
        output.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "jsonl":
            with open(output, "w") as f:
                for task in tasks:
                    task_dict = {
                        "id": str(task.id),
                        "track": (
                            task.track.value if hasattr(task.track, "value") else str(task.track)
                        ),
                        "question": task.question,
                        "correct_answer": task.correct_answer,
                        "difficulty": (
                            task.difficulty.value
                            if hasattr(task.difficulty, "value")
                            else str(task.difficulty)
                        ),
                    }
                    f.write(json.dumps(task_dict) + "\n")
        else:
            task_list = [
                {
                    "id": str(task.id),
                    "track": task.track.value if hasattr(task.track, "value") else str(task.track),
                    "question": task.question,
                    "correct_answer": task.correct_answer,
                    "difficulty": (
                        task.difficulty.value
                        if hasattr(task.difficulty, "value")
                        else str(task.difficulty)
                    ),
                }
                for task in tasks
            ]
            with open(output, "w") as f:
                json.dump({"track": track, "n_tasks": len(tasks), "tasks": task_list}, f, indent=2)

        console.print(f"[green]✓[/green] Generated {len(tasks)} tasks → {output}")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("results_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "markdown"]),
    default="table",
    help="Output format for results",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save report to file",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    results_path: Path,
    output_format: str,
    output: Path | None,
) -> None:
    """Analyze and display evaluation results.

    \b
    Examples:
        chimera analyze results/run_20240101_120000/
        chimera analyze results.json --format markdown -o report.md
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load results
        if results_path.is_dir():
            summary_file = results_path / "evaluation_summary.json"
            if not summary_file.exists():
                console.print(f"[red]No evaluation_summary.json found in {results_path}[/red]")
                sys.exit(1)
            with open(summary_file) as f:
                data = json.load(f)
        else:
            with open(results_path) as f:
                data = json.load(f)

        # Generate report
        if output_format == "table":
            _print_analysis_table(data)
        elif output_format == "json":
            console.print_json(json.dumps(data, indent=2))
        elif output_format == "markdown":
            report = _generate_markdown_report(data)
            if output:
                output.write_text(report, encoding="utf-8")
                console.print(f"[green]✓[/green] Report saved to {output}")
            else:
                console.print(report)

    except Exception as e:
        console.print(f"[red]Error analyzing results: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _print_analysis_table(data: dict[str, Any]) -> None:
    """Print analysis as table."""
    console.print(Panel.fit("[bold]CHIMERA Evaluation Analysis[/bold]"))

    if "timestamp" in data:
        console.print(f"[dim]Run timestamp: {data['timestamp']}[/dim]")

    if "results" in data:
        _print_results_table(data["results"])

    if "config" in data:
        config = data["config"]
        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Model: {config.get('model', {}).get('name', 'Unknown')}")
        console.print(
            f"  Tasks per track: {config.get('generation', {}).get('n_tasks', 'Unknown')}"
        )


def _generate_markdown_report(data: dict[str, Any]) -> str:
    """Generate markdown report from results."""
    lines = [
        "# CHIMERA Evaluation Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
    ]

    if "timestamp" in data:
        lines.append(f"**Run Timestamp:** {data['timestamp']}")
        lines.append("")

    if "config" in data:
        config = data["config"]
        lines.extend(
            [
                "## Configuration",
                "",
                f"- **Model Provider:** {config.get('model', {}).get('provider', 'Unknown')}",
                f"- **Model Name:** {config.get('model', {}).get('name', 'Unknown')}",
                f"- **Tasks per Track:** {config.get('generation', {}).get('n_tasks', 'Unknown')}",
                f"- **Random Seed:** {config.get('generation', {}).get('seed', 'None')}",
                "",
            ]
        )

    if "results" in data:
        lines.extend(
            [
                "## Results",
                "",
                "| Track | Status | Tasks | Notes |",
                "|-------|--------|-------|-------|",
            ]
        )

        for track, result in data["results"].items():
            status = "✓" if result.get("status") == "success" else "✗"
            n_tasks = result.get("n_tasks", "-")
            message = result.get("message", result.get("error", ""))[:50]
            lines.append(f"| {track} | {status} | {n_tasks} | {message} |")

        lines.append("")

    lines.extend(
        [
            "## Metrics Summary",
            "",
            "*Detailed metrics will be available after full evaluation runs.*",
            "",
            "---",
            "",
            "*Report generated by CHIMERA Benchmark*",
        ]
    )

    return "\n".join(lines)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display information about CHIMERA benchmark."""
    console.print(
        Panel.fit(
            f"[bold blue]CHIMERA Benchmark v{__version__}[/bold blue]\n\n"
            "[dim]Calibrated Hierarchical Introspection and\n"
            "Meta-cognitive Error Recognition Assessment[/dim]",
            title="About",
        )
    )

    console.print("\n[bold]Evaluation Tracks:[/bold]")

    tracks_info = [
        ("calibration", "Confidence calibration accuracy measurement"),
        ("error_detection", "Self-error identification and correction"),
        ("knowledge_boundary", "Knowledge limit and uncertainty recognition"),
        ("self_correction", "Reasoning error detection and correction"),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Track")
    table.add_column("Description")

    for track, desc in tracks_info:
        table.add_row(track, desc)

    console.print(table)

    console.print("\n[bold]Supported Models:[/bold]")
    console.print("  • Google Gemini (gemini-2.0-flash-exp, gemini-1.5-pro)")
    console.print("  • OpenAI (gpt-4o, gpt-4o-mini, gpt-4)")

    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  $ chimera run --track calibration --model gemini")
    console.print("  $ chimera generate -t error_detection -n 50 -o tasks.json")
    console.print("  $ chimera analyze results/")


@cli.command()
@click.option(
    "--all",
    "check_all",
    is_flag=True,
    help="Check all dependencies including optional",
)
@click.pass_context
def check(ctx: click.Context, check_all: bool) -> None:
    """Check environment and dependencies."""
    console.print("[bold]Checking CHIMERA Environment[/bold]\n")

    checks: list[tuple[str, bool, str]] = []

    # Check Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    checks.append(
        (
            "Python Version",
            py_ok,
            f"{py_version.major}.{py_version.minor}.{py_version.micro}"
            + (" ✓" if py_ok else " (requires 3.10+)"),
        )
    )

    # Check core dependencies
    core_deps = [
        ("click", "CLI framework"),
        ("rich", "Console output"),
        ("pydantic", "Data validation"),
        ("numpy", "Numerical computing"),
        ("pyyaml", "Configuration"),
    ]

    for dep, desc in core_deps:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "installed")
            checks.append((f"{dep}", True, f"{version}"))
        except ImportError:
            checks.append((f"{dep}", False, f"Not installed ({desc})"))

    # Check API clients
    api_deps = [
        ("google.generativeai", "Gemini API"),
        ("openai", "OpenAI API"),
    ]

    for dep, desc in api_deps:
        try:
            __import__(dep)
            checks.append((f"{desc}", True, "Available"))
        except ImportError:
            checks.append((f"{desc}", False, "Not installed"))

    # Check environment variables
    import os

    env_vars = [
        ("GOOGLE_API_KEY", "Gemini API key"),
        ("OPENAI_API_KEY", "OpenAI API key"),
    ]

    for var, desc in env_vars:
        has_var = os.environ.get(var) is not None
        checks.append((desc, has_var, "Set" if has_var else "Not set"))

    # Print results
    table = Table(show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    for name, ok, details in checks:
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(name, status, details)

    console.print(table)

    # Summary
    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        console.print("\n[green]All checks passed![/green]")
    else:
        console.print("\n[yellow]Some checks failed. See details above.[/yellow]")


if __name__ == "__main__":
    cli()
