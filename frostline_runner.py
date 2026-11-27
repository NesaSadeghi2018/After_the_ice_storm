
"""Central experiment runner for the Greyhaven Frostline Layout project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def run_assignment_a(
    runs: int,
    max_evaluations: int,
    output_dir: Path,
) -> None:
    from ice_storm.assignments.assignment_a import experiment

    output_file = output_dir / (
        f"assignment_a_runs_{max_evaluations}.csv"
    )

    print("\n" + "=" * 65)
    print("ASSIGNMENT A: CONTINUOUS OPTIMIZATION")
    print("=" * 65)

    experiment.run_experiments(
        runs=runs,
        max_evaluations=max_evaluations,
        seed_base=experiment.DEFAULT_SEED_BASE,
        output_file=str(output_file),
    )


def run_assignment_b(
    runs: int,
    max_evaluations: int,
    output_dir: Path,
) -> None:
    from ice_storm.assignments.assignment_b import experiment

    output_file = output_dir / (
        f"assignment_b_runs_{max_evaluations}.csv"
    )

    print("\n" + "=" * 65)
    print("ASSIGNMENT B: MIXED-VARIABLE SINGLE-OBJECTIVE OPTIMIZATION")
    print("=" * 65)

    experiment.run_experiments(
        runs=runs,
        max_evaluations=max_evaluations,
        seed_base=experiment.DEFAULT_SEED_BASE,
        output_file=str(output_file),
        selected_algorithms=None,
    )


def run_assignment_c(
    runs: int,
    max_evaluations: int,
    output_dir: Path,
) -> None:
    from ice_storm.assignments.assignment_c import experiment

    runs_output = output_dir / (
        f"assignment_c_runs_{max_evaluations}.csv"
    )
    pareto_output = output_dir / (
        f"assignment_c_pareto_{max_evaluations}.csv"
    )

    print("\n" + "=" * 65)
    print("ASSIGNMENT C: MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 65)

    experiment.run_experiments(
        runs=runs,
        max_evaluations=max_evaluations,
        seed_base=experiment.DEFAULT_SEED_BASE,
        runs_output_file=str(runs_output),
        pareto_output_file=str(pareto_output),
        algorithms=tuple(experiment.DEFAULT_ALGORITHMS),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Central experiment runner for the Greyhaven "
            "Frostline Layout optimization project."
        )
    )

    parser.add_argument(
        "--assignment",
        required=True,
        choices=["A", "B", "C", "all"],
        help="Select Assignment A, B, C, or all assignments.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=31,
        help="Independent runs per algorithm and scenario (default: 31).",
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=10000,
        help="Maximum evaluations per run (default: 10000).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/frostline_runner",
        help="Directory for experiment output files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display the execution plan without running experiments.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1.")

    if args.max_evaluations < 1:
        parser.error("--max-evaluations must be at least 1.")

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    selected = (
        ["A", "B", "C"]
        if args.assignment == "all"
        else [args.assignment]
    )

    print("\nGREYHAVEN FROSTLINE EXPERIMENT RUNNER")
    print("-" * 65)
    print(f"Selected assignments : {', '.join(selected)}")
    print(f"Independent runs      : {args.runs}")
    print(f"Evaluation budget     : {args.max_evaluations:,}")
    print(f"Output directory      : {output_dir}")
    print()
    print("-" * 65)

    if args.dry_run:
        print("\nDry run completed. No experiments were executed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "A": run_assignment_a,
        "B": run_assignment_b,
        "C": run_assignment_c,
    }

    for assignment in selected:
        runners[assignment](
            runs=args.runs,
            max_evaluations=args.max_evaluations,
            output_dir=output_dir,
        )

    print("\n" + "=" * 65)
    print("ALL REQUESTED FROSTLINE EXPERIMENTS COMPLETED")
    print(f"Results directory: {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
