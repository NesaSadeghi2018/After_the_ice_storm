
import argparse
import csv
import json
from pathlib import Path

from ice_storm.assignments.assignment_a.problem import (
    AssignmentAProblem,
    SCENARIO_NAMES,
)
from ice_storm.optimization.algorithms.differential_evolution import (
    differential_evolution,
)
from ice_storm.optimization.algorithms.particle_swarm import (
    particle_swarm,
)
from ice_storm.optimization.algorithms.cma_es import cma_es


ALGORITHMS = {
    "DE": differential_evolution,
    "PSO": particle_swarm,
    "CMA-ES": cma_es,
}

RUNS_PER_COMBINATION = 31
DEFAULT_MAX_EVALUATIONS = 10_000
DEFAULT_SEED_BASE = 2027000

RESULT_FIELDS = [
    "scenario",
    "algorithm",
    "run",
    "seed",
    "best_f",
    "n_evaluations",
    "runtime_seconds",
    "best_x",
    "max_evaluations",
]


def make_seed(scenario_index, run_index, seed_base):
    """
    Generate reproducible seeds.

    The same run index uses the same seed across algorithms
    within a scenario, allowing run-index-based comparisons.
    Different runs and scenarios receive different seeds.
    """
    return seed_base + scenario_index * 1000 + run_index


def load_completed_runs(output_path, max_evaluations):
    """Load previously completed experiment keys for resuming."""
    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if int(row["max_evaluations"]) != max_evaluations:
                continue

            key = (
                row["scenario"],
                row["algorithm"],
                int(row["run"]),
            )
            completed.add(key)

    return completed


def append_result(output_path, result_row):
    """Append one completed run and flush it immediately."""
    file_exists = output_path.exists()

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        if not file_exists or output_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(result_row)
        file.flush()


def run_experiments(
    runs=RUNS_PER_COMBINATION,
    max_evaluations=DEFAULT_MAX_EVALUATIONS,
    seed_base=DEFAULT_SEED_BASE,
    output_file="results/assignment_a_runs_10000.csv",
):
    """Run all Assignment A experiments with incremental CSV saving."""

    if runs < 1:
        raise ValueError("runs must be at least 1.")

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1.")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_runs(
        output_path,
        max_evaluations,
    )

    total_runs = len(SCENARIO_NAMES) * len(ALGORITHMS) * runs
    completed_count = 0

    for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
        problem = AssignmentAProblem(scenario_name)

        for algorithm_name, algorithm_function in ALGORITHMS.items():
            for run_index in range(1, runs + 1):

                key = (
                    scenario_name,
                    algorithm_name,
                    run_index,
                )

                if key in completed:
                    completed_count += 1
                    print(
                        f"[SKIP] {scenario_name} | "
                        f"{algorithm_name} | Run {run_index}"
                    )
                    continue

                seed = make_seed(
                    scenario_index,
                    run_index,
                    seed_base,
                )

                print(
                    f"\n[RUN] {scenario_name} | "
                    f"{algorithm_name} | "
                    f"Run {run_index}/{runs} | "
                    f"Seed {seed}"
                )

                result = algorithm_function(
                    problem=problem,
                    seed=seed,
                    max_evaluations=max_evaluations,
                )

                if result["n_evaluations"] > max_evaluations:
                    raise RuntimeError(
                        f"{algorithm_name} exceeded the evaluation budget."
                    )

                result_row = {
                    "scenario": scenario_name,
                    "algorithm": algorithm_name,
                    "run": run_index,
                    "seed": seed,
                    "best_f": result["best_f"],
                    "n_evaluations": result["n_evaluations"],
                    "runtime_seconds": result["runtime_seconds"],
                    "best_x": json.dumps(
                        result["best_x"].tolist()
                        if hasattr(result["best_x"], "tolist")
                        else result["best_x"]
                    ),
                    "max_evaluations": max_evaluations,
                }

                append_result(output_path, result_row)
                completed.add(key)
                completed_count += 1

                print(
                    f"[DONE] Best Q: {result['best_f']:.8f} | "
                    f"Evaluations: {result['n_evaluations']} | "
                    f"Runtime: {result['runtime_seconds']:.2f}s"
                )

                print(
                    f"Progress: {completed_count}/{total_runs}"
                )

    print("\nAll requested experiments are complete.")
    print(f"Results saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Assignment A optimization experiments."
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_COMBINATION,
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=DEFAULT_MAX_EVALUATIONS,
    )

    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/assignment_a_runs_10000.csv",
    )

    args = parser.parse_args()

    run_experiments(
        runs=args.runs,
        max_evaluations=args.max_evaluations,
        seed_base=args.seed_base,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
