
import argparse
import csv
import json
import time
from pathlib import Path

from ice_storm.assignments.assignment_b.problem import (
    AssignmentBProblem,
    SCENARIO_NAMES,
)
from ice_storm.optimization.algorithms.mixed_variable_ga import (
    mixed_variable_ga,
)
from ice_storm.optimization.algorithms.mixed_variable_de import (
    mixed_variable_de,
)
from ice_storm.optimization.algorithms.mixed_variable_sa import (
    mixed_variable_sa,
)


# ============================================================
# Algorithms
# ============================================================

ALGORITHMS = {
    "MVGA": mixed_variable_ga,
    "MVDE": mixed_variable_de,
    "MVSA": mixed_variable_sa,
}


# Provisional configurations.
# These parameters must be finalized after independent tuning.
ALGORITHM_PARAMETERS = {
    "MVGA": {
        "population_size": 30,
        "crossover_rate": 0.9,
        "mutation_rate": 1 / 24,
        "tournament_size": 3,
        "elitism": 2,
        "mutation_scale": 0.10,
    },
    "MVDE": {
        "population_size": 40,
        "mutation_factor": 0.3,
        "crossover_rate": 0.9,
    },
    "MVSA": {
        "initial_population_size": 40,
        "initial_temperature": 1.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
}


# ============================================================
# Experiment configuration
# ============================================================

RUNS_PER_COMBINATION = 31

DEFAULT_MAX_EVALUATIONS = 10_000

DEFAULT_SEED_BASE = 2028000

DEFAULT_OUTPUT_FILE = (
    "results/assignment_b_runs_10000.csv"
)


RESULT_FIELDS = [
    "scenario",
    "algorithm",
    "run",
    "seed",
    "best_f",
    "Q",
    "C",
    "E",
    "optimization_evaluations",
    "reporting_evaluations",
    "n_evaluations",
    "optimization_runtime_seconds",
    "runtime_seconds",
    "best_x",
    "max_evaluations",
    "parameters",
]


# ============================================================
# Reproducibility
# ============================================================

def make_seed(scenario_index, run_index, seed_base):
    """
    Generate reproducible seeds.

    The same run index uses the same seed across algorithms
    within a scenario.

    Different scenarios and runs receive different seeds.
    """

    return (
        seed_base
        + scenario_index * 1000
        + run_index
    )


def serialize_parameters(parameters):
    """Create a stable JSON representation of parameters."""

    return json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
    )


# ============================================================
# Resume support
# ============================================================

def load_completed_runs(
    output_path,
    max_evaluations,
):
    """
    Load previously completed experiment keys.

    A run is considered complete only when its scenario,
    algorithm, run number, evaluation budget and parameter
    configuration match the current experiment.
    """

    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_fields = {
            "scenario",
            "algorithm",
            "run",
            "max_evaluations",
            "parameters",
        }

        if not required_fields.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Existing results file has an incompatible schema. "
                "Use a new output file."
            )

        for row in reader:

            if int(row["max_evaluations"]) != max_evaluations:
                continue

            key = (
                row["scenario"],
                row["algorithm"],
                int(row["run"]),
                int(row["seed"]),
                row["parameters"],
            )

            completed.add(key)

    return completed


# ============================================================
# Result persistence
# ============================================================

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


# ============================================================
# Single experiment run
# ============================================================

def execute_single_run(
    problem,
    algorithm_name,
    algorithm_function,
    parameters,
    seed,
    max_evaluations,
):
    """
    Execute one optimization run and evaluate its final metrics.

    One evaluation is reserved for final reporting of Q, C and E.
    Therefore, the optimizer receives max_evaluations - 1.
    """

    if max_evaluations < 2:
        raise ValueError(
            "max_evaluations must be at least 2."
        )

    optimization_budget = max_evaluations - 1

    total_start = time.perf_counter()

    result = algorithm_function(
        problem=problem,
        seed=seed,
        max_evaluations=optimization_budget,
        **parameters,
    )

    optimization_runtime = result["runtime_seconds"]

    optimization_evaluations = result["n_evaluations"]

    if optimization_evaluations != optimization_budget:
        raise RuntimeError(
            f"{algorithm_name} did not use its exact optimization budget: "
            f"{optimization_evaluations}/{optimization_budget}"
        )

    best_x = result["best_x"]

    problem.validate_solution(best_x)

    # Final reporting evaluation.
    # This is counted as one additional deployment evaluation.
    metrics = problem.evaluate(best_x)

    reporting_evaluations = 1

    total_evaluations = (
        optimization_evaluations
        + reporting_evaluations
    )

    total_runtime = time.perf_counter() - total_start

    if total_evaluations > max_evaluations:
        raise RuntimeError(
            f"{algorithm_name} exceeded the total evaluation budget."
        )

    if abs(metrics["Q"] - result["best_f"]) > 1e-10:
        raise RuntimeError(
            "Final Q does not match the optimizer's best objective."
        )

    return {
        "best_f": float(result["best_f"]),
        "Q": float(metrics["Q"]),
        "C": float(metrics["C"]),
        "E": float(metrics["E"]),
        "best_x": best_x,
        "optimization_evaluations": optimization_evaluations,
        "reporting_evaluations": reporting_evaluations,
        "n_evaluations": total_evaluations,
        "optimization_runtime_seconds": optimization_runtime,
        "runtime_seconds": total_runtime,
    }


# ============================================================
# Experiment runner
# ============================================================

def run_experiments(
    runs=RUNS_PER_COMBINATION,
    max_evaluations=DEFAULT_MAX_EVALUATIONS,
    seed_base=DEFAULT_SEED_BASE,
    output_file=DEFAULT_OUTPUT_FILE,
    selected_algorithms=None,
):
    """
    Run Assignment B experiments.

    Default design:
        3 scenarios
        3 algorithms
        31 independent runs per combination

    Total:
        279 optimization runs.
    """

    if runs < 1:
        raise ValueError(
            "runs must be at least 1."
        )

    if max_evaluations < 2:
        raise ValueError(
            "max_evaluations must be at least 2."
        )

    if selected_algorithms is None:
        selected_algorithms = list(ALGORITHMS.keys())

    unknown_algorithms = (
        set(selected_algorithms) - set(ALGORITHMS.keys())
    )

    if unknown_algorithms:
        raise ValueError(
            f"Unknown algorithms: {sorted(unknown_algorithms)}"
        )

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed_runs(
        output_path,
        max_evaluations,
    )

    total_runs = (
        len(SCENARIO_NAMES)
        * len(selected_algorithms)
        * runs
    )

    completed_count = 0

    for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):

        problem = AssignmentBProblem(scenario_name)

        for algorithm_name in selected_algorithms:

            algorithm_function = ALGORITHMS[algorithm_name]

            parameters = ALGORITHM_PARAMETERS[algorithm_name]

            parameters_json = serialize_parameters(
                parameters
            )

            for run_index in range(runs):
                seed = make_seed(
                    scenario_index,
                    run_index,
                    seed_base,
                )

                key = (
                    scenario_name,
                    algorithm_name,
                    run_index,
                    seed,
                    parameters_json,
                )

                if key in completed:
                    print(
                        f"Skipping completed run: "
                        f"{scenario_name} / {algorithm_name} / Run {run_index + 1}"
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

                run_result = execute_single_run(
                    problem=problem,
                    algorithm_name=algorithm_name,
                    algorithm_function=algorithm_function,
                    parameters=parameters,
                    seed=seed,
                    max_evaluations=max_evaluations,
                )

                result_row = {
                    "scenario": scenario_name,
                    "algorithm": algorithm_name,
                    "run": run_index,
                    "seed": seed,
                    "best_f": run_result["best_f"],
                    "Q": run_result["Q"],
                    "C": run_result["C"],
                    "E": run_result["E"],
                    "optimization_evaluations": (
                        run_result["optimization_evaluations"]
                    ),
                    "reporting_evaluations": (
                        run_result["reporting_evaluations"]
                    ),
                    "n_evaluations": run_result["n_evaluations"],
                    "optimization_runtime_seconds": (
                        run_result["optimization_runtime_seconds"]
                    ),
                    "runtime_seconds": run_result["runtime_seconds"],
                    "best_x": json.dumps(
                        run_result["best_x"].tolist()
                    ),
                    "max_evaluations": max_evaluations,
                    "parameters": parameters_json,
                }

                append_result(
                    output_path,
                    result_row,
                )

                completed.add(key)

                completed_count += 1

                print(
                    f"[DONE] Q: {run_result['Q']:.8f} | "
                    f"C: {run_result['C']:.4f} | "
                    f"E: {run_result['E']:.4f}"
                )

                print(
                    f"Evaluations: "
                    f"{run_result['n_evaluations']}/"
                    f"{max_evaluations} | "
                    f"Runtime: "
                    f"{run_result['runtime_seconds']:.2f}s"
                )

                print(
                    f"Progress: {completed_count}/{total_runs}"
                )

    print("\nAll requested experiments are complete.")

    print(
        f"Results saved to: {output_path.resolve()}"
    )


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Assignment B mixed-variable optimization experiments."
        )
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
        default=DEFAULT_OUTPUT_FILE,
    )

    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default=None,
    )

    args = parser.parse_args()

    selected_algorithms = (
        [args.algorithm]
        if args.algorithm
        else None
    )

    run_experiments(
        runs=args.runs,
        max_evaluations=args.max_evaluations,
        seed_base=args.seed_base,
        output_file=args.output,
        selected_algorithms=selected_algorithms,
    )


if __name__ == "__main__":
    main()
