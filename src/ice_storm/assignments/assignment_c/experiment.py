from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ice_storm.assignments.assignment_b.problem import SCENARIO_NAMES
from ice_storm.assignments.assignment_c.problem import AssignmentCProblem
from ice_storm.optimization.algorithms.nsga_ii import mixed_variable_nsga_ii
from ice_storm.optimization.algorithms.multiobjective import (
    mixed_variable_moead,
    mixed_variable_spea2,
)


# ============================================================
# Algorithm configuration
# ============================================================

RUNS_PER_COMBINATION = 31
DEFAULT_MAX_EVALUATIONS = 10_000
DEFAULT_SEED_BASE = 2_029_000

ALGORITHMS: dict[str, tuple[Callable, dict]] = {
    "MVNSGA-II": (
        mixed_variable_nsga_ii,
        {
            "population_size": 40,
            "crossover_rate": 0.9,
            "mutation_rate": 1 / 24,
            "tournament_size": 2,
            "eta_c": 20.0,
            "eta_m": 20.0,
        },
    ),
    "MOEA/D": (
        mixed_variable_moead,
        {
            "population_size": 40,
            "neighborhood_size": 10,
            "neighborhood_mating_probability": 0.9,
            "max_replacements": 2,
            "crossover_rate": 0.9,
            "mutation_rate": 1 / 24,
            "eta_c": 20.0,
            "eta_m": 20.0,
        },
    ),
    "SPEA2": (
        mixed_variable_spea2,
        {
            "population_size": 40,
            "archive_size": 40,
            "crossover_rate": 0.9,
            "mutation_rate": 1 / 24,
            "eta_c": 20.0,
            "eta_m": 20.0,
        },
    ),
}

DEFAULT_ALGORITHMS = tuple(ALGORITHMS)


# ============================================================
# Output configuration and CSV schemas
# ============================================================

DEFAULT_RUNS_OUTPUT_FILE = "results/assignment_c_runs_10000.csv"
DEFAULT_PARETO_OUTPUT_FILE = "results/assignment_c_pareto_10000.csv"

RUN_RESULT_FIELDS = [
    "scenario", "algorithm", "run", "seed", "n_evaluations",
    "max_evaluations", "population_size", "n_pareto_solutions",
    "runtime_seconds", "parameters",
]

PARETO_RESULT_FIELDS = [
    "scenario", "algorithm", "run", "seed", "candidate_index",
    *[f"x_{index}" for index in range(24)],
    "Q", "C", "E", "is_pareto",
]


# ============================================================
# Reproducibility and serialization
# ============================================================

def make_seed(scenario_index: int, run_index: int, seed_base: int) -> int:
    """Generate a reproducible, independent seed for each scenario/run pair."""
    return seed_base + scenario_index * 1000 + run_index


def serialize_parameters(parameters: dict) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def algorithm_output_path(base_path: str | Path, algorithm_name: str) -> Path:
    """Keep the historical NSGA-II filenames; give other algorithms distinct files."""
    path = Path(base_path)
    if algorithm_name == "MVNSGA-II":
        return path
    suffix = "moead" if algorithm_name == "MOEA/D" else "spea2"
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


# ============================================================
# CSV persistence
# ============================================================

def append_csv_rows(output_path, fieldnames, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


def load_completed_runs(
    output_path,
    algorithm_name: str,
    algorithm_parameters: dict,
    max_evaluations: int,
    seed_base: int,
):
    """Load only completed runs matching this algorithm's exact configuration."""
    completed = set()
    output_path = Path(output_path)
    if not output_path.exists():
        return completed

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not set(RUN_RESULT_FIELDS).issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Existing runs CSV has an incompatible schema: {output_path}. "
                "Use a new output file."
            )
        expected_parameters = serialize_parameters(algorithm_parameters)
        for row in reader:
            if row["algorithm"] != algorithm_name:
                continue
            if int(row["max_evaluations"]) != max_evaluations:
                continue
            if row["parameters"] != expected_parameters:
                continue
            scenario_name = row["scenario"]
            if scenario_name not in SCENARIO_NAMES:
                continue
            run_index = int(row["run"])
            scenario_index = SCENARIO_NAMES.index(scenario_name)
            expected_seed = make_seed(scenario_index, run_index, seed_base)
            if int(row["seed"]) == expected_seed:
                completed.add((scenario_name, algorithm_name, run_index, expected_seed))
    return completed


# ============================================================
# Single experiment run and validation
# ============================================================

def execute_single_run(
    problem,
    algorithm_name: str,
    seed: int,
    max_evaluations: int,
):
    optimizer, parameters = ALGORITHMS[algorithm_name]
    start_time = time.perf_counter()
    result = optimizer(
        problem=problem,
        seed=seed,
        max_evaluations=max_evaluations,
        **parameters,
    )
    runtime_seconds = time.perf_counter() - start_time

    if result["algorithm"] != algorithm_name:
        raise RuntimeError(
            f"Unexpected algorithm label: {result['algorithm']!r}; "
            f"expected {algorithm_name!r}."
        )
    if result["n_evaluations"] != max_evaluations:
        raise RuntimeError(
            f"{algorithm_name} did not use its exact evaluation budget: "
            f"{result['n_evaluations']}/{max_evaluations}."
        )

    population = np.asarray(result["population"], dtype=float)
    objectives = np.asarray(result["objectives"], dtype=float)
    pareto_solutions = np.asarray(result["pareto_solutions"], dtype=float)
    pareto_objectives = np.asarray(result["pareto_objectives"], dtype=float)

    expected_population_size = min(parameters["population_size"], max_evaluations)
    if algorithm_name == "SPEA2":
        expected_population_size = min(parameters["archive_size"], max_evaluations)

    if population.shape != (expected_population_size, problem.dimension):
        raise RuntimeError(
            f"{algorithm_name}: incorrect final population shape {population.shape}."
        )
    if objectives.shape != (expected_population_size, problem.n_objectives):
        raise RuntimeError(
            f"{algorithm_name}: incorrect objective matrix shape {objectives.shape}."
        )
    if pareto_solutions.ndim != 2 or pareto_solutions.shape[1] != problem.dimension:
        raise RuntimeError(f"{algorithm_name}: invalid Pareto decision matrix shape.")
    if pareto_objectives.shape != (len(pareto_solutions), problem.n_objectives):
        raise RuntimeError(f"{algorithm_name}: invalid Pareto objective matrix shape.")
    if len(pareto_solutions) == 0:
        raise RuntimeError(f"{algorithm_name}: final Pareto front is empty.")
    for name, values in (
        ("population", population),
        ("objectives", objectives),
        ("Pareto solutions", pareto_solutions),
        ("Pareto objectives", pareto_objectives),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{algorithm_name}: {name} contain non-finite values.")
    for solution in population:
        problem.validate_solution(solution)

    return {
        "population": population,
        "objectives": objectives,
        "pareto_solutions": pareto_solutions,
        "pareto_objectives": pareto_objectives,
        "n_evaluations": result["n_evaluations"],
        "runtime_seconds": runtime_seconds,
        "history": result["history"],
    }


# ============================================================
# Result preparation
# ============================================================

def prepare_result_rows(
    scenario_name: str,
    algorithm_name: str,
    algorithm_parameters: dict,
    run_index: int,
    seed: int,
    max_evaluations: int,
    run_result: dict,
):
    parameters_json = serialize_parameters(algorithm_parameters)
    population = run_result["population"]
    objectives = run_result["objectives"]
    pareto_solutions = run_result["pareto_solutions"]

    summary_row = {
        "scenario": scenario_name,
        "algorithm": algorithm_name,
        "run": run_index,
        "seed": seed,
        "n_evaluations": run_result["n_evaluations"],
        "max_evaluations": max_evaluations,
        "population_size": len(population),
        "n_pareto_solutions": len(pareto_solutions),
        "runtime_seconds": run_result["runtime_seconds"],
        "parameters": parameters_json,
    }

    pareto_keys = {
        np.asarray(solution, dtype=float).tobytes()
        for solution in pareto_solutions
    }
    population_rows = []
    for candidate_index, (solution, values) in enumerate(zip(population, objectives)):
        solution = np.asarray(solution, dtype=float)
        row = {
            "scenario": scenario_name,
            "algorithm": algorithm_name,
            "run": run_index,
            "seed": seed,
            "candidate_index": candidate_index,
            "Q": float(values[0]),
            "C": float(values[1]),
            "E": float(values[2]),
            "is_pareto": solution.tobytes() in pareto_keys,
        }
        for variable_index, value in enumerate(solution):
            row[f"x_{variable_index}"] = float(value)
        population_rows.append(row)

    if sum(bool(row["is_pareto"]) for row in population_rows) != len(pareto_solutions):
        raise RuntimeError(
            f"{algorithm_name}: Pareto membership count does not match its output."
        )
    return summary_row, population_rows


def remove_incomplete_population_rows(output_path, completed):
    """Drop population rows whose corresponding summary completion record is absent."""
    output_path = Path(output_path)
    if not output_path.exists():
        return
    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not set(PARETO_RESULT_FIELDS).issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Existing population CSV has an incompatible schema: {output_path}. "
                "Use a new output file."
            )
        rows = list(reader)

    valid_rows = []
    for row in rows:
        key = (row["scenario"], row["algorithm"], int(row["run"]), int(row["seed"]))
        if key in completed:
            valid_rows.append(row)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PARETO_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(valid_rows)
        file.flush()
    temporary_path.replace(output_path)


# ============================================================
# Experiment runner
# ============================================================

def run_experiments(
    runs: int = RUNS_PER_COMBINATION,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
    seed_base: int = DEFAULT_SEED_BASE,
    runs_output_file: str | Path = DEFAULT_RUNS_OUTPUT_FILE,
    pareto_output_file: str | Path = DEFAULT_PARETO_OUTPUT_FILE,
    algorithms: tuple[str, ...] | list[str] = DEFAULT_ALGORITHMS,
):
    """Run the requested algorithms over all Assignment C scenarios."""
    if runs < 1:
        raise ValueError("runs must be at least 1.")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1.")
    if not algorithms:
        raise ValueError("At least one algorithm must be selected.")
    unknown = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. Available: {list(ALGORITHMS)}")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("Algorithm names must not contain duplicates.")

    total_runs = len(SCENARIO_NAMES) * runs * len(algorithms)
    completed_count = 0
    print(
        f"Starting Assignment C experiments: {len(SCENARIO_NAMES)} scenarios, "
        f"{len(algorithms)} algorithms, {runs} runs per combination, "
        f"{total_runs} total runs."
    )
    print(f"Evaluation budget per run: {max_evaluations}")

    for algorithm_name in algorithms:
        parameters = ALGORITHMS[algorithm_name][1]
        runs_path = algorithm_output_path(runs_output_file, algorithm_name)
        pareto_path = algorithm_output_path(pareto_output_file, algorithm_name)
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        pareto_path.parent.mkdir(parents=True, exist_ok=True)

        completed = load_completed_runs(
            runs_path, algorithm_name, parameters, max_evaluations, seed_base
        )
        remove_incomplete_population_rows(pareto_path, completed)
        print(f"\n{'=' * 68}\nALGORITHM: {algorithm_name}\n{'=' * 68}")

        for scenario_index, scenario_name in enumerate(SCENARIO_NAMES):
            problem = AssignmentCProblem(scenario_name)
            for run_index in range(runs):
                seed = make_seed(scenario_index, run_index, seed_base)
                key = (scenario_name, algorithm_name, run_index, seed)
                if key in completed:
                    print(f"[SKIP] {scenario_name} | Run {run_index + 1}/{runs}")
                    completed_count += 1
                    continue

                print(
                    f"\n[RUN] {algorithm_name} | {scenario_name} | "
                    f"Run {run_index + 1}/{runs} | Seed {seed}"
                )
                run_result = execute_single_run(
                    problem, algorithm_name, seed, max_evaluations
                )
                summary_row, population_rows = prepare_result_rows(
                    scenario_name=scenario_name,
                    algorithm_name=algorithm_name,
                    algorithm_parameters=parameters,
                    run_index=run_index,
                    seed=seed,
                    max_evaluations=max_evaluations,
                    run_result=run_result,
                )

                # Population details first; summary row is the completion marker.
                append_csv_rows(pareto_path, PARETO_RESULT_FIELDS, population_rows)
                append_csv_rows(runs_path, RUN_RESULT_FIELDS, [summary_row])
                completed.add(key)
                completed_count += 1
                print(
                    f"[DONE] Pareto: {summary_row['n_pareto_solutions']} | "
                    f"Evaluations: {run_result['n_evaluations']}/{max_evaluations} | "
                    f"Runtime: {run_result['runtime_seconds']:.2f}s"
                )
                print(f"Progress: {completed_count}/{total_runs}")

        print(f"\nRun summaries: {runs_path.resolve()}")
        print(f"Population details: {pareto_path.resolve()}")

    print("\nAll requested experiments are complete.")


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Assignment C mixed-variable multi-objective optimization "
            "experiments using MVNSGA-II, MOEA/D, and SPEA2."
        )
    )
    parser.add_argument("--runs", type=int, default=RUNS_PER_COMBINATION,
                        help="Number of independent runs per scenario and algorithm.")
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS,
                        help="Exact objective-evaluation budget per run.")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                        help="Base value used for reproducible random seeds.")
    parser.add_argument("--runs-output", type=str, default=DEFAULT_RUNS_OUTPUT_FILE,
                        help="Base CSV path for experiment summaries.")
    parser.add_argument("--pareto-output", type=str, default=DEFAULT_PARETO_OUTPUT_FILE,
                        help="Base CSV path for final population details.")
    parser.add_argument(
        "--algorithms", nargs="+", choices=list(ALGORITHMS),
        default=list(DEFAULT_ALGORITHMS),
        help="Algorithms to run. Defaults to all three.",
    )
    args = parser.parse_args()
    run_experiments(
        runs=args.runs,
        max_evaluations=args.max_evaluations,
        seed_base=args.seed_base,
        runs_output_file=args.runs_output,
        pareto_output_file=args.pareto_output,
        algorithms=args.algorithms,
    )


if __name__ == "__main__":
    main()
