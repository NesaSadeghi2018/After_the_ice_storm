import csv
import json
import time
from pathlib import Path

import numpy as np

from ice_storm.assignments.assignment_b.problem import (
    AssignmentBProblem,
    SCENARIO_NAMES,
)
from ice_storm.optimization.algorithms.mixed_variable_de import (
    mixed_variable_de,
)


CONFIGURATIONS = {
    "DE01": {
        "population_size": 40,
        "mutation_factor": 0.5,
        "crossover_rate": 0.9,
    },
    "DE02": {
        "population_size": 30,
        "mutation_factor": 0.5,
        "crossover_rate": 0.9,
    },
    "DE03": {
        "population_size": 50,
        "mutation_factor": 0.5,
        "crossover_rate": 0.9,
    },
    "DE04": {
        "population_size": 60,
        "mutation_factor": 0.5,
        "crossover_rate": 0.9,
    },
    "DE05": {
        "population_size": 40,
        "mutation_factor": 0.3,
        "crossover_rate": 0.9,
    },
    "DE06": {
        "population_size": 40,
        "mutation_factor": 0.7,
        "crossover_rate": 0.9,
    },
    "DE07": {
        "population_size": 40,
        "mutation_factor": 1.0,
        "crossover_rate": 0.9,
    },
    "DE08": {
        "population_size": 40,
        "mutation_factor": 0.5,
        "crossover_rate": 0.6,
    },
    "DE09": {
        "population_size": 40,
        "mutation_factor": 0.5,
        "crossover_rate": 0.8,
    },
    "DE10": {
        "population_size": 40,
        "mutation_factor": 0.5,
        "crossover_rate": 1.0,
    },
}


SEEDS = [2029301, 2029302, 2029303]

BUDGET = 1000

OUTPUT = Path("results/assignment_b_mvde_tuning.csv")

RESULT_FIELDS = [
    "configuration",
    "scenario",
    "seed",
    "parameters",
    "initial_best_f",
    "final_best_f",
    "improvement",
    "evaluations",
    "runtime_seconds",
]


def run_tuning():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}. "
            "Choose a new output file or move the old file."
        )

    total_runs = (
        len(CONFIGURATIONS)
        * len(SCENARIO_NAMES)
        * len(SEEDS)
    )

    completed = 0

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()

        for scenario_name in SCENARIO_NAMES:

            problem = AssignmentBProblem(scenario_name)

            for config_name, parameters in CONFIGURATIONS.items():

                parameters_json = json.dumps(
                    parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                )

                for seed in SEEDS:

                    start_time = time.perf_counter()

                    result = mixed_variable_de(
                        problem=problem,
                        max_evaluations=BUDGET,
                        seed=seed,
                        **parameters,
                    )

                    runtime = time.perf_counter() - start_time

                    initial_best = float(
                        result["history"][0]["best_f"]
                    )

                    final_best = float(result["best_f"])

                    improvement = initial_best - final_best

                    if result["n_evaluations"] != BUDGET:
                        raise RuntimeError(
                            f"Unexpected evaluation count: "
                            f"{result['n_evaluations']}"
                        )

                    row = {
                        "configuration": config_name,
                        "scenario": scenario_name,
                        "seed": seed,
                        "parameters": parameters_json,
                        "initial_best_f": initial_best,
                        "final_best_f": final_best,
                        "improvement": improvement,
                        "evaluations": result["n_evaluations"],
                        "runtime_seconds": runtime,
                    }

                    writer.writerow(row)
                    file.flush()

                    completed += 1

                    print(
                        f"Progress: {completed}/{total_runs} "
                        f"({100 * completed / total_runs:.1f}%)"
                    )

    print("\nMVDE tuning completed.")
    print(f"Results saved to: {OUTPUT}")

    print("\nSUMMARY BY CONFIGURATION AND SCENARIO")
    print("=" * 85)

    with OUTPUT.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(csv.DictReader(file))

    for config_name in CONFIGURATIONS:

        print(f"\nCONFIGURATION {config_name}")

        scenario_means = []

        for scenario_name in SCENARIO_NAMES:

            selected = [
                float(row["improvement"])
                for row in rows
                if row["configuration"] == config_name
                and row["scenario"] == scenario_name
            ]

            mean_value = float(np.mean(selected))
            median_value = float(np.median(selected))
            std_value = float(np.std(selected, ddof=1))

            scenario_means.append(mean_value)

            print(
                f"{scenario_name}: "
                f"mean={mean_value:.6f}, "
                f"median={median_value:.6f}, "
                f"std={std_value:.6f}"
            )

        print(
            "Overall macro-average improvement: "
            f"{np.mean(scenario_means):.6f}"
        )


if __name__ == "__main__":
    run_tuning()
