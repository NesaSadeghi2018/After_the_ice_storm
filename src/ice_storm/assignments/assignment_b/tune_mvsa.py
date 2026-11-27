
import csv
import json
import time
from pathlib import Path

import numpy as np

from ice_storm.assignments.assignment_b.problem import (
    AssignmentBProblem,
    SCENARIO_NAMES,
)
from ice_storm.optimization.algorithms.mixed_variable_sa import (
    mixed_variable_sa,
)


CONFIGURATIONS = {
    "SA01": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA02": {
        "initial_population_size": 30,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA03": {
        "initial_population_size": 50,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA04": {
        "initial_population_size": 60,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA05": {
        "initial_population_size": 40,
        "initial_temperature": 1.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA06": {
        "initial_population_size": 40,
        "initial_temperature": 10.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA07": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.990,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA08": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.999,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA09": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.05,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA10": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.20,
        "reheat_after": 500,
        "reheat_factor": 2.0,
    },
    "SA11": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 250,
        "reheat_factor": 2.0,
    },
    "SA12": {
        "initial_population_size": 40,
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 1e-8,
        "continuous_step_size": 0.10,
        "reheat_after": 500,
        "reheat_factor": 10.0,
    },
}


SEEDS = [2029301, 2029302, 2029303]

BUDGET = 1000

OUTPUT = Path("results/assignment_b_mvsa_tuning.csv")

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

                    result = mixed_variable_sa(
                        problem=problem,
                        max_evaluations=BUDGET,
                        seed=seed,
                        **parameters,
                    )

                    runtime = time.perf_counter() - start_time

                    if result["n_evaluations"] != BUDGET:
                        raise RuntimeError(
                            f"Unexpected evaluation count: "
                            f"{result['n_evaluations']}"
                        )

                    initial_population_size = (
                        parameters["initial_population_size"]
                    )

                    initial_best = float(
                        result["history"][
                            initial_population_size - 1
                        ]["best_f"]
                    )

                    final_best = float(result["best_f"])

                    improvement = initial_best - final_best

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

    print("\nMVSA tuning completed.")
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
