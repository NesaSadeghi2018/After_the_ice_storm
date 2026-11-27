
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCENARIOS = [
    "initial_response",
    "evening_pressure",
    "northern_outage",
]

ALGORITHM_NAME = "MVNSGA-II"

EXPECTED_RUNS_PER_SCENARIO = 31
EXPECTED_POPULATION_SIZE = 40
EXPECTED_DIMENSION = 24

OBJECTIVES = ["Q", "C", "E"]

DECISION_COLUMNS = [
    f"x_{i}"
    for i in range(EXPECTED_DIMENSION)
]

RUN_REQUIRED_COLUMNS = {
    "scenario",
    "algorithm",
    "run",
    "seed",
    "n_evaluations",
    "max_evaluations",
    "population_size",
    "n_pareto_solutions",
    "runtime_seconds",
    "parameters",
}

POPULATION_REQUIRED_COLUMNS = {
    "scenario",
    "algorithm",
    "run",
    "seed",
    "candidate_index",
    *DECISION_COLUMNS,
    *OBJECTIVES,
    "is_pareto",
}


def load_results(
    runs_path: str | Path,
    population_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate Assignment C experiment results."""

    runs_path = Path(runs_path)
    population_path = Path(population_path)

    for path in (runs_path, population_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Results file not found: {path}"
            )

    runs = pd.read_csv(runs_path)
    population = pd.read_csv(population_path)

    if runs.empty:
        raise ValueError("Run summary file is empty.")

    if population.empty:
        raise ValueError("Population results file is empty.")

    missing_runs = RUN_REQUIRED_COLUMNS - set(runs.columns)

    if missing_runs:
        raise ValueError(
            f"Missing columns in run summary: "
            f"{sorted(missing_runs)}"
        )

    missing_population = (
        POPULATION_REQUIRED_COLUMNS - set(population.columns)
    )

    if missing_population:
        raise ValueError(
            f"Missing columns in population results: "
            f"{sorted(missing_population)}"
        )

    expected_runs = (
        len(SCENARIOS) * EXPECTED_RUNS_PER_SCENARIO
    )

    expected_population = (
        expected_runs * EXPECTED_POPULATION_SIZE
    )

    if len(runs) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} run records, "
            f"found {len(runs)}."
        )

    if len(population) != expected_population:
        raise ValueError(
            f"Expected {expected_population} population records, "
            f"found {len(population)}."
        )

    if set(runs["scenario"].unique()) != set(SCENARIOS):
        raise ValueError(
            "Run summary contains unexpected scenarios."
        )

    if set(population["scenario"].unique()) != set(SCENARIOS):
        raise ValueError(
            "Population data contains unexpected scenarios."
        )

    if not (runs["algorithm"] == ALGORITHM_NAME).all():
        raise ValueError("Unexpected algorithm in run summary.")

    if not (population["algorithm"] == ALGORITHM_NAME).all():
        raise ValueError("Unexpected algorithm in population data.")

    numeric_run_columns = [
        "run",
        "seed",
        "n_evaluations",
        "max_evaluations",
        "population_size",
        "n_pareto_solutions",
        "runtime_seconds",
    ]

    numeric_population_columns = [
        "run",
        "seed",
        "candidate_index",
        *DECISION_COLUMNS,
        *OBJECTIVES,
    ]

    for column in numeric_run_columns:
        runs[column] = pd.to_numeric(
            runs[column],
            errors="raise",
        )

        if not np.isfinite(runs[column]).all():
            raise ValueError(
                f"Non-finite values found in run column: {column}"
            )

    for column in numeric_population_columns:
        population[column] = pd.to_numeric(
            population[column],
            errors="raise",
        )

        if not np.isfinite(population[column]).all():
            raise ValueError(
                f"Non-finite values found in population column: {column}"
            )

    if (runs["runtime_seconds"] <= 0).any():
        raise ValueError("Runtime values must be positive.")

    if not (
        runs["n_evaluations"] == runs["max_evaluations"]
    ).all():
        raise ValueError(
            "Some runs did not use their exact evaluation budget."
        )

    if not (
        runs["population_size"] == EXPECTED_POPULATION_SIZE
    ).all():
        raise ValueError("Unexpected population size.")

    if not runs["n_pareto_solutions"].between(
        1,
        EXPECTED_POPULATION_SIZE,
    ).all():
        raise ValueError("Invalid Pareto solution count.")

    run_keys = ["scenario", "algorithm", "run"]

    if runs.duplicated(run_keys).any():
        raise ValueError("Duplicate run summary records found.")

    if population.duplicated(
        run_keys + ["candidate_index"]
    ).any():
        raise ValueError("Duplicate candidate records found.")

    for scenario in SCENARIOS:
        scenario_runs = runs[
            runs["scenario"] == scenario
        ]

        scenario_population = population[
            population["scenario"] == scenario
        ]

        expected_run_indices = set(
            range(EXPECTED_RUNS_PER_SCENARIO)
        )

        if set(scenario_runs["run"]) != expected_run_indices:
            raise ValueError(
                f"Incomplete run indices for scenario: {scenario}"
            )

        for _, run_record in scenario_runs.iterrows():
            run_id = run_record["run"]

            group = scenario_population[
                scenario_population["run"] == run_id
            ]

            if len(group) != EXPECTED_POPULATION_SIZE:
                raise ValueError(
                    f"Incorrect population size for "
                    f"{scenario}, run {run_id}."
                )

            expected_indices = set(
                range(EXPECTED_POPULATION_SIZE)
            )

            if set(group["candidate_index"]) != expected_indices:
                raise ValueError(
                    f"Invalid candidate indices for "
                    f"{scenario}, run {run_id}."
                )

            if not (group["seed"] == run_record["seed"]).all():
                raise ValueError(
                    f"Seed mismatch for {scenario}, run {run_id}."
                )

            pareto_count = int(
                group["is_pareto"].astype(bool).sum()
            )

            if pareto_count != int(
                run_record["n_pareto_solutions"]
            ):
                raise ValueError(
                    f"Pareto count mismatch for "
                    f"{scenario}, run {run_id}."
                )

    return runs, population


def descriptive_statistics(
    runs: pd.DataFrame,
    population: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive statistics for runs and objectives."""

    run_metrics = [
        "runtime_seconds",
        "n_evaluations",
        "n_pareto_solutions",
    ]

    run_records = []

    for scenario, group in runs.groupby("scenario", sort=False):
        for metric in run_metrics:
            values = group[metric].to_numpy(dtype=float)

            run_records.append({
                "scenario": scenario,
                "metric": metric,
                "n": len(values),
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "median": np.median(values),
                "q1": np.percentile(values, 25),
                "q3": np.percentile(values, 75),
                "min": np.min(values),
                "max": np.max(values),
            })

    run_statistics = pd.DataFrame(run_records)

    objective_records = []

    for scenario, scenario_group in population.groupby(
        "scenario",
        sort=False,
    ):
        for pareto_status, label in [
            (False, "population"),
            (True, "pareto"),
        ]:
            selected = scenario_group[
                scenario_group["is_pareto"].astype(bool)
                == pareto_status
            ]

            for objective in OBJECTIVES:
                values = selected[objective].to_numpy(dtype=float)

                if len(values) == 0:
                    continue

                objective_records.append({
                    "scenario": scenario,
                    "subset": label,
                    "objective": objective,
                    "n": len(values),
                    "mean": np.mean(values),
                    "std": (
                        np.std(values, ddof=1)
                        if len(values) > 1
                        else 0.0
                    ),
                    "median": np.median(values),
                    "q1": np.percentile(values, 25),
                    "q3": np.percentile(values, 75),
                    "min": np.min(values),
                    "max": np.max(values),
                })

    objective_statistics = pd.DataFrame(objective_records)

    return run_statistics, objective_statistics



def no_service_summary(
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Count fully inactive and zero-objective solutions."""

    records = []

    active_columns = ["x_5", "x_11", "x_17", "x_23"]

    for scenario, group in population.groupby(
        "scenario",
        sort=False,
    ):
        fully_inactive = (
            group[active_columns] == 0
        ).all(axis=1)

        zero_objectives = np.isclose(
            group[OBJECTIVES].to_numpy(dtype=float),
            0.0,
        ).all(axis=1)

        records.append({
            "scenario": scenario,
            "total_solutions": len(group),
            "fully_inactive_solutions": int(
                fully_inactive.sum()
            ),
            "zero_objective_solutions": int(
                zero_objectives.sum()
            ),
            "pareto_fully_inactive": int(
                (
                    fully_inactive
                    & group["is_pareto"].astype(bool)
                ).sum()
            ),
        })

    return pd.DataFrame(records)



def plot_objective_distributions(
    population: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Generate objective distribution boxplots by scenario."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for objective in OBJECTIVES:
        data = [
            population.loc[
                population["scenario"] == scenario,
                objective,
            ].to_numpy(dtype=float)
            for scenario in SCENARIOS
        ]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.boxplot(
            data,
            tick_labels=SCENARIOS,
            showfliers=True,
        )

        ax.set_title(
            f"{objective} Distribution Across Scenarios"
        )
        ax.set_xlabel("Scenario")
        ax.set_ylabel(objective)
        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()

        fig.savefig(
            output_path / f"objective_{objective.lower()}_boxplot.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)



def plot_pareto_fronts(
    population: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Plot pairwise objective spaces for each scenario."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    objective_pairs = [
        ("Q", "C"),
        ("Q", "E"),
        ("C", "E"),
    ]

    for scenario in SCENARIOS:
        scenario_data = population[
            population["scenario"] == scenario
        ].copy()

        no_service = np.isclose(
            scenario_data[OBJECTIVES].to_numpy(dtype=float),
            0.0,
        ).all(axis=1)

        for x_objective, y_objective in objective_pairs:
            fig, ax = plt.subplots(figsize=(8, 6))

            ax.scatter(
                scenario_data.loc[
                    ~no_service, x_objective
                ],
                scenario_data.loc[
                    ~no_service, y_objective
                ],
                alpha=0.55,
                s=22,
                label="Active deployment",
            )

            if no_service.any():
                ax.scatter(
                    scenario_data.loc[
                        no_service, x_objective
                    ],
                    scenario_data.loc[
                        no_service, y_objective
                    ],
                    marker="X",
                    s=75,
                    label="No-service deployment",
                )

            ax.set_title(
                f"{scenario}: {x_objective} vs {y_objective}"
            )
            ax.set_xlabel(x_objective)
            ax.set_ylabel(y_objective)
            ax.grid(alpha=0.3)
            ax.legend()

            fig.tight_layout()

            filename = (
                f"pareto_{scenario}_{x_objective.lower()}_"
                f"{y_objective.lower()}.png"
            )

            fig.savefig(
                output_path / filename,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)



def save_analysis_tables(
    runs: pd.DataFrame,
    population: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save descriptive and no-service analysis tables."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_statistics, objective_statistics = (
        descriptive_statistics(runs, population)
    )

    no_service_statistics = no_service_summary(population)

    run_statistics.to_csv(
        output_path / "run_descriptive_statistics.csv",
        index=False,
    )

    objective_statistics.to_csv(
        output_path / "objective_descriptive_statistics.csv",
        index=False,
    )

    no_service_statistics.to_csv(
        output_path / "no_service_summary.csv",
        index=False,
    )



def solution_diversity_summary(
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Measure decision-space and objective-space uniqueness per run."""

    decision_columns = [
        f"x_{i}" for i in range(EXPECTED_DIMENSION)
    ]

    continuous_indices = [
        0, 1, 2,
        6, 7, 8,
        12, 13, 14,
        18, 19, 20,
    ]

    records = []

    group_columns = ["scenario", "algorithm", "run", "seed"]

    for keys, group in population.groupby(
        group_columns,
        sort=False,
    ):
        scenario, algorithm, run, seed = keys

        decisions = group[decision_columns].to_numpy(
            dtype=float
        )

        rounded_decisions = decisions.copy()
        rounded_decisions[:, continuous_indices] = np.round(
            rounded_decisions[:, continuous_indices],
            decimals=6,
        )

        objectives = group[OBJECTIVES].to_numpy(
            dtype=float
        )

        rounded_objectives = np.round(
            objectives,
            decimals=8,
        )

        unique_decisions = np.unique(
            rounded_decisions,
            axis=0,
        ).shape[0]

        unique_objectives = np.unique(
            rounded_objectives,
            axis=0,
        ).shape[0]

        records.append({
            "scenario": scenario,
            "algorithm": algorithm,
            "run": int(run),
            "seed": int(seed),
            "population_size": len(group),
            "unique_decision_vectors": unique_decisions,
            "duplicate_decision_vectors": (
                len(group) - unique_decisions
            ),
            "unique_objective_vectors": unique_objectives,
            "duplicate_objective_vectors": (
                len(group) - unique_objectives
            ),
        })

    return pd.DataFrame(records)




def save_diversity_analysis(
    population: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Save per-run solution diversity and scenario summaries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    diversity = solution_diversity_summary(population)

    diversity.to_csv(
        output_path / "solution_diversity_by_run.csv",
        index=False,
    )

    metrics = [
        "unique_decision_vectors",
        "duplicate_decision_vectors",
        "unique_objective_vectors",
        "duplicate_objective_vectors",
    ]

    scenario_summary = (
        diversity.groupby("scenario")[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    scenario_summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in scenario_summary.columns
    ]

    scenario_summary.to_csv(
        output_path / "solution_diversity_summary.csv",
        index=False,
    )

    return diversity


def generate_report(
    runs_path: str | Path,
    population_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Run the complete Assignment C analysis pipeline."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading and validating experiment results...")

    runs, population = load_results(
        runs_path,
        population_path,
    )

    print("[2/5] Calculating descriptive statistics...")

    save_analysis_tables(
        runs,
        population,
        output_path,
    )

    print("[3/5] Calculating solution diversity...")

    save_diversity_analysis(
        population,
        output_path,
    )

    print("[4/5] Generating objective distribution plots...")

    plot_objective_distributions(
        population,
        output_path,
    )

    print("[5/5] Generating Pareto front plots...")

    plot_pareto_fronts(
        population,
        output_path,
    )

    print("\nASSIGNMENT C ANALYSIS COMPLETED")
    print("Output directory:", output_path.resolve())
    print("Runs analyzed:", len(runs))
    print("Population records analyzed:", len(population))
    print(
        "Generated CSV files:",
        len(list(output_path.glob("*.csv"))),
    )
    print(
        "Generated PNG files:",
        len(list(output_path.glob("*.png"))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Assignment C multi-objective optimization results."
    )

    parser.add_argument(
        "--runs",
        default="results/assignment_c_runs_10000.csv",
        help="Path to the experiment summary CSV.",
    )

    parser.add_argument(
        "--population",
        default="results/assignment_c_pareto_10000.csv",
        help="Path to the final population CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/assignment_c_analysis",
        help="Directory for analysis tables and plots.",
    )

    args = parser.parse_args()

    generate_report(
        runs_path=args.runs,
        population_path=args.population,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
