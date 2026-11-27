
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PARETO_DIR = ROOT / "results" / "final" / "assignment_c" / "pareto"
OUTPUT_DIR = ROOT / "results" / "final" / "assignment_c" / "analysis"

INPUT_FILES = [
    "assignment_c_pareto_10000_v2.csv",
    "assignment_c_pareto_10000_v2_moead.csv",
    "assignment_c_pareto_10000_v2_spea2.csv",
]

MIN_SERVICE_QUALITY = 0.85
MAX_DEPLOYMENT_COST = 4734.427
MAX_ENERGY_CONSUMPTION = 2304.0

OBJECTIVE_COLUMNS = ["Q", "C", "E"]


def load_pareto_populations():
    frames = []

    for filename in INPUT_FILES:
        path = PARETO_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        frame = pd.read_csv(path)

        required = {
            "scenario",
            "algorithm",
            "Q",
            "C",
            "E",
            "is_pareto",
        }

        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{filename} is missing columns: {sorted(missing)}"
            )

        pareto_mask = (
            frame["is_pareto"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1"])
        )

        frames.append(frame.loc[pareto_mask].copy())

    combined = pd.concat(frames, ignore_index=True)

    if combined.empty:
        raise ValueError("No Pareto solutions were found.")

    combined[OBJECTIVE_COLUMNS] = combined[
        OBJECTIVE_COLUMNS
    ].apply(pd.to_numeric, errors="raise")

    return combined


def global_nondominated_mask(values):
    """Return the nondominated rows for minimization objectives."""
    n = len(values)
    nondominated = np.ones(n, dtype=bool)

    for i in range(n):
        if not nondominated[i]:
            continue

        dominates_i = (
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )

        if np.any(dominates_i):
            nondominated[i] = False

    return nondominated


def select_operational_solutions(combined):
    selected_rows = []
    scenario_summaries = []

    for scenario, group in combined.groupby("scenario"):
        group = (
            group.copy()
            .drop_duplicates(subset=OBJECTIVE_COLUMNS, keep="first")
            .reset_index(drop=True)
        )

        values = group[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
        global_mask = global_nondominated_mask(values)

        global_pareto = group.loc[global_mask].copy()

        global_pareto["service_quality"] = -global_pareto["Q"]

        eligible = global_pareto.loc[
            global_pareto["service_quality"]
            >= MIN_SERVICE_QUALITY
        ].copy()

        if eligible.empty:
            raise ValueError(
                f"No solution meets the service threshold in {scenario}."
            )

        eligible["normalized_cost_energy_distance"] = np.sqrt(
            (eligible["C"] / MAX_DEPLOYMENT_COST) ** 2
            + (eligible["E"] / MAX_ENERGY_CONSUMPTION) ** 2
        )

        eligible = eligible.sort_values(
            "normalized_cost_energy_distance",
            ascending=True,
            kind="stable",
        )

        best = eligible.iloc[0].copy()
        best["global_pareto_count"] = len(global_pareto)
        best["service_eligible_count"] = len(eligible)

        selected_rows.append(best)

        scenario_summaries.append(
            {
                "scenario": scenario,
                "combined_pareto_candidates": len(group),
                "global_pareto_solutions": len(global_pareto),
                "service_eligible_solutions": len(eligible),
                "selected_algorithm": best["algorithm"],
                "selected_service_quality": best["service_quality"],
                "selected_cost": best["C"],
                "selected_energy": best["E"],
                "normalized_cost_energy_distance": (
                    best["normalized_cost_energy_distance"]
                ),
            }
        )

    return (
        pd.DataFrame(selected_rows),
        pd.DataFrame(scenario_summaries),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_pareto_populations()

    selected, summary = select_operational_solutions(combined)

    output_columns = [
        "scenario",
        "algorithm",
        "run",
        "seed",
        "candidate_index",
        "Q",
        "service_quality",
        "C",
        "E",
        "normalized_cost_energy_distance",
        "global_pareto_count",
        "service_eligible_count",
    ]

    decision_variables = [
        column for column in selected.columns
        if column.startswith("x_")
    ]

    output_columns.extend(decision_variables)

    selected_path = (
        OUTPUT_DIR / "assignment_c_operational_solutions.csv"
    )
    summary_path = (
        OUTPUT_DIR / "assignment_c_operational_selection_summary.csv"
    )

    selected[output_columns].to_csv(
        selected_path, index=False, float_format="%.10f"
    )
    summary.to_csv(
        summary_path, index=False, float_format="%.10f"
    )

    print("\nSelected operational solutions:")
    print(
        selected[
            [
                "scenario",
                "algorithm",
                "service_quality",
                "C",
                "E",
                "normalized_cost_energy_distance",
            ]
        ].to_string(index=False)
    )

    print("\nScenario-level selection summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved: {selected_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
