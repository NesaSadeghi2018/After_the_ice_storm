
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


ALGORITHMS = ["DE", "PSO", "CMA-ES"]

METRICS = [
    "best_f",
    "n_evaluations",
    "runtime_seconds",
]

LOWER_IS_BETTER = True


def load_results(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the completed experiment results."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "scenario",
        "algorithm",
        "run",
        "seed",
        "best_f",
        "n_evaluations",
        "runtime_seconds",
        "best_x",
        "max_evaluations",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Results file is empty.")

    for metric in METRICS:
        df[metric] = pd.to_numeric(
            df[metric],
            errors="raise",
        )

        if not np.isfinite(df[metric]).all():
            raise ValueError(
                f"Non-finite values found in {metric}."
            )

    return df


def descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive statistics for each group."""

    records = []

    grouped = df.groupby(
        ["scenario", "algorithm"],
        sort=False,
    )

    for (scenario, algorithm), group in grouped:
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)

            records.append({
                "scenario": scenario,
                "algorithm": algorithm,
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

    return pd.DataFrame(records)


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """Apply Holm-Bonferroni correction."""

    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    if m == 0:
        return []

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, original_index in enumerate(order):
        corrected = (m - rank) * p_values[original_index]

        running_max = max(
            running_max,
            corrected,
        )

        adjusted[original_index] = min(
            running_max,
            1.0,
        )

    return adjusted.tolist()


def cliff_delta(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate Cliff's delta.

    Standard definition:
    delta = P(A > B) - P(A < B)

    Since all project metrics are minimized,
    a negative delta favors algorithm A.
    """

    greater = 0
    less = 0

    for value_a in a:
        greater += np.sum(value_a > b)
        less += np.sum(value_a < b)

    total = len(a) * len(b)

    if total == 0:
        return np.nan

    return float((greater - less) / total)


def cliff_effect_label(delta: float) -> str:
    """Interpret the magnitude of Cliff's delta."""

    magnitude = abs(delta)

    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"

    return "large"


def omnibus_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run Kruskal-Wallis tests for each scenario and metric."""

    records = []

    scenarios = sorted(df["scenario"].unique())

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]

        for metric in METRICS:
            samples = []

            for algorithm in ALGORITHMS:
                values = scenario_df.loc[
                    scenario_df["algorithm"] == algorithm,
                    metric,
                ].to_numpy(dtype=float)

                if len(values) == 0:
                    raise ValueError(
                        f"Missing data: {scenario}, {algorithm}, {metric}"
                    )

                samples.append(values)

            try:
                result = stats.kruskal(*samples)

                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            except ValueError as exc:
                if "identical" not in str(exc).lower():
                    raise

                statistic = 0.0
                p_value = 1.0

            records.append({
                "scenario": scenario,
                "metric": metric,
                "test": "Kruskal-Wallis",
                "statistic": statistic,
                "p_value": p_value,
                "significant_0.05": p_value < 0.05,
            })

    return pd.DataFrame(records)


def pairwise_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run pairwise Mann-Whitney U tests.

    Holm correction is applied separately within each
    scenario-metric family of three pairwise comparisons.
    """

    records = []

    scenarios = sorted(df["scenario"].unique())

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]

        for metric in METRICS:
            family_records = []

            for algorithm_a, algorithm_b in combinations(
                ALGORITHMS,
                2,
            ):
                a = scenario_df.loc[
                    scenario_df["algorithm"] == algorithm_a,
                    metric,
                ].to_numpy(dtype=float)

                b = scenario_df.loc[
                    scenario_df["algorithm"] == algorithm_b,
                    metric,
                ].to_numpy(dtype=float)


                if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
                    u_statistic = len(a) * len(b) / 2
                    p_raw = 1.0
                else:
                    result = stats.mannwhitneyu(
                        a,
                        b,
                        alternative="two-sided",
                        method="auto",
                    )
                    u_statistic = float(result.statistic)
                    p_raw = float(result.pvalue)

                if not np.isfinite(p_raw):
                    raise ValueError(
                        f"Non-finite p-value for {scenario}, "
                        f"{metric}, {algorithm_a} vs {algorithm_b}"
                    )


                delta = cliff_delta(a, b)

                family_records.append({
                    "scenario": scenario,
                    "metric": metric,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "n_a": len(a),
                    "n_b": len(b),
                    "u_statistic": u_statistic,
                    "p_raw": p_raw,
                    "cliff_delta": delta,
                    "effect_magnitude": cliff_effect_label(delta),
                    "direction": (
                        f"{algorithm_a} tends to have lower values"
                        if delta < 0
                        else (
                            f"{algorithm_b} tends to have lower values"
                            if delta > 0
                            else "No directional difference"
                        )
                    ),
                })

            adjusted = holm_adjust([
                item["p_raw"]
                for item in family_records
            ])

            for item, adjusted_p in zip(
                family_records,
                adjusted,
            ):
                item["p_holm"] = adjusted_p
                item["significant_0.05"] = adjusted_p < 0.05

                records.append(item)

    return pd.DataFrame(records)


def generate_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Generate boxplots for all metrics across scenarios."""

    output_dir = Path(output_dir)
    scenarios = sorted(df["scenario"].unique())

    for metric in METRICS:
        fig, axes = plt.subplots(
            1,
            len(scenarios),
            figsize=(15, 5),
        )

        if len(scenarios) == 1:
            axes = [axes]

        for ax, scenario in zip(axes, scenarios):
            scenario_df = df[df["scenario"] == scenario]

            data = [
                scenario_df.loc[
                    scenario_df["algorithm"] == algorithm,
                    metric,
                ].to_numpy(dtype=float)
                for algorithm in ALGORITHMS
            ]

            ax.boxplot(
                data,
                tick_labels=ALGORITHMS,
                showmeans=True,
            )

            ax.set_title(scenario.replace("_", " ").title())
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Algorithm Comparison: {metric}",
            fontsize=14,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / f"{metric}_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def generate_report(
    input_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate all statistical analysis outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results(input_path)

    descriptive = descriptive_statistics(df)
    omnibus = omnibus_tests(df)
    pairwise = pairwise_tests(df)
    generate_plots(df, output_dir)

    descriptive.to_csv(
        output_dir / "descriptive_statistics.csv",
        index=False,
    )

    omnibus.to_csv(
        output_dir / "omnibus_tests.csv",
        index=False,
    )

    pairwise.to_csv(
        output_dir / "pairwise_tests.csv",
        index=False,
    )

    print("\nSTATISTICAL REPORT GENERATED")
    print("--------------------------------")
    print(f"Input records: {len(df)}")
    print(f"Scenarios: {df['scenario'].nunique()}")
    print(f"Algorithms: {df['algorithm'].nunique()}")

    print("\nDescriptive statistics:")
    print(descriptive.to_string(index=False))

    print("\nKruskal-Wallis tests:")
    print(omnibus.to_string(index=False))

    print("\nPairwise Mann-Whitney tests:")
    print(pairwise.to_string(index=False))

    print("\nOutput files:")
    print(output_dir / "descriptive_statistics.csv")
    print(output_dir / "omnibus_tests.csv")
    print(output_dir / "pairwise_tests.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical analysis for Assignment A."
    )

    parser.add_argument(
        "--input",
        default="results/assignment_a_runs_10000.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="results/assignment_a_analysis",
    )

    args = parser.parse_args()

    generate_report(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
