
import numpy as np
import pytest

from ice_storm.assignments.assignment_b.problem import AssignmentBProblem
from ice_storm.optimization.algorithms.mixed_variable_de import (
    mixed_variable_de,
)


@pytest.fixture
def problem():
    return AssignmentBProblem("initial_response")


def test_result_structure(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=50,
        seed=42,
        population_size=10,
    )

    expected_keys = {
        "algorithm",
        "best_x",
        "best_f",
        "n_evaluations",
        "runtime_seconds",
        "seed",
        "history",
    }

    assert set(result.keys()) == expected_keys
    assert result["algorithm"] == "MVDE"
    assert result["best_x"].shape == (24,)
    assert np.isfinite(result["best_f"])
    assert result["runtime_seconds"] >= 0


@pytest.mark.parametrize(
    "budget",
    [1, 2, 3, 7, 13, 43, 100],
)
def test_exact_evaluation_budget(problem, budget):
    result = mixed_variable_de(
        problem,
        max_evaluations=budget,
        seed=42,
        population_size=10,
    )

    assert result["n_evaluations"] == budget


def test_best_solution_is_valid(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=10,
    )

    problem.validate_solution(result["best_x"])


def test_best_f_matches_objective(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=10,
    )

    actual_f = problem.objective(result["best_x"])

    assert result["best_f"] == pytest.approx(actual_f)


def test_reproducibility(problem):
    result_1 = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=123,
        population_size=10,
    )

    result_2 = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=123,
        population_size=10,
    )

    assert np.array_equal(
        result_1["best_x"],
        result_2["best_x"],
    )

    assert result_1["best_f"] == result_2["best_f"]


def test_different_seeds_can_produce_different_solutions(problem):
    result_1 = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=10,
        population_size=10,
    )

    result_2 = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=20,
        population_size=10,
    )

    assert (
        not np.array_equal(
            result_1["best_x"],
            result_2["best_x"],
        )
        or result_1["best_f"] != result_2["best_f"]
    )


def test_history_is_monotonic(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=10,
    )

    history_values = [
        entry["best_f"]
        for entry in result["history"]
    ]

    assert all(
        history_values[i] <= history_values[i - 1]
        for i in range(1, len(history_values))
    )


def test_history_tracks_evaluation_count(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=43,
        seed=42,
        population_size=10,
    )

    evaluation_counts = [
        entry["n_evaluations"]
        for entry in result["history"]
    ]

    assert evaluation_counts[0] == 10
    assert evaluation_counts[-1] == 43

    assert all(
        evaluation_counts[i] > evaluation_counts[i - 1]
        for i in range(1, len(evaluation_counts))
    )


def test_discrete_variables_remain_valid(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=200,
        seed=42,
        population_size=20,
    )

    solution = result["best_x"]

    for index in problem.height_indices:
        assert solution[index] in (2, 3, 4, 5, 6)

    for index in problem.channel_indices:
        assert solution[index] in (1, 2, 3)

    for index in problem.active_indices:
        assert solution[index] in (0, 1)


def test_small_budget_returns_valid_solution(problem):
    result = mixed_variable_de(
        problem,
        max_evaluations=2,
        seed=42,
        population_size=10,
    )

    assert result["n_evaluations"] == 2
    problem.validate_solution(result["best_x"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 0},
        {"population_size": 3},
        {"mutation_factor": 0},
        {"mutation_factor": 2.1},
        {"crossover_rate": -0.1},
        {"crossover_rate": 1.1},
    ],
)
def test_invalid_parameters_raise_error(problem, kwargs):
    with pytest.raises(ValueError):
        mixed_variable_de(
            problem,
            seed=42,
            **kwargs,
        )
