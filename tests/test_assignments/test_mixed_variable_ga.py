
import numpy as np
import pytest

from ice_storm.assignments.assignment_b.problem import (
    AssignmentBProblem,
)

from ice_storm.optimization.algorithms.mixed_variable_ga import (
    mixed_variable_ga,
)


@pytest.fixture
def problem():
    return AssignmentBProblem("initial_response")


def test_mvga_returns_expected_result_structure(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
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

    assert expected_keys.issubset(result.keys())
    assert result["algorithm"] == "MVGA"


def test_mvga_respects_exact_evaluation_budget(problem):

    budget = 103

    result = mixed_variable_ga(
        problem,
        max_evaluations=budget,
        seed=42,
        population_size=20,
    )

    assert result["n_evaluations"] == budget


def test_mvga_returns_valid_solution(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    assert problem.validate_solution(result["best_x"])


def test_mvga_best_f_matches_objective(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    expected = problem.objective(result["best_x"])

    assert result["best_f"] == pytest.approx(expected)


def test_mvga_is_reproducible(problem):

    result_1 = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    result_2 = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    np.testing.assert_array_equal(
        result_1["best_x"],
        result_2["best_x"],
    )

    assert result_1["best_f"] == pytest.approx(
        result_2["best_f"]
    )


def test_mvga_history_is_monotonically_nonincreasing(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=200,
        seed=42,
        population_size=20,
    )

    values = [
        item["best_f"]
        for item in result["history"]
    ]

    assert all(
        values[i] <= values[i - 1]
        for i in range(1, len(values))
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 0},
        {"population_size": 1},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"tournament_size": 0},
        {"elitism": 20, "population_size": 20},
    ],
)
def test_mvga_rejects_invalid_parameters(problem, kwargs):

    with pytest.raises(ValueError):

        mixed_variable_ga(
            problem,
            seed=42,
            **kwargs,
        )


def test_mvga_handles_budget_smaller_than_population(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=7,
        seed=42,
        population_size=40,
    )

    assert result["n_evaluations"] == 7
    assert problem.validate_solution(result["best_x"])


def test_mvga_accepts_custom_mutation_scale(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
        mutation_scale=0.2,
    )

    assert result["n_evaluations"] == 100
    assert problem.validate_solution(result["best_x"])


def test_mvga_accepts_zero_mutation_scale(problem):

    result = mixed_variable_ga(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
        mutation_scale=0.0,
    )

    assert result["n_evaluations"] == 100
    assert problem.validate_solution(result["best_x"])


@pytest.mark.parametrize(
    "mutation_scale",
    [-0.1, float("nan"), float("inf")],
)
def test_mvga_rejects_invalid_mutation_scale(
    problem,
    mutation_scale,
):

    with pytest.raises(ValueError):

        mixed_variable_ga(
            problem,
            seed=42,
            mutation_scale=mutation_scale,
        )
