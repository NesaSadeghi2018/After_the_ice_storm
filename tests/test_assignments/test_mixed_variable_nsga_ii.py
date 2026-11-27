
import numpy as np
import pytest

from ice_storm.assignments.assignment_c.problem import (
    AssignmentCProblem,
)

from ice_storm.optimization.algorithms.multiobjective_utils import (
    dominates,
)

from ice_storm.optimization.algorithms.nsga_ii import (
    mixed_variable_nsga_ii,
)


@pytest.fixture
def problem():
    return AssignmentCProblem("initial_response")


def test_nsga_ii_returns_expected_result_structure(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    expected_keys = {
        "algorithm",
        "population",
        "objectives",
        "pareto_solutions",
        "pareto_objectives",
        "n_evaluations",
        "runtime_seconds",
        "seed",
        "history",
    }

    assert expected_keys.issubset(result.keys())
    assert result["algorithm"] == "MVNSGA-II"


def test_nsga_ii_returns_correct_array_shapes(problem):

    population_size = 20

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=population_size,
    )

    assert result["population"].shape == (
        population_size,
        problem.dimension,
    )

    assert result["objectives"].shape == (
        population_size,
        problem.n_objectives,
    )

    assert result["pareto_solutions"].ndim == 2
    assert result["pareto_solutions"].shape[1] == problem.dimension

    assert result["pareto_objectives"].shape == (
        len(result["pareto_solutions"]),
        problem.n_objectives,
    )


def test_nsga_ii_respects_exact_evaluation_budget(problem):

    budget = 103

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=budget,
        seed=42,
        population_size=20,
    )

    assert result["n_evaluations"] == budget


def test_nsga_ii_handles_budget_smaller_than_population(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=7,
        seed=42,
        population_size=40,
    )

    assert result["n_evaluations"] == 7
    assert len(result["population"]) == 7


def test_nsga_ii_returns_valid_solutions(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    for solution in result["population"]:
        assert problem.validate_solution(solution)

    for solution in result["pareto_solutions"]:
        assert problem.validate_solution(solution)


def test_nsga_ii_objectives_match_problem_evaluation(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    expected_objectives = np.asarray(
        [
            problem.objectives(solution)
            for solution in result["population"]
        ]
    )

    np.testing.assert_allclose(
        result["objectives"],
        expected_objectives,
    )


def test_nsga_ii_pareto_objectives_match_solutions(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    expected_objectives = np.asarray(
        [
            problem.objectives(solution)
            for solution in result["pareto_solutions"]
        ]
    )

    np.testing.assert_allclose(
        result["pareto_objectives"],
        expected_objectives,
    )


def test_nsga_ii_pareto_front_is_non_dominated(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=200,
        seed=42,
        population_size=30,
    )

    pareto_objectives = result["pareto_objectives"]

    for i in range(len(pareto_objectives)):

        for j in range(len(pareto_objectives)):

            if i == j:
                continue

            assert not dominates(
                pareto_objectives[i],
                pareto_objectives[j],
            )


def test_nsga_ii_is_reproducible(problem):

    result_1 = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    result_2 = mixed_variable_nsga_ii(
        problem,
        max_evaluations=100,
        seed=42,
        population_size=20,
    )

    np.testing.assert_array_equal(
        result_1["population"],
        result_2["population"],
    )

    np.testing.assert_array_equal(
        result_1["objectives"],
        result_2["objectives"],
    )

    np.testing.assert_array_equal(
        result_1["pareto_solutions"],
        result_2["pareto_solutions"],
    )

    np.testing.assert_array_equal(
        result_1["pareto_objectives"],
        result_2["pareto_objectives"],
    )


def test_nsga_ii_history_tracks_evaluation_budget(problem):

    result = mixed_variable_nsga_ii(
        problem,
        max_evaluations=103,
        seed=42,
        population_size=20,
    )

    evaluation_counts = [
        item["n_evaluations"]
        for item in result["history"]
    ]

    assert evaluation_counts[0] == 20
    assert evaluation_counts[-1] == 103

    assert all(
        evaluation_counts[i] > evaluation_counts[i - 1]
        for i in range(1, len(evaluation_counts))
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 0},
        {"population_size": 1},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"tournament_size": 0},
        {"eta_c": 0.0},
        {"eta_m": -1.0},
    ],
)
def test_nsga_ii_rejects_invalid_parameters(problem, kwargs):

    with pytest.raises(ValueError):

        mixed_variable_nsga_ii(
            problem,
            seed=42,
            **kwargs,
        )
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 0},
        {"population_size": 1},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"tournament_size": 0},
        {"eta_c": 0.0},
        {"eta_m": -1.0},
        {"eta_c": np.nan},
        {"eta_c": np.inf},
        {"eta_m": np.nan},
        {"eta_m": np.inf},
    ],
)
def test_nsga_ii_rejects_invalid_parameters(problem, kwargs):

    with pytest.raises(ValueError):

        mixed_variable_nsga_ii(
            problem,
            seed=42,
            **kwargs,
        )