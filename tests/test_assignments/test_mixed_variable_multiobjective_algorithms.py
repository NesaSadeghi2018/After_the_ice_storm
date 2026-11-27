import numpy as np
import pytest

from ice_storm.assignments.assignment_c.problem import AssignmentCProblem
from ice_storm.optimization.algorithms.multiobjective import (
    mixed_variable_moead,
    mixed_variable_spea2,
)
from ice_storm.optimization.algorithms.multiobjective_utils import dominates


@pytest.fixture

def problem():
    return AssignmentCProblem("initial_response")


@pytest.mark.parametrize(
    "algorithm",
    [mixed_variable_moead, mixed_variable_spea2],
)
@pytest.mark.parametrize("budget", [1, 40, 41, 93])
def test_algorithm_respects_exact_evaluation_budget(problem, algorithm, budget):
    result = algorithm(problem, max_evaluations=budget, seed=17, population_size=20)
    assert result["n_evaluations"] == budget


@pytest.mark.parametrize(
    "algorithm, expected_name",
    [
        (mixed_variable_moead, "MOEA/D"),
        (mixed_variable_spea2, "SPEA2"),
    ],
)
def test_algorithm_returns_valid_result_and_mixed_variables(problem, algorithm, expected_name):
    result = algorithm(problem, max_evaluations=100, seed=123, population_size=20)
    assert result["algorithm"] == expected_name
    assert result["population"].ndim == 2
    assert result["population"].shape[1] == problem.dimension
    assert result["objectives"].shape == (len(result["population"]), 3)
    assert len(result["pareto_solutions"]) >= 1
    assert result["pareto_objectives"].shape == (len(result["pareto_solutions"]), 3)
    assert np.all(np.isfinite(result["objectives"]))
    for solution in result["population"]:
        assert problem.validate_solution(solution)
    for solution in result["pareto_solutions"]:
        assert problem.validate_solution(solution)


def test_moead_is_reproducible(problem):
    first = mixed_variable_moead(problem, max_evaluations=100, seed=321, population_size=20)
    second = mixed_variable_moead(problem, max_evaluations=100, seed=321, population_size=20)
    np.testing.assert_array_equal(first["population"], second["population"])
    np.testing.assert_array_equal(first["objectives"], second["objectives"])


def test_spea2_is_reproducible(problem):
    first = mixed_variable_spea2(problem, max_evaluations=100, seed=321, population_size=20)
    second = mixed_variable_spea2(problem, max_evaluations=100, seed=321, population_size=20)
    np.testing.assert_array_equal(first["population"], second["population"])
    np.testing.assert_array_equal(first["objectives"], second["objectives"])


@pytest.mark.parametrize("algorithm", [mixed_variable_moead, mixed_variable_spea2])
def test_reported_pareto_front_is_nondominated(problem, algorithm):
    result = algorithm(problem, max_evaluations=100, seed=77, population_size=20)
    values = result["pareto_objectives"]
    for i in range(len(values)):
        for j in range(len(values)):
            if i != j:
                assert not dominates(values[i], values[j])
