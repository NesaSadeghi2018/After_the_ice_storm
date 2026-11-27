
import numpy as np
import pytest

from ice_storm.core.scenarios import SCENARIOS
from ice_storm.assignments.assignment_c.problem import (
    AssignmentCProblem,
)

@pytest.mark.parametrize(
    "scenario_name",
    list(SCENARIOS.keys()),
)
def test_objectives_return_three_finite_values(scenario_name):
    problem = AssignmentCProblem(scenario_name)

    rng = np.random.default_rng(12345)
    solution = problem.sample_solution(rng)

    objectives = problem.objectives(solution)

    assert isinstance(objectives, np.ndarray)
    assert objectives.shape == (3,)
    assert np.all(np.isfinite(objectives))

@pytest.mark.parametrize(
    "scenario_name",
    list(SCENARIOS.keys()),
)
def test_objectives_match_evaluation_results(scenario_name):
    problem = AssignmentCProblem(scenario_name)

    rng = np.random.default_rng(12345)
    solution = problem.sample_solution(rng)

    results = problem.evaluate(solution)
    objectives = problem.objectives(solution)

    expected = np.array(
        [
            results["Q"],
            results["C"],
            results["E"],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        objectives,
        expected,
        rtol=0.0,
        atol=1e-12,
    )

def test_assignment_c_inherits_mixed_variable_structure():
    problem = AssignmentCProblem("initial_response")

    assert problem.dimension == 24
    assert len(problem.continuous_indices) == 12
    assert len(problem.height_indices) == 4
    assert len(problem.channel_indices) == 4
    assert len(problem.active_indices) == 4


def test_invalid_solution_is_rejected():
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution[3] = 3.5

    with pytest.raises(ValueError):
        problem.objectives(solution)

def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        AssignmentCProblem("unknown_scenario")

def test_assignment_c_has_correct_objective_metadata():
    problem = AssignmentCProblem("initial_response")

    assert problem.n_objectives == 3
    assert problem.objective_names == ("Q", "C", "E")

def test_objectives_reject_invalid_solution_shape():
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    with pytest.raises(ValueError):
        problem.objectives(solution[:-1])

@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_objectives_reject_non_finite_variables(invalid_value):
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution[0] = invalid_value

    with pytest.raises(ValueError):
        problem.objectives(solution)

def test_objectives_reject_variables_outside_bounds():
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution[0] = problem.upper_bounds[0] + 1.0

    with pytest.raises(ValueError):
        problem.objectives(solution)

def test_objectives_reject_non_integer_height():
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution[3] = 3.5

    with pytest.raises(ValueError):
        problem.objectives(solution)

def test_objectives_reject_invalid_channel():
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution[4] = 1.5

    with pytest.raises(ValueError):
        problem.objectives(solution)

def test_objectives_reject_invalid_activity_value():
    problem = AssignmentCProblem("initial_response")

    solution = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution[5] = 0.5

    with pytest.raises(ValueError):
        problem.objectives(solution)

def test_sample_solution_is_reproducible():
    problem = AssignmentCProblem("initial_response")

    solution_1 = problem.sample_solution(
        np.random.default_rng(12345)
    )

    solution_2 = problem.sample_solution(
        np.random.default_rng(12345)
    )

    np.testing.assert_array_equal(
        solution_1,
        solution_2,
    )
