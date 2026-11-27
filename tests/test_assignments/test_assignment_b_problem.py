import numpy as np
import pytest

from ice_storm.assignments.assignment_b.problem import (
    AssignmentBProblem,
    SCENARIO_NAMES,
    N_VARIABLES,
)


def valid_solution():
    """
    Return a valid 24-dimensional mixed-variable solution.
    """

    return np.array(
        [
            0.0, 0.0, 0.2, 2.0, 1.0, 0.0,
            10.0, 10.0, 2.0, 6.0, 3.0, 1.0,
            5.0, 5.0, 1.0, 4.0, 2.0, 1.0,
            3.0, 7.0, 1.5, 3.0, 1.0, 0.0,
        ],
        dtype=float,
    )


@pytest.fixture
def problem():
    return AssignmentBProblem("initial_response")


def test_problem_dimension_and_bounds(problem):
    assert problem.dimension == 24
    assert problem.lower_bounds.shape == (24,)
    assert problem.upper_bounds.shape == (24,)
    assert problem.bounds.shape == (24, 2)


def test_problem_rejects_unknown_scenario():
    with pytest.raises(ValueError):
        AssignmentBProblem("unknown_scenario")


def test_problem_accepts_all_defined_scenarios():
    for scenario in SCENARIO_NAMES:
        instance = AssignmentBProblem(scenario)
        assert instance.scenario_name == scenario


def test_validate_solution_accepts_valid_boundaries(problem):
    x = valid_solution()

    assert problem.validate_solution(x) is True


def test_build_deployment_returns_correct_shape_and_values(problem):
    x = valid_solution()

    deployment = problem.build_deployment(x)

    assert deployment.shape == (4, 6)

    np.testing.assert_array_equal(
        deployment,
        x.reshape(4, 6),
    )


def test_validate_solution_rejects_wrong_dimension(problem):
    with pytest.raises(ValueError):
        problem.validate_solution(np.zeros(23))


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_validate_solution_rejects_non_finite_values(
    problem,
    invalid_value,
):
    x = valid_solution()
    x[0] = invalid_value

    with pytest.raises(ValueError):
        problem.validate_solution(x)


@pytest.mark.parametrize(
    "index, invalid_value",
    [
        (0, -0.1),
        (0, 10.1),
        (1, -0.1),
        (1, 10.1),
        (2, 0.19),
        (2, 2.01),
        (6, -0.1),
        (6, 10.1),
        (8, 0.19),
        (8, 2.01),
    ],
)
def test_validate_solution_rejects_continuous_values_outside_bounds(
    problem,
    index,
    invalid_value,
):
    x = valid_solution()
    x[index] = invalid_value

    with pytest.raises(ValueError):
        problem.validate_solution(x)


@pytest.mark.parametrize(
    "index, invalid_value",
    [
        (3, 1.0),
        (3, 4.5),
        (3, 7.0),
        (9, 0.0),
        (9, 5.5),
        (15, 7.0),
    ],
)
def test_validate_solution_rejects_invalid_heights(
    problem,
    index,
    invalid_value,
):
    x = valid_solution()
    x[index] = invalid_value

    with pytest.raises(ValueError):
        problem.validate_solution(x)


@pytest.mark.parametrize(
    "index, invalid_value",
    [
        (4, 0.0),
        (4, 1.5),
        (4, 4.0),
        (10, 0.0),
        (10, 2.5),
        (16, 4.0),
    ],
)
def test_validate_solution_rejects_invalid_channels(
    problem,
    index,
    invalid_value,
):
    x = valid_solution()
    x[index] = invalid_value

    with pytest.raises(ValueError):
        problem.validate_solution(x)


@pytest.mark.parametrize(
    "index, invalid_value",
    [
        (5, -1.0),
        (5, 0.5),
        (5, 2.0),
        (11, 1.5),
        (17, -1.0),
        (23, 2.0),
    ],
)
def test_validate_solution_rejects_invalid_active_values(
    problem,
    index,
    invalid_value,
):
    x = valid_solution()
    x[index] = invalid_value

    with pytest.raises(ValueError):
        problem.validate_solution(x)


def test_sample_solution_generates_valid_solution(problem):
    rng = np.random.default_rng(42)

    x = problem.sample_solution(rng)

    assert x.shape == (N_VARIABLES,)
    assert problem.validate_solution(x) is True


def test_sample_solution_is_reproducible(problem):
    x1 = problem.sample_solution(np.random.default_rng(42))
    x2 = problem.sample_solution(np.random.default_rng(42))

    np.testing.assert_array_equal(x1, x2)


def test_evaluate_returns_expected_metrics(problem):
    results = problem.evaluate(valid_solution())

    assert set(results.keys()) >= {"Q", "C", "E"}

    assert np.isfinite(results["Q"])
    assert np.isfinite(results["C"])
    assert np.isfinite(results["E"])


def test_objective_returns_q_value(problem):
    x = valid_solution()

    results = problem.evaluate(x)
    objective_value = problem.objective(x)

    assert objective_value == pytest.approx(results["Q"])