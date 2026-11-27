
import numpy as np
import pytest

from ice_storm.assignments.assignment_a.problem import (
    AssignmentAProblem,
    SCENARIO_NAMES,
    N_VARIABLES,
)


def valid_solution():
    """Return a valid 12-dimensional Assignment A solution."""
    return np.array(
        [
            0.0, 0.0, 0.2,
            10.0, 10.0, 2.0,
            5.0, 5.0, 1.0,
            3.0, 7.0, 1.5,
        ],
        dtype=float,
    )


@pytest.fixture
def problem():
    return AssignmentAProblem("initial_response")


def test_problem_dimension_and_bounds(problem):
    assert problem.dimension == 12
    assert problem.lower_bounds.shape == (12,)
    assert problem.upper_bounds.shape == (12,)
    assert problem.bounds.shape == (12, 2)


def test_problem_rejects_unknown_scenario():
    with pytest.raises(ValueError):
        AssignmentAProblem("unknown_scenario")


def test_problem_accepts_all_defined_scenarios():
    for scenario in SCENARIO_NAMES:
        instance = AssignmentAProblem(scenario)
        assert instance.scenario_name == scenario


def test_validate_solution_accepts_valid_boundaries(problem):
    assert problem.validate_solution(valid_solution()) is True


def test_validate_solution_rejects_wrong_dimension(problem):
    with pytest.raises(ValueError):
        problem.validate_solution(np.zeros(11))


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
        (3, -0.1),
        (3, 10.1),
        (5, 2.01),
        (11, 0.19),
    ],
)
def test_validate_solution_rejects_values_outside_bounds(
    problem,
    index,
    invalid_value,
):
    x = valid_solution()
    x[index] = invalid_value

    with pytest.raises(ValueError):
        problem.validate_solution(x)


def test_build_deployment_returns_correct_shape(problem):
    deployment = problem.build_deployment(valid_solution())

    assert deployment.shape == (4, 6)


def test_build_deployment_preserves_decision_variables(problem):
    x = valid_solution()
    deployment = problem.build_deployment(x)

    np.testing.assert_allclose(
        deployment[:, 0:3],
        x.reshape(4, 3),
    )


def test_build_deployment_uses_fixed_parameters(problem):
    deployment = problem.build_deployment(valid_solution())

    np.testing.assert_array_equal(
        deployment[:, 3],
        np.full(4, 4),
    )

    np.testing.assert_array_equal(
        deployment[:, 4],
        np.full(4, 1),
    )

    np.testing.assert_array_equal(
        deployment[:, 5],
        np.full(4, 1),
    )


def test_evaluate_returns_expected_metrics(problem):
    results = problem.evaluate(valid_solution())

    assert set(results.keys()) >= {"Q", "C", "E"}

    assert np.isfinite(results["Q"])
    assert np.isfinite(results["C"])
    assert np.isfinite(results["E"])

    assert -1.0 <= results["Q"] <= 0.0


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_objective_is_consistent_with_evaluation(scenario):
    instance = AssignmentAProblem(scenario)
    x = valid_solution()

    results = instance.evaluate(x)
    objective_value = instance.objective(x)

    assert objective_value == pytest.approx(results["Q"])


def test_sample_solution_generates_valid_solution(problem):
    rng = np.random.default_rng(42)

    x = problem.sample_solution(rng)

    assert x.shape == (N_VARIABLES,)
    assert problem.validate_solution(x) is True


def test_sample_solution_is_reproducible(problem):
    x1 = problem.sample_solution(np.random.default_rng(42))
    x2 = problem.sample_solution(np.random.default_rng(42))

    np.testing.assert_array_equal(x1, x2)
