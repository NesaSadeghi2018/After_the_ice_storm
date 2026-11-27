
import numpy as np
import pytest

from ice_storm.assignments.assignment_b.problem import AssignmentBProblem
from ice_storm.optimization.algorithms.mixed_variable_sa import (
    mixed_variable_sa,
    _generate_neighbor,
)


@pytest.fixture
def problem():
    return AssignmentBProblem("initial_response")


def test_result_structure(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=50,
        seed=42,
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
    assert result["algorithm"] == "MVSA"
    assert result["best_x"].shape == (24,)
    assert np.isfinite(result["best_f"])
    assert result["runtime_seconds"] >= 0


@pytest.mark.parametrize(
    "budget",
    [1, 2, 3, 7, 13, 43, 100],
)
def test_exact_evaluation_budget(problem, budget):
    result = mixed_variable_sa(
        problem,
        max_evaluations=budget,
        seed=42,
    )

    assert result["n_evaluations"] == budget
    assert len(result["history"]) == budget


def test_best_solution_is_valid(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=42,
    )

    problem.validate_solution(result["best_x"])


def test_best_f_matches_objective(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=42,
    )

    actual_f = problem.objective(result["best_x"])

    assert result["best_f"] == pytest.approx(actual_f)


def test_reproducibility(problem):
    result_1 = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=123,
    )

    result_2 = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=123,
    )

    assert np.array_equal(
        result_1["best_x"],
        result_2["best_x"],
    )

    assert result_1["best_f"] == result_2["best_f"]


def test_different_seeds_can_produce_different_solutions(problem):
    result_1 = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=10,
    )

    result_2 = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=20,
    )

    assert (
        not np.array_equal(
            result_1["best_x"],
            result_2["best_x"],
        )
        or result_1["best_f"] != result_2["best_f"]
    )


def test_history_is_monotonic(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=42,
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
    result = mixed_variable_sa(
        problem,
        max_evaluations=43,
        seed=42,
    )

    evaluation_counts = [
        entry["n_evaluations"]
        for entry in result["history"]
    ]

    assert evaluation_counts[0] == 1
    assert evaluation_counts[-1] == 43

    assert evaluation_counts == list(range(1, 44))


def test_temperature_is_positive(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=42,
    )

    temperatures = [
        entry["temperature"]
        for entry in result["history"]
    ]

    assert all(
        temperature > 0
        for temperature in temperatures
    )


def test_neighbor_is_valid(problem):
    rng = np.random.default_rng(42)
    solution = problem.sample_solution(rng)

    for _ in range(100):

        neighbor = _generate_neighbor(
            solution,
            problem,
            rng,
            continuous_step_size=0.1,
        )

        problem.validate_solution(neighbor)


def test_neighbor_changes_only_one_variable_group(problem):
    rng = np.random.default_rng(42)
    solution = problem.sample_solution(rng)

    groups = [
        problem.continuous_indices,
        problem.height_indices,
        problem.channel_indices,
        problem.active_indices,
    ]

    for _ in range(100):

        neighbor = _generate_neighbor(
            solution,
            problem,
            rng,
            continuous_step_size=0.1,
        )

        changed_indices = np.flatnonzero(
            neighbor != solution
        )

        if len(changed_indices) == 0:
            continue

        changed_groups = sum(
            any(index in group for index in changed_indices)
            for group in groups
        )

        assert changed_groups == 1


def test_discrete_variables_remain_valid(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=200,
        seed=42,
    )

    solution = result["best_x"]

    for index in problem.height_indices:
        assert solution[index] in (2, 3, 4, 5, 6)

    for index in problem.channel_indices:
        assert solution[index] in (1, 2, 3)

    for index in problem.active_indices:
        assert solution[index] in (0, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 0},
        {"initial_temperature": 0},
        {"cooling_rate": 0},
        {"cooling_rate": 1},
        {"minimum_temperature": 0},
        {"minimum_temperature": 101},
        {"continuous_step_size": 0},
        {"reheat_after": 0},
        {"reheat_factor": 1},
    ],
)
def test_invalid_parameters_raise_error(problem, kwargs):
    with pytest.raises(ValueError):
        mixed_variable_sa(
            problem,
            seed=42,
            **kwargs,
        )
def test_initialization_uses_best_of_initial_population(problem):
    seed = 2029000
    initial_population_size = 40

    rng = np.random.default_rng(seed)

    candidates = [
        problem.sample_solution(rng)
        for _ in range(initial_population_size)
    ]

    expected_best_f = min(
        problem.objective(candidate)
        for candidate in candidates
    )

    result = mixed_variable_sa(
        problem,
        max_evaluations=100,
        seed=seed,
        initial_population_size=initial_population_size,
    )

    assert result["history"][initial_population_size - 1]["best_f"] == pytest.approx(
        expected_best_f
    )


def test_initial_population_respects_evaluation_budget(problem):
    result = mixed_variable_sa(
        problem,
        max_evaluations=10,
        seed=42,
        initial_population_size=40,
    )

    assert result["n_evaluations"] == 10
    assert len(result["history"]) == 10