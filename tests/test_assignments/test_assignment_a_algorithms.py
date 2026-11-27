
import numpy as np
import pytest

from ice_storm.optimization.algorithms.cma_es import cma_es
from ice_storm.optimization.algorithms.differential_evolution import (
    differential_evolution,
)
from ice_storm.optimization.algorithms.particle_swarm import (
    particle_swarm,
)

class SphereProblem:
    """A simple bounded optimization problem for testing CMA-ES."""

    dimension = 3
    lower_bounds = np.array([-5.0, -5.0, -5.0])
    upper_bounds = np.array([5.0, 5.0, 5.0])

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2))


def test_cma_es_returns_expected_result_structure():
    problem = SphereProblem()

    result = cma_es(
        problem,
        seed=42,
        max_evaluations=40,
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
    assert result["algorithm"] == "CMA-ES"
    assert result["seed"] == 42
    assert isinstance(result["best_x"], np.ndarray)
    assert result["best_x"].shape == (problem.dimension,)
    assert np.isfinite(result["best_f"])
    assert np.isfinite(result["runtime_seconds"])
    assert result["runtime_seconds"] >= 0
    assert isinstance(result["history"], list)


def test_cma_es_respects_evaluation_budget_and_complete_generations():
    problem = SphereProblem()
    population_size = 8
    max_evaluations = 30

    result = cma_es(
        problem,
        seed=123,
        max_evaluations=max_evaluations,
        population_size=population_size,
    )

    assert 0 < result["n_evaluations"] <= max_evaluations
    assert result["n_evaluations"] % population_size == 0


def test_cma_es_returns_solution_within_bounds():
    problem = SphereProblem()

    result = cma_es(
        problem,
        seed=42,
        max_evaluations=40,
        population_size=10,
    )

    assert np.all(result["best_x"] >= problem.lower_bounds)
    assert np.all(result["best_x"] <= problem.upper_bounds)


def test_cma_es_best_objective_matches_returned_solution():
    problem = SphereProblem()

    result = cma_es(
        problem,
        seed=42,
        max_evaluations=40,
        population_size=10,
    )

    expected_value = problem.objective(result["best_x"])

    assert result["best_f"] == pytest.approx(expected_value)


def test_cma_es_is_reproducible_with_same_seed():
    problem = SphereProblem()

    first = cma_es(
        problem,
        seed=2027,
        max_evaluations=40,
        population_size=10,
    )

    second = cma_es(
        problem,
        seed=2027,
        max_evaluations=40,
        population_size=10,
    )

    assert first["n_evaluations"] == second["n_evaluations"]
    assert first["best_f"] == pytest.approx(second["best_f"])
    np.testing.assert_allclose(first["best_x"], second["best_x"])


def test_cma_es_rejects_nonpositive_evaluation_budget():
    with pytest.raises(ValueError, match="max_evaluations must be positive"):
        cma_es(SphereProblem(), max_evaluations=0)


def test_cma_es_rejects_population_size_below_two():
    with pytest.raises(ValueError, match="Population size must be at least 2"):
        cma_es(SphereProblem(), population_size=1)


def test_cma_es_rejects_incorrect_bounds_dimensions():
    class InvalidProblem:
        dimension = 3
        lower_bounds = np.array([-5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])

        def objective(self, x):
            return float(np.sum(x**2))

    with pytest.raises(ValueError, match="Problem bounds have incorrect dimensions"):
        cma_es(InvalidProblem())


def test_cma_es_rejects_invalid_bounds():
    class InvalidProblem:
        dimension = 2
        lower_bounds = np.array([5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])

        def objective(self, x):
            return float(np.sum(x**2))

    with pytest.raises(ValueError, match="Invalid variable bounds"):
        cma_es(InvalidProblem())

def test_cma_es_rejects_budget_smaller_than_population_size():
    with pytest.raises(
        ValueError,
        match="max_evaluations must be at least population_size",
    ):
        cma_es(
            SphereProblem(),
            max_evaluations=5,
            population_size=10,
        )


def test_de_returns_expected_result_structure():
    problem = SphereProblem()

    result = differential_evolution(
        problem,
        seed=42,
        max_evaluations=40,
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
    assert result["algorithm"] == "Differential Evolution"
    assert result["seed"] == 42
    assert isinstance(result["best_x"], np.ndarray)
    assert result["best_x"].shape == (problem.dimension,)
    assert np.isfinite(result["best_f"])
    assert np.isfinite(result["runtime_seconds"])
    assert result["runtime_seconds"] >= 0
    assert isinstance(result["history"], list)


def test_de_respects_evaluation_budget_with_partial_generation():
    problem = SphereProblem()

    result = differential_evolution(
        problem,
        seed=123,
        max_evaluations=23,
        population_size=10,
    )

    assert result["n_evaluations"] == 23
    assert result["n_evaluations"] <= 23
    assert result["history"][-1]["evaluations"] == 23


def test_de_returns_solution_within_bounds():
    problem = SphereProblem()

    result = differential_evolution(
        problem,
        seed=42,
        max_evaluations=40,
        population_size=10,
    )

    assert np.all(result["best_x"] >= problem.lower_bounds)
    assert np.all(result["best_x"] <= problem.upper_bounds)


def test_de_best_objective_matches_returned_solution():
    problem = SphereProblem()

    result = differential_evolution(
        problem,
        seed=42,
        max_evaluations=40,
        population_size=10,
    )

    expected_value = problem.objective(result["best_x"])

    assert result["best_f"] == pytest.approx(expected_value)


def test_de_is_reproducible_with_same_seed():
    problem = SphereProblem()

    first = differential_evolution(
        problem,
        seed=2027,
        max_evaluations=40,
        population_size=10,
    )

    second = differential_evolution(
        problem,
        seed=2027,
        max_evaluations=40,
        population_size=10,
    )

    assert first["n_evaluations"] == second["n_evaluations"]
    assert first["best_f"] == pytest.approx(second["best_f"])
    np.testing.assert_allclose(first["best_x"], second["best_x"])


def test_de_rejects_nonpositive_evaluation_budget():
    with pytest.raises(ValueError, match="max_evaluations must be positive"):
        differential_evolution(SphereProblem(), max_evaluations=0)


def test_de_rejects_population_size_below_four():
    with pytest.raises(
        ValueError,
        match="Population size must be at least 4",
    ):
        differential_evolution(SphereProblem(), population_size=3)


def test_de_rejects_population_larger_than_evaluation_budget():
    with pytest.raises(
        ValueError,
        match="Population size cannot exceed evaluation budget",
    ):
        differential_evolution(
            SphereProblem(),
            max_evaluations=5,
            population_size=10,
        )


def test_de_rejects_invalid_mutation_factor():
    with pytest.raises(ValueError, match="mutation_factor must be in"):
        differential_evolution(
            SphereProblem(),
            mutation_factor=0,
        )


def test_de_rejects_invalid_crossover_rate():
    with pytest.raises(ValueError, match="crossover_rate must be in"):
        differential_evolution(
            SphereProblem(),
            crossover_rate=1.5,
        )


def test_de_rejects_incorrect_bounds_dimensions():
    class InvalidProblem:
        dimension = 3
        lower_bounds = np.array([-5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])

        def objective(self, x):
            return float(np.sum(x**2))

    with pytest.raises(
        ValueError,
        match="Problem bounds have incorrect dimensions",
    ):
        differential_evolution(InvalidProblem())


def test_de_rejects_invalid_bounds():
    class InvalidProblem:
        dimension = 2
        lower_bounds = np.array([5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])

        def objective(self, x):
            return float(np.sum(x**2))

    with pytest.raises(
        ValueError,
        match="Every lower bound must be below its upper bound",
    ):
        differential_evolution(InvalidProblem())


def test_pso_returns_expected_result_structure():
    problem = SphereProblem()

    result = particle_swarm(
        problem,
        seed=42,
        max_evaluations=40,
        swarm_size=10,
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
    assert result["algorithm"] == "Particle Swarm Optimization"
    assert result["seed"] == 42
    assert isinstance(result["best_x"], np.ndarray)
    assert result["best_x"].shape == (problem.dimension,)
    assert np.isfinite(result["best_f"])
    assert np.isfinite(result["runtime_seconds"])
    assert result["runtime_seconds"] >= 0
    assert isinstance(result["history"], list)


def test_pso_respects_evaluation_budget_with_partial_iteration():
    problem = SphereProblem()

    result = particle_swarm(
        problem,
        seed=123,
        max_evaluations=23,
        swarm_size=10,
    )

    assert result["n_evaluations"] == 23
    assert result["n_evaluations"] <= 23
    assert result["history"][-1]["evaluations"] == 23


def test_pso_returns_solution_within_bounds():
    problem = SphereProblem()

    result = particle_swarm(
        problem,
        seed=42,
        max_evaluations=40,
        swarm_size=10,
    )

    assert np.all(result["best_x"] >= problem.lower_bounds)
    assert np.all(result["best_x"] <= problem.upper_bounds)


def test_pso_best_objective_matches_returned_solution():
    problem = SphereProblem()

    result = particle_swarm(
        problem,
        seed=42,
        max_evaluations=40,
        swarm_size=10,
    )

    expected_value = problem.objective(result["best_x"])

    assert result["best_f"] == pytest.approx(expected_value)


def test_pso_is_reproducible_with_same_seed():
    problem = SphereProblem()

    first = particle_swarm(
        problem,
        seed=2027,
        max_evaluations=40,
        swarm_size=10,
    )

    second = particle_swarm(
        problem,
        seed=2027,
        max_evaluations=40,
        swarm_size=10,
    )

    assert first["n_evaluations"] == second["n_evaluations"]
    assert first["best_f"] == pytest.approx(second["best_f"])
    np.testing.assert_allclose(first["best_x"], second["best_x"])


def test_pso_rejects_nonpositive_evaluation_budget():
    with pytest.raises(ValueError, match="max_evaluations must be positive"):
        particle_swarm(SphereProblem(), max_evaluations=0)


def test_pso_rejects_nonpositive_swarm_size():
    with pytest.raises(ValueError, match="swarm_size must be positive"):
        particle_swarm(SphereProblem(), swarm_size=0)


def test_pso_rejects_swarm_larger_than_evaluation_budget():
    with pytest.raises(
        ValueError,
        match="Swarm size cannot exceed evaluation budget",
    ):
        particle_swarm(
            SphereProblem(),
            max_evaluations=5,
            swarm_size=10,
        )


def test_pso_rejects_incorrect_bounds_dimensions():
    class InvalidProblem:
        dimension = 3
        lower_bounds = np.array([-5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])

        def objective(self, x):
            return float(np.sum(x**2))

    with pytest.raises(
        ValueError,
        match="Problem bounds have incorrect dimensions",
    ):
        particle_swarm(InvalidProblem())


def test_pso_rejects_invalid_bounds():
    class InvalidProblem:
        dimension = 2
        lower_bounds = np.array([5.0, -5.0])
        upper_bounds = np.array([5.0, 5.0])

        def objective(self, x):
            return float(np.sum(x**2))

    with pytest.raises(
        ValueError,
        match="Invalid variable bounds",
    ):
        particle_swarm(InvalidProblem())
