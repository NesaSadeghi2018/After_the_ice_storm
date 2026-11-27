"""Mixed-variable MOEA/D for multi-objective minimization problems."""

from __future__ import annotations

import time

import numpy as np

from ice_storm.optimization.algorithms.multiobjective_utils import (
    fast_non_dominated_sort,
)
from ice_storm.optimization.algorithms.multiobjective.objective_scaling import (
    normalize_objectives,
)
from ice_storm.optimization.algorithms.nsga_ii import (
    _mixed_variable_crossover,
    _mutate,
)


def _weight_vectors(n_vectors: int, rng: np.random.Generator) -> np.ndarray:
    """Create reproducible simplex weights, including objective extremes."""
    if n_vectors <= 0:
        raise ValueError("n_vectors must be positive.")
    if n_vectors == 1:
        return np.full((1, 3), 1.0 / 3.0)

    extremes = np.eye(3, dtype=float)
    if n_vectors <= 3:
        return extremes[:n_vectors].copy()

    interior = rng.dirichlet(np.ones(3), size=n_vectors - 3)
    return np.vstack((extremes, interior))


def _tchebycheff(values, weights, ideal):
    return np.max(np.maximum(weights, 1e-6) * np.abs(values - ideal), axis=1)


def _pareto_indices(objectives: np.ndarray) -> np.ndarray:
    return np.asarray(fast_non_dominated_sort(objectives)[0], dtype=int)


def mixed_variable_moead(
    problem,
    max_evaluations: int = 10_000,
    seed: int | None = None,
    population_size: int = 40,
    neighborhood_size: int = 10,
    neighborhood_mating_probability: float = 0.9,
    max_replacements: int = 2,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
):
    """Run MOEA/D using Tchebycheff decomposition and mixed-variable variation."""
    start = time.perf_counter()
    if max_evaluations < 1 or population_size < 1:
        raise ValueError("max_evaluations and population_size must be positive.")
    if neighborhood_size < 1 or max_replacements < 1:
        raise ValueError("neighborhood_size and max_replacements must be positive.")
    if not 0.0 <= neighborhood_mating_probability <= 1.0:
        raise ValueError("neighborhood_mating_probability must be in [0, 1].")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1].")
    if mutation_rate is None:
        mutation_rate = 1.0 / problem.dimension
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1].")
    if eta_c <= 0 or eta_m <= 0:
        raise ValueError("eta_c and eta_m must be positive.")

    rng = np.random.default_rng(seed)
    size = min(population_size, max_evaluations)
    population = np.asarray([problem.sample_solution(rng) for _ in range(size)], dtype=float)
    for x in population:
        problem.validate_solution(x)
    objectives = np.asarray([problem.objectives(x) for x in population], dtype=float)
    if objectives.shape != (size, problem.n_objectives) or not np.all(np.isfinite(objectives)):
        raise RuntimeError("Objective evaluation returned invalid values.")

    weights = _weight_vectors(size, rng)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighborhood_size = min(neighborhood_size, size)
    neighborhoods = np.argsort(distances, axis=1, kind="stable")[:, :neighborhood_size]
    normalized_objectives = normalize_objectives(objectives)
    ideal = np.min(normalized_objectives, axis=0)
    n_evaluations = size
    history = [{"n_evaluations": n_evaluations, "n_pareto_solutions": len(_pareto_indices(objectives))}]

    while n_evaluations < max_evaluations:
        for subproblem in rng.permutation(size):
            if n_evaluations >= max_evaluations:
                break
            if rng.random() < neighborhood_mating_probability:
                pool = neighborhoods[subproblem]
            else:
                pool = np.arange(size)
            if len(pool) >= 2:
                parents = rng.choice(pool, size=2, replace=False)
            else:
                parents = np.array([pool[0], pool[0]])
            p1, p2 = population[int(parents[0])], population[int(parents[1])]
            if rng.random() < crossover_rate:
                child, _ = _mixed_variable_crossover(p1, p2, problem, rng, eta_c)
            else:
                child = p1.copy()
            child = _mutate(child, problem, rng, mutation_rate, eta_m)
            child_objectives = np.asarray(problem.objectives(child), dtype=float)
            if child_objectives.shape != (problem.n_objectives,) or not np.all(np.isfinite(child_objectives)):
                raise RuntimeError("Objective evaluation returned invalid values.")
            n_evaluations += 1
            normalized_child_objectives = normalize_objectives(child_objectives)
            ideal = np.minimum(ideal, normalized_child_objectives)

            replace_candidates = neighborhoods[subproblem].copy()
            rng.shuffle(replace_candidates)
            replacements = 0
            for index in replace_candidates:
                old_value = _tchebycheff(
                    normalized_objectives[index:index + 1],
                    weights[index:index + 1],
                    ideal,
                )[0]

                new_value = _tchebycheff(
                    normalized_child_objectives[None, :],
                    weights[index:index + 1],
                    ideal,
                )[0]

                if new_value <= old_value:
                    population[index] = child
                    objectives[index] = child_objectives
                    normalized_objectives[index] = normalized_child_objectives

                    replacements += 1

                    if replacements >= max_replacements:
                        break

        pareto_count = len(_pareto_indices(objectives))
        history.append({"n_evaluations": n_evaluations, "n_pareto_solutions": pareto_count})

    front = _pareto_indices(objectives)
    return {
        "algorithm": "MOEA/D",
        "population": population.copy(),
        "objectives": objectives.copy(),
        "pareto_solutions": population[front].copy(),
        "pareto_objectives": objectives[front].copy(),
        "n_evaluations": n_evaluations,
        "runtime_seconds": time.perf_counter() - start,
        "seed": seed,
        "history": history,
    }
