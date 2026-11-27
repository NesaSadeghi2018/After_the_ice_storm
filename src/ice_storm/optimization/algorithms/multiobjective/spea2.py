"""Mixed-variable Strength Pareto Evolutionary Algorithm 2 (SPEA2)."""

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


def _spea2_fitness(objectives: np.ndarray):
    """Return SPEA2 fitness and normalized pairwise objective-space distances."""
    objectives = np.asarray(objectives, dtype=float)

    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError("Expected a 2D objective array with three objectives.")

    if not np.all(np.isfinite(objectives)):
        raise ValueError("Objective values must be finite.")

    normalized_objectives = normalize_objectives(objectives)

    n = len(objectives)

    distances = np.linalg.norm(
        normalized_objectives[:, None, :] - normalized_objectives[None, :, :],
        axis=2,
    )

    strength = np.zeros(n, dtype=float)
    raw = np.zeros(n, dtype=float)

    for i in range(n):
        strength[i] = sum(
            1
            for j in range(n)
            if i != j
            and np.all(objectives[i] <= objectives[j])
            and np.any(objectives[i] < objectives[j])
        )

    for i in range(n):
        raw[i] = sum(
            strength[j]
            for j in range(n)
            if i != j
            and np.all(objectives[j] <= objectives[i])
            and np.any(objectives[j] < objectives[i])
        )

    k = max(1, min(int(np.sqrt(n)), n - 1)) if n > 1 else 0

    density = np.zeros(n, dtype=float)

    if k:
        for i in range(n):
            neighbor_distances = np.sort(
                np.delete(distances[i], i),
                kind="stable",
            )
            sigma = neighbor_distances[k - 1]
            density[i] = 1.0 / (sigma + 2.0)

    return raw + density, distances


def _truncate_archive(indices, distances, target_size):
    """SPEA2 nearest-neighbor truncation for an oversized archive."""
    survivors = list(map(int, indices))
    while len(survivors) > target_size:
        local = distances[np.ix_(survivors, survivors)].copy()
        np.fill_diagonal(local, np.inf)
        ordered = np.sort(local, axis=1, kind="stable")
        remove_position = min(
            range(len(survivors)),
            key=lambda pos: tuple(ordered[pos].tolist()),
        )
        survivors.pop(remove_position)
    return np.asarray(survivors, dtype=int)


def _environmental_selection(population, objectives, archive_size):
    fitness, distances = _spea2_fitness(objectives)
    nondominated = np.flatnonzero(fitness < 1.0)
    if len(nondominated) > archive_size:
        selected = _truncate_archive(nondominated, distances, archive_size)
    elif len(nondominated) < archive_size:
        dominated = np.flatnonzero(fitness >= 1.0)
        order = dominated[np.argsort(fitness[dominated], kind="stable")]
        selected = np.concatenate((nondominated, order[:archive_size - len(nondominated)]))
    else:
        selected = nondominated
    return population[selected].copy(), objectives[selected].copy(), fitness[selected].copy()


def _tournament(archive, fitness, rng):
    if len(archive) == 1:
        return archive[0].copy()
    contestants = rng.choice(len(archive), size=2, replace=False)
    winner = contestants[np.argmin(fitness[contestants])]
    return archive[int(winner)].copy()


def mixed_variable_spea2(
    problem,
    max_evaluations: int = 10_000,
    seed: int | None = None,
    population_size: int = 40,
    archive_size: int | None = None,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
):
    """Run SPEA2 with strength/raw fitness, density, and archive truncation."""
    start = time.perf_counter()
    if max_evaluations < 1 or population_size < 1:
        raise ValueError("max_evaluations and population_size must be positive.")
    if archive_size is None:
        archive_size = population_size
    if archive_size < 1:
        raise ValueError("archive_size must be positive.")
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
    archive = np.empty((0, problem.dimension), dtype=float)
    archive_objectives = np.empty((0, problem.n_objectives), dtype=float)
    n_evaluations = size
    history = []

    while True:
        combined_population = np.vstack((population, archive))
        combined_objectives = np.vstack((objectives, archive_objectives))
        archive, archive_objectives, archive_fitness = _environmental_selection(
            combined_population, combined_objectives, min(archive_size, max_evaluations)
        )
        front = fast_non_dominated_sort(archive_objectives)[0]
        history.append({"n_evaluations": n_evaluations, "n_pareto_solutions": len(front)})
        if n_evaluations >= max_evaluations:
            break

        offspring_count = min(population_size, max_evaluations - n_evaluations)
        offspring = []
        while len(offspring) < offspring_count:
            parent_1 = _tournament(archive, archive_fitness, rng)
            parent_2 = _tournament(archive, archive_fitness, rng)
            if rng.random() < crossover_rate:
                child_1, child_2 = _mixed_variable_crossover(
                    parent_1, parent_2, problem, rng, eta_c
                )
            else:
                child_1, child_2 = parent_1.copy(), parent_2.copy()
            for child in (child_1, child_2):
                if len(offspring) >= offspring_count:
                    break
                child = _mutate(child, problem, rng, mutation_rate, eta_m)
                offspring.append(child)

        population = np.asarray(offspring, dtype=float)
        objectives = np.asarray([problem.objectives(x) for x in population], dtype=float)
        if not np.all(np.isfinite(objectives)):
            raise RuntimeError("Objective evaluation returned non-finite values.")
        n_evaluations += len(population)

    front = np.asarray(fast_non_dominated_sort(archive_objectives)[0], dtype=int)
    return {
        "algorithm": "SPEA2",
        "population": archive.copy(),
        "objectives": archive_objectives.copy(),
        "pareto_solutions": archive[front].copy(),
        "pareto_objectives": archive_objectives[front].copy(),
        "n_evaluations": n_evaluations,
        "runtime_seconds": time.perf_counter() - start,
        "seed": seed,
        "history": history,
    }
