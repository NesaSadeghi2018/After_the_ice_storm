
import time

import numpy as np

import pytest


def _tournament_selection(
    population,
    fitness,
    rng,
    tournament_size,
):
    """Select one parent using tournament selection."""

    candidates = rng.choice(
        len(population),
        size=min(tournament_size, len(population)),
        replace=False,
    )

    winner = candidates[np.argmin(fitness[candidates])]

    return population[winner].copy()


def _crossover(parent_1, parent_2, rng):
    """Uniform crossover for mixed-variable chromosomes."""

    mask = rng.random(len(parent_1)) < 0.5

    child = np.where(mask, parent_1, parent_2)

    return child.astype(float)


def _mutate(
    solution,
    problem,
    rng,
    mutation_rate,
    mutation_scale,
):
    """
    Apply type-aware mutation.

    Continuous variables:
        Gaussian mutation with configurable scale.

    Height and channel:
        Random selection from their valid domains.

    Active variables:
        Binary flip.
    """

    child = solution.copy()

    # Continuous variables
    for index in problem.continuous_indices:

        if rng.random() < mutation_rate:

            variable_range = (
                problem.upper_bounds[index]
                - problem.lower_bounds[index]
            )

            child[index] += rng.normal(
                loc=0.0,
                scale=mutation_scale * variable_range,
            )

            child[index] = np.clip(
                child[index],
                problem.lower_bounds[index],
                problem.upper_bounds[index],
            )

    # Mast height
    for index in problem.height_indices:

        if rng.random() < mutation_rate:

            alternatives = [
                value
                for value in problem.integer_domains["height"]
                if value != child[index]
            ]

            child[index] = rng.choice(alternatives)

    # Channel
    for index in problem.channel_indices:

        if rng.random() < mutation_rate:

            alternatives = [
                value
                for value in problem.integer_domains["channel"]
                if value != child[index]
            ]

            child[index] = rng.choice(alternatives)

    # Active / inactive
    for index in problem.active_indices:

        if rng.random() < mutation_rate:

            child[index] = 1 - child[index]

    return child


def mixed_variable_ga(
    problem,
    max_evaluations=10_000,
    seed=None,
    population_size=40,
    crossover_rate=0.9,
    mutation_rate=None,
    tournament_size=3,
    elitism=2,
    mutation_scale=0.1,
):
    """
    Mixed-Variable Genetic Algorithm.

    Supports:
        - Continuous variables
        - Integer categorical variables
        - Binary variables

    Parameters:
        mutation_scale:
            Gaussian mutation standard deviation as a fraction
            of each continuous variable's range.

    Returns:
        Dictionary containing:
            algorithm
            best_x
            best_f
            n_evaluations
            runtime_seconds
            seed
            history
    """

    start_time = time.perf_counter()

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1.")

    if population_size < 2:
        raise ValueError("population_size must be at least 2.")

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1.")

    if not 0 <= elitism < population_size:
        raise ValueError(
            "elitism must be non-negative and smaller "
            "than population_size."
        )

    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1.")

    if mutation_rate is None:
        mutation_rate = 1.0 / problem.dimension

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")

    if not np.isfinite(mutation_scale) or mutation_scale < 0.0:
        raise ValueError(
            "mutation_scale must be a finite non-negative number."
        )

    rng = np.random.default_rng(seed)

    population_size = min(population_size, max_evaluations)

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = np.array(
        [
            problem.sample_solution(rng)
            for _ in range(population_size)
        ],
        dtype=float,
    )

    fitness = np.empty(population_size, dtype=float)

    n_evaluations = 0

    for i in range(population_size):

        fitness[i] = problem.objective(population[i])

        n_evaluations += 1

    best_index = int(np.argmin(fitness))

    best_x = population[best_index].copy()
    best_f = float(fitness[best_index])

    history = [
        {
            "n_evaluations": n_evaluations,
            "best_f": best_f,
        }
    ]

    # --------------------------------------------------------
    # Evolutionary search
    # --------------------------------------------------------

    while n_evaluations < max_evaluations:

        elite_indices = np.argsort(fitness)[:elitism]

        new_population = [
            population[index].copy()
            for index in elite_indices
        ]

        new_fitness = [
            float(fitness[index])
            for index in elite_indices
        ]

        while (
            len(new_population) < population_size
            and n_evaluations < max_evaluations
        ):

            parent_1 = _tournament_selection(
                population,
                fitness,
                rng,
                tournament_size,
            )

            parent_2 = _tournament_selection(
                population,
                fitness,
                rng,
                tournament_size,
            )

            if rng.random() < crossover_rate:

                child = _crossover(
                    parent_1,
                    parent_2,
                    rng,
                )

            else:
                child = parent_1.copy()

            child = _mutate(
                child,
                problem,
                rng,
                mutation_rate,
                mutation_scale,
            )

            problem.validate_solution(child)

            child_f = float(problem.objective(child))

            n_evaluations += 1

            new_population.append(child)
            new_fitness.append(child_f)

            if child_f < best_f:

                best_f = child_f
                best_x = child.copy()

        population = np.asarray(
            new_population,
            dtype=float,
        )

        fitness = np.asarray(
            new_fitness,
            dtype=float,
        )

        history.append(
            {
                "n_evaluations": n_evaluations,
                "best_f": best_f,
            }
        )

    runtime_seconds = time.perf_counter() - start_time

    return {
        "algorithm": "MVGA",
        "best_x": best_x,
        "best_f": best_f,
        "n_evaluations": n_evaluations,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "history": history,
    }
