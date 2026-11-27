
import time

import numpy as np


def _select_donors(population, target_index, rng):
    """Select three distinct donors, excluding the target."""

    candidates = np.delete(
        np.arange(len(population)),
        target_index,
    )

    indices = rng.choice(
        candidates,
        size=3,
        replace=False,
    )

    return population[indices[0]], population[indices[1]], population[indices[2]]


def _create_mutant(
    target,
    donor_1,
    donor_2,
    donor_3,
    problem,
    rng,
    mutation_factor,
):
    """
    Create a type-aware mutant vector.

    Continuous:
        DE/rand/1 mutation.

    Categorical:
        Category-aware differential mutation.

    Binary:
        XOR-based differential mutation.
    """

    mutant = donor_1.copy()

    # --------------------------------------------------------
    # Continuous variables: DE/rand/1
    # --------------------------------------------------------

    indices = problem.continuous_indices

    mutant[indices] = (
        donor_1[indices]
        + mutation_factor
        * (donor_2[indices] - donor_3[indices])
    )

    mutant[indices] = np.clip(
        mutant[indices],
        problem.lower_bounds[indices],
        problem.upper_bounds[indices],
    )

    # --------------------------------------------------------
    # Categorical variables: height and channel
    # --------------------------------------------------------

    categorical_domains = (
    (problem.height_indices, problem.integer_domains["height"]),
    (problem.channel_indices, problem.integer_domains["channel"]),
    )

    for indices, domain in categorical_domains:

        for index in indices:

            # A categorical difference exists when
            # donor 2 and donor 3 have different categories.
            if donor_2[index] != donor_3[index]:

                if rng.random() < mutation_factor / 2.0:
                    mutant[index] = donor_2[index]

            if mutant[index] not in domain:
                raise ValueError(
                    "Categorical mutation generated an invalid value."
                )

    # --------------------------------------------------------
    # Binary variables: XOR differential mutation
    # --------------------------------------------------------

    for index in problem.active_indices:

        if rng.random() < mutation_factor / 2.0:

            mutant[index] = np.logical_xor(
                bool(donor_1[index]),
                np.logical_xor(
                    bool(donor_2[index]),
                    bool(donor_3[index]),
                ),
            )

    return mutant.astype(float)


def _binomial_crossover(
    target,
    mutant,
    rng,
    crossover_rate,
):
    """Apply binomial crossover with at least one mutant gene."""

    dimension = len(target)

    mask = rng.random(dimension) < crossover_rate

    forced_index = rng.integers(dimension)
    mask[forced_index] = True

    trial = np.where(mask, mutant, target)

    return trial.astype(float)


def mixed_variable_de(
    problem,
    max_evaluations=10_000,
    seed=None,
    population_size=40,
    mutation_factor=0.5,
    crossover_rate=0.9,
):
    """
    Mixed-Variable Differential Evolution.

    Supports:
        - Continuous variables
        - Integer categorical variables
        - Binary variables

    Strategy:
        DE/rand/1 with type-aware discrete mutation.

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

    # --------------------------------------------------------
    # Parameter validation
    # --------------------------------------------------------

    if max_evaluations < 1:
        raise ValueError(
            "max_evaluations must be at least 1."
        )

    if population_size < 4:
        raise ValueError(
            "population_size must be at least 4."
        )

    if not 0.0 < mutation_factor <= 2.0:
        raise ValueError(
            "mutation_factor must be in (0, 2]."
        )

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError(
            "crossover_rate must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    effective_population_size = min(
        population_size,
        max_evaluations,
    )

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = np.array(
        [
            problem.sample_solution(rng)
            for _ in range(effective_population_size)
        ],
        dtype=float,
    )

    fitness = np.empty(
        effective_population_size,
        dtype=float,
    )

    n_evaluations = 0

    for i in range(effective_population_size):

        fitness[i] = float(
            problem.objective(population[i])
        )

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
    # Differential Evolution search
    # --------------------------------------------------------

    while n_evaluations < max_evaluations:

        generation_best_f = best_f

        for target_index in range(effective_population_size):

            if n_evaluations >= max_evaluations:
                break

            target = population[target_index]

            donor_1, donor_2, donor_3 = _select_donors(
                population,
                target_index,
                rng,
            )

            mutant = _create_mutant(
                target,
                donor_1,
                donor_2,
                donor_3,
                problem,
                rng,
                mutation_factor,
            )

            trial = _binomial_crossover(
                target,
                mutant,
                rng,
                crossover_rate,
            )

            problem.validate_solution(trial)

            trial_f = float(
                problem.objective(trial)
            )

            n_evaluations += 1

            # Greedy selection
            if trial_f <= fitness[target_index]:

                population[target_index] = trial
                fitness[target_index] = trial_f

                if trial_f < best_f:

                    best_f = trial_f
                    best_x = trial.copy()

        history.append(
            {
                "n_evaluations": n_evaluations,
                "best_f": best_f,
            }
        )

        if best_f > generation_best_f:
            raise RuntimeError(
                "Best objective value unexpectedly increased."
            )

    runtime_seconds = time.perf_counter() - start_time

    return {
        "algorithm": "MVDE",
        "best_x": best_x,
        "best_f": best_f,
        "n_evaluations": n_evaluations,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "history": history,
    }
