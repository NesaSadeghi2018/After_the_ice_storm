
import time
import numpy as np


def differential_evolution(
    problem,
    seed=None,
    max_evaluations=10_000,
    population_size=50,
    mutation_factor=0.8,
    crossover_rate=0.9,
):
    """
    Differential Evolution (DE/rand/1/bin).

    Minimizes problem.objective(x) subject to box constraints.

    Parameters
    ----------
    problem:
        Problem object exposing:
        - dimension
        - lower_bounds
        - upper_bounds
        - objective(x)

    seed : int or None
        Random seed for reproducibility.

    max_evaluations : int
        Maximum number of objective evaluations.

    population_size : int
        Number of individuals in the population.

    mutation_factor : float
        Differential mutation factor F.

    crossover_rate : float
        Binomial crossover probability CR.

    Returns
    -------
    dict
        Best solution, objective value, evaluation count,
        runtime, and convergence history.
    """

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive.")

    if population_size < 4:
        raise ValueError(
            "Population size must be at least 4 for DE/rand/1."
        )

    if population_size > max_evaluations:
        raise ValueError(
            "Population size cannot exceed evaluation budget."
        )

    if not (0.0 < mutation_factor <= 2.0):
        raise ValueError("mutation_factor must be in (0, 2].")

    if not (0.0 <= crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be in [0, 1].")

    rng = np.random.default_rng(seed)

    dimension = problem.dimension
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError("Problem bounds have incorrect dimensions.")

    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be below its upper bound.")

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 1. Initialize population
    # --------------------------------------------------------

    population = rng.uniform(
        lower,
        upper,
        size=(population_size, dimension),
    )

    fitness = np.empty(population_size, dtype=float)

    evaluations = 0

    for i in range(population_size):
        fitness[i] = problem.objective(population[i])
        evaluations += 1

    best_index = int(np.argmin(fitness))

    best_x = population[best_index].copy()
    best_f = float(fitness[best_index])

    history = [
        {
            "evaluations": evaluations,
            "best_objective": best_f,
        }
    ]

    # --------------------------------------------------------
    # 2. Evolutionary search
    # --------------------------------------------------------

    while evaluations < max_evaluations:

        old_population = population.copy()
        old_fitness = fitness.copy()

        next_population = old_population.copy()
        next_fitness = old_fitness.copy()

        evaluated_in_generation = 0

        for i in range(population_size):

            if evaluations >= max_evaluations:
                break

            # Select three distinct individuals, excluding i.
            candidates = np.delete(
                np.arange(population_size),
                i,
            )

            r1, r2, r3 = rng.choice(
                candidates,
                size=3,
                replace=False,
            )

            # Mutation: DE/rand/1
            mutant = (
                old_population[r1]
                + mutation_factor
                * (old_population[r2] - old_population[r3])
            )

            # Repair bound violations.
            mutant = np.clip(mutant, lower, upper)

            # Binomial crossover.
            crossover_mask = rng.random(dimension) < crossover_rate

            # Ensure at least one mutant component is selected.
            forced_index = rng.integers(dimension)
            crossover_mask[forced_index] = True

            trial = np.where(
                crossover_mask,
                mutant,
                old_population[i],
            )

            trial_fitness = float(problem.objective(trial))
            evaluations += 1
            evaluated_in_generation += 1

            # Greedy selection.
            if trial_fitness <= old_fitness[i]:
                next_population[i] = trial
                next_fitness[i] = trial_fitness

        population = next_population
        fitness = next_fitness

        generation_best_index = int(np.argmin(fitness))

        if fitness[generation_best_index] < best_f:
            best_f = float(fitness[generation_best_index])
            best_x = population[generation_best_index].copy()

        history.append(
            {
                "evaluations": evaluations,
                "best_objective": best_f,
            }
        )

        if evaluated_in_generation == 0:
            break

    runtime = time.perf_counter() - start_time

    return {
        "algorithm": "Differential Evolution",
        "best_x": best_x,
        "best_f": best_f,
        "n_evaluations": evaluations,
        "runtime_seconds": runtime,
        "seed": seed,
        "history": history,
    }
