
import time

import numpy as np

from ice_storm.optimization.algorithms.multiobjective_utils import (
    crowding_distance,
    environmental_selection,
    fast_non_dominated_sort,
)


def _evaluate_population(problem, population):
    """Evaluate all objectives for each candidate."""

    objectives = np.asarray(
        [
            problem.objectives(solution)
            for solution in population
        ],
        dtype=float,
    )

    if objectives.ndim != 2:
        raise RuntimeError(
            "Objective evaluation must return a two-dimensional array."
        )

    if not np.all(np.isfinite(objectives)):
        raise RuntimeError(
            "Objective values must be finite."
        )

    return objectives


def _calculate_rank_and_crowding(objectives):
    """Calculate non-dominated ranks and crowding distances."""

    fronts = fast_non_dominated_sort(objectives)

    ranks = np.empty(len(objectives), dtype=int)
    distances = np.zeros(len(objectives), dtype=float)

    for rank, front in enumerate(fronts):

        ranks[front] = rank

        distances[front] = crowding_distance(
            objectives,
            front,
        )

    return fronts, ranks, distances


def _tournament_selection(
    population,
    ranks,
    distances,
    rng,
    tournament_size=2,
):
    """Select a parent using rank and crowding distance."""

    candidates = rng.choice(
        len(population),
        size=min(tournament_size, len(population)),
        replace=False,
    )

    winner = int(candidates[0])

    for candidate in candidates[1:]:

        candidate = int(candidate)

        if ranks[candidate] < ranks[winner]:

            winner = candidate

        elif ranks[candidate] == ranks[winner]:

            if distances[candidate] > distances[winner]:

                winner = candidate

            elif distances[candidate] == distances[winner]:

                if rng.random() < 0.5:
                    winner = candidate

    return population[winner].copy()


def _simulated_binary_crossover(
    parent_1,
    parent_2,
    problem,
    rng,
    eta_c,
):
    """Apply bounded SBX to continuous variables."""

    child_1 = parent_1.copy()
    child_2 = parent_2.copy()

    for index in problem.continuous_indices:

        if rng.random() > 0.5:
            continue

        x1 = parent_1[index]
        x2 = parent_2[index]

        if np.isclose(x1, x2):
            continue

        lower = problem.lower_bounds[index]
        upper = problem.upper_bounds[index]

        y1 = min(x1, x2)
        y2 = max(x1, x2)

        rand = rng.random()

        # Generate the first child.
        beta = 1.0 + (
            2.0 * (y1 - lower) / (y2 - y1)
        )

        alpha = 2.0 - beta ** (-(eta_c + 1.0))

        if rand <= 1.0 / alpha:

            beta_q = (
                rand * alpha
            ) ** (1.0 / (eta_c + 1.0))

        else:

            beta_q = (
                1.0 / (
                    2.0 - rand * alpha
                )
            ) ** (1.0 / (eta_c + 1.0))

        c1 = 0.5 * (
            (y1 + y2)
            - beta_q * (y2 - y1)
        )

        # Generate the second child.
        beta = 1.0 + (
            2.0 * (upper - y2) / (y2 - y1)
        )

        alpha = 2.0 - beta ** (-(eta_c + 1.0))

        if rand <= 1.0 / alpha:

            beta_q = (
                rand * alpha
            ) ** (1.0 / (eta_c + 1.0))

        else:

            beta_q = (
                1.0 / (
                    2.0 - rand * alpha
                )
            ) ** (1.0 / (eta_c + 1.0))

        c2 = 0.5 * (
            (y1 + y2)
            + beta_q * (y2 - y1)
        )

        c1 = np.clip(c1, lower, upper)
        c2 = np.clip(c2, lower, upper)

        if rng.random() < 0.5:
            child_1[index] = c2
            child_2[index] = c1
        else:
            child_1[index] = c1
            child_2[index] = c2

    return child_1, child_2

def _mixed_variable_crossover(
    parent_1,
    parent_2,
    problem,
    rng,
    eta_c,
):
    """Apply type-aware crossover to mixed-variable chromosomes."""

    child_1, child_2 = _simulated_binary_crossover(
        parent_1,
        parent_2,
        problem,
        rng,
        eta_c,
    )

    discrete_indices = (
        list(problem.height_indices)
        + list(problem.channel_indices)
        + list(problem.active_indices)
    )

    for index in discrete_indices:

        if rng.random() < 0.5:

            child_1[index] = parent_2[index]
            child_2[index] = parent_1[index]

    return child_1, child_2


def _polynomial_mutation(
    solution,
    problem,
    rng,
    mutation_rate,
    eta_m,
):
    """Apply bounded polynomial mutation to continuous variables."""

    child = solution.copy()

    for index in problem.continuous_indices:

        if rng.random() >= mutation_rate:
            continue

        lower = problem.lower_bounds[index]
        upper = problem.upper_bounds[index]

        variable_range = upper - lower

        if variable_range <= 0.0:
            continue

        x = child[index]

        delta_1 = (x - lower) / variable_range
        delta_2 = (upper - x) / variable_range

        u = rng.random()
        mutation_power = 1.0 / (eta_m + 1.0)

        if u <= 0.5:

            xy = 1.0 - delta_1

            value = (
                2.0 * u
                + (1.0 - 2.0 * u)
                * (xy ** (eta_m + 1.0))
            )

            delta_q = (
                value ** mutation_power
            ) - 1.0

        else:

            xy = 1.0 - delta_2

            value = (
                2.0 * (1.0 - u)
                + 2.0 * (u - 0.5)
                * (xy ** (eta_m + 1.0))
            )

            delta_q = (
                1.0 - value ** mutation_power
            )

        child[index] = np.clip(
            x + delta_q * variable_range,
            lower,
            upper,
        )

    return child


def _mutate_discrete_variables(
    solution,
    problem,
    rng,
    mutation_rate,
):
    """Apply categorical and binary mutation."""

    child = solution.copy()

    for index in problem.height_indices:

        if rng.random() < mutation_rate:

            alternatives = [
                value
                for value in problem.integer_domains["height"]
                if value != child[index]
            ]

            if alternatives:
                child[index] = rng.choice(alternatives)

    for index in problem.channel_indices:

        if rng.random() < mutation_rate:

            alternatives = [
                value
                for value in problem.integer_domains["channel"]
                if value != child[index]
            ]

            if alternatives:
                child[index] = rng.choice(alternatives)

    for index in problem.active_indices:

        if rng.random() < mutation_rate:
            child[index] = 1 - child[index]

    return child


def _mutate(
    solution,
    problem,
    rng,
    mutation_rate,
    eta_m,
):
    """Apply type-aware mutation to a candidate solution."""

    child = _polynomial_mutation(
        solution,
        problem,
        rng,
        mutation_rate,
        eta_m,
    )

    child = _mutate_discrete_variables(
        child,
        problem,
        rng,
        mutation_rate,
    )

    problem.validate_solution(child)

    return child


def mixed_variable_nsga_ii(
    problem,
    max_evaluations=10_000,
    seed=None,
    population_size=40,
    crossover_rate=0.9,
    mutation_rate=None,
    tournament_size=2,
    eta_c=20.0,
    eta_m=20.0,
):
    """
    Mixed-Variable Non-dominated Sorting Genetic Algorithm II.

    Supports:
        - Continuous variables
        - Integer categorical variables
        - Binary variables
        - Multiple minimization objectives

    Returns:
        Dictionary containing:
            algorithm
            population
            objectives
            pareto_solutions
            pareto_objectives
            n_evaluations
            runtime_seconds
            seed
            history
    """

    start_time = time.perf_counter()

    if max_evaluations < 1:
        raise ValueError(
            "max_evaluations must be at least 1."
        )

    if population_size < 2:
        raise ValueError(
            "population_size must be at least 2."
        )

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError(
            "crossover_rate must be between 0 and 1."
        )

    if tournament_size < 1:
        raise ValueError(
            "tournament_size must be at least 1."
        )

    if (
        not np.isfinite(eta_c)
        or not np.isfinite(eta_m)
        or eta_c <= 0.0
        or eta_m <= 0.0
    ):
        raise ValueError(
            "eta_c and eta_m must be finite and positive."
        )

    if mutation_rate is None:
        mutation_rate = 1.0 / problem.dimension

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "mutation_rate must be between 0 and 1."
        )

    rng = np.random.default_rng(seed)

    population_size = min(
        population_size,
        max_evaluations,
    )

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = np.asarray(
        [
            problem.sample_solution(rng)
            for _ in range(population_size)
        ],
        dtype=float,
    )

    for solution in population:
        problem.validate_solution(solution)

    objectives = _evaluate_population(
        problem,
        population,
    )

    n_evaluations = len(population)

    history = [
        {
            "n_evaluations": n_evaluations,
            "n_pareto_solutions": len(
                fast_non_dominated_sort(objectives)[0]
            ),
        }
    ]

    # --------------------------------------------------------
    # Evolutionary search
    # --------------------------------------------------------

    while n_evaluations < max_evaluations:

        fronts, ranks, distances = (
            _calculate_rank_and_crowding(objectives)
        )

        offspring = []

        remaining_budget = (
            max_evaluations - n_evaluations
        )

        offspring_target = min(
            population_size,
            remaining_budget,
        )

        while len(offspring) < offspring_target:

            parent_1 = _tournament_selection(
                population,
                ranks,
                distances,
                rng,
                tournament_size,
            )

            parent_2 = _tournament_selection(
                population,
                ranks,
                distances,
                rng,
                tournament_size,
            )

            if rng.random() < crossover_rate:

                child_1, child_2 = (
                    _mixed_variable_crossover(
                        parent_1,
                        parent_2,
                        problem,
                        rng,
                        eta_c,
                    )
                )

            else:

                child_1 = parent_1.copy()
                child_2 = parent_2.copy()

            for child in (child_1, child_2):

                if len(offspring) >= offspring_target:
                    break

                child = _mutate(
                    child,
                    problem,
                    rng,
                    mutation_rate,
                    eta_m,
                )

                offspring.append(child)

        offspring = np.asarray(
            offspring,
            dtype=float,
        )

        offspring_objectives = _evaluate_population(
            problem,
            offspring,
        )

        n_evaluations += len(offspring)

        # ----------------------------------------------------
        # Environmental selection
        # ----------------------------------------------------

        combined_population = np.vstack(
            (population, offspring)
        )

        combined_objectives = np.vstack(
            (objectives, offspring_objectives)
        )

        (
            population,
            objectives,
            _,
        ) = environmental_selection(
            combined_population,
            combined_objectives,
            population_size,
        )

        final_fronts = fast_non_dominated_sort(
            objectives
        )

        history.append(
            {
                "n_evaluations": n_evaluations,
                "n_pareto_solutions": len(final_fronts[0]),
            }
        )

    # --------------------------------------------------------
    # Final Pareto front
    # --------------------------------------------------------

    final_fronts = fast_non_dominated_sort(
        objectives
    )

    pareto_indices = np.asarray(
        final_fronts[0],
        dtype=int,
    )

    pareto_solutions = population[
        pareto_indices
    ].copy()

    pareto_objectives = objectives[
        pareto_indices
    ].copy()

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    return {
        "algorithm": "MVNSGA-II",
        "population": population.copy(),
        "objectives": objectives.copy(),
        "pareto_solutions": pareto_solutions,
        "pareto_objectives": pareto_objectives,
        "n_evaluations": n_evaluations,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "history": history,
    }
