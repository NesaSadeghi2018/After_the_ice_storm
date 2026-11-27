
import numpy as np


def dominates(objectives_a, objectives_b):
    """
    Return True if solution A Pareto-dominates solution B.

    All objectives are assumed to be minimized.
    """

    a = np.asarray(objectives_a, dtype=float)
    b = np.asarray(objectives_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            "Objective vectors must have the same shape."
        )

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(
            "Objective values must be finite."
        )

    return bool(
        np.all(a <= b)
        and np.any(a < b)
    )


def fast_non_dominated_sort(objectives):
    """
    Perform fast non-dominated sorting.

    Parameters:
        objectives:
            Array of shape (population_size, n_objectives).

    Returns:
        List of fronts.
        Each front contains integer population indices.
    """

    objectives = np.asarray(objectives, dtype=float)

    if objectives.ndim != 2:
        raise ValueError(
            "Objectives must be a two-dimensional array."
        )

    if len(objectives) == 0:
        return []

    if not np.all(np.isfinite(objectives)):
        raise ValueError(
            "Objective values must be finite."
        )

    population_size = len(objectives)

    domination_counts = np.zeros(
        population_size,
        dtype=int,
    )

    dominated_solutions = [
        []
        for _ in range(population_size)
    ]

    first_front = []

    for p in range(population_size):

        for q in range(population_size):

            if p == q:
                continue

            if dominates(objectives[p], objectives[q]):

                dominated_solutions[p].append(q)

            elif dominates(objectives[q], objectives[p]):

                domination_counts[p] += 1

        if domination_counts[p] == 0:
            first_front.append(p)

    fronts = [first_front]

    current_front_index = 0

    while current_front_index < len(fronts):

        next_front = []

        for p in fronts[current_front_index]:

            for q in dominated_solutions[p]:

                domination_counts[q] -= 1

                if domination_counts[q] == 0:
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)

        current_front_index += 1

    return fronts


def crowding_distance(objectives, front):
    """
    Calculate normalized crowding distances for one Pareto front.

    Boundary solutions receive infinite distance.

    Objectives with zero range are ignored because they provide
    no diversity information within the current front.
    """

    objectives = np.asarray(objectives, dtype=float)
    front = np.asarray(front, dtype=int)

    if objectives.ndim != 2:
        raise ValueError(
            "Objectives must be a two-dimensional array."
        )

    if not np.all(np.isfinite(objectives)):
        raise ValueError(
            "Objective values must be finite."
        )

    if np.any(front < 0) or np.any(front >= len(objectives)):
        raise ValueError(
            "Front contains invalid population indices."
        )

    distances = np.zeros(
        len(front),
        dtype=float,
    )

    if len(front) == 0:
        return distances

    if len(front) <= 2:
        distances[:] = np.inf
        return distances

    front_objectives = objectives[front]

    n_objectives = objectives.shape[1]

    for objective_index in range(n_objectives):

        sorted_positions = np.argsort(
            front_objectives[:, objective_index],
            kind="stable",
        )

        objective_min = front_objectives[
            sorted_positions[0],
            objective_index,
        ]

        objective_max = front_objectives[
            sorted_positions[-1],
            objective_index,
        ]

        objective_range = objective_max - objective_min

        # Ignore objectives with no variation in this front.
        if objective_range == 0.0:
            continue

        # Boundary solutions receive infinite crowding distance.
        distances[sorted_positions[0]] = np.inf
        distances[sorted_positions[-1]] = np.inf

        for position in range(1, len(front) - 1):

            current_position = sorted_positions[position]

            if np.isinf(distances[current_position]):
                continue

            previous_value = front_objectives[
                sorted_positions[position - 1],
                objective_index,
            ]

            next_value = front_objectives[
                sorted_positions[position + 1],
                objective_index,
            ]

            distances[current_position] += (
                next_value - previous_value
            ) / objective_range

    return distances


def environmental_selection(
    population,
    objectives,
    population_size,
):
    """
    Select the next generation using NSGA-II environmental selection.

    Selection priority:
        1. Lower non-dominated rank.
        2. Higher crowding distance.

    Returns:
        selected_population:
            Selected candidate solutions.

        selected_objectives:
            Objective vectors corresponding to selected solutions.

        selected_indices:
            Indices of selected solutions in the original arrays.
    """

    population = np.asarray(
        population,
        dtype=float,
    )

    objectives = np.asarray(
        objectives,
        dtype=float,
    )

    if population.ndim != 2:
        raise ValueError(
            "Population must be a two-dimensional array."
        )

    if objectives.ndim != 2:
        raise ValueError(
            "Objectives must be a two-dimensional array."
        )

    if len(population) != len(objectives):
        raise ValueError(
            "Population and objective arrays must have equal lengths."
        )

    if population_size < 1:
        raise ValueError(
            "Population size must be positive."
        )

    if population_size > len(population):
        raise ValueError(
            "Population size cannot exceed the candidate count."
        )

    fronts = fast_non_dominated_sort(objectives)

    selected_indices = []

    for front in fronts:

        remaining = population_size - len(selected_indices)

        if remaining <= 0:
            break

        if len(front) <= remaining:

            selected_indices.extend(front)

        else:

            distances = crowding_distance(
                objectives,
                front,
            )

            order = np.argsort(
                -distances,
                kind="stable",
            )

            selected_indices.extend(
                np.asarray(front, dtype=int)[
                    order[:remaining]
                ].tolist()
            )

            break

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    return (
        population[selected_indices].copy(),
        objectives[selected_indices].copy(),
        selected_indices,
    )
