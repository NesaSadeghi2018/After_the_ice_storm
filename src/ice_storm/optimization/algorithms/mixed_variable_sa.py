
import time

import numpy as np


def _generate_neighbor(
    solution,
    problem,
    rng,
    continuous_step_size,
):
    """
    Generate a type-aware neighboring solution.

    Continuous variables:
        Gaussian perturbation.

    Height and channel:
        Random selection of a different valid category.

    Binary variables:
        Bit flip.
    """

    neighbor = solution.copy()

    variable_groups = [
        problem.continuous_indices,
        problem.height_indices,
        problem.channel_indices,
        problem.active_indices,
    ]

    nonempty_groups = [
        group for group in variable_groups if len(group) > 0
    ]

    selected_group = nonempty_groups[
        rng.integers(len(nonempty_groups))
    ]

    selected_index = int(rng.choice(selected_group))

    # Continuous variables
    if selected_index in problem.continuous_indices:

        variable_range = (
            problem.upper_bounds[selected_index]
            - problem.lower_bounds[selected_index]
        )

        step = continuous_step_size * variable_range

        neighbor[selected_index] += rng.normal(
            loc=0.0,
            scale=step,
        )

        neighbor[selected_index] = np.clip(
            neighbor[selected_index],
            problem.lower_bounds[selected_index],
            problem.upper_bounds[selected_index],
        )

    # Mast height
    elif selected_index in problem.height_indices:

        allowed_values = problem.integer_domains["height"]
        current_value = neighbor[selected_index]

        alternatives = allowed_values[
            allowed_values != current_value
        ]

        neighbor[selected_index] = rng.choice(alternatives)

    # Channel
    elif selected_index in problem.channel_indices:

        allowed_values = problem.integer_domains["channel"]
        current_value = neighbor[selected_index]

        alternatives = allowed_values[
            allowed_values != current_value
        ]

        neighbor[selected_index] = rng.choice(alternatives)

    # Active / inactive
    elif selected_index in problem.active_indices:

        neighbor[selected_index] = 1.0 - neighbor[selected_index]

    return neighbor.astype(float)


def mixed_variable_sa(
    problem,
    max_evaluations=10_000,
    seed=None,
    initial_population_size=40,
    initial_temperature=100.0,
    cooling_rate=0.995,
    minimum_temperature=1e-8,
    continuous_step_size=0.10,
    reheat_after=500,
    reheat_factor=2.0,
):
    """
    Mixed-Variable Simulated Annealing.

    Initial solution:
        Best solution from a randomly generated initial population.

    Supports:
        - Continuous variables
        - Categorical integer variables
        - Binary variables

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

    if initial_population_size < 1:
        raise ValueError(
            "initial_population_size must be at least 1."
        )

    if initial_temperature <= 0:
        raise ValueError(
            "initial_temperature must be positive."
        )

    if not 0.0 < cooling_rate < 1.0:
        raise ValueError(
            "cooling_rate must be between 0 and 1."
        )

    if minimum_temperature <= 0:
        raise ValueError(
            "minimum_temperature must be positive."
        )

    if minimum_temperature >= initial_temperature:
        raise ValueError(
            "minimum_temperature must be smaller than initial_temperature."
        )

    if continuous_step_size <= 0:
        raise ValueError(
            "continuous_step_size must be positive."
        )

    if reheat_after < 1:
        raise ValueError(
            "reheat_after must be at least 1."
        )

    if reheat_factor <= 1:
        raise ValueError(
            "reheat_factor must be greater than 1."
        )

    rng = np.random.default_rng(seed)

    effective_population_size = min(
        initial_population_size,
        max_evaluations,
    )

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    best_x = None
    best_f = float("inf")

    n_evaluations = 0

    temperature = float(initial_temperature)

    history = []

    for _ in range(effective_population_size):

        candidate_x = problem.sample_solution(rng)

        problem.validate_solution(candidate_x)

        candidate_f = float(
            problem.objective(candidate_x)
        )

        n_evaluations += 1

        if candidate_f < best_f:

            best_x = candidate_x.copy()
            best_f = candidate_f

        history.append(
            {
                "n_evaluations": n_evaluations,
                "best_f": best_f,
                "temperature": temperature,
            }
        )

    # Start Simulated Annealing from the best initial candidate.
    current_x = best_x.copy()
    current_f = best_f

    evaluations_since_improvement = 0

    # --------------------------------------------------------
    # Simulated Annealing search
    # --------------------------------------------------------

    while n_evaluations < max_evaluations:

        neighbor_x = _generate_neighbor(
            current_x,
            problem,
            rng,
            continuous_step_size,
        )

        problem.validate_solution(neighbor_x)

        neighbor_f = float(
            problem.objective(neighbor_x)
        )

        n_evaluations += 1

        delta = neighbor_f - current_f

        # Metropolis acceptance criterion
        if delta <= 0:

            accept = True

        else:

            acceptance_probability = np.exp(
                -delta / temperature
            )

            accept = (
                rng.random() < acceptance_probability
            )

        if accept:

            current_x = neighbor_x
            current_f = neighbor_f

        # Update global best
        if current_f < best_f:

            best_x = current_x.copy()
            best_f = float(current_f)

            evaluations_since_improvement = 0

        else:

            evaluations_since_improvement += 1

        # Cooling schedule
        temperature = max(
            minimum_temperature,
            temperature * cooling_rate,
        )

        # Reheating after prolonged stagnation
        if evaluations_since_improvement >= reheat_after:

            temperature = min(
                initial_temperature,
                max(
                    temperature * reheat_factor,
                    initial_temperature * 0.1,
                ),
            )

            evaluations_since_improvement = 0

        history.append(
            {
                "n_evaluations": n_evaluations,
                "best_f": best_f,
                "temperature": temperature,
            }
        )

    runtime_seconds = time.perf_counter() - start_time

    return {
        "algorithm": "MVSA",
        "best_x": best_x,
        "best_f": best_f,
        "n_evaluations": n_evaluations,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
        "history": history,
    }
