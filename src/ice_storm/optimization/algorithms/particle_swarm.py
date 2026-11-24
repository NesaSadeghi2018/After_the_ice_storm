
import time
import numpy as np


def particle_swarm(
    problem,
    seed=None,
    max_evaluations=10_000,
    swarm_size=50,
    inertia=0.729,
    cognitive=1.49445,
    social=1.49445,
):
    """
    Global-best Particle Swarm Optimization (PSO).

    Minimizes problem.objective(x) subject to box constraints.
    """

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive.")

    if swarm_size < 1:
        raise ValueError("swarm_size must be positive.")

    if swarm_size > max_evaluations:
        raise ValueError(
            "Swarm size cannot exceed evaluation budget."
        )

    rng = np.random.default_rng(seed)

    dimension = problem.dimension
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError("Problem bounds have incorrect dimensions.")

    if np.any(lower >= upper):
        raise ValueError("Invalid variable bounds.")

    start_time = time.perf_counter()

    # Initialize particle positions and velocities.
    positions = rng.uniform(
        lower,
        upper,
        size=(swarm_size, dimension),
    )

    velocity_limit = 0.2 * (upper - lower)

    velocities = rng.uniform(
        -velocity_limit,
        velocity_limit,
        size=(swarm_size, dimension),
    )

    # Evaluate initial swarm.
    fitness = np.empty(swarm_size, dtype=float)

    evaluations = 0

    for i in range(swarm_size):
        fitness[i] = problem.objective(positions[i])
        evaluations += 1

    # Personal best positions.
    personal_best_positions = positions.copy()
    personal_best_fitness = fitness.copy()

    # Global best.
    best_index = int(np.argmin(personal_best_fitness))

    global_best_position = personal_best_positions[
        best_index
    ].copy()

    global_best_fitness = float(
        personal_best_fitness[best_index]
    )

    history = [
        {
            "evaluations": evaluations,
            "best_objective": global_best_fitness,
        }
    ]

    # PSO iterations.
    while evaluations < max_evaluations:

        r1 = rng.random((swarm_size, dimension))
        r2 = rng.random((swarm_size, dimension))

        velocities = (
            inertia * velocities
            + cognitive
            * r1
            * (personal_best_positions - positions)
            + social
            * r2
            * (global_best_position - positions)
        )

        velocities = np.clip(
            velocities,
            -velocity_limit,
            velocity_limit,
        )

        positions = positions + velocities

        # Reflect particles at violated boundaries.
        below = positions < lower
        above = positions > upper

        positions = np.clip(positions, lower, upper)

        velocities[below | above] *= -0.5

        # Evaluate particles within remaining budget.
        evaluated_in_iteration = 0

        for i in range(swarm_size):

            if evaluations >= max_evaluations:
                break

            current_fitness = float(
                problem.objective(positions[i])
            )

            evaluations += 1
            evaluated_in_iteration += 1

            if current_fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = current_fitness
                personal_best_positions[i] = positions[i].copy()

                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = positions[i].copy()

        history.append(
            {
                "evaluations": evaluations,
                "best_objective": global_best_fitness,
            }
        )

        if evaluated_in_iteration == 0:
            break

    runtime = time.perf_counter() - start_time

    return {
        "algorithm": "Particle Swarm Optimization",
        "best_x": global_best_position,
        "best_f": global_best_fitness,
        "n_evaluations": evaluations,
        "runtime_seconds": runtime,
        "seed": seed,
        "history": history,
    }
