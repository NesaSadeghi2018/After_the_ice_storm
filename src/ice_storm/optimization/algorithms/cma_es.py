
import time
import numpy as np
import cma


def cma_es(
    problem,
    seed=None,
    max_evaluations=10_000,
    population_size=20,
):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    Minimizes problem.objective(x) under box constraints.
    """

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive.")

    if population_size < 2:
        raise ValueError("Population size must be at least 2.")

    if max_evaluations < population_size:
        raise ValueError("max_evaluations must be at least population_size.")

    dimension = problem.dimension

    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)

    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError("Problem bounds have incorrect dimensions.")

    if np.any(lower >= upper):
        raise ValueError("Invalid variable bounds.")

    start_time = time.perf_counter()

    # Initial point and initial step size.
    x0 = (lower + upper) / 2.0
    sigma0 = 0.25 * float(np.mean(upper - lower))

    options = {
        "bounds": [lower.tolist(), upper.tolist()],
        "popsize": population_size,
        "maxfevals": max_evaluations,
        "seed": seed,
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
        "verb_time": False,
    }

    strategy = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        options,
    )

    evaluations = 0
    best_x = None
    best_f = float("inf")

    history = []

    # CMA-ES optimization loop.
    # CMA-ES optimization loop.
    while (
        not strategy.stop()
        and evaluations + population_size <= max_evaluations
    ):

        # Ask for a complete population.
        solutions = strategy.ask()

        fitness_values = []

        for solution in solutions:

            value = float(
                problem.objective(np.asarray(solution, dtype=float))
            )

            fitness_values.append(value)
            evaluations += 1

            if value < best_f:
                best_f = value
                best_x = np.asarray(solution, dtype=float).copy()

        # Tell CMA-ES the evaluated candidates.
        strategy.tell(solutions, fitness_values)

        history.append(
            {
                "evaluations": evaluations,
                "best_objective": best_f,
            }
        )

    runtime = time.perf_counter() - start_time

    return {
        "algorithm": "CMA-ES",
        "best_x": best_x,
        "best_f": best_f,
        "n_evaluations": evaluations,
        "runtime_seconds": runtime,
        "seed": seed,
        "history": history,
    }
