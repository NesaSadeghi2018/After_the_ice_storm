
import numpy as np

from ice_storm.core.data import (
    BASE_WEIGHTS,
    DEMAND_TYPES,
    N_DEMAND_POINTS,
)
from ice_storm.core.scenarios import SCENARIOS


def compute_service_quality(sinr):
    """
    Compute the service quality of each demand point.

    q_j = max_i [SINR_ji^2 / (1 + SINR_ji^2)]

    Returns:
        Array with shape (n_demand_points,).
    """
    sinr = np.asarray(sinr, dtype=float)

    if sinr.ndim != 2:
        raise ValueError("SINR must be a two-dimensional matrix.")

    if not np.all(np.isfinite(sinr)) or np.any(sinr < 0):
        raise ValueError(
            "SINR values must be finite and non-negative."
        )

    if sinr.shape[1] == 0:
        raise ValueError("SINR must contain at least one station.")

    station_quality = sinr**2 / (1.0 + sinr**2)

    return np.max(station_quality, axis=1)


def compute_priority_weights(scenario_name):
    """
    Compute priority-adjusted weights:

    W_j = w_j * m_j
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    multipliers = SCENARIOS[scenario_name]["priority_multipliers"]

    priority_factors = np.array(
        [multipliers[demand_type] for demand_type in DEMAND_TYPES],
        dtype=float
    )

    return BASE_WEIGHTS * priority_factors


def compute_objective_q(sinr, scenario_name):
    """
    Compute the weighted negative service quality.

    Q(X) = -sum(W_j * q_j) / 18
    """
    service_quality = compute_service_quality(sinr)
    weights = compute_priority_weights(scenario_name)

    if service_quality.shape != weights.shape:
        raise ValueError(
            "SINR demand-point count does not match the data."
        )

    objective_q = (
        -np.sum(weights * service_quality) / N_DEMAND_POINTS
    )

    return float(objective_q)
