
import numpy as np

from ice_storm.core.data import DEPOT_COORDINATES, N_STATIONS


def compute_deployment_cost(deployment):
    """
    Compute total deployment cost:

    C(X) = sum_i z_i * [400 + 35(h_i - 2)^2 + 20 L_i]

    L_i is the distance from station i to the depot.
    """
    deployment = np.asarray(deployment, dtype=float)

    if deployment.shape != (N_STATIONS, 6):
        raise ValueError("Deployment must have shape (4, 6).")

    positions = deployment[:, 0:2]
    heights = deployment[:, 3]
    active_status = deployment[:, 5]

    distances_to_depot = np.sqrt(
        np.sum((positions - DEPOT_COORDINATES) ** 2, axis=1)
    )

    station_costs = (
        400
        + 35 * (heights - 2) ** 2
        + 20 * distances_to_depot
    )

    total_cost = np.sum(active_status * station_costs)

    return float(total_cost)


def compute_energy_consumption(deployment):
    """
    Compute energy consumption over 12 hours:

    E(X) = 12 * sum_i z_i *
           [18 + 14 p_i + 0.5(h_i - 2)]
    """
    deployment = np.asarray(deployment, dtype=float)

    if deployment.shape != (N_STATIONS, 6):
        raise ValueError("Deployment must have shape (4, 6).")

    powers = deployment[:, 2]
    heights = deployment[:, 3]
    active_status = deployment[:, 5]

    station_energy = (
        18
        + 14 * powers
        + 0.5 * (heights - 2)
    )

    total_energy = 12 * np.sum(active_status * station_energy)

    return float(total_energy)
