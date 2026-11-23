
import numpy as np

from ice_storm.core.data import DEMAND_COORDINATES, N_STATIONS


def compute_squared_distances(deployment, demand_coordinates=DEMAND_COORDINATES):
    """
    Compute squared Euclidean distances between demand points
    and radio stations.

    Returns an array with shape (number_of_demand_points, 4).
    Rows represent demand points; columns represent stations.
    """
    deployment = np.asarray(deployment, dtype=float)
    demand_coordinates = np.asarray(demand_coordinates, dtype=float)

    if deployment.shape != (N_STATIONS, 6):
        raise ValueError("Deployment must have shape (4, 6).")

    if demand_coordinates.ndim != 2 or demand_coordinates.shape[1] != 2:
        raise ValueError("Demand coordinates must have shape (n, 2).")

    differences = (
        demand_coordinates[:, np.newaxis, :]
        - deployment[np.newaxis, :, 0:2]
    )

    squared_distances = np.sum(differences ** 2, axis=2)

    return squared_distances


def compute_station_gains(deployment):
    """
    Compute the station gain factor:

        g_i = 1 + 0.5 * (h_i - 2)
    """
    deployment = np.asarray(deployment, dtype=float)

    if deployment.shape != (N_STATIONS, 6):
        raise ValueError("Deployment must have shape (4, 6).")

    heights = deployment[:, 3]

    gains = 1.0 + 0.5 * (heights - 2.0)

    return gains


def compute_received_power(deployment, demand_coordinates=DEMAND_COORDINATES):
    """
    Compute received signal power:

        R_ji = z_i * p_i / (1 + d_ji^2 / g_i)

    Returns an array with shape (n_demand_points, 4).
    """
    deployment = np.asarray(deployment, dtype=float)

    squared_distances = compute_squared_distances(
        deployment,
        demand_coordinates
    )

    gains = compute_station_gains(deployment)

    powers = deployment[:, 2]
    active_status = deployment[:, 5]

    received_power = (
        active_status[np.newaxis, :] * powers[np.newaxis, :]
    ) / (
        1.0 + squared_distances / gains[np.newaxis, :]
    )

    return received_power
