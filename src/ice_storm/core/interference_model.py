
import numpy as np

from ice_storm.core.data import DEMAND_COORDINATES, N_STATIONS
from ice_storm.core.signal_model import compute_received_power


def compute_noise_levels(
    scenario,
    demand_coordinates=DEMAND_COORDINATES
):
    """
    Compute background interference/noise for each demand point.
    """

    demand_coordinates = np.asarray(demand_coordinates, dtype=float)

    nu_0 = scenario["nu_0"]
    rho = scenario["rho"]
    a = scenario["a"]
    b = scenario["b"]

    squared_distance = (
        (demand_coordinates[:, 0] - a) ** 2
        + (demand_coordinates[:, 1] - b) ** 2
    )

    noise = nu_0 * (
        1.0 + rho * np.exp(-squared_distance / 8.0)
    )

    return noise


def compute_sinr(
    deployment,
    scenario,
    demand_coordinates=DEMAND_COORDINATES
):
    """
    Compute SINR for every demand point and radio station.

    Returns an array with shape (n_demand_points, 4).
    """

    deployment = np.asarray(deployment, dtype=float)

    if deployment.shape != (N_STATIONS, 6):
        raise ValueError("Deployment must have shape (4, 6).")

    received_power = compute_received_power(
        deployment,
        demand_coordinates
    )

    noise = compute_noise_levels(
        scenario,
        demand_coordinates
    )

    channels = deployment[:, 4]

    sinr = np.zeros_like(received_power)

    for i in range(N_STATIONS):

        same_channel = channels == channels[i]

        same_channel[i] = False

        interference = np.sum(
            received_power[:, same_channel],
            axis=1
        )

        denominator = noise + interference

        sinr[:, i] = received_power[:, i] / denominator

    return sinr
