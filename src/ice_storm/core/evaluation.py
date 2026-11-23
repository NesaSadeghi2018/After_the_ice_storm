
from ice_storm.core.data import DEMAND_COORDINATES
from ice_storm.core.scenarios import SCENARIOS

from ice_storm.core.interference_model import (
    compute_noise_levels as _compute_noise_levels,
    compute_sinr as _compute_sinr,
)

from ice_storm.core.service_model import compute_objective_q

from ice_storm.core.objectives import (
    compute_deployment_cost,
    compute_energy_consumption,
)


def compute_background_noise(scenario, demand_coordinates):
    """
    Compute background noise for each demand point.

    This function delegates the calculation to interference_model.py
    to maintain a single source of truth.
    """
    return _compute_noise_levels(
        scenario,
        demand_coordinates
    )


def compute_sinr(deployment, scenario_name):
    """
    Compute SINR for every demand point and station.

    The scenario is selected by name and passed to the
    centralized SINR implementation in interference_model.py.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = SCENARIOS[scenario_name]

    return _compute_sinr(
        deployment,
        scenario,
        DEMAND_COORDINATES
    )


def evaluate_deployment(deployment, scenario_name):
    """
    Evaluate a deployment under one operating scenario.

    Returns:
        Q: Negative weighted service quality
        C: Deployment cost
        E: 12-hour energy consumption
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    sinr = compute_sinr(deployment, scenario_name)

    q = compute_objective_q(sinr, scenario_name)

    cost = compute_deployment_cost(deployment)

    energy = compute_energy_consumption(deployment)

    return {
        "Q": q,
        "C": cost,
        "E": energy,
    }
