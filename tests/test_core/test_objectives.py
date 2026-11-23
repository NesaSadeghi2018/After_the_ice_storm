
import numpy as np
import pytest

from ice_storm.core.objectives import (
    compute_deployment_cost,
    compute_energy_consumption,
)


def sample_deployment():
    return np.array(
        [
            [5.0, 0.0, 1.0, 2.0, 1.0, 1.0],
            [5.0, 3.0, 2.0, 4.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 3.0, 0.0],
            [8.0, 4.0, 0.5, 5.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_deployment_cost_for_known_values():
    deployment = sample_deployment()

    result = compute_deployment_cost(deployment)

    expected = (
        400.0
        + (400.0 + 35.0 * (4.0 - 2.0) ** 2 + 20.0 * 3.0)
        + (400.0 + 35.0 * (5.0 - 2.0) ** 2 + 20.0 * 5.0)
    )

    assert np.isclose(result, expected)


def test_inactive_station_has_no_deployment_cost():
    deployment = sample_deployment()

    original_cost = compute_deployment_cost(deployment)

    deployment[2, 5] = 1.0
    modified_cost = compute_deployment_cost(deployment)

    assert modified_cost > original_cost


def test_zero_active_stations_have_zero_deployment_cost():
    deployment = sample_deployment()
    deployment[:, 5] = 0.0

    assert compute_deployment_cost(deployment) == 0.0


def test_deployment_cost_is_nonnegative():
    deployment = sample_deployment()

    assert compute_deployment_cost(deployment) >= 0.0


def test_energy_consumption_for_known_values():
    deployment = sample_deployment()

    result = compute_energy_consumption(deployment)

    expected = 12.0 * (
        (18.0 + 14.0 * 1.0 + 0.5 * (2.0 - 2.0))
        + (18.0 + 14.0 * 2.0 + 0.5 * (4.0 - 2.0))
        + (18.0 + 14.0 * 0.5 + 0.5 * (5.0 - 2.0))
    )

    assert np.isclose(result, expected)


def test_inactive_station_has_no_energy_consumption():
    deployment = sample_deployment()

    original_energy = compute_energy_consumption(deployment)

    deployment[2, 5] = 1.0
    modified_energy = compute_energy_consumption(deployment)

    assert modified_energy > original_energy


def test_zero_active_stations_have_zero_energy_consumption():
    deployment = sample_deployment()
    deployment[:, 5] = 0.0

    assert compute_energy_consumption(deployment) == 0.0


def test_energy_consumption_is_nonnegative():
    deployment = sample_deployment()

    assert compute_energy_consumption(deployment) >= 0.0


@pytest.mark.parametrize(
    "invalid_deployment",
    [
        np.zeros((3, 6)),
        np.zeros((4, 5)),
        np.zeros((4, 7)),
    ],
)
def test_invalid_deployment_shape_raises_error(invalid_deployment):
    with pytest.raises(ValueError):
        compute_deployment_cost(invalid_deployment)

    with pytest.raises(ValueError):
        compute_energy_consumption(invalid_deployment)
