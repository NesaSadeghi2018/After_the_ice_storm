
import numpy as np

from ice_storm.core.signal_model import (
    compute_squared_distances,
    compute_station_gains,
    compute_received_power,
)


def sample_deployment():
    """
    Create a deployment with four stations for testing.
    Columns: x, y, p, h, c, z.
    """
    return np.array(
        [
            [0.0, 0.0, 1.0, 2.0, 1.0, 1.0],
            [3.0, 4.0, 2.0, 4.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 3.0, 0.0],
            [2.0, 2.0, 1.0, 5.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_squared_distances():
    deployment = sample_deployment()

    demand_points = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )

    distances = compute_squared_distances(
        deployment,
        demand_points,
    )

    assert distances.shape == (2, 4)

    np.testing.assert_allclose(
        distances[0],
        [0.0, 25.0, 2.0, 8.0],
    )

    np.testing.assert_allclose(
        distances[1],
        [25.0, 0.0, 13.0, 5.0],
    )


def test_station_gains():
    deployment = sample_deployment()

    gains = compute_station_gains(deployment)

    assert gains.shape == (4,)

    np.testing.assert_allclose(
        gains,
        [1.0, 2.0, 1.5, 2.5],
    )


def test_received_power():
    deployment = sample_deployment()

    demand_points = np.array([[0.0, 0.0]])

    received_power = compute_received_power(
        deployment,
        demand_points,
    )

    assert received_power.shape == (1, 4)

    expected = np.array(
        [[
            1.0,
            2.0 / 13.5,
            0.0,
            1.0 / 4.2,
        ]]
    )

    np.testing.assert_allclose(
        received_power,
        expected,
    )


def test_inactive_station_has_zero_received_power():
    deployment = sample_deployment()

    demand_points = np.array([[0.0, 0.0]])

    received_power = compute_received_power(
        deployment,
        demand_points,
    )

    # Station 3 (index 2) is inactive.
    assert deployment[2, 5] == 0.0
    assert received_power[0, 2] == 0.0


def test_received_power_is_finite_and_nonnegative():
    deployment = sample_deployment()

    received_power = compute_received_power(deployment)

    assert np.all(np.isfinite(received_power))
    assert np.all(received_power >= 0.0)
