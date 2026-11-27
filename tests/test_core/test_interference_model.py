
import numpy as np
import pytest

from ice_storm.core.interference_model import (
    compute_noise_levels,
    compute_sinr,
)
from ice_storm.core.scenarios import SCENARIOS


def sample_deployment():
    return np.array(
        [
            [0.0, 0.0, 1.0, 2.0, 1.0, 1.0],
            [3.0, 4.0, 2.0, 4.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 3.0, 3.0, 0.0],
            [2.0, 2.0, 1.0, 5.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_initial_response_noise_is_constant():
    noise = compute_noise_levels(
        SCENARIOS["initial_response"]
    )

    np.testing.assert_allclose(
        noise,
        np.full(18, 0.04),
    )


def test_noise_is_highest_at_interference_center():
    scenario = {
        "nu_0": 0.04,
        "rho": 3.0,
        "a": 5.0,
        "b": 5.0,
    }

    coordinates = np.array(
        [
            [5.0, 5.0],
            [0.0, 0.0],
        ]
    )

    noise = compute_noise_levels(
        scenario,
        coordinates,
    )

    assert noise[0] > noise[1]


def test_noise_increases_with_rho():
    coordinates = np.array([[5.0, 5.0]])

    low_rho = {
        "nu_0": 0.04,
        "rho": 1.0,
        "a": 5.0,
        "b": 5.0,
    }

    high_rho = {
        "nu_0": 0.04,
        "rho": 3.0,
        "a": 5.0,
        "b": 5.0,
    }

    noise_low = compute_noise_levels(low_rho, coordinates)
    noise_high = compute_noise_levels(high_rho, coordinates)

    assert noise_high[0] > noise_low[0]


def test_sinr_shape_matches_demand_points_and_stations():
    sinr = compute_sinr(
        sample_deployment(),
        SCENARIOS["initial_response"],
    )

    assert sinr.shape == (18, 4)


def test_different_channels_do_not_interfere():
    deployment = sample_deployment()

    # Four stations, three of which are active.
    # Columns: x, y, p, h, c, z.

    deployment[2, 5] = 1.0
    deployment[3, 5] = 0.0

    sinr = compute_sinr(
        deployment,
        SCENARIOS["initial_response"],
    )

    from ice_storm.core.signal_model import compute_received_power

    received_power = compute_received_power(deployment)

    expected_sinr = received_power[0] / 0.04

    np.testing.assert_allclose(
        sinr[0],
        expected_sinr,
    )


def test_same_channel_creates_interference():
    deployment = sample_deployment()

    deployment[:, 4] = 1.0

    sinr = compute_sinr(
        deployment,
        SCENARIOS["initial_response"],
    )

    from ice_storm.core.signal_model import compute_received_power

    received_power = compute_received_power(deployment)

    expected = received_power[0, 0] / (
        0.04 + received_power[0, 1] + received_power[0, 3]
    )

    assert np.isclose(sinr[0, 0], expected)


def test_inactive_station_has_zero_sinr():
    deployment = sample_deployment()

    sinr = compute_sinr(
        deployment,
        SCENARIOS["initial_response"],
    )

    np.testing.assert_array_equal(
        sinr[:, 2],
        np.zeros(18),
    )


def test_invalid_deployment_shape_raises_error():
    with pytest.raises(ValueError):
        compute_sinr(
            np.zeros((3, 6)),
            SCENARIOS["initial_response"],
        )


def test_sinr_values_are_finite_and_nonnegative():
    sinr = compute_sinr(
        sample_deployment(),
        SCENARIOS["northern_outage"],
    )

    assert np.all(np.isfinite(sinr))
    assert np.all(sinr >= 0.0)
