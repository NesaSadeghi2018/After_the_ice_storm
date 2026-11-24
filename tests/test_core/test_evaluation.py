
import numpy as np

from ice_storm.core.data import DEMAND_COORDINATES
from ice_storm.core.scenarios import SCENARIOS
from ice_storm.core.evaluation import (
    compute_background_noise,
    compute_sinr,
    evaluate_deployment,
)


def sample_deployment():
    """
    Four active stations with different channels.
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


def test_initial_response_noise_is_constant():
    scenario = SCENARIOS["initial_response"]

    noise = compute_background_noise(
        scenario,
        DEMAND_COORDINATES,
    )

    assert noise.shape == (18,)

    np.testing.assert_allclose(
        noise,
        np.full(18, 0.04),
    )


def test_evening_pressure_noise_is_not_below_baseline():
    scenario = SCENARIOS["evening_pressure"]

    noise = compute_background_noise(
        scenario,
        DEMAND_COORDINATES,
    )

    assert noise.shape == (18,)
    assert np.all(noise >= scenario["nu_0"])


def test_northern_outage_noise_is_not_below_baseline():
    scenario = SCENARIOS["northern_outage"]

    noise = compute_background_noise(
        scenario,
        DEMAND_COORDINATES,
    )

    assert noise.shape == (18,)
    assert np.all(noise >= scenario["nu_0"])



def test_different_channels_do_not_interfere():
    deployment = sample_deployment()

    # Keep the first three stations active on distinct channels.
    # Turn off the fourth station to avoid channel reuse.
    deployment[3, 5] = 0.0

    sinr = compute_sinr(
        deployment,
        "initial_response",
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

    # Put all stations on channel 1.
    deployment[:, 4] = 1.0

    sinr = compute_sinr(
        deployment,
        "initial_response",
    )

    from ice_storm.core.signal_model import compute_received_power

    received_power = compute_received_power(deployment)

    # The first station receives interference from
    # the second and fourth active stations.
    expected = received_power[0, 0] / (
        0.04 + received_power[0, 1] + received_power[0, 3]
    )

    assert np.isclose(sinr[0, 0], expected)


def test_inactive_station_has_zero_sinr():
    deployment = sample_deployment()

    sinr = compute_sinr(
        deployment,
        "initial_response",
    )

    # Third station is inactive.
    assert deployment[2, 5] == 0.0
    assert sinr[:, 2].max() == 0.0


def test_sinr_is_finite_and_nonnegative():
    deployment = sample_deployment()

    sinr = compute_sinr(
        deployment,
        "northern_outage",
    )

    assert sinr.shape == (18, 4)
    assert np.all(np.isfinite(sinr))
    assert np.all(sinr >= 0.0)


def test_full_evaluation_returns_three_objectives():
    deployment = sample_deployment()

    result = evaluate_deployment(
        deployment,
        "initial_response",
    )

    assert set(result.keys()) == {"Q", "C", "E"}

    assert np.isfinite(result["Q"])
    assert np.isfinite(result["C"])
    assert np.isfinite(result["E"])

    assert -109.0 / 18.0 <= result["Q"] <= 0.0
    assert result["C"] >= 0.0
    assert result["E"] >= 0.0
