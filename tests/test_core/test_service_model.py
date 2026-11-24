
import numpy as np
import pytest

from ice_storm.core.data import BASE_WEIGHTS, DEMAND_TYPES
from ice_storm.core.scenarios import SCENARIOS
from ice_storm.core.service_model import (
    compute_service_quality,
    compute_priority_weights,
    compute_objective_q,
)


def test_service_quality_uses_best_station():
    sinr = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 2.0, 1.0, 0.0],
        ]
    )

    quality = compute_service_quality(sinr)

    expected = np.array([0.5, 0.8])

    np.testing.assert_allclose(quality, expected)


def test_service_quality_is_between_zero_and_one():
    sinr = np.array(
        [
            [0.0, 0.5, 1.0, 2.0],
            [0.2, 0.4, 0.6, 0.8],
        ]
    )

    quality = compute_service_quality(sinr)

    assert np.all(quality >= 0.0)
    assert np.all(quality < 1.0)


def test_zero_sinr_gives_zero_service_quality():
    sinr = np.zeros((18, 4))

    quality = compute_service_quality(sinr)

    np.testing.assert_array_equal(
        quality,
        np.zeros(18),
    )


def test_initial_response_weights_equal_base_weights():
    weights = compute_priority_weights("initial_response")

    np.testing.assert_array_equal(
        weights,
        BASE_WEIGHTS,
    )


def test_evening_pressure_priority_weights():
    weights = compute_priority_weights("evening_pressure")

    expected = np.array(
        [
            BASE_WEIGHTS[i]
            * SCENARIOS["evening_pressure"]["priority_multipliers"][
                DEMAND_TYPES[i]
            ]
            for i in range(len(DEMAND_TYPES))
        ]
    )

    np.testing.assert_array_equal(weights, expected)


def test_northern_outage_priority_weights():
    weights = compute_priority_weights("northern_outage")

    expected = np.array(
        [
            BASE_WEIGHTS[i]
            * SCENARIOS["northern_outage"]["priority_multipliers"][
                DEMAND_TYPES[i]
            ]
            for i in range(len(DEMAND_TYPES))
        ]
    )

    np.testing.assert_array_equal(weights, expected)


def test_unknown_scenario_raises_error():
    with pytest.raises(ValueError):
        compute_priority_weights("unknown_scenario")


def test_objective_q_is_negative_weighted_service_quality():
    sinr = np.zeros((18, 4))

    sinr[0, 0] = 1.0
    sinr[1, 0] = 2.0

    result = compute_objective_q(
        sinr,
        "initial_response",
    )

    expected = -(4.0 * 0.5 + 8.0 * 0.8) / 18.0

    assert np.isclose(result, expected)


def test_objective_q_is_zero_when_all_sinr_values_are_zero():
    sinr = np.zeros((18, 4))

    result = compute_objective_q(
        sinr,
        "initial_response",
    )

    assert result == 0.0


def test_invalid_sinr_dimensions_raise_error():
    sinr = np.zeros(18)

    with pytest.raises(ValueError):
        compute_service_quality(sinr)


def test_negative_sinr_raises_error():
    sinr = np.zeros((18, 4))
    sinr[0, 0] = -1.0

    with pytest.raises(ValueError):
        compute_service_quality(sinr)
