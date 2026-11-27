import numpy as np
import pytest

from ice_storm.optimization.algorithms.multiobjective.objective_scaling import (
    MAX_DEPLOYMENT_COST,
    MAX_ENERGY_CONSUMPTION,
    normalize_objectives,
)


def test_normalize_objectives_maps_theoretical_bounds_to_zero_and_one():
    values = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, MAX_DEPLOYMENT_COST, MAX_ENERGY_CONSUMPTION],
        ]
    )

    normalized = normalize_objectives(values)

    np.testing.assert_allclose(
        normalized,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_normalize_objectives_handles_single_solution():
    values = np.array([-0.5, MAX_DEPLOYMENT_COST / 2, MAX_ENERGY_CONSUMPTION / 2])

    normalized = normalize_objectives(values)

    assert normalized.shape == (3,)
    np.testing.assert_allclose(normalized, [0.5, 0.5, 0.5])


def test_normalize_objectives_handles_population_matrix():
    values = np.array(
        [
            [-0.75, 1000.0, 500.0],
            [-0.25, 2000.0, 1500.0],
        ]
    )

    normalized = normalize_objectives(values)

    assert normalized.shape == values.shape
    assert np.all(np.isfinite(normalized))


@pytest.mark.parametrize(
    "values",
    [
        np.array(1.0),
        np.array([1.0, 2.0]),
        np.zeros((2, 4)),
    ],
)
def test_normalize_objectives_rejects_invalid_shapes(values):
    with pytest.raises(ValueError):
        normalize_objectives(values)


@pytest.mark.parametrize(
    "values",
    [
        np.array([np.nan, 0.0, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        np.array([[0.0, 0.0, -np.inf]]),
    ],
)
def test_normalize_objectives_rejects_non_finite_values(values):
    with pytest.raises(ValueError):
        normalize_objectives(values)
