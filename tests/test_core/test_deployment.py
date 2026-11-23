
import numpy as np
import pytest

from ice_storm.core.deployment import (
    build_deployment_a,
    build_deployment_b,
    validate_deployment,
)


def valid_deployment():
    return np.array(
        [
            [0.0, 0.0, 0.2, 2.0, 1.0, 0.0],
            [10.0, 10.0, 2.0, 6.0, 3.0, 1.0],
            [5.0, 5.0, 1.0, 4.0, 2.0, 1.0],
            [3.0, 7.0, 1.5, 3.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def test_build_deployment_a_shape_and_fixed_values():
    x = np.array(
        [
            1.0, 2.0, 0.5,
            3.0, 4.0, 1.0,
            5.0, 6.0, 1.5,
            7.0, 8.0, 2.0,
        ]
    )

    deployment = build_deployment_a(x)

    assert deployment.shape == (4, 6)

    np.testing.assert_array_equal(deployment[:, 3], 4.0)
    np.testing.assert_array_equal(deployment[:, 4], 1.0)
    np.testing.assert_array_equal(deployment[:, 5], 1.0)

    np.testing.assert_array_equal(
        deployment[:, 0:3],
        x.reshape(4, 3),
    )


def test_build_deployment_a_rejects_wrong_number_of_variables():
    with pytest.raises(ValueError):
        build_deployment_a(np.zeros(11))


def test_build_deployment_a_rejects_out_of_range_power():
    x = np.tile([5.0, 5.0, 1.0], 4)
    x[2] = 2.1

    with pytest.raises(ValueError):
        build_deployment_a(x)


def test_build_deployment_b_shape_and_values():
    x = valid_deployment().flatten()

    deployment = build_deployment_b(x)

    assert deployment.shape == (4, 6)

    np.testing.assert_array_equal(
        deployment,
        valid_deployment(),
    )


def test_build_deployment_b_rejects_wrong_number_of_variables():
    with pytest.raises(ValueError):
        build_deployment_b(np.zeros(23))


def test_validate_deployment_accepts_valid_boundaries():
    deployment = valid_deployment()

    assert validate_deployment(deployment) is True


def test_validate_deployment_rejects_wrong_shape():
    with pytest.raises(ValueError):
        validate_deployment(np.zeros((3, 6)))


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_validate_deployment_rejects_non_finite_values(invalid_value):
    deployment = valid_deployment()
    deployment[0, 0] = invalid_value

    with pytest.raises(ValueError):
        validate_deployment(deployment)


@pytest.mark.parametrize(
    "column, invalid_value",
    [
        (0, -0.1),
        (0, 10.1),
        (1, -0.1),
        (1, 10.1),
        (2, 0.19),
        (2, 2.01),
        (3, 1.0),
        (3, 4.5),
        (3, 7.0),
        (4, 0.0),
        (4, 4.0),
        (5, -1.0),
        (5, 2.0),
    ],
)
def test_validate_deployment_rejects_values_outside_domains(
    column,
    invalid_value,
):
    deployment = valid_deployment()
    deployment[0, column] = invalid_value

    with pytest.raises(ValueError):
        validate_deployment(deployment)
