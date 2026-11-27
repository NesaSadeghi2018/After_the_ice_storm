"""Shared objective normalization for multi-objective algorithms."""

from __future__ import annotations

import numpy as np


# Theoretical objective bounds for Assignment C: minimize Q, C, and E.
MAX_DEPLOYMENT_COST = 4 * (
    400 + 35 * (6 - 2) ** 2 + 20 * np.sqrt(125)
)

MAX_ENERGY_CONSUMPTION = 2304.0

OBJECTIVE_LOWER_BOUNDS = np.array([-1.0, 0.0, 0.0], dtype=float)

OBJECTIVE_UPPER_BOUNDS = np.array(
    [0.0, MAX_DEPLOYMENT_COST, MAX_ENERGY_CONSUMPTION],
    dtype=float,
)

OBJECTIVE_RANGES = OBJECTIVE_UPPER_BOUNDS - OBJECTIVE_LOWER_BOUNDS


def normalize_objectives(values: np.ndarray) -> np.ndarray:
    """Normalize Q, C, and E using fixed theoretical objective bounds."""
    values = np.asarray(values, dtype=float)

    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("Expected objective values with a final dimension of 3.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Objective values must be finite.")

    return (values - OBJECTIVE_LOWER_BOUNDS) / OBJECTIVE_RANGES
