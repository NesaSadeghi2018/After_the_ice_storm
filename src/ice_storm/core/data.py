
import numpy as np


# ============================================================
# General problem constants
# ============================================================

N_STATIONS = 4
N_DEMAND_POINTS = 18

AREA_BOUNDS = (0.0, 10.0)

DEPOT_COORDINATES = np.array(
    [5.0, 0.0],
    dtype=float
)


# ============================================================
# Demand point data
# Source: Table 2 of "After the Ice Storm"
# ============================================================

DEMAND_COORDINATES = np.array(
    [
        [0.8, 1.2],
        [2.0, 1.8],
        [4.6, 1.0],
        [7.0, 1.5],
        [9.0, 0.8],
        [1.0, 4.2],
        [2.8, 3.4],
        [5.0, 4.5],
        [7.4, 3.6],
        [9.2, 4.8],
        [0.7, 7.2],
        [3.2, 6.4],
        [5.6, 7.1],
        [8.5, 6.5],
        [1.8, 9.0],
        [4.0, 8.8],
        [7.0, 9.3],
        [9.3, 8.8],
    ],
    dtype=float
)


DEMAND_TYPES = (
    "Residential",
    "Shelter",
    "Depot",
    "Residential",
    "Clinic",
    "Clinic",
    "Residential",
    "Shelter",
    "Depot",
    "Residential",
    "Residential",
    "Shelter",
    "Clinic",
    "Residential",
    "Depot",
    "Residential",
    "Shelter",
    "Residential",
)


BASE_WEIGHTS = np.array(
    [
        4, 8, 5, 4, 10, 10,
        4, 8, 5, 4, 4, 8,
        10, 4, 5, 4, 8, 4
    ],
    dtype=float
)


# ============================================================
# Basic data validation
# ============================================================

assert DEMAND_COORDINATES.shape == (N_DEMAND_POINTS, 2)
assert len(DEMAND_TYPES) == N_DEMAND_POINTS
assert BASE_WEIGHTS.shape == (N_DEMAND_POINTS,)

assert np.all(DEMAND_COORDINATES >= 0.0)
assert np.all(DEMAND_COORDINATES <= 10.0)
assert np.all(BASE_WEIGHTS > 0.0)
