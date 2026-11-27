
import numpy as np

from ice_storm.core.data import N_STATIONS
from ice_storm.core.scenarios import SCENARIOS
from ice_storm.core.deployment import (
    POSITION_MIN,
    POSITION_MAX,
    POWER_MIN,
    POWER_MAX,
    HEIGHT_VALUES,
    CHANNEL_VALUES,
    ACTIVE_VALUES,
    build_deployment_b,
)
from ice_storm.core.evaluation import evaluate_deployment


# ============================================================
# Assignment B: Decision variables and domains
# ============================================================

VARIABLES_PER_STATION = 6
N_VARIABLES = N_STATIONS * VARIABLES_PER_STATION

SCENARIO_NAMES = tuple(SCENARIOS.keys())

LOWER_BOUNDS = np.tile(
    [
        POSITION_MIN,
        POSITION_MIN,
        POWER_MIN,
        min(HEIGHT_VALUES),
        min(CHANNEL_VALUES),
        min(ACTIVE_VALUES),
    ],
    N_STATIONS,
).astype(float)

UPPER_BOUNDS = np.tile(
    [
        POSITION_MAX,
        POSITION_MAX,
        POWER_MAX,
        max(HEIGHT_VALUES),
        max(CHANNEL_VALUES),
        max(ACTIVE_VALUES),
    ],
    N_STATIONS,
).astype(float)

BOUNDS = np.column_stack(
    (LOWER_BOUNDS, UPPER_BOUNDS)
)


# Variable indices by type
CONTINUOUS_INDICES = np.array(
    [
        i
        for station in range(N_STATIONS)
        for i in (
            station * 6,
            station * 6 + 1,
            station * 6 + 2,
        )
    ],
    dtype=int,
)

HEIGHT_INDICES = np.arange(3, N_VARIABLES, 6)
CHANNEL_INDICES = np.arange(4, N_VARIABLES, 6)
ACTIVE_INDICES = np.arange(5, N_VARIABLES, 6)

INTEGER_DOMAINS = {
    "height": np.asarray(HEIGHT_VALUES, dtype=int),
    "channel": np.asarray(CHANNEL_VALUES, dtype=int),
    "active": np.asarray(ACTIVE_VALUES, dtype=int),
}


# ============================================================
# Assignment B problem definition
# ============================================================

class AssignmentBProblem:
    """
    Assignment B: Equipment choices.

    Optimize all six deployment settings for four stations.

    Decision vector:
        [x1, y1, p1, h1, c1, z1, ..., x4, y4, p4, h4, c4, z4]

    Variable types:
        x, y, p: continuous
        h: integer categorical choice
        c: integer categorical choice
        z: binary

    Objective:
        Minimize negative weighted service quality Q(X).
    """

    def __init__(self, scenario_name):

        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. "
                f"Available scenarios: {SCENARIO_NAMES}"
            )

        self.scenario_name = scenario_name
        self.dimension = N_VARIABLES

        self.lower_bounds = LOWER_BOUNDS.copy()
        self.upper_bounds = UPPER_BOUNDS.copy()
        self.bounds = BOUNDS.copy()

        self.continuous_indices = CONTINUOUS_INDICES.copy()
        self.height_indices = HEIGHT_INDICES.copy()
        self.channel_indices = CHANNEL_INDICES.copy()
        self.active_indices = ACTIVE_INDICES.copy()
        self.integer_domains = {
        name: values.copy()
        for name, values in INTEGER_DOMAINS.items()
        }

    def validate_solution(self, x):
        """
        Validate the 24-dimensional mixed-variable decision vector.

        Discrete variables must belong to their exact domains.
        No automatic rounding is performed.
        """

        x = np.asarray(x, dtype=float)

        if x.shape != (self.dimension,):
            raise ValueError(
                f"Decision vector must have shape "
                f"({self.dimension},)."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                "Decision variables must be finite."
            )

        if np.any(x < self.lower_bounds) or np.any(
            x > self.upper_bounds
        ):
            raise ValueError(
                "Decision vector violates variable bounds."
            )

        if not np.all(
            np.isin(x[self.height_indices], HEIGHT_VALUES)
        ):
            raise ValueError(
                "Heights must be integers from 2 to 6."
            )

        if not np.all(
            np.isin(x[self.channel_indices], CHANNEL_VALUES)
        ):
            raise ValueError(
                "Channels must be integers from 1 to 3."
            )

        if not np.all(
            np.isin(x[self.active_indices], ACTIVE_VALUES)
        ):
            raise ValueError(
                "Activity variables must be 0 or 1."
            )

        return True

    def build_deployment(self, x):
        """
        Convert the decision vector into a standard deployment.
        """

        self.validate_solution(x)

        return build_deployment_b(x)

    def evaluate(self, x):
        """
        Evaluate one deployment under the selected scenario.

        Returns:
            Dictionary containing:
                Q: Objective value
                C: Deployment cost
                E: 12-hour energy consumption
        """

        deployment = self.build_deployment(x)

        return evaluate_deployment(
            deployment,
            self.scenario_name,
        )

    def objective(self, x):
        """
        Return scalar objective value for minimization.
        """

        results = self.evaluate(x)

        return results["Q"]

    def sample_solution(self, rng=None):
        """
        Generate one random feasible mixed-variable solution.
        """

        if rng is None:
            rng = np.random.default_rng()

        x = np.empty(self.dimension, dtype=float)

        x[self.continuous_indices] = rng.uniform(
            self.lower_bounds[self.continuous_indices],
            self.upper_bounds[self.continuous_indices],
        )

        x[self.height_indices] = rng.choice(HEIGHT_VALUES, size=N_STATIONS)
        x[self.channel_indices] = rng.choice(CHANNEL_VALUES, size=N_STATIONS)
        x[self.active_indices] = rng.choice(ACTIVE_VALUES, size=N_STATIONS)

        self.validate_solution(x)

        return x
