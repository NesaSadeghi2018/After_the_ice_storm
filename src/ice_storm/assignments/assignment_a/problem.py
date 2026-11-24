
import numpy as np

from ice_storm.core.data import N_STATIONS
from ice_storm.core.scenarios import SCENARIOS
from ice_storm.core.deployment import build_deployment_a
from ice_storm.core.evaluation import evaluate_deployment


# ============================================================
# Assignment A: Decision variables and bounds
# ============================================================

N_VARIABLES = N_STATIONS * 3

POSITION_MIN = 0.0
POSITION_MAX = 10.0

POWER_MIN = 0.2
POWER_MAX = 2.0

SCENARIO_NAMES = tuple(SCENARIOS.keys())


LOWER_BOUNDS = np.tile(
    [POSITION_MIN, POSITION_MIN, POWER_MIN],
    N_STATIONS
)

UPPER_BOUNDS = np.tile(
    [POSITION_MAX, POSITION_MAX, POWER_MAX],
    N_STATIONS
)

BOUNDS = np.column_stack(
    (LOWER_BOUNDS, UPPER_BOUNDS)
)


# ============================================================
# Assignment A problem definition
# ============================================================

class AssignmentAProblem:
    """
    Assignment A: The first shift.

    Optimize station positions and transmission powers.

    Decision vector:
        [x1, y1, p1, x2, y2, p2, ..., x4, y4, p4]

    Fixed deployment parameters:
        h_i = 4
        c_i = 1
        z_i = 1

    Objective:
        Minimize Q(X), the negative weighted service quality.
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

    def validate_solution(self, x):
        """
        Validate the 12-dimensional decision vector.
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

        return True

    def build_deployment(self, x):
        """
        Convert the decision vector into a standard deployment.
        """

        self.validate_solution(x)

        return build_deployment_a(x)

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
            self.scenario_name
        )

    def objective(self, x):
        """
        Return the scalar objective value for optimization.

        The optimization algorithms minimize this value.
        """

        results = self.evaluate(x)

        return results["Q"]

    def sample_solution(self, rng=None):
        """
        Generate one random feasible solution.
        """

        if rng is None:
            rng = np.random.default_rng()

        return rng.uniform(
            self.lower_bounds,
            self.upper_bounds
        )
