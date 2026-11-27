
import numpy as np

from ice_storm.assignments.assignment_b.problem import (
    AssignmentBProblem,
)


class AssignmentCProblem(AssignmentBProblem):
    """
    Assignment C: Mixed-variable multi-objective optimization.

    Decision variables:
        24 mixed variables describing four radio stations.

    Objectives:
        Minimize:
            Q(X): Negative weighted service quality
            C(X): Deployment cost
            E(X): 12-hour energy consumption

    All three objectives are evaluated together for each
    candidate deployment. One call to objectives() therefore
    represents one deployment evaluation.
    """

    n_objectives = 3

    objective_names = ("Q", "C", "E")

    def objectives(self, x):
        """
        Return the three objective values as a NumPy array.

        Returns:
            np.ndarray:
                [Q(X), C(X), E(X)]
        """

        results = self.evaluate(x)

        values = np.array(
            [
                results["Q"],
                results["C"],
                results["E"],
            ],
            dtype=float,
        )

        if values.shape != (self.n_objectives,):
            raise RuntimeError(
                "Objective vector has an incorrect shape."
            )

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Objective values must be finite."
            )

        return values
