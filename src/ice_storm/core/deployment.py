
import numpy as np

from ice_storm.core.data import N_STATIONS


# Variable limits from Table 5
POSITION_MIN = 0.0
POSITION_MAX = 10.0

POWER_MIN = 0.2
POWER_MAX = 2.0

HEIGHT_VALUES = (2, 3, 4, 5, 6)
CHANNEL_VALUES = (1, 2, 3)
ACTIVE_VALUES = (0, 1)


def build_deployment_a(x):
    """
    Convert Assignment A's 12 continuous variables
    into a standard 4 x 6 deployment matrix.

    Variable order per station: x, y, p.
    Fixed values: h=4, c=1, z=1.
    """
    x = np.asarray(x, dtype=float)

    if x.shape != (N_STATIONS * 3,):
        raise ValueError("Assignment A requires exactly 12 variables.")

    deployment = np.zeros((N_STATIONS, 6), dtype=float)

    deployment[:, 0:3] = x.reshape(N_STATIONS, 3)
    deployment[:, 3] = 4
    deployment[:, 4] = 1
    deployment[:, 5] = 1

    validate_deployment(deployment)

    return deployment


def build_deployment_b(x):
    """
    Convert Assignment B's 24 variables
    into a standard 4 x 6 deployment matrix.

    Variable order per station: x, y, p, h, c, z.
    """
    x = np.asarray(x, dtype=float)

    if x.shape != (N_STATIONS * 6,):
        raise ValueError("Assignment B requires exactly 24 variables.")

    deployment = x.reshape(N_STATIONS, 6)

    validate_deployment(deployment)

    return deployment


def validate_deployment(deployment):
    """
    Check whether a deployment satisfies all variable domains.
    """
    deployment = np.asarray(deployment, dtype=float)

    if deployment.shape != (N_STATIONS, 6):
        raise ValueError("Deployment must have shape (4, 6).")

    if not np.all(np.isfinite(deployment)):
        raise ValueError("All deployment variables must be finite.")

    x = deployment[:, 0]
    y = deployment[:, 1]
    p = deployment[:, 2]
    h = deployment[:, 3]
    c = deployment[:, 4]
    z = deployment[:, 5]

    if not np.all((POSITION_MIN <= x) & (x <= POSITION_MAX)):
        raise ValueError("Station x-coordinates must be between 0 and 10.")

    if not np.all((POSITION_MIN <= y) & (y <= POSITION_MAX)):
        raise ValueError("Station y-coordinates must be between 0 and 10.")

    if not np.all((POWER_MIN <= p) & (p <= POWER_MAX)):
        raise ValueError("Transmit power must be between 0.2 and 2.")

    if not np.all(np.isin(h, HEIGHT_VALUES)):
        raise ValueError("Station heights must be integers from 2 to 6.")

    if not np.all(np.isin(c, CHANNEL_VALUES)):
        raise ValueError("Channel values must be 1, 2, or 3.")

    if not np.all(np.isin(z, ACTIVE_VALUES)):
        raise ValueError("Station activity values must be 0 or 1.")

    return True
