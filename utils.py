"""
Utility functions for gpsp_tools.
"""

# Imports
import numpy as np


def interpolate_to_match(x: np.ndarray,
                         y: np.ndarray,
                         x_new: np.ndarray) -> np.ndarray:
    """
    Interpolate an array linearly onto a new set of points.

    This is used to bring data defined on different grids, such as a radial distribution function and a
    property curve, onto a common grid.

    Usage: ``y_new = interpolate_to_match(x, y, x_new)``

    Parameters
    ----------
    x : ndarray
        Original points in ascending order.
    y : ndarray
        Values at the original points.
    x_new : ndarray
        New points.

    Returns
    -------
    y_new : ndarray
        Linearly interpolated values at the new points.
    """

    # Linear interpolation onto the new points
    return np.interp(x_new, x, y)
