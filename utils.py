"""
Utility functions for gpsp_tools.
"""

##########################################################################################################
# Imports
##########################################################################################################
import numpy as np

##########################################################################################################
# Utility functions
##########################################################################################################
# Function to interpolate an array to have the density of points of another array
def interpolate_to_match(x, y, x_new):
    """
    Interpolates the array y defined at points x to the new points x_new.

    Parameters:
    - x: Original array of x values (must be sorted).
    - y: Original array of y values corresponding to x.
    - x_new: New array of x values to interpolate to.

    Returns:
    - y_new: Interpolated array of y values corresponding to x_new.
    """
    return np.interp(x_new, x, y)
