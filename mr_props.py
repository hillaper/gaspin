"""
Magnetic resonance properties of gas phase systems.
"""

############################################################################################################
# Imports
############################################################################################################
import numpy as np
import scipy.constants as const

############################################################################################################
# Averages
############################################################################################################
def ensemble_average_from_rdf(r, p_r, rdf, V=1.0, cumulative=False):
    """
    Calculates the ensemble average of the weighted property values, normalized by the
    system volume. Optionally returns the running cumulative integral.

    Parameters:
    - r: Array of radial distances corresponding to the property values.
    - p_r: Property values as a function of distance (same length as r).
    - rdf: Radial distribution function values at the corresponding distances (same length as r).
    - V: Volume of the system (default is 1.0 for normalization).
    - cumulative: If True, also return the cumulative integral as a function of distance.

    Returns:
    - Asymptotic ensemble average value.
    - If cumulative is True, also returns the running ensemble average array.
    """
    # Build the integrand for the radial integral
    integrand = rdf * p_r * 4 * np.pi * r**2

    # Compute the total (asymptotic) average normalized by volume
    asymptotic = np.trapz(integrand, r) / V

    if not cumulative:
        return asymptotic

    # Compute cumulative integral using the trapezoidal rule
    dr = np.diff(r)
    cumulative_integral = np.zeros_like(r)
    cumulative_integral[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)
    
    # Normalize cumulative integral by volume
    cumulative_average = cumulative_integral / V

    return asymptotic, cumulative_average

def coordination_number_from_rdf(r, rdf, V=1.0, cumulative=False):
    """
    Calculates the coordination number by integrating the RDF.

    Parameters:
    - r: Array of radial distances corresponding to the RDF values.
    - rdf: Radial distribution function values at the corresponding distances (same length as r).
    - V: Volume of the system (default is 1.0 for normalization).
    - cumulative: If True, also return the cumulative coordination number.

    Returns:
    - Asymptotic coordination number.
    - If cumulative is True, also returns the running coordination number array.
    """
    # Coordination number is the integral of the RDF (property is unity)
    p_r = np.ones_like(r)
    return ensemble_average_from_rdf(r, p_r, rdf, V, cumulative=cumulative)

def first_order_virial_coefficient(r, p_r, V_r, T):
    """
    Calculates the first order virial coefficient for a property p(R) using the formula:
    B(T) = ∫ [p(R) * exp(-V_r(R)/kT)] 4 * π * R^2 dR

    Parameters:
    - r: Array of radial distances corresponding to the property values.
    - p_r: Property values as a function of distance (same length as r).
    - V_r: Potential energy as a function of distance (same length as r).
    - T: Temperature in Kelvin.

    Returns:
    - B_T: First order virial coefficient B(T).
    """
    # Inverse temperature
    beta = 1 / (const.k * T)

    # Calculate the integrand
    integrand = p_r * np.exp(-beta * V_r) * 4 * np.pi * r**2

    # Perform numerical integration using the trapezoidal rule
    B_T = np.trapz(integrand, r)

    return B_T
