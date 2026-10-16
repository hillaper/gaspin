"""
Magnetic resonance properties of gas-phase systems.

This module provides the ensemble averages of distance-dependent pair properties, such as the nuclear shielding
or the spin-spin coupling of an atom pair at distance R. The averages are obtained either from a radial
distribution function (RDF) sampled in a simulation or from the Boltzmann factor of the pair potential through
the second virial coefficient of the property.
"""

# Imports
import numpy as np
import scipy.constants as const


def ensemble_average_from_rdf(r: np.ndarray,
                              p_r: np.ndarray,
                              rdf: np.ndarray,
                              V: float=1.0,
                              cumulative: bool=False,
                              N_other_atoms: int | None=None) -> float | tuple[float, np.ndarray]:
    """
    Calculate the ensemble average of a distance-dependent pair property from the radial distribution function.

    The average of the property summed over all neighbours of a reference atom is

    .. math::

        \\langle A \\rangle = \\frac{N}{V} \\int A(r) g(r) 4 \\pi r^2 \\, dr,

    where N is the number of other atoms (N_other_atoms) and V is the volume. The integral is evaluated with
    the trapezoidal rule. Optionally, the running integral is returned as a function of the upper limit r.

    Usage: ``average = ensemble_average_from_rdf(r, p_r, rdf, V, N_other_atoms=N)``

    Parameters
    ----------
    r : ndarray
        Radial distances.
    p_r : ndarray
        Property values at the distances r.
    rdf : ndarray
        Radial distribution function at the distances r.
    V : float, default=1.0
        Volume of the system in the units of r cubed.
    cumulative : bool, default=False
        If True, also return the running integral as a function of the upper integration limit.
    N_other_atoms : int, default=None
        Number of other atoms that the reference atom interacts with. If given, the result is multiplied by it.
        For an RDF normalised by the number of distinct pairs, as in MDAnalysis with the self-pairs excluded,
        this is the total number of atoms minus one.

    Returns
    -------
    asymptotic : float
        Ensemble average of the property over the full range of r.
    cumulative_average : ndarray
        Running ensemble average as a function of the upper integration limit. Returned only if cumulative is
        True.
    """

    # Integrand of the radial integral
    integrand = rdf * p_r * 4 * np.pi * r**2

    # Integral over the full range normalised by the volume
    asymptotic = np.trapezoid(integrand, r) / V

    # Scale by the number of other atoms if it is given
    if N_other_atoms is not None:
        asymptotic *= N_other_atoms

    # Return only the full-range average unless the running integral is requested
    if not cumulative:
        return asymptotic

    # Running integral with the trapezoidal rule
    dr = np.diff(r)
    cumulative_integral = np.zeros_like(r)
    cumulative_integral[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)

    # Normalise the running integral by the volume
    cumulative_average = cumulative_integral / V

    # Scale the running integral by the number of other atoms if it is given
    if N_other_atoms is not None:
        cumulative_average *= N_other_atoms

    return asymptotic, cumulative_average


def coordination_number_from_rdf(r: np.ndarray,
                                 rdf: np.ndarray,
                                 V: float=1.0,
                                 cumulative: bool=False,
                                 N_other_atoms: int | None=None) -> float | tuple[float, np.ndarray]:
    """
    Calculate the coordination number by integrating the radial distribution function.

    The coordination number is the ensemble average of a property that equals unity at all distances
    (see ensemble_average_from_rdf()).

    Usage: ``coordination_number = coordination_number_from_rdf(r, rdf, V, N_other_atoms=N)``

    Parameters
    ----------
    r : ndarray
        Radial distances.
    rdf : ndarray
        Radial distribution function at the distances r.
    V : float, default=1.0
        Volume of the system in the units of r cubed.
    cumulative : bool, default=False
        If True, also return the running coordination number as a function of the upper integration limit.
    N_other_atoms : int, default=None
        Number of other atoms that the reference atom interacts with. If given, the result is multiplied by it.

    Returns
    -------
    asymptotic : float
        Coordination number over the full range of r.
    cumulative_average : ndarray
        Running coordination number as a function of the upper integration limit. Returned only if cumulative
        is True.
    """

    # Coordination number as the average of a unit property
    p_r = np.ones_like(r)
    return ensemble_average_from_rdf(r,
                                     p_r,
                                     rdf,
                                     V,
                                     cumulative=cumulative,
                                     N_other_atoms=N_other_atoms)


def first_order_virial_coefficient(r: np.ndarray,
                                   p_r: np.ndarray,
                                   V_r: np.ndarray,
                                   T: float) -> float:
    """
    Calculate the second virial coefficient of a distance-dependent pair property.

    The coefficient is the first-order term of the density expansion of the property,

    .. math::

        B(T) = \\int p(r) \\exp\\left[-\\frac{V(r)}{k_\\mathrm{B} T}\\right] 4 \\pi r^2 \\, dr,

    and the average of the property summed over all neighbours of an atom at number density n is n B(T). The
    integral is evaluated with the trapezoidal rule.

    Usage: ``B_T = first_order_virial_coefficient(r, p_r, V_r, T)``

    Parameters
    ----------
    r : ndarray
        Radial distances.
    p_r : ndarray
        Property values at the distances r.
    V_r : ndarray
        Pair potential energy at the distances r in J.
    T : float
        Temperature in K.

    Returns
    -------
    B_T : float
        Second virial coefficient of the property, in the units of the property times the units of r cubed.
    """

    # Inverse thermal energy
    beta = 1 / (const.k * T)

    # Integrand weighted by the Boltzmann factor of the pair potential
    integrand = p_r * np.exp(-beta * V_r) * 4 * np.pi * r**2

    # Radial integral with the trapezoidal rule
    B_T = np.trapezoid(integrand, r)

    return B_T
