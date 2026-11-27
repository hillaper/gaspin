"""
Trajectory helpers: centring and wrapping, the radial distribution function, and the angular momentum and
connecting vector of two groups of atoms.
"""

# Imports
import MDAnalysis as mda
import MDAnalysis.transformations as mda_trans
import numpy as np
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.coordinates.timestep import Timestep


def center_and_wrap(u: mda.Universe,
                    ts: Timestep,
                    group: mda.AtomGroup) -> None:
    """
    Centre the selected group in the simulation box and wrap all atoms back into the box.

    The timestep is modified in place. The centring is done before the wrapping, so that the atoms around the
    selected group are not split across the periodic boundaries.

    Usage: ``center_and_wrap(u, u.trajectory.ts, group)``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory and topology.
    ts : Timestep
        Current timestep of the trajectory, modified in place.
    group : AtomGroup
        Group of atoms placed at the centre of the box.
    """

    # Move the centre of mass of the selected group to the centre of the box
    mda_trans.center_in_box(group)(ts)

    # Wrap the coordinates of all atoms back into the box
    mda_trans.wrap(u.atoms)(ts)


def rdf(group1: mda.AtomGroup,
        group2: mda.AtomGroup,
        nbins: int=100,
        range: tuple[float, float]=(0.0, 10.0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the radial distribution function (RDF) between two groups of atoms over the trajectory.

    The calculation uses MDAnalysis InterRDF. When the two groups are the same, the self-pairs are excluded, and
    the RDF is normalised by the number of distinct pairs N(N - 1).

    Usage: ``r, g_r = rdf(group1, group2)``

    Parameters
    ----------
    group1 : AtomGroup
        First group of atoms.
    group2 : AtomGroup
        Second group of atoms.
    nbins : int, default=100
        Number of bins of the RDF histogram.
    range : tuple of float, default=(0.0, 10.0)
        Range of distances of the RDF histogram in Å.

    Returns
    -------
    r : ndarray
        Centres of the distance bins in Å.
    g_r : ndarray
        Radial distribution function in each bin.
    """

    # Exclude the self-pairs when the two groups are the same
    if group1 == group2:
        rdf_analysis = InterRDF(group1,
                                group2,
                                exclusion_block=(1, 1),
                                range=range,
                                nbins=nbins)
    else:
        rdf_analysis = InterRDF(group1,
                                group2,
                                range=range,
                                nbins=nbins)

    # Accumulate the distance histogram over the trajectory
    rdf_analysis.run()

    return rdf_analysis.results.bins, rdf_analysis.results.rdf


def angular_momentum(group1: mda.AtomGroup,
                     group2: mda.AtomGroup) -> np.ndarray:
    """
    Calculate the angular momentum of two groups of atoms about their combined centre of mass.

    The angular momentum is evaluated for the current timestep of the trajectory.

    Usage: ``L = angular_momentum(group1, group2)``

    Parameters
    ----------
    group1 : AtomGroup
        First group of atoms.
    group2 : AtomGroup
        Second group of atoms.

    Returns
    -------
    L : ndarray
        Angular momentum vector in the MDAnalysis units of mass, length and velocity.
    """

    # Combine both groups
    combined_group = group1 + group2

    # Centre of mass of the combined group
    com = combined_group.center_of_mass()

    # Positions relative to the centre of mass, velocities and masses shaped for broadcasting
    r_rel = combined_group.positions - com
    v = combined_group.velocities
    masses = combined_group.masses[:, np.newaxis]

    # Sum of the angular momenta of all atoms
    L = np.sum(np.cross(r_rel, masses * v), axis=0)

    return L


def connector(group1: mda.AtomGroup,
              group2: mda.AtomGroup) -> np.ndarray:
    """
    Calculate the vector from the centre of mass of one group of atoms to that of another.

    The vector is evaluated for the current timestep of the trajectory. For a group of one atom, its position
    is used.

    Usage: ``r_vec = connector(group1, group2)``

    Parameters
    ----------
    group1 : AtomGroup
        Group at the start of the vector.
    group2 : AtomGroup
        Group at the end of the vector.

    Returns
    -------
    r_vec : ndarray
        Vector from the centre of mass of group1 to that of group2 in Å.
    """

    # Centre of mass of each group, or the position for a single atom
    com1 = group1.center_of_mass() if len(group1) > 1 else group1.positions[0]
    com2 = group2.center_of_mass() if len(group2) > 1 else group2.positions[0]

    # Vector from the first centre of mass to the second
    r_vec = com2 - com1

    return r_vec
