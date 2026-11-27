"""
Small synthetic systems with known answers, shared by the tests.
"""

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.coordinates.memory import MemoryReader

# Edge of the cubic box in Å and the collision cutoff in Å
BOX_LENGTH = 100.0
CUTOFF = 9.0

# Distances of atoms 1 (along x) and 2 (along y) from atom 0 at the centre of the box, one value per frame
DISTANCES_ATOM_1 = [20.0, 20.0, 8.0, 6.0, 5.0, 7.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
DISTANCES_ATOM_2 = [20.0, 20.0, 20.0, 20.0, 8.0, 7.0, 6.0, 8.0, 8.0, 20.0, 20.0, 20.0]


def three_atom_universe(velocities: np.ndarray | None=None) -> mda.Universe:
    """
    Create a three-atom trajectory in which atoms 1 and 2 approach atom 0 one after the other.

    Atom 0 stays at the centre of the box. Atom 1 is within the cutoff of atom 0 in frames 2-5 and atom 2 in
    frames 4-8. Atoms 1 and 2 never come within the cutoff of each other.

    Parameters
    ----------
    velocities : ndarray, default=None
        Velocities with the shape (N_frames, 3, 3). If None, all velocities are zero.

    Returns
    -------
    u : Universe
        Universe with the trajectory in memory and unit masses.
    """

    # Positions of the three atoms in each frame
    n_frames = len(DISTANCES_ATOM_1)
    centre = np.full(3, BOX_LENGTH / 2)
    positions = np.zeros((n_frames, 3, 3))
    for frame in range(n_frames):
        positions[frame, 0] = centre
        positions[frame, 1] = centre + [DISTANCES_ATOM_1[frame], 0.0, 0.0]
        positions[frame, 2] = centre + [0.0, DISTANCES_ATOM_2[frame], 0.0]

    # Universe with unit masses and the trajectory in memory
    u = mda.Universe.empty(3, trajectory=True)
    u.add_TopologyAttr('masses', [1.0, 1.0, 1.0])
    if velocities is None:
        velocities = np.zeros_like(positions)
    u.load_new(positions,
               format=MemoryReader,
               velocities=velocities,
               dimensions=[BOX_LENGTH, BOX_LENGTH, BOX_LENGTH, 90.0, 90.0, 90.0])

    return u


def event_dataframe(distances_by_partner: dict,
                    start_frame: int=0) -> pd.DataFrame:
    """
    Create a collision event in the format of process_collisions() from distance series of the partners.

    Parameters
    ----------
    distances_by_partner : dict
        Atom index of each partner mapped to its distance in each frame of the event. NaN marks the frames in
        which the partner is outside the cutoff.
    start_frame : int, default=0
        Frame number of the first frame of the event.

    Returns
    -------
    event_df : DataFrame
        Event with the columns 'frame', 'atoms2' and 'distances'.
    """

    # One row per frame with the partners present in it, in ascending atom-index order
    n_frames = len(next(iter(distances_by_partner.values())))
    rows = []
    for i in range(n_frames):
        atoms2 = [atom for atom in sorted(distances_by_partner) if not np.isnan(distances_by_partner[atom][i])]
        rows.append({'frame': start_frame + i,
                     'atoms2': atoms2,
                     'distances': [distances_by_partner[atom][i] for atom in atoms2]})

    return pd.DataFrame(rows)
