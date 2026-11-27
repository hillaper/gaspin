"""
Extraction of the positions and velocities of the atoms of classified collision events.
"""

# Imports
import os

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

from gaspin._core._collisions import main_partner_index
from gaspin._core._trajectory import center_and_wrap


def collision_event_trajectories(u: mda.Universe,
                                 categorized_collisions: dict,
                                 category: str,
                                 atom_index: int,
                                 event_index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the positions and velocities of all atoms involved in a collision event.

    In each frame, the reference atom is centred in the box and all atoms are wrapped before the positions are
    read.

    Usage: ``positions, velocities = collision_event_trajectories(u, categorized, 'vdw', atom_index, event_index)``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory and topology.
    categorized_collisions : dict
        Output of categorize_collisions().
    category : str
        Category of the event, e.g. 'two_body_scattering'.
    atom_index : int
        Atom index of the reference atom.
    event_index : int
        Index of the event in the event list of the reference atom in the category.

    Returns
    -------
    positions : ndarray
        Positions in Å with the shape (N_frames, N_atoms, 3). Index 0 of the second axis is the reference atom,
        index 1 the main partner and indices 2 onwards the other partners in ascending atom-index order.
    velocities : ndarray
        Velocities in the MDAnalysis units with the same shape and atom order as positions.
    """

    # Event and its main partner
    event_df = categorized_collisions[category][atom_index][event_index]
    main_partner_idx = main_partner_index(event_df)

    # Other partners of the event in ascending order
    all_partner_indices = set(event_df['atoms2'].explode().unique())
    all_partner_indices.discard(main_partner_idx)
    other_partner_indices = sorted(list(all_partner_indices))

    # Number of atoms (reference atom, main partner and other partners) and number of frames
    n_atoms = 2 + len(other_partner_indices)
    n_frames = len(event_df)

    # Arrays of the positions and velocities
    positions = np.zeros((n_frames, n_atoms, 3))
    velocities = np.zeros((n_frames, n_atoms, 3))

    # AtomGroups of all atoms of the event
    atom_groups = {idx: u.atoms[[idx]] for idx in [atom_index, main_partner_idx] + other_partner_indices}

    # Go through the frames of the event
    for frame_idx, row in enumerate(event_df.itertuples()):
        frame = row.frame

        # Move the trajectory to the current frame
        u.trajectory[frame]

        # Reference atom, centred in the box with all atoms wrapped
        atom_group = atom_groups[atom_index]
        center_and_wrap(u, u.trajectory.ts, atom_group)
        positions[frame_idx, 0] = atom_group.positions[0]
        velocities[frame_idx, 0] = atom_group.velocities[0]

        # Main partner
        partner_group = atom_groups[main_partner_idx]
        positions[frame_idx, 1] = partner_group.positions[0]
        velocities[frame_idx, 1] = partner_group.velocities[0]

        # Other partners
        for idx, partner_index in enumerate(other_partner_indices):
            partner_group = atom_groups[partner_index]
            positions[frame_idx, 2 + idx] = partner_group.positions[0]
            velocities[frame_idx, 2 + idx] = partner_group.velocities[0]

    return positions, velocities


def save_collision_event_trajectories(u: mda.Universe,
                                      categorized_collisions: dict,
                                      directory: str='collision_trajectories') -> None:
    """
    Save the positions and velocities of all classified collision events to .npy files.

    The files are named positions_<category>_atom<atom_index>_event<event_index>.npy and
    velocities_<category>_atom<atom_index>_event<event_index>.npy (see collision_event_trajectories()).

    Usage: ``save_collision_event_trajectories(u, categorized, directory='collision_trajectories')``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory and topology.
    categorized_collisions : dict
        Output of categorize_collisions().
    directory : str, default='collision_trajectories'
        Directory of the saved files, created if it does not exist.
    """

    # Total number of events for the progress bar
    total_events = sum(len(categorized_collisions[category][atom_index])
                       for category in categorized_collisions for atom_index in categorized_collisions[category])

    # Create the output directory
    os.makedirs(directory, exist_ok=True)

    # Save the trajectories of all events
    with tqdm(total=total_events, desc='Saving collision event trajectories') as pbar:
        for category in categorized_collisions:
            for atom_index in categorized_collisions[category]:
                for event_index in range(len(categorized_collisions[category][atom_index])):

                    # Positions and velocities of the event
                    positions, velocities = collision_event_trajectories(u, categorized_collisions,
                                                                         category, atom_index, event_index)

                    # File names from the category, atom index and event index
                    positions_filename = f'{directory}/positions_{category}_atom{atom_index}_event{event_index}.npy'
                    velocities_filename = f'{directory}/velocities_{category}_atom{atom_index}_event{event_index}.npy'

                    # Save the arrays
                    np.save(positions_filename, positions)
                    np.save(velocities_filename, velocities)

                    pbar.update(1)
