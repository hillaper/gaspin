"""
Analysis of molecular dynamics trajectories of atomic gases.

The module finds the collision events between the atoms of a gas-phase MD trajectory, classifies them into
two-body scattering (TBS), multi-body scattering (MBS), perturbed scattering (PS) and van der Waals (vdW) events,
and extracts their statistics and the averages of distance-dependent properties:

1. find_collisions() records, for every frame, the atoms within a cutoff distance of each reference atom.
2. process_collisions() groups these frame-wise records into events from the point of view of each reference
   atom.
3. categorize_collisions() classifies the events by the local minima of the distance of the main colliding
   pair, the fraction of the event spent in the potential well, and the number of atoms involved.
4. collision_statistics() and the functions that follow it give the event lifetimes, the numbers of atoms
   involved, the atom-time fractions, the equilibrium constants and the collision rates.
5. average_property_R_time_weighted() gives the time-weighted category averages of a property that depends only
   on the interatomic distance R.
6. find_bound_clusters() and bound_cluster_events() find the energetically bound clusters of three or more atoms
   (vdW trimers and larger complexes) independently of the classification, from the positions and velocities.

All distances are in Å, and the time step between frames is given in seconds.
"""

# TODO: OMP parallelization, thread number control (to 1)

# Imports
import os
from collections import Counter
from collections.abc import Callable

import MDAnalysis as mda
import MDAnalysis.transformations as mda_trans
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.coordinates.timestep import Timestep
from MDAnalysis.lib import distances as mda_distances
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


###################################################################################################################
# Trajectory transformations
###################################################################################################################
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


###################################################################################################################
# General functions
###################################################################################################################
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


###################################################################################################################
# Atomic collision extraction and processing functions
###################################################################################################################
def find_collisions(u: mda.Universe,
                    group1: mda.AtomGroup,
                    group2: mda.AtomGroup,
                    cutoff: float) -> dict:
    """
    Find the atoms of one group within a cutoff distance of each atom of another group in every frame.

    The distances follow the minimum-image convention of the periodic box. Self-pairs are skipped.

    Usage: ``collisions = find_collisions(u, group1, group2, cutoff)``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory and topology.
    group1 : AtomGroup
        Reference atoms.
    group2 : AtomGroup
        Partner atoms.
    cutoff : float
        Cutoff distance in Å.

    Returns
    -------
    collisions : dict
        Atom index of each reference atom mapped to a list of frame records in chronological order. Each frame
        record is a dictionary with the keys 'frame' (frame number), 'atoms2' (list of the atom indices of the
        partners within the cutoff) and 'distances' (list of the corresponding distances in Å).
    """

    # Frame records of each reference atom
    collisions = {atom.index: [] for atom in group1}

    # Frame records of each reference atom indexed by the frame number, for appending further partners
    frame_dict = {atom.index: {} for atom in group1}

    # Iterate over all frames of the trajectory
    for ts in tqdm(u.trajectory, desc='Finding collisions'):

        # Distance matrix between the two groups with the minimum-image convention
        dists = distances.distance_array(group1.positions,
                                         group2.positions,
                                         box=u.dimensions)

        # Pairs of atoms within the cutoff
        g1_indices, g2_indices = np.where(dists < cutoff)

        # Loop over all pairs and skip the self-pairs
        for i, j in zip(g1_indices, g2_indices):
            if group1.indices[i] != group2.indices[j]:

                # Append the partner to the record of this frame if the record already exists
                if ts.frame in frame_dict[group1.indices[i]]:
                    frame_entry = frame_dict[group1.indices[i]][ts.frame]
                    frame_entry['atoms2'].append(group2.indices[j])
                    frame_entry['distances'].append(dists[i, j])

                # Otherwise create a new record for this frame
                else:
                    new_entry = {'frame': ts.frame,
                                 'atoms2': [group2.indices[j]],
                                 'distances': [dists[i, j]]}
                    collisions[group1.indices[i]].append(new_entry)
                    frame_dict[group1.indices[i]][ts.frame] = new_entry

    return collisions


# TODO: Later: Allowed atom indices can be generalized to MDAnalysis AtomGroups
def process_collisions(collisions: dict,
                       allowed_atom_indices: list[int] | None=None) -> dict:
    """
    Group the frame-wise collision records of each reference atom into events (first-partner rule).

    An event of a reference atom starts when a partner enters the cutoff sphere. The first partner of the event
    (the first atom in 'atoms2' of its first frame) anchors the event, which continues while

    1. the frames remain consecutive, and
    2. the first partner remains within the cutoff.

    Other atoms may enter and leave during the event. When the first partner leaves while other partners remain,
    a new event anchored to one of them starts in the same frame. Each frame of a reference atom therefore
    belongs to exactly one of its events. Unlike process_collisions_partner_specific(), no overlapping
    partner-specific events are created.

    Usage: ``events = process_collisions(collisions)``

    Parameters
    ----------
    collisions : dict
        Output of find_collisions().
    allowed_atom_indices : list of int, default=None
        If given, frames in which none of these atoms is within the cutoff are skipped.

    Returns
    -------
    processed_collisions : dict
        Atom index of each reference atom mapped to a list of events. Each event is a DataFrame with the columns
        'frame', 'atoms2' and 'distances', one row per frame.
    """

    # Events of all reference atoms and the optional partner filter as a set
    processed_collisions = {}
    allowed_set = set(allowed_atom_indices) if allowed_atom_indices is not None else None

    # Process the frame records of each reference atom separately
    for atom_index in tqdm(collisions, desc='Processing collisions'):

        # Finished events, the frames of the ongoing event, the previous frame and the anchor of the event
        events = []
        current_event = []
        previous_frame = None
        first_atom = None

        # Go through the frame records in chronological order
        for collision in collisions[atom_index]:
            frame = collision['frame']
            atoms2 = collision['atoms2']

            # Skip the frame if none of the allowed atoms is within the cutoff
            if allowed_set is not None and not any(atom in allowed_set for atom in atoms2):
                continue

            # Set the anchor of a new event to the first partner of its first frame
            if first_atom is None and atoms2:
                first_atom = atoms2[0]

            # End the ongoing event at a frame gap or when its anchor has left the cutoff sphere
            if (previous_frame is not None and frame - previous_frame > 1) or (first_atom not in atoms2):
                if current_event:
                    events.append(current_event)
                current_event = []
                first_atom = None

            # Anchor a newly started event to the first partner of the current frame. This is needed because the
            # anchor is reset above, after the anchor assignment at the start of the iteration.
            if first_atom not in atoms2:
                first_atom = atoms2[0] if atoms2 else None

            # Add the frame to the ongoing event
            current_event.append(collision)
            previous_frame = frame

        # Add the last event
        if current_event:
            events.append(current_event)

        # Convert each event into a DataFrame
        processed_collisions[atom_index] = []
        for event in events:
            processed_collisions[atom_index].append(pd.DataFrame(event))

    return processed_collisions


# TODO: Later: Allowed atom indices can be generalized to MDAnalysis AtomGroups
# TODO: Human check this function
def process_collisions_partner_specific(collisions: dict,
                                        allowed_atom_indices: list[int] | None=None) -> dict:
    """
    Group the frame-wise collision records of each reference atom into partner-specific events.

    Unlike process_collisions(), an event is created separately for every partner of the reference atom. A
    partner-specific event starts when the partner enters the cutoff sphere and ends when

    1. the partner is absent in the next processed frame, or
    2. there is a frame gap.

    A frame with several partners therefore belongs to several events. In each row of an event, the partner of
    the event is moved to the first position of 'atoms2', and its distance to the first position of
    'distances'.

    Usage: ``events = process_collisions_partner_specific(collisions)``

    Parameters
    ----------
    collisions : dict
        Output of find_collisions().
    allowed_atom_indices : list of int, default=None
        If given, only these atoms define partner-specific events.

    Returns
    -------
    processed_collisions : dict
        Atom index of each reference atom mapped to a list of partner-specific events, sorted by the start frame
        and the partner index. Each event is a DataFrame with the columns 'frame', 'atoms2' and 'distances'.
    """

    # Events of all reference atoms and the optional partner filter as a set
    processed_collisions = {}
    allowed_set = set(allowed_atom_indices) if allowed_atom_indices is not None else None

    def collision_for_partner(collision: dict,
                              partner_atom: int) -> dict:
        """
        Reorder a frame record so that the given partner and its distance come first.

        Parameters
        ----------
        collision : dict
            Frame record with the keys 'frame', 'atoms2' and 'distances'.
        partner_atom : int
            Atom index of the partner that is moved to the first position.

        Returns
        -------
        reordered_collision : dict
            Frame record with the partner first and the other partners in their original order.
        """

        # Copy the partners and distances and locate the selected partner
        atoms2 = list(collision['atoms2'])
        distances2 = list(collision['distances'])
        partner_position = atoms2.index(partner_atom)

        # Start with the selected partner
        reordered_atoms2 = [atoms2[partner_position]]
        reordered_distances = [distances2[partner_position]]

        # Keep the other simultaneous partners after the selected one
        for i, atom2 in enumerate(atoms2):
            if i == partner_position:
                continue
            reordered_atoms2.append(atom2)
            reordered_distances.append(distances2[i])

        return {'frame': collision['frame'],
                'atoms2': reordered_atoms2,
                'distances': reordered_distances}

    # Process the frame records of each reference atom separately
    for atom_index in tqdm(collisions, desc='Processing collisions'):

        # Finished events as tuples (start frame, partner atom index, list of frame records)
        event_records = []

        # Ongoing events keyed by the partner atom index, which allows overlapping events
        active_events = {}

        # Go through the frame records in chronological order
        for collision in collisions[atom_index]:
            frame = collision['frame']
            atoms2 = list(collision['atoms2'])

            # Partners that may define events in this frame
            if allowed_set is None:
                current_partners = atoms2
            else:
                current_partners = [atom for atom in atoms2 if atom in allowed_set]

            # Skip frames without relevant partners
            if not current_partners:
                continue

            # End the ongoing events at a frame gap or when their partner is no longer within the cutoff
            for partner_atom in list(active_events):
                previous_frame = active_events[partner_atom]['previous_frame']
                if frame - previous_frame > 1 or partner_atom not in current_partners:
                    event_records.append((active_events[partner_atom]['start_frame'],
                                          partner_atom,
                                          active_events[partner_atom]['rows']))
                    del active_events[partner_atom]

            # Continue or start an event for every partner present in this frame
            for partner_atom in current_partners:
                if partner_atom not in active_events:
                    active_events[partner_atom] = {'rows': [],
                                                   'previous_frame': frame,
                                                   'start_frame': frame}

                # Append the frame record with this partner first
                active_events[partner_atom]['rows'].append(collision_for_partner(collision, partner_atom))
                active_events[partner_atom]['previous_frame'] = frame

        # Finish the events still ongoing after the last frame
        for partner_atom, event_data in active_events.items():
            event_records.append((event_data['start_frame'],
                                  partner_atom,
                                  event_data['rows']))

        # Sort the events by start frame and partner index and convert each into a DataFrame
        processed_collisions[atom_index] = []
        for _, _, event_rows in sorted(event_records, key=lambda item: (item[0], item[1])):
            processed_collisions[atom_index].append(pd.DataFrame(event_rows))

    return processed_collisions


def categorize_collisions(processed_collisions: dict,
                          cutoff: float,
                          dt: float,
                          vdw_bond_length: float,
                          distance_minima_prominence: float=0.25,
                          n_jobs: int=-1,
                          allowed_atom_indices: list[int] | None=None,
                          vdw_well_occupancy: float | None=0.75) -> dict:
    """
    Classify the collision events into two-body scattering, multi-body scattering, perturbed scattering and
    van der Waals events.

    The classification uses the distance R(t) between the reference atom and the main partner (see
    main_partner_index()) and involves no lifetime criterion. With the well region defined as
    vdw_bond_length ± 0.75 Å, the rules are applied in the following order:

    1. vdW if the fraction of the frames with R(t) in the well region is at least vdw_well_occupancy. This
       catches bound pairs on nearly circular orbits, whose radial oscillations are too shallow to be
       resolved as distance minima.
    2. Otherwise, the local minima of R(t) with the given prominence are counted (N_min), together with the
       minima below vdw_bond_length + 0.75 Å (n_min). The event is

       - two-body scattering if N_min <= 1 and only one partner is within the cutoff in every frame,
       - multi-body scattering if N_min <= 1 and two or more partners are within the cutoff in some frame,
       - perturbed scattering if N_min > 1 and n_min <= 2, and
       - vdW if N_min > 1 and n_min > 2.

    Usage: ``categorized = categorize_collisions(events, cutoff, dt, vdw_bond_length)``

    Parameters
    ----------
    processed_collisions : dict
        Output of process_collisions() or process_collisions_partner_specific(). Both are valid input but give
        different results: process_collisions() uses the first-partner rule, whereas
        process_collisions_partner_specific() creates partner-specific events.
    cutoff : float
        Cutoff distance used in the collision detection in Å.
    dt : float
        Time step between frames in seconds. It is not used by the classification and is kept for
        compatibility with earlier calls.
    vdw_bond_length : float
        Distance of the minimum of the pair potential in Å.
    distance_minima_prominence : float, default=0.25
        Minimum prominence of a local minimum of R(t) in Å.
    n_jobs : int, default=-1
        Number of parallel processes over the reference atoms. The value -1 uses all available CPUs.
    allowed_atom_indices : list of int, default=None
        If given, events in which none of these atoms is a partner are skipped, and only these atoms are
        considered as the main partner.
    vdw_well_occupancy : float or None, default=0.75
        Minimum fraction of the frames in the well region for the direct vdW assignment of rule 1. None
        disables the rule. For gas-phase Xe, all events that meet the criterion last longer than 50 ps.

    Returns
    -------
    categorized_collisions : dict
        Dictionary with the keys 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering' and
        'vdw'. Each maps the atom index of a reference atom to the list of its events in that category, as
        DataFrames. Reference atoms without events in a category are omitted from it. Every processed event,
        including short events without a resolved minimum, is assigned to one category.
    """

    # Well region of ±0.75 Å around the minimum of the pair potential
    vdw_bond_threshold = (vdw_bond_length - 0.75, vdw_bond_length + 0.75)

    def process_atom(atom_index: int,
                     collisions: list) -> tuple[int, dict]:
        """
        Classify the events of one reference atom.

        Parameters
        ----------
        atom_index : int
            Atom index of the reference atom.
        collisions : list
            Events of the reference atom as DataFrames.

        Returns
        -------
        atom_index : int
            Atom index of the reference atom.
        categorized : dict
            Category mapped to the list of events of the reference atom in that category.
        """

        # Event lists of the four categories
        categorized = {'two_body_scattering': [],
                       'multi_body_scattering': [],
                       'perturbed_scattering': [],
                       'vdw': []}

        # Set of the allowed atom indices for fast lookup
        allowed_set = set(allowed_atom_indices) if allowed_atom_indices is not None else None

        # Classify each event of the reference atom
        for collision_df in collisions:

            # Skip the event if none of the allowed atoms is a partner in it
            if allowed_set is not None:
                atoms2_all = set(atom for sublist in collision_df['atoms2'] for atom in sublist)
                if not atoms2_all.intersection(allowed_set):
                    continue

            # Multi-body event if two or more partners are within the cutoff in some frame
            is_multi_body = collision_df['atoms2'].str.len().max() > 1

            # Distance series of the main pair
            main_atom_distances = main_partner_distance_series(collision_df,
                                                               allowed_atom_indices=allowed_atom_indices)

            # Classify events in which the pair spends most of the time in the potential well directly as vdW
            if vdw_well_occupancy is not None:

                # Fraction of the frames in the well region
                within_threshold = np.sum((main_atom_distances >= vdw_bond_threshold[0]) &
                                          (main_atom_distances <= vdw_bond_threshold[1]))
                fraction_within_threshold = within_threshold / len(main_atom_distances)

                # Assign to vdW if the fraction reaches the well-occupancy threshold
                if fraction_within_threshold >= vdw_well_occupancy:
                    categorized['vdw'].append(collision_df)
                    continue

            # Local minima of the distance as the peaks of the negated distance, shifted to start from zero
            shifted_neg_distances = -main_atom_distances + cutoff
            minima_indices, _ = find_peaks(shifted_neg_distances,
                                           prominence=distance_minima_prominence)

            # Number of all local minima and of those below the upper edge of the well region
            num_minima = len(minima_indices)
            num_minima_below_vdw = np.sum(main_atom_distances[minima_indices] < (vdw_bond_length + 0.75))

            # Classify by the number of minima. Events with no resolved minimum go to the scattering categories.
            if num_minima <= 1:
                if is_multi_body:
                    categorized['multi_body_scattering'].append(collision_df)
                else:
                    categorized['two_body_scattering'].append(collision_df)
            elif num_minima > 1:
                if num_minima_below_vdw <= 2:
                    categorized['perturbed_scattering'].append(collision_df)
                elif num_minima_below_vdw > 2:
                    categorized['vdw'].append(collision_df)

        return atom_index, categorized

    # Classify the events of all reference atoms in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_atom)(atom_index, collisions)
                                      for atom_index, collisions in tqdm(processed_collisions.items(),
                                                                         desc='Categorizing collisions'))

    # Combine the results of all reference atoms, omitting empty event lists
    categorized_collisions = {'two_body_scattering': {},
                              'multi_body_scattering': {},
                              'perturbed_scattering': {},
                              'vdw': {}}
    for atom_index, categorized in results:
        for category in categorized_collisions:
            if categorized[category]:
                categorized_collisions[category][atom_index] = categorized[category]

    return categorized_collisions


###################################################################################################################
# Helper functions
###################################################################################################################
def main_partner_index(collision: pd.DataFrame,
                       allowed_atom_indices: list[int] | None=None) -> int | None:
    """
    Identify the main partner of a collision event as the partner present in the largest number of frames.

    Ties are resolved in favour of the partner that appears first in the event. Under the first-partner rule of
    process_collisions(), the anchor of the event is present in every frame, so the main partner is always
    present in every frame as well.

    Usage: ``main_partner = main_partner_index(event_df)``

    Parameters
    ----------
    collision : DataFrame
        Collision event with the columns 'frame', 'atoms2' and 'distances'.
    allowed_atom_indices : list of int, default=None
        If given, only these atoms are considered as the main partner.

    Returns
    -------
    main_partner_index : int or None
        Atom index of the main partner, or None if the event has no eligible partner.
    """

    # Number of frames in which each partner is present
    partner_counts = Counter()
    for row in collision.itertuples():
        for atom in row.atoms2:

            # Skip atoms outside the allowed list
            if allowed_atom_indices is not None and atom not in allowed_atom_indices:
                continue

            partner_counts[atom] += 1

    # No eligible partner in the event
    if not partner_counts:
        return None

    # Partner present in the largest number of frames
    main_partner_index = partner_counts.most_common(1)[0][0]

    return main_partner_index


def main_partner_distance_series(collision: pd.DataFrame,
                                 allowed_atom_indices: list[int] | None=None) -> np.ndarray:
    """
    Extract the distance series between the reference atom and the main partner of a collision event.

    The main partner is the partner present in the largest number of frames (see main_partner_index()).

    Usage: ``distances = main_partner_distance_series(event_df)``

    Parameters
    ----------
    collision : DataFrame
        Collision event with the columns 'frame', 'atoms2' and 'distances'.
    allowed_atom_indices : list of int, default=None
        If given, only these atoms are considered as the main partner.

    Returns
    -------
    distances : ndarray
        Distance in Å between the reference atom and the main partner in each frame of the event. The distance
        is NaN in the frames where the main partner is absent, and in all frames if there is no main partner.
    """

    # Main partner of the event
    main_partner_idx = main_partner_index(collision,
                                          allowed_atom_indices=allowed_atom_indices)

    # Distances initialised to NaN for the frames without the main partner
    distances = np.full(len(collision), np.nan)

    # No main partner in the event
    if main_partner_idx is None:
        return distances

    # Distance to the main partner in each frame where it is present
    for i, row in enumerate(collision.itertuples()):
        if main_partner_idx in row.atoms2:
            partner_index = row.atoms2.index(main_partner_idx)
            distances[i] = row.distances[partner_index]

    return distances


def distance_series(collision: pd.DataFrame,
                    allowed_atom_indices: list[int] | None=None) -> dict:
    """
    Extract the distance series between the reference atom and each partner of a collision event.

    Usage: ``distance_dict = distance_series(event_df)``

    Parameters
    ----------
    collision : DataFrame
        Collision event with the columns 'frame', 'atoms2' and 'distances'.
    allowed_atom_indices : list of int, default=None
        If given, only these atoms are included.

    Returns
    -------
    distance_dict : dict
        Atom index of each partner mapped to an array of its distances in Å from the reference atom in each
        frame of the event. The distance is NaN in the frames where the partner is absent.
    """

    # Distance series of each partner
    distance_dict = {}

    # Go through the frames of the event
    for i, row in enumerate(collision.itertuples()):

        # Go through the partners present in this frame
        for j, atom_index in enumerate(row.atoms2):

            # Skip atoms outside the allowed list
            if allowed_atom_indices is not None and atom_index not in allowed_atom_indices:
                continue

            # Initialise the distance series of a new partner with NaN
            if atom_index not in distance_dict:
                distance_dict[atom_index] = np.full(len(collision), np.nan)

            # Store the distance of this frame
            distance_dict[atom_index][i] = row.distances[j]

    return distance_dict


###################################################################################################################
# Atomic collision analysis functions
###################################################################################################################
# TODO: Check everything below
def collision_statistics(categorized_events: dict,
                         dt: float,
                         total_frames: int,
                         return_cluster_changes: bool=False) -> tuple[dict, dict] | tuple[dict, dict, dict]:
    """
    Extract the statistics of the classified events and the number of clusters of each category in each frame.

    For each event, the lifetime, the number of atoms involved, the mean number of atoms per frame and the mean
    distance of the main pair are recorded. In addition, the atoms of all events of a category present in the
    same frame are merged into connected clusters, which gives the number of clusters of each category in each
    frame. Optionally, the births and deaths of the clusters are counted in each frame: a cluster is born in
    frame t if none of its atoms belongs to a cluster of the same category in frame t - 1, and a cluster of
    frame t - 1 dies in frame t if none of its atoms belongs to a cluster of the same category in frame t.
    Clusters that merge, split or change their members are neither born nor dead.

    Usage: ``collision_stats, cluster_counts, cluster_changes = collision_statistics(categorized, dt,
    total_frames, return_cluster_changes=True)``

    Parameters
    ----------
    categorized_events : dict
        Output of categorize_collisions().
    dt : float
        Time step between frames in seconds.
    total_frames : int
        Total number of frames in the trajectory.
    return_cluster_changes : bool, default=False
        If True, also return the numbers of cluster births and deaths in each frame.

    Returns
    -------
    collision_stats : dict
        Same nested structure as categorized_events, with each event replaced by a dictionary with the keys

        - 'event_index': index of the event in the event list of the reference atom in the category,
        - 'lifetime_ps': lifetime of the event in ps,
        - 'num_atoms_involved': number of distinct atoms involved, including the reference atom,
        - 'avg_num_atoms': mean number of atoms per frame, including the reference atom, and
        - 'avg_main_partner_distance': mean distance between the reference atom and the main partner in Å.

    cluster_counts : dict
        Category mapped to an integer array of length total_frames with the number of connected clusters of
        that category in each frame.
    cluster_changes : dict
        Category mapped to a dictionary with the keys 'births' and 'deaths', integer arrays of length
        total_frames with the numbers of cluster births and deaths in each frame. Both are zero in the first
        frame. Returned only if return_cluster_changes is True.
    """

    def merge_frame_cluster(frame_clusters: list,
                            cluster_atoms: set) -> list:
        """
        Merge a set of atoms into the connected clusters of a frame.

        Parameters
        ----------
        frame_clusters : list of set
            Connected clusters of the frame found so far.
        cluster_atoms : set
            Atoms of one event row.

        Returns
        -------
        remaining_clusters : list of set
            Connected clusters of the frame, with all clusters overlapping cluster_atoms merged into one.
        """

        # Merge every existing cluster that shares an atom with the new set
        merged_cluster = set(cluster_atoms)
        remaining_clusters = []
        for existing_cluster in frame_clusters:
            if merged_cluster.intersection(existing_cluster):
                merged_cluster.update(existing_cluster)
            else:
                remaining_clusters.append(existing_cluster)

        # Add the merged cluster
        remaining_clusters.append(merged_cluster)

        return remaining_clusters

    # Event statistics, cluster counts and the connected clusters of each frame for each category
    collision_stats = {category: {} for category in categorized_events}
    cluster_counts = {category: np.zeros(total_frames, dtype=int) for category in categorized_events}
    cluster_set = {category: {} for category in categorized_events}

    # Go through the categories
    for category in categorized_events:

        # Go through the reference atoms of the category
        for atom_index in tqdm(categorized_events[category], desc=f'Extracting statistics for {category}'):

            # Statistics of the events of this reference atom
            collision_stats[category][atom_index] = []

            # Go through the events of the reference atom
            for event_index, event_df in enumerate(categorized_events[category][atom_index]):

                # Lifetime of the event in ps
                event_duration = event_df['frame'].max() - event_df['frame'].min() + 1
                lifetime = event_duration * dt * 1e12

                # Number of distinct atoms involved, including the reference atom
                unique_atoms = set(atom for sublist in event_df['atoms2'] for atom in sublist)
                num_atoms_involved = len(unique_atoms) + 1

                # Mean number of atoms per frame, including the reference atom
                num_frames = len(event_df)
                total_atoms_per_frame = sum(len(atoms2_list) + 1 for atoms2_list in event_df['atoms2'])
                avg_num_atoms = total_atoms_per_frame / num_frames if num_frames > 0 else 0

                # Mean distance to the main partner
                main_partner_distances = main_partner_distance_series(event_df)
                if np.all(np.isnan(main_partner_distances)):
                    avg_main_partner_distance = np.nan
                else:
                    avg_main_partner_distance = np.nanmean(main_partner_distances)

                # Store the statistics of the event
                collision_stats[category][atom_index].append({'event_index': event_index,
                                                              'lifetime_ps': lifetime,
                                                              'num_atoms_involved': num_atoms_involved,
                                                              'avg_num_atoms': avg_num_atoms,
                                                              'avg_main_partner_distance': avg_main_partner_distance})

                # Merge the atoms of each frame of the event into the connected clusters of that frame
                for row in event_df.itertuples():
                    frame = row.frame
                    cluster_atoms = set([atom_index] + list(row.atoms2))
                    if frame not in cluster_set[category]:
                        cluster_set[category][frame] = []
                    cluster_set[category][frame] = merge_frame_cluster(cluster_set[category][frame],
                                                                       cluster_atoms)

    # Number of connected clusters in each frame for each category
    for category, clusters in cluster_set.items():
        for frame, clusters_in_frame in clusters.items():
            cluster_counts[category][frame] = len(clusters_in_frame)

    # Return without the cluster births and deaths unless they are requested
    if not return_cluster_changes:
        return collision_stats, cluster_counts

    # Births and deaths of the clusters of each category in each frame
    cluster_changes = {category: {'births': np.zeros(total_frames, dtype=int),
                                  'deaths': np.zeros(total_frames, dtype=int)}
                       for category in categorized_events}
    for category, clusters in cluster_set.items():
        for frame in range(1, total_frames):

            # Clusters of the previous and the current frame and the atoms in them
            previous_clusters = clusters.get(frame - 1, [])
            current_clusters = clusters.get(frame, [])
            if not previous_clusters and not current_clusters:
                continue
            previous_atoms = set().union(*previous_clusters)
            current_atoms = set().union(*current_clusters)

            # Clusters with no atom in a cluster of the other frame
            cluster_changes[category]['births'][frame] = sum(1 for cluster in current_clusters
                                                             if not cluster & previous_atoms)
            cluster_changes[category]['deaths'][frame] = sum(1 for cluster in previous_clusters
                                                             if not cluster & current_atoms)

    return collision_stats, cluster_counts, cluster_changes


def combine_collision_statistics(collision_statistics_results: list) -> tuple[dict, dict]:
    """
    Combine several outputs of collision_statistics() from consecutive trajectory segments.

    The event statistics are concatenated per category and reference atom, and the event indices are
    renumbered consecutively. The cluster-count arrays are concatenated in time.

    Usage: ``combined_stats, combined_counts = combine_collision_statistics([(stats1, counts1), (stats2, counts2)])``

    Parameters
    ----------
    collision_statistics_results : list of tuple
        Outputs (collision_stats, cluster_counts) of collision_statistics() in chronological order. The first
        output defines the categories.

    Returns
    -------
    combined_collision_stats : dict
        Combined event statistics with the same nested structure as a single output of collision_statistics().
    combined_cluster_counts : dict
        Category mapped to the concatenated cluster-count array.

    Raises
    ------
    ValueError
        Raised if no outputs are given, or if a later output contains a category missing from the first one or
        lacks one of its categories.
    """

    # Require at least one output to define the categories
    if not collision_statistics_results:
        raise ValueError('collision_statistics_results must contain at least one result.')

    # Use the first output as the reference structure
    reference_collision_stats, reference_cluster_counts = collision_statistics_results[0]

    # Copy the event statistics of the first output
    combined_collision_stats = {}
    for category, atoms_dict in reference_collision_stats.items():
        combined_collision_stats[category] = {}
        for atom_index, stats_list in atoms_dict.items():
            combined_collision_stats[category][atom_index] = [dict(event_stats)
                                                              for event_stats in stats_list]

    # Copy the cluster counts of the first output
    combined_cluster_counts = {category: np.array(counts, copy=True)
                               for category, counts in reference_cluster_counts.items()}

    # Append the following outputs in chronological order
    for collision_stats, cluster_counts in collision_statistics_results[1:]:

        # Extend the event statistics of each category, including reference atoms that appear only later
        for category, next_atoms_dict in collision_stats.items():
            if category not in combined_collision_stats:
                raise ValueError(f"Missing collision_stats category '{category}' in reference result.")

            for atom_index, next_stats_list in next_atoms_dict.items():
                if atom_index not in combined_collision_stats[category]:
                    combined_collision_stats[category][atom_index] = []

                # Append the events with consecutive event indices
                combined_stats_list = combined_collision_stats[category][atom_index]
                next_event_index = len(combined_stats_list)
                for event_stats in next_stats_list:
                    merged_event_stats = dict(event_stats)
                    merged_event_stats['event_index'] = next_event_index
                    combined_stats_list.append(merged_event_stats)
                    next_event_index += 1

        # Concatenate the cluster counts of each category
        for category, combined_counts in combined_cluster_counts.items():
            next_counts = cluster_counts.get(category)
            if next_counts is None:
                raise ValueError(f"Missing cluster_counts category '{category}' in merged result.")
            combined_cluster_counts[category] = np.concatenate((combined_counts,
                                                                next_counts))

    return combined_collision_stats, combined_cluster_counts


def collision_counts(collision_stats: dict) -> dict:
    """
    Count the events of each category.

    Usage: ``counts = collision_counts(collision_stats)``

    Parameters
    ----------
    collision_stats : dict
        Output of collision_statistics() or combine_collision_statistics().

    Returns
    -------
    collision_counts : dict
        Category mapped to the total number of events.
    """

    # Number of events summed over the reference atoms of each category
    return {category: sum(len(stats) for stats in atoms_dict.values())
            for category, atoms_dict in collision_stats.items()}


def collision_counts_by_lifetime(collision_stats: dict,
                                 lifetime_ranges: list[tuple[float, float]]) -> dict:
    """
    Count the events of each category in given lifetime ranges.

    An event is counted in the first range for which min_lifetime <= lifetime < max_lifetime.

    Usage: ``counts = collision_counts_by_lifetime(collision_stats, [(0, 1), (1, 3), (3, np.inf)])``

    Parameters
    ----------
    collision_stats : dict
        Output of collision_statistics() or combine_collision_statistics().
    lifetime_ranges : list of tuple
        Lifetime ranges (min_lifetime, max_lifetime) in ps.

    Returns
    -------
    counts_by_lifetime : dict
        Category mapped to a list of event counts, one per lifetime range.
    """

    # One counter per lifetime range for each category
    counts_by_lifetime = {category: [0] * len(lifetime_ranges) for category in collision_stats}

    # Go through all events of all categories
    for category, atoms_dict in collision_stats.items():
        for atom_index, stats_list in atoms_dict.items():
            for event in stats_list:
                lifetime = event['lifetime_ps']

                # Count the event in the first matching lifetime range
                for i, (min_lifetime, max_lifetime) in enumerate(lifetime_ranges):
                    if min_lifetime <= lifetime < max_lifetime:
                        counts_by_lifetime[category][i] += 1
                        break

    return counts_by_lifetime


def cluster_formation_equilibrium_values(cluster_counts: dict,
                                         collision_stats: dict,
                                         total_monomers: int,
                                         volume: float | None=None,
                                         free_monomers: float | None=None) -> tuple[dict, dict]:
    """
    Calculate the equilibrium constants and the cluster fractions of the event categories.

    The equilibrium constant of category c is

    .. math::

        K_c = \\frac{\\langle N_c \\rangle}{\\langle N_\\mathrm{mon} \\rangle^2} V,

    where N_c is the number of clusters of the category and N_mon the number of monomers, i.e., atoms in no
    event. The number of atoms in the clusters of a category is estimated as the mean number of clusters times
    the mean number of atoms per frame of its events. The mean number of monomers should be given as
    free_monomers, e.g. from the atom-time fractions as f_0 N (see atom_time_fractions()), which is exact.
    Otherwise it is estimated as the atoms not counted in any cluster, which is approximate because the mean
    cluster sizes are not time-weighted and a pair can belong to clusters of two categories.

    Usage: ``K, fractions = cluster_formation_equilibrium_values(cluster_counts, collision_stats, N, volume=V)``

    Parameters
    ----------
    cluster_counts : dict
        Category mapped to the number of clusters in each frame, from collision_statistics().
    collision_stats : dict
        Output of collision_statistics(). The field 'avg_num_atoms' gives the mean cluster size of each
        category.
    total_monomers : int
        Total number of atoms in the system.
    volume : float, default=None
        Volume of the system. If given, the equilibrium constants are multiplied by it.
    free_monomers : float, default=None
        Mean number of monomers. If None, it is estimated from the cluster counts and sizes.

    Returns
    -------
    equilibrium_constants : dict
        Category mapped to the equilibrium constant, in units of volume if the volume is given.
    equilibrium_fractions : dict
        Category mapped to the fraction of all atoms in its clusters, plus the key 'free_monomers' for the
        fraction of monomers.
    """

    # Equilibrium constants and fractions of each category
    equilibrium_constants = {}
    equilibrium_fractions = {}

    # Mean number of atoms per frame of the events of each category
    avg_cluster_sizes = {}
    for category in cluster_counts:
        category_event_stats = [event_stat
                                for atom_stats in collision_stats.get(category, {}).values()
                                for event_stat in atom_stats]
        if category_event_stats:
            avg_cluster_sizes[category] = np.mean([event_stat['avg_num_atoms']
                                                   for event_stat in category_event_stats])
        else:
            avg_cluster_sizes[category] = 0.0

    # Mean number of atoms in the clusters of all categories
    total_bound_monomers = 0
    for category, counts in cluster_counts.items():
        avg_num_clusters = np.mean(counts)
        avg_bound_monomers = avg_num_clusters * avg_cluster_sizes[category]
        total_bound_monomers += avg_bound_monomers

    # Mean number of monomers, given or estimated as the atoms not counted in any cluster
    if free_monomers is not None:
        avg_free_monomers = free_monomers
    else:
        avg_free_monomers = total_monomers - total_bound_monomers

    # Equilibrium constant and cluster fraction of each category
    for category, counts in cluster_counts.items():
        avg_num_clusters = np.mean(counts)

        # Equilibrium constant K = [cluster] / [monomer]^2, infinite if there are no monomers
        avg_free_monomer_pairs = avg_free_monomers**2
        if avg_free_monomer_pairs > 0:
            equilibrium_constant = avg_num_clusters / avg_free_monomer_pairs
        else:
            equilibrium_constant = np.inf

        # Convert the ratio of numbers into the ratio of concentrations
        if volume is not None:
            equilibrium_constant *= volume
        equilibrium_constants[category] = equilibrium_constant

        # Fraction of all atoms in the clusters of this category
        avg_bound_monomers = avg_num_clusters * avg_cluster_sizes[category]
        equilibrium_fractions[category] = avg_bound_monomers / total_monomers if total_monomers > 0 else 0

    # Fraction of monomers
    equilibrium_fractions['free_monomers'] = avg_free_monomers / total_monomers if total_monomers > 0 else 0

    return equilibrium_constants, equilibrium_fractions


def cluster_formation_rates(cluster_counts: dict,
                            dt: float,
                            volume: float | None=None,
                            cluster_changes: dict | None=None) -> dict:
    """
    Calculate the formation and break-up rates of the clusters of each category.

    If the cluster births and deaths of collision_statistics() are given, the formations (break-ups) are their
    total numbers. Otherwise, they are estimated as the sum of the positive (negative) changes of the cluster
    count between consecutive frames. This estimate misses formations and break-ups that coincide in the same
    frame, and therefore underestimates the rates of frequent events (by about 9 % for two-body scattering in
    Xe gas at 1 atm with 50 fs frames). The numbers are divided by the time spanned by the frames,
    (N_frames - 1) dt.

    Usage: ``rates = cluster_formation_rates(cluster_counts, dt, volume=V, cluster_changes=cluster_changes)``

    Parameters
    ----------
    cluster_counts : dict
        Category mapped to the number of clusters in each frame, from collision_statistics().
    dt : float
        Time step between frames in seconds.
    volume : float, default=None
        Volume of the system in Å^3. If given, the rates are divided by the volume and given in s^-1 cm^-3.
    cluster_changes : dict, default=None
        Cluster births and deaths in each frame, the third output of collision_statistics() with
        return_cluster_changes=True.

    Returns
    -------
    rates : dict
        Category mapped to a dictionary with the keys 'formation_rate' and 'breakup_rate', in s^-1, or in
        s^-1 cm^-3 if the volume is given.
    """

    # Rates of each category
    rates = {}

    # Volume in cm^3
    angstrom_to_cm = 1e-8
    volume_cm3 = volume * (angstrom_to_cm ** 3) if volume is not None else None

    # Go through the categories
    for category, counts in cluster_counts.items():

        # Formations and break-ups as the total numbers of cluster births and deaths
        if cluster_changes is not None:
            formations = np.sum(cluster_changes[category]['births'])
            breakups = np.sum(cluster_changes[category]['deaths'])

        # Otherwise as the sums of the positive and negative changes of the cluster count between frames
        else:
            count_diff = np.diff(counts)
            formations = np.sum(count_diff[count_diff > 0])
            breakups = -np.sum(count_diff[count_diff < 0])

        # Time spanned by the frames in seconds
        total_time = (len(counts) - 1) * dt

        # Rates per unit time
        formation_rate = formations / total_time if total_time > 0 else 0
        breakup_rate = breakups / total_time if total_time > 0 else 0

        # Rates per unit volume if the volume is given
        if volume_cm3 is not None:
            formation_rate /= volume_cm3
            breakup_rate /= volume_cm3

        rates[category] = {'formation_rate': formation_rate,
                           'breakup_rate': breakup_rate}

    return rates


def atom_time_fractions(collision_stats: dict,
                        n_atoms: int,
                        total_frames: int,
                        dt: float) -> dict:
    """
    Calculate the fraction of the total atom-time spent by the reference atoms in events of each category.

    Each frame of a reference atom belongs to at most one of its events, so these fractions and the remaining
    fraction of free atoms sum to unity. Together with the time-weighted category averages of
    average_property_R_time_weighted(), they decompose the trajectory average of a single-atom property exactly:

    .. math::

        \\langle A \\rangle = \\sum_c f_c \\langle A_c \\rangle.

    Usage: ``time_fractions = atom_time_fractions(collision_stats, n_atoms, total_frames, dt)``

    Parameters
    ----------
    collision_stats : dict
        Output of collision_statistics() or combine_collision_statistics(). The field 'lifetime_ps' of each
        event is used.
    n_atoms : int
        Number of reference atoms.
    total_frames : int
        Total number of frames in the trajectory.
    dt : float
        Time step between frames in seconds.

    Returns
    -------
    time_fractions : dict
        Category mapped to its fraction of the total atom-time, plus the key 'free_monomers' for the fraction
        spent outside any event.
    """

    # Total atom-time of the trajectory in ps
    total_atom_time_ps = n_atoms * total_frames * dt * 1e12

    # Summed event lifetimes of each category normalised by the total atom-time
    time_fractions = {}
    for category, atoms_dict in collision_stats.items():
        category_time_ps = sum(event['lifetime_ps'] for stats_list in atoms_dict.values() for event in stats_list)
        time_fractions[category] = category_time_ps / total_atom_time_ps

    # The remaining atom-time is spent outside any event
    time_fractions['free_monomers'] = 1.0 - sum(time_fractions.values())

    return time_fractions


###################################################################################################################
# Trajectory analysis functions using categorized events
###################################################################################################################
def property_time_series(u: mda.Universe,
                         categorized_collisions: dict,
                         category: str,
                         atom_index: int,
                         event_index: int,
                         property_function: Callable,
                         *args) -> np.ndarray:
    """
    Calculate the time series of a property of the main pair of a collision event from the trajectory.

    In each frame of the event, the reference atom is centred in the box and all atoms are wrapped, after which
    the property function is evaluated for the reference atom and the main partner.

    Usage: ``series = property_time_series(u, categorized, 'vdw', atom_index, event_index, angular_momentum)``

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
    property_function : callable
        Function of two AtomGroups and the additional arguments that returns the property.
    *args
        Additional arguments passed to property_function.

    Returns
    -------
    time_series : ndarray
        Property value in each frame of the event.
    """

    # Event and its main partner
    event_df = categorized_collisions[category][atom_index][event_index]
    main_partner_idx = main_partner_index(event_df)

    # Property value in each frame of the event
    time_series = []
    for row in event_df.itertuples():
        frame = row.frame

        # Move the trajectory to the current frame
        u.trajectory[frame]

        # AtomGroups of the reference atom and the main partner
        atom_group1 = u.atoms[[atom_index]]
        atom_group2 = u.atoms[[main_partner_idx]]

        # Centre the reference atom and wrap all atoms into the box
        center_and_wrap(u, u.trajectory.ts, atom_group1)

        # Evaluate the property
        property_value = property_function(atom_group1, atom_group2, *args)
        time_series.append(property_value)

    return np.array(time_series)


def property_R_main_partner_time_series(categorized_collisions: dict,
                                        category: str,
                                        atom_index: int,
                                        event_index: int,
                                        property_function: Callable,
                                        *args) -> np.ndarray:
    """
    Calculate the time series of a distance-dependent property of the main pair of a collision event.

    Usage: ``series = property_R_main_partner_time_series(categorized, 'vdw', atom_index, event_index, A_of_R)``

    Parameters
    ----------
    categorized_collisions : dict
        Output of categorize_collisions().
    category : str
        Category of the event, e.g. 'two_body_scattering'.
    atom_index : int
        Atom index of the reference atom.
    event_index : int
        Index of the event in the event list of the reference atom in the category.
    property_function : callable
        Function of a distance array in Å and the additional arguments that returns the property.
    *args
        Additional arguments passed to property_function.

    Returns
    -------
    time_series : ndarray
        Property value of the main pair in each frame of the event.
    """

    # Event of the reference atom
    event_df = categorized_collisions[category][atom_index][event_index]

    # Distance series of the main pair
    main_partner_distances = main_partner_distance_series(event_df)

    # Property in each frame of the event
    time_series = property_function(main_partner_distances, *args)

    return np.array(time_series)


def property_R_all_partners_time_series(categorized_collisions: dict,
                                        category: str,
                                        atom_index: int,
                                        event_index: int,
                                        property_function: Callable,
                                        *args,
                                        use_pairwise_sum: bool=False) -> dict | tuple[dict, np.ndarray]:
    """
    Calculate the time series of a distance-dependent property for each partner of a collision event.

    Usage: ``series, pairwise_sum = property_R_all_partners_time_series(categorized, 'vdw', atom_index,
    event_index, A_of_R, use_pairwise_sum=True)``

    Parameters
    ----------
    categorized_collisions : dict
        Output of categorize_collisions().
    category : str
        Category of the event, e.g. 'two_body_scattering'.
    atom_index : int
        Atom index of the reference atom.
    event_index : int
        Index of the event in the event list of the reference atom in the category.
    property_function : callable
        Function of a distance array in Å and the additional arguments that returns the property.
    *args
        Additional arguments passed to property_function.
    use_pairwise_sum : bool, default=False
        If True, also return the sum of the property over all partners in each frame (pairwise additivity).

    Returns
    -------
    time_series : dict
        Atom index of each partner mapped to the property in each frame of the event. The value is NaN in the
        frames where the partner is absent.
    pairwise_sum : ndarray
        Sum of the property over all partners present in each frame. Returned only if use_pairwise_sum is True.
    """

    # Event of the reference atom
    event_df = categorized_collisions[category][atom_index][event_index]

    # Distance series of each partner
    partner_distances = distance_series(event_df)

    # Property series of each partner
    time_series = {partner: property_function(distances, *args)
                   for partner, distances in partner_distances.items()}

    # Sum over the partners present in each frame, ignoring the NaN values of absent partners
    if use_pairwise_sum:
        pairwise_sum = np.nansum(list(time_series.values()), axis=0)
        return time_series, pairwise_sum

    return time_series


def average_property_R_per_event(categorized_collisions: dict,
                                 property_function: Callable,
                                 *args,
                                 use_pairwise_sum: bool=False) -> dict:
    """
    Calculate the average of a distance-dependent property over each collision event.

    Usage: ``event_averages = average_property_R_per_event(categorized, A_of_R, use_pairwise_sum=True)``

    Parameters
    ----------
    categorized_collisions : dict
        Output of categorize_collisions().
    property_function : callable
        Function of a distance array in Å and the additional arguments that returns the property.
    *args
        Additional arguments passed to property_function.
    use_pairwise_sum : bool, default=False
        If True, the property of a frame is the sum over all partners. If False, only the main partner is used.

    Returns
    -------
    event_average_properties : dict
        Same nested structure as categorized_collisions, with each event replaced by a dictionary with the key
        'average_property', the mean of the property over the frames of the event.
    """

    # Nested structure of the per-event averages
    event_average_properties = {category: {atom_index: [] for atom_index in categorized_collisions[category]}
                                for category in categorized_collisions}

    # Go through the categories
    for category in categorized_collisions:

        # Go through the reference atoms of the category
        for atom_index in tqdm(categorized_collisions[category],
                               desc=f'Calculating average collision properties for {category}'):

            # Go through the events of the reference atom
            for event_index, _ in enumerate(categorized_collisions[category][atom_index]):

                # Property series summed over all partners or of the main partner only
                if use_pairwise_sum:
                    _, property_values = property_R_all_partners_time_series(categorized_collisions,
                                                                             category,
                                                                             atom_index,
                                                                             event_index,
                                                                             property_function,
                                                                             *args,
                                                                             use_pairwise_sum=True)
                else:
                    property_values = property_R_main_partner_time_series(categorized_collisions,
                                                                          category,
                                                                          atom_index,
                                                                          event_index,
                                                                          property_function,
                                                                          *args)

                # Mean of the property over the frames of the event
                average_value = np.nanmean(property_values)
                event_average_properties[category][atom_index].append({'average_property': average_value})

    return event_average_properties


def average_property_R_over_events(event_average_properties: dict) -> dict:
    """
    Average the per-event property averages over the events of each reference atom.

    Every event has the same weight, regardless of its lifetime.

    Usage: ``atom_averages = average_property_R_over_events(event_averages)``

    Parameters
    ----------
    event_average_properties : dict
        Output of average_property_R_per_event().

    Returns
    -------
    average_properties : dict
        Category mapped to a dictionary that maps the atom index of each reference atom to a dictionary with
        the keys 'average_property' (mean over the events) and 'total_num_events' (number of events).
    """

    # Summaries of each category
    average_properties = {}
    for category, atoms_dict in event_average_properties.items():
        average_properties[category] = {}

        # Mean over the events of each reference atom
        for atom_index, atom_events in atoms_dict.items():
            if atom_events:
                average_property_values = [event['average_property'] for event in atom_events]
                average_properties[category][atom_index] = {'average_property': np.mean(average_property_values),
                                                            'total_num_events': len(atom_events)}
            else:
                average_properties[category][atom_index] = {'average_property': np.nan,
                                                            'total_num_events': 0}

    return average_properties


def average_property_R_over_atoms(average_properties_over_events: dict) -> dict:
    """
    Average the per-atom property averages over the reference atoms of each category.

    Every reference atom with at least one event has the same weight.

    Usage: ``category_averages = average_property_R_over_atoms(atom_averages)``

    Parameters
    ----------
    average_properties_over_events : dict
        Output of average_property_R_over_events().

    Returns
    -------
    average_properties : dict
        Category mapped to a dictionary with the keys 'average_property' (mean over the reference atoms) and
        'total_num_events' (number of events in the category).
    """

    # Summaries of each category
    average_properties = {}
    for category, atoms_dict in average_properties_over_events.items():

        # Reference atoms with at least one event
        atom_summaries = [summary for summary in atoms_dict.values()
                          if summary['total_num_events'] > 0]

        # Mean over the reference atoms and the total number of events
        if atom_summaries:
            average_property_values = [summary['average_property']
                                       for summary in atom_summaries]
            total_event_counts = [summary['total_num_events']
                                  for summary in atom_summaries]
            average_properties[category] = {'average_property': np.mean(average_property_values),
                                            'total_num_events': np.sum(total_event_counts)}
        else:
            average_properties[category] = {'average_property': np.nan,
                                            'total_num_events': 0}

    return average_properties


def average_property_R(categorized_collisions: dict,
                       property_function: Callable,
                       *args,
                       average_over_events: bool=True,
                       average_over_atoms: bool=False,
                       use_pairwise_sum: bool=False) -> dict:
    """
    Calculate the event-weighted averages of a distance-dependent property for each category.

    The property is first averaged over each event (average_property_R_per_event()) and then, optionally, over
    the events of each reference atom and over the reference atoms. Every event, and every reference atom, has
    the same weight regardless of its lifetime. For averages consistent with the atom-time fractions, use
    average_property_R_time_weighted() instead.

    Usage: ``averages = average_property_R(categorized, A_of_R, average_over_atoms=True)``

    Parameters
    ----------
    categorized_collisions : dict
        Output of categorize_collisions().
    property_function : callable
        Function of a distance array in Å and the additional arguments that returns the property.
    *args
        Additional arguments passed to property_function.
    average_over_events : bool, default=True
        If True, average over the events of each reference atom. If False, return the per-event averages.
    average_over_atoms : bool, default=False
        If True, also average over the reference atoms of each category. Requires average_over_events=True.
    use_pairwise_sum : bool, default=False
        If True, the property of a frame is the sum over all partners. If False, only the main partner is used.

    Returns
    -------
    average_properties : dict
        Output of average_property_R_per_event(), average_property_R_over_events() or
        average_property_R_over_atoms(), depending on the averaging options.
    """

    # Average over each event
    event_average_properties = average_property_R_per_event(categorized_collisions,
                                                            property_function,
                                                            *args,
                                                            use_pairwise_sum=use_pairwise_sum)

    # Return the per-event averages if no further averaging is requested
    if not average_over_events:
        return event_average_properties

    # Average over the events of each reference atom
    average_properties = average_property_R_over_events(event_average_properties)

    # Optionally average over the reference atoms of each category
    if average_over_atoms:
        return average_property_R_over_atoms(average_properties)

    return average_properties


def average_property_R_time_weighted(categorized_collisions: dict,
                                     property_function: Callable,
                                     *args,
                                     use_pairwise_sum: bool=True,
                                     n_jobs: int=1) -> dict:
    """
    Calculate the time-weighted average of a distance-dependent property for each event category.

    The property values of all frames of all events in a category are pooled, so that each event contributes in
    proportion to its lifetime. This is the category average that is consistent with the atom-time fractions of
    atom_time_fractions(), and the two together reproduce the trajectory average of the property exactly.

    Usage: ``averages = average_property_R_time_weighted(categorized, A_of_R, n_jobs=8)``

    Parameters
    ----------
    categorized_collisions : dict
        Output of categorize_collisions().
    property_function : callable
        Function of a distance array in Å and the additional arguments that returns the property. It must return
        NaN for NaN distances.
    *args
        Additional arguments passed to property_function.
    use_pairwise_sum : bool, default=True
        If True, the property of a frame is the sum over all partners within the cutoff (pairwise additivity).
        If False, only the main partner is used.
    n_jobs : int, default=1
        Number of parallel processes over the reference atoms.

    Returns
    -------
    average_properties : dict
        Category mapped to a dictionary with the keys

        - 'average_property': time-weighted average of the property in the category,
        - 'total_num_events': number of events in the category,
        - 'total_frames': number of frames of all events in the category, and
        - 'property_sum': sum of the property over all these frames, for pooling several trajectories.
    """

    def process_atom(events: list) -> tuple[float, int, int]:
        """
        Sum the property over all frames of the events of one reference atom.

        Only the event list is passed, so that the parallel workers do not receive the whole dictionary.

        Parameters
        ----------
        events : list
            Events of the reference atom as DataFrames.

        Returns
        -------
        property_sum : float
            Sum of the property over all frames of the events.
        num_frames : int
            Number of frames of the events.
        num_events : int
            Number of events.
        """

        # Accumulate the property sum and the number of frames over the events
        property_sum = 0.0
        num_frames = 0
        for event_df in events:

            # Property of each partner in each frame, or of the main partner only
            if use_pairwise_sum:
                partner_distances = distance_series(event_df)
                property_values = [property_function(distances, *args) for distances in partner_distances.values()]
            else:
                property_values = property_function(main_partner_distance_series(event_df), *args)

            # Sum over the frames and partners, ignoring the NaN values of absent partners
            property_sum += np.nansum(property_values)
            num_frames += len(event_df)

        return property_sum, num_frames, len(events)

    # Time-weighted average of each category
    average_properties = {}
    for category in categorized_collisions:

        # Property sums of all reference atoms of the category in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(process_atom)(events)
                                          for events in tqdm(categorized_collisions[category].values(),
                                                             desc=f'Time-weighted averages for {category}'))

        # Pool the sums and frame counts of all reference atoms
        property_sum = sum(result[0] for result in results)
        total_frames = sum(result[1] for result in results)
        total_num_events = sum(result[2] for result in results)

        average_properties[category] = {'average_property': property_sum / total_frames if total_frames > 0
                                                            else np.nan,
                                        'total_num_events': total_num_events,
                                        'total_frames': total_frames,
                                        'property_sum': property_sum}

    return average_properties


def average_property_R_statistics(average_properties: dict,
                                  equilibrium_fractions: dict,
                                  collision_stats: dict) -> pd.DataFrame:
    """
    Tabulate the event-weighted category averages of a property with lifetime statistics and their
    contributions to the ensemble average.

    The contribution of each category is its average property multiplied by its equilibrium fraction. The final
    row 'all' gives the sum of the contributions and the average property of the events of all categories.

    Usage: ``table = average_property_R_statistics(averages, fractions, collision_stats)``

    Parameters
    ----------
    average_properties : dict
        Output of average_property_R() with average_over_events=True and average_over_atoms=True, so that each
        category maps to a dictionary with the keys 'average_property' and 'total_num_events'.
    equilibrium_fractions : dict
        Output of cluster_formation_equilibrium_values(). The category fractions weight the contributions.
    collision_stats : dict
        Output of collision_statistics(), used for the event lifetimes.

    Returns
    -------
    property_statistics_df : DataFrame
        One row per category and a final row 'all', with the columns 'category', 'average_property',
        'total_num_events', 'average_lifetime_ps', 'equilibrium_fraction' and 'weighted_average'. The
        floating-point columns are formatted as strings in scientific notation.
    """

    # Rows of the output table
    property_statistics_rows = []

    # Summary statistics of each category
    for category, category_summary in average_properties.items():

        # Event lifetimes and equilibrium fraction of the category
        category_collision_stats = collision_stats.get(category, {})
        event_lifetimes = [event['lifetime_ps']
                           for stats_list in category_collision_stats.values()
                           for event in stats_list]
        equilibrium_fraction = equilibrium_fractions.get(category, np.nan)

        # Mean lifetime and the contribution to the ensemble average
        average_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan
        weighted_average = category_summary['average_property'] * equilibrium_fraction

        property_statistics_rows.append({'category': category,
                                         'average_property': category_summary['average_property'],
                                         'total_num_events': category_summary['total_num_events'],
                                         'average_lifetime_ps': average_lifetime_ps,
                                         'equilibrium_fraction': equilibrium_fraction,
                                         'weighted_average': weighted_average})

    # Sum of the fractions and of the contributions over the categories
    equilibrium_fraction_values = [row['equilibrium_fraction'] for row in property_statistics_rows]
    all_equilibrium_fraction = np.sum(equilibrium_fraction_values)
    all_weighted_average = np.sum([row['average_property'] * row['equilibrium_fraction']
                                   for row in property_statistics_rows])

    # Mean lifetime over the events of all categories
    event_lifetimes = [event['lifetime_ps']
                       for category_stats in collision_stats.values()
                       for stats_list in category_stats.values()
                       for event in stats_list]
    all_average_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan

    # Average property of the events of all categories
    all_average_property = all_weighted_average / all_equilibrium_fraction \
        if all_equilibrium_fraction > 0 else np.nan

    # Final row for all categories
    property_statistics_rows.append({'category': 'all',
                                     'average_property': all_average_property,
                                     'total_num_events': np.sum([row['total_num_events']
                                                                 for row in property_statistics_rows]),
                                     'average_lifetime_ps': all_average_lifetime_ps,
                                     'equilibrium_fraction': all_equilibrium_fraction,
                                     'weighted_average': all_weighted_average})

    # Convert to a DataFrame and format the floating-point columns in scientific notation
    property_statistics_df = pd.DataFrame(property_statistics_rows)
    for column in ['average_property',
                   'average_lifetime_ps',
                   'equilibrium_fraction',
                   'weighted_average']:
        # property_statistics_df[column] = property_statistics_df[column].map(lambda value: f'{value:.2e}')
        property_statistics_df[column] = property_statistics_df[column].map(lambda value: f'{value:.3e}')

    return property_statistics_df

# def average_property_R_statistics(average_properties,
#                                   collision_stats,
#                                   dt,
#                                   total_frames):
#     """
#     Calculate category-level lifetime statistics and weighted average
#     contributions for distance-dependent properties.

#     This function assumes that average_property_R() was called with
#     average_over_events=True and average_over_atoms=True, so that each
#     category maps directly to a category-level summary.

#     Inputs:
#     - average_properties: dict
#         Output of average_property_R() when both average_over_events and
#         average_over_atoms are True. Each category value must contain the
#         keys 'average_property' and 'total_num_events'.
#     - collision_stats: dict
#         Output of collision_statistics(). Used to obtain event lifetimes for
#         the trajectory-fraction statistics.
#     - dt: float
#         The time step between frames in seconds.
#     - total_frames: int
#         The total number of frames in the trajectory.

#     Returns:
#     - property_statistics_df: pandas DataFrame
#         A DataFrame containing one row for each event category and one final
#         all row. The columns are:
#         - 'category': The event category label.
#         - 'average_property': The average property value for the category.
#                 - 'total_num_events': The total number of events in the category.
#         - 'average_lifetime_ps': The average event lifetime in picoseconds.
#         - 'total_lifetime_ps': The total event lifetime in picoseconds summed
#           over all atoms.
#         - 'fraction_of_trajectory': The fraction of trajectory assigned to
#           the category.
#         - 'weighted_average': The contribution of the category to the
#           ensemble average.
#     """
#     # Calculate the total trajectory length in picoseconds from dt and the
#     # total number of frames.
#     total_trajectory_length_ps = total_frames * dt * 1e12

#     # Initialise a list of rows for the output table.
#     property_statistics_rows = []

#     # Calculate one set of summary statistics for each category.
#     for category, category_summary in average_properties.items():
#         category_collision_stats = collision_stats.get(category, {})
#         atom_lifetime_totals = [np.sum([event['lifetime_ps'] for event in stats_list])
#                                for stats_list in category_collision_stats.values()
#                                if stats_list]
#         event_lifetimes = [event['lifetime_ps']
#                            for stats_list in category_collision_stats.values()
#                            for event in stats_list]

#         average_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan
#         total_lifetime_ps = np.sum(event_lifetimes) if event_lifetimes else 0.0
#         fraction_of_trajectory = np.mean([lifetime / total_trajectory_length_ps
#                                           for lifetime in atom_lifetime_totals]) if atom_lifetime_totals else np.nan
#         weighted_average = category_summary['average_property'] * fraction_of_trajectory

#         # Store the category contribution as one row of the output table.
#         property_statistics_rows.append({'category': category,
#                                         'average_property': category_summary['average_property'],
#                                         'total_num_events': category_summary['total_num_events'],
#                                         'average_lifetime_ps': average_lifetime_ps,
#                                         'total_lifetime_ps': total_lifetime_ps,
#                                         'fraction_of_trajectory': fraction_of_trajectory,
#                                         'weighted_average': weighted_average})

#     # Sum the category contributions to obtain the ensemble-average
#     # contribution from all event categories.
#     trajectory_fractions = [row['fraction_of_trajectory'] for row in property_statistics_rows]
#     weighted_averages = [row['weighted_average'] for row in property_statistics_rows]
#     all_fraction_of_trajectory = np.sum(trajectory_fractions)
#     all_weighted_average = np.sum(weighted_averages)
#     all_total_lifetime_ps = np.sum([row['total_lifetime_ps'] for row in property_statistics_rows])

#     event_lifetimes = [event['lifetime_ps']
#                        for category_stats in collision_stats.values()
#                        for stats_list in category_stats.values()
#                        for event in stats_list]
#     all_average_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan

#     # Calculate the all-category average property represented by the included
#     # event categories.
#     all_average_property = all_weighted_average / all_fraction_of_trajectory \
#         if all_fraction_of_trajectory > 0 else np.nan

#     # Append a final row containing the all-category weighted average.
#     property_statistics_rows.append({'category': 'all',
#                                      'average_property': all_average_property,
#                                      'total_num_events': np.sum([row['total_num_events'] for row in property_statistics_rows]),
#                                      'average_lifetime_ps': all_average_lifetime_ps,
#                                      'total_lifetime_ps': all_total_lifetime_ps,
#                                      'fraction_of_trajectory': all_fraction_of_trajectory,
#                                      'weighted_average': all_weighted_average})

#     # Convert the table to a DataFrame and format the numeric columns for
#     # display in scientific notation with two decimals.
#     property_statistics_df = pd.DataFrame(property_statistics_rows)
#     for column in ['average_property',
#                    'average_lifetime_ps',
#                    'total_lifetime_ps',
#                    'fraction_of_trajectory',
#                    'weighted_average']:
#         property_statistics_df[column] = property_statistics_df[column].map(lambda value: f'{value:.2e}')

#     return property_statistics_df


###################################################################################################################
# Energetically bound clusters
###################################################################################################################
def hill_bound_clusters(positions: np.ndarray,
                        velocities: np.ndarray,
                        masses: np.ndarray,
                        box: np.ndarray,
                        cutoff: float,
                        pair_potential: Callable,
                        kinetic_energy_factor: float,
                        min_size: int=2,
                        criterion: str='cluster') -> list[tuple[int, ...]]:
    """
    Find the energetically bound clusters of atoms in one frame.

    A pair of atoms i and j within the cutoff is bound if its energy of relative motion is negative,

    .. math::

        E_{ij} = \\frac{1}{2} \\mu_{ij} |\\mathbf{v}_i - \\mathbf{v}_j|^2 + V(r_{ij}) < 0,

    where mu_ij is the reduced mass and V the pair potential (T. L. Hill, J. Chem. Phys. 23, 617 (1955)). The
    connected components of the graph of bound pairs are the bound clusters of the 'pair' criterion.

    With the 'cluster' criterion, these components are the cores that grow by the atoms bound to them. An atom k
    within the cutoff of a member of cluster C is bound to C if its energy of motion relative to C is negative,

    .. math::

        E_{k} = \\frac{1}{2} \\mu_{kC} |\\mathbf{v}_k - \\mathbf{v}_C|^2 + \\sum_{j \\in C} V(r_{kj}) < 0,

    where mu_kC is the reduced mass of k and C, v_C the centre-of-mass velocity of C, and the sum runs over the
    members within the cutoff. The most strongly bound atom is added first, and the growth is repeated until no
    further atom is bound. This criterion counts a third atom bound to a dimer as a whole, even when its pair
    energy with one of the two atoms is momentarily positive. For a pair, both criteria coincide. Distances
    follow the minimum-image convention of the periodic box.

    Usage: ``clusters = hill_bound_clusters(positions, velocities, masses, box, 9.0, V, factor, min_size=3)``

    Parameters
    ----------
    positions : ndarray
        Positions of the atoms with the shape (N, 3).
    velocities : ndarray
        Velocities of the atoms with the shape (N, 3).
    masses : ndarray
        Masses of the atoms with the shape (N,).
    box : ndarray
        Box dimensions [lx, ly, lz, alpha, beta, gamma] in the MDAnalysis format.
    cutoff : float
        Largest distance of a pair considered as bound, in the units of the positions.
    pair_potential : callable
        Vectorised pair potential V(r) in the energy units of the result.
    kinetic_energy_factor : float
        Factor converting mass times velocity squared into the energy units of pair_potential, e.g.
        1.2027e6 K per amu Å^2 fs^-2.
    min_size : int, default=2
        Smallest number of atoms in a returned cluster.
    criterion : str, default='cluster'
        'pair' for the bound-pair components of Hill, or 'cluster' for these components grown by the atoms bound
        to them as a whole.

    Returns
    -------
    clusters : list of tuple
        Indices (into the arrays) of the atoms of each bound cluster with at least min_size atoms, in ascending
        order within each cluster.
    """

    # Pairs within the cutoff and their distances with the minimum-image convention
    pairs, pair_distances = mda_distances.self_capped_distance(positions, cutoff, box=box)
    if len(pairs) < min_size - 1:
        return []

    # Energy of relative motion of each pair
    i, j = pairs[:, 0], pairs[:, 1]
    reduced_masses = masses[i] * masses[j] / (masses[i] + masses[j])
    relative_speed_squared = np.sum((velocities[i] - velocities[j])**2, axis=1)
    pair_energies = 0.5*kinetic_energy_factor*reduced_masses*relative_speed_squared + pair_potential(pair_distances)

    # Graph of the bound pairs, which needs at least one bound pair for the cluster criterion
    bound = pair_energies < 0
    min_bound_pairs = min_size - 1 if criterion == 'pair' else 1
    if np.count_nonzero(bound) < min_bound_pairs:
        return []
    n_atoms = len(positions)
    graph = coo_matrix((np.ones(np.count_nonzero(bound)), (i[bound], j[bound])), shape=(n_atoms, n_atoms))

    # Connected components of the bound pairs, which are the clusters of the pair criterion
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    cores = [set(int(k) for k in np.flatnonzero(labels == label)) for label in np.flatnonzero(sizes >= 2)]
    if criterion == 'pair':
        return [tuple(sorted(core)) for core in cores if len(core) >= min_size]

    # Neighbours within the cutoff and the pair distances, for growing the cores
    neighbours = {}
    distance_of_pair = {}
    for a, b, distance in zip(i, j, pair_distances):
        neighbours.setdefault(int(a), set()).add(int(b))
        neighbours.setdefault(int(b), set()).add(int(a))
        distance_of_pair[(int(a), int(b))] = distance
        distance_of_pair[(int(b), int(a))] = distance

    # Grow each core by the atom most strongly bound to it, until no further atom is bound
    clusters = []
    for cluster in cores:
        while True:
            members = np.array(sorted(cluster))
            cluster_mass = np.sum(masses[members])
            cluster_velocity = np.sum(masses[members, None]*velocities[members], axis=0) / cluster_mass

            # Energy of motion relative to the cluster of each candidate atom within the cutoff of a member
            candidates = set().union(*(neighbours.get(member, set()) for member in cluster)) - cluster
            best_atom, best_energy = None, 0.0
            for k in candidates:
                reduced_mass = masses[k] * cluster_mass / (masses[k] + cluster_mass)
                kinetic = 0.5*kinetic_energy_factor*reduced_mass*np.sum((velocities[k] - cluster_velocity)**2)
                member_distances = np.array([distance_of_pair[(k, member)] for member in cluster
                                             if (k, member) in distance_of_pair])
                energy = kinetic + np.sum(pair_potential(member_distances))
                if energy < best_energy:
                    best_atom, best_energy = k, energy

            # Add the most strongly bound atom, or stop if none is bound
            if best_atom is None:
                break
            cluster.add(best_atom)
        clusters.append(cluster)

    # Merge grown clusters that share atoms and keep those with at least min_size atoms
    merged = []
    for cluster in clusters:
        for other in [other for other in merged if other & cluster]:
            cluster |= other
            merged.remove(other)
        merged.append(cluster)
    return [tuple(sorted(cluster)) for cluster in merged if len(cluster) >= min_size]


def find_bound_clusters(u: mda.Universe,
                        group: mda.AtomGroup,
                        cutoff: float,
                        pair_potential: Callable,
                        kinetic_energy_factor: float,
                        min_size: int=3,
                        criterion: str='cluster') -> dict:
    """
    Find the energetically bound clusters of a group of atoms in every frame of the trajectory.

    The clusters are defined by the energy criteria of hill_bound_clusters(). The trajectory must contain
    velocities.

    Usage: ``frame_clusters = find_bound_clusters(u, xenon, 9.0, V, 1.2027e6, min_size=3)``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory with velocities.
    group : AtomGroup
        Atoms considered.
    cutoff : float
        Largest distance of a pair considered as bound, in Å.
    pair_potential : callable
        Vectorised pair potential V(r) of r in Å.
    kinetic_energy_factor : float
        Factor converting mass times velocity squared, in the units of the trajectory, into the energy units of
        pair_potential.
    min_size : int, default=3
        Smallest number of atoms in a recorded cluster.
    criterion : str, default='cluster'
        'pair' or 'cluster' (see hill_bound_clusters()).

    Returns
    -------
    frame_clusters : dict
        Frame number mapped to the list of bound clusters in that frame, each a tuple of atom indices of the
        Universe. Frames without clusters of at least min_size atoms are omitted.
    """

    # Masses and atom indices of the group
    masses = group.masses
    atom_indices = group.indices

    # Bound clusters in each frame, converted from group positions to atom indices
    frame_clusters = {}
    for ts in tqdm(u.trajectory, desc='Finding bound clusters'):
        clusters = hill_bound_clusters(group.positions, group.velocities, masses, ts.dimensions, cutoff,
                                       pair_potential, kinetic_energy_factor, min_size=min_size,
                                       criterion=criterion)
        if clusters:
            frame_clusters[ts.frame] = [tuple(int(atom_indices[k]) for k in cluster) for cluster in clusters]

    return frame_clusters


def bound_cluster_events(frame_clusters: dict,
                         dt: float,
                         max_gap_frames: int=0) -> list[dict]:
    """
    Link the bound clusters of consecutive frames into cluster events.

    A cluster continues an ongoing event if it shares at least one atom with the cluster of that event in the
    last frame where the event was present, and if at most max_gap_frames frames without such a cluster lie in
    between. A cluster that shares atoms with several ongoing events merges them. Short gaps allow for the
    momentary breaking of a single bound pair inside a persisting cluster.

    Usage: ``events = bound_cluster_events(frame_clusters, 25e-15, max_gap_frames=40)``

    Parameters
    ----------
    frame_clusters : dict
        Output of find_bound_clusters().
    dt : float
        Time step between frames in seconds.
    max_gap_frames : int, default=0
        Largest number of consecutive frames without the cluster within one event.

    Returns
    -------
    events : list of dict
        Cluster events sorted by their first frame. Each is a dictionary with the keys

        - 'start_frame' and 'end_frame': first and last frame with the cluster,
        - 'lifetime_ps': time from the first to the last frame with the cluster, inclusive, in ps,
        - 'atoms': tuple of all atom indices that belonged to the cluster during the event,
        - 'max_size': largest number of atoms in the cluster in one frame,
        - 'frames': array of the frames with the cluster, and
        - 'sizes': array of the number of atoms in the cluster in those frames.
    """

    # Ongoing events, each with its frames, sizes, atoms and the atoms of its last frame
    active = []
    finished = []
    for frame in sorted(frame_clusters):

        # Finish the events whose gap has become too long
        still_active = []
        for event in active:
            if frame - event['frames'][-1] > max_gap_frames + 1:
                finished.append(event)
            else:
                still_active.append(event)
        active = still_active

        # Attach each cluster of this frame to the overlapping ongoing events, merging them if there are several
        for cluster in frame_clusters[frame]:
            cluster_atoms = set(cluster)
            overlapping = [event for event in active if event['last_atoms'] & cluster_atoms]

            # Start a new event if the cluster overlaps none
            if not overlapping:
                active.append({'frames': [frame], 'sizes': [len(cluster)], 'atoms': set(cluster_atoms),
                               'last_atoms': set(cluster_atoms)})
                continue

            # Merge all overlapping events into the earliest one
            event = min(overlapping, key=lambda item: item['frames'][0])
            for other in overlapping:
                if other is event:
                    continue
                merged = sorted(zip(event['frames'] + other['frames'], event['sizes'] + other['sizes']))
                event['frames'] = [f for f, _ in merged]
                event['sizes'] = [s for _, s in merged]
                event['atoms'] |= other['atoms']
                active.remove(other)

            # Add the cluster of this frame, keeping the larger cluster if the frame is already present
            if event['frames'][-1] == frame:
                event['sizes'][-1] = max(event['sizes'][-1], len(cluster))
                event['last_atoms'] |= cluster_atoms
            else:
                event['frames'].append(frame)
                event['sizes'].append(len(cluster))
                event['last_atoms'] = set(cluster_atoms)
            event['atoms'] |= cluster_atoms

    # Finish the events still ongoing after the last frame
    finished.extend(active)

    # Summary of each event, sorted by the first frame
    events = []
    for event in sorted(finished, key=lambda item: item['frames'][0]):
        frames = np.array(event['frames'])
        events.append({'start_frame': int(frames[0]),
                       'end_frame': int(frames[-1]),
                       'lifetime_ps': (frames[-1] - frames[0] + 1) * dt * 1e12,
                       'atoms': tuple(sorted(event['atoms'])),
                       'max_size': int(max(event['sizes'])),
                       'frames': frames,
                       'sizes': np.array(event['sizes'])})

    return events


def bound_cluster_time_series(u: mda.Universe,
                              atom_indices: tuple[int, ...],
                              start_frame: int,
                              stop_frame: int,
                              pair_potential: Callable,
                              kinetic_energy_factor: float) -> dict:
    """
    Calculate the pair distances and the binding energies of a set of atoms over a range of frames.

    The pair energy is the energy of relative motion of the pair (see hill_bound_clusters()). The atom energy is
    the energy of motion of one atom relative to the rest of the set, i.e. the kinetic energy with the reduced
    mass of the atom and the rest, relative to the centre-of-mass velocity of the rest, plus the pair potentials
    with the other atoms. An atom is bound to the others where it is negative. The internal energy is the
    kinetic energy in the centre-of-mass frame of the set plus the sum of all pair potentials. Distances follow
    the minimum-image convention.

    Usage: ``series = bound_cluster_time_series(u, (12, 57, 140), 1000, 5000, V, 1.2027e6)``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory with velocities.
    atom_indices : tuple of int
        Atom indices of the set.
    start_frame : int
        First frame.
    stop_frame : int
        Last frame, inclusive.
    pair_potential : callable
        Vectorised pair potential V(r) of r in Å.
    kinetic_energy_factor : float
        Factor converting mass times velocity squared, in the units of the trajectory, into the energy units of
        pair_potential.

    Returns
    -------
    series : dict
        Dictionary with the keys

        - 'frames': array of the frame numbers,
        - 'pairs': list of the atom-index pairs (i, j) with i < j, in the order of the columns below,
        - 'distances': array of the pair distances in Å with the shape (N_frames, N_pairs),
        - 'pair_energies': array of the pair energies with the shape (N_frames, N_pairs),
        - 'atoms': list of the atom indices, in the order of the columns of 'atom_energies',
        - 'atom_energies': array of the atom energies relative to the rest with the shape (N_frames, N_atoms), and
        - 'internal_energy': array of the internal energy of the set with the shape (N_frames,).
    """

    # Atoms of the set, their masses and their pairs in the order of the condensed distance array
    atoms = u.atoms[list(atom_indices)]
    masses = atoms.masses
    i, j = np.triu_indices(len(atoms), k=1)
    reduced_masses = masses[i] * masses[j] / (masses[i] + masses[j])

    # Number and total mass of the atoms of the set
    n_set = len(atoms)
    total_mass = np.sum(masses)

    # Distances and energies in each frame
    frames, all_distances, all_pair_energies, all_atom_energies, internal_energies = [], [], [], [], []
    for ts in u.trajectory[start_frame:stop_frame + 1]:
        positions = atoms.positions
        velocities = atoms.velocities

        # Pair distances with the minimum-image convention and the pair potentials
        pair_distances = mda_distances.self_distance_array(positions, box=ts.dimensions)
        pair_potentials = pair_potential(pair_distances)

        # Energies of relative motion of the pairs
        relative_speed_squared = np.sum((velocities[i] - velocities[j])**2, axis=1)
        pair_energies = 0.5*kinetic_energy_factor*reduced_masses*relative_speed_squared + pair_potentials

        # Energy of each atom relative to the rest: reduced-mass kinetic energy and pair potentials with the rest
        potential_matrix = np.zeros((n_set, n_set))
        potential_matrix[i, j] = pair_potentials
        potential_matrix[j, i] = pair_potentials
        momentum = np.sum(masses[:, None]*velocities, axis=0)
        rest_masses = total_mass - masses
        rest_velocities = (momentum[None, :] - masses[:, None]*velocities) / rest_masses[:, None]
        atom_reduced_masses = masses * rest_masses / total_mass
        atom_kinetic = 0.5*kinetic_energy_factor*atom_reduced_masses*np.sum((velocities - rest_velocities)**2, axis=1)
        atom_energies = atom_kinetic + np.sum(potential_matrix, axis=1)

        # Internal energy: kinetic energy in the centre-of-mass frame plus all pair potentials
        com_velocity = momentum / total_mass
        kinetic_energy = 0.5*kinetic_energy_factor*np.sum(masses*np.sum((velocities - com_velocity)**2, axis=1))

        frames.append(ts.frame)
        all_distances.append(pair_distances)
        all_pair_energies.append(pair_energies)
        all_atom_energies.append(atom_energies)
        internal_energies.append(kinetic_energy + np.sum(pair_potentials))

    return {'frames': np.array(frames),
            'pairs': [(int(atom_indices[a]), int(atom_indices[b])) for a, b in zip(i, j)],
            'distances': np.array(all_distances),
            'pair_energies': np.array(all_pair_energies),
            'atoms': [int(atom) for atom in atom_indices],
            'atom_energies': np.array(all_atom_energies),
            'internal_energy': np.array(internal_energies)}


###################################################################################################################
# Trajectory extraction functions using categorized events
###################################################################################################################
# TODO: Human check this function
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


###################################################################################################################
# Data saving functions
###################################################################################################################
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


###################################################################################################################
# Miscellaneous helper functions
###################################################################################################################
def sort_collision_statistics(collision_stats: dict,
                              property_name: str,
                              atomwise: bool=True,
                              reverse: bool=True) -> dict:
    """
    Sort the event statistics by a given field.

    Usage: ``sorted_stats = sort_collision_statistics(collision_stats, 'lifetime_ps', atomwise=False)``

    Parameters
    ----------
    collision_stats : dict
        Output of collision_statistics() or combine_collision_statistics().
    property_name : str
        Field of the event statistics to sort by, e.g. 'lifetime_ps'.
    atomwise : bool, default=True
        If True, sort the events of each reference atom separately and keep the nested structure. If False,
        pool the events of all reference atoms of each category into one sorted list.
    reverse : bool, default=True
        If True, sort in descending order. If False, sort in ascending order.

    Returns
    -------
    sorted_stats : dict
        If atomwise is True, the same nested structure as collision_stats with sorted event lists. If atomwise
        is False, category mapped to a sorted list of event statistics, each with the additional key
        'atom_index' for its reference atom.
    """

    # Sorted statistics of each category
    sorted_stats = {}
    for category, atoms_dict in collision_stats.items():

        # Sort the events of each reference atom separately
        if atomwise:
            sorted_stats[category] = {}
            for atom_index, stats_list in atoms_dict.items():
                sorted_stats[category][atom_index] = sorted(stats_list,
                                                            key=lambda x: x[property_name],
                                                            reverse=reverse)

        # Pool the events of all reference atoms, recording the reference atom of each, and sort them
        else:
            combined_stats = []
            for atom_index, stats_list in atoms_dict.items():
                for event_stats in stats_list:
                    event_stats_with_atom = dict(event_stats)
                    event_stats_with_atom['atom_index'] = atom_index
                    combined_stats.append(event_stats_with_atom)
            sorted_stats[category] = sorted(combined_stats,
                                            key=lambda x: x[property_name],
                                            reverse=reverse)

    return sorted_stats
