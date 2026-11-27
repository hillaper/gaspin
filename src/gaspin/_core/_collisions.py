"""
Detection, grouping and classification of the collision events of an atomic gas.

find_collisions() records, for every frame, the atoms within a cutoff distance of each reference atom.
process_collisions() groups these records into events, and categorize_collisions() classifies the events into
two-body scattering (TBS), multi-body scattering (MBS), perturbed scattering (PS) and van der Waals (vdW)
events. All distances are in Å.
"""

# Imports
from collections import Counter

import MDAnalysis as mda
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from MDAnalysis.analysis import distances
from scipy.signal import find_peaks
from tqdm import tqdm


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
        If given, only these atoms anchor events, and frames in which none of them is within the cutoff are
        skipped.

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

            # Partners that may anchor an event, and skip the frame if there are none
            if allowed_set is None:
                candidates = collision['atoms2']
            else:
                candidates = [atom for atom in collision['atoms2'] if atom in allowed_set]
            if not candidates:
                continue

            # End the ongoing event at a frame gap or when its anchor has left the cutoff sphere
            if current_event and (frame - previous_frame > 1 or first_atom not in candidates):
                events.append(current_event)
                current_event = []

            # Anchor a new event to the first eligible partner of its first frame
            if not current_event:
                first_atom = candidates[0]

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
                          vdw_bond_length: float,
                          well_tolerance: float,
                          distance_minima_prominence: float,
                          vdw_well_occupancy: float | None=0.75,
                          n_jobs: int=1,
                          allowed_atom_indices: list[int] | None=None) -> dict:
    """
    Classify the collision events into two-body scattering, multi-body scattering, perturbed scattering and
    van der Waals events.

    The classification uses the distance R(t) between the reference atom and the main partner (see
    main_partner_index()) and involves no lifetime criterion. With the well region defined as
    vdw_bond_length ± well_tolerance, the rules are applied in the following order:

    1. vdW if the fraction of the frames with R(t) in the well region is at least vdw_well_occupancy. This
       catches bound pairs on nearly circular orbits, whose radial oscillations are too shallow to be
       resolved as distance minima.
    2. Otherwise, the local minima of R(t) with the given prominence are counted (N_min), together with the
       minima below vdw_bond_length + well_tolerance (n_min). The event is

       - two-body scattering if N_min <= 1 and only one partner is within the cutoff in every frame,
       - multi-body scattering if N_min <= 1 and two or more partners are within the cutoff in some frame,
       - perturbed scattering if N_min > 1 and n_min <= 2, and
       - vdW if N_min > 1 and n_min > 2.

    The well tolerance and the prominence are length scales of the pair potential and the collision dynamics,
    and they are therefore chosen for each gas.

    Usage: ``categorized = categorize_collisions(events, 9.0, 4.378, 0.75, 0.25)``

    Parameters
    ----------
    processed_collisions : dict
        Output of process_collisions() or process_collisions_partner_specific(). Both are valid input but give
        different results: process_collisions() uses the first-partner rule, whereas
        process_collisions_partner_specific() creates partner-specific events.
    cutoff : float
        Cutoff distance used in the collision detection in Å.
    vdw_bond_length : float
        Distance of the minimum of the pair potential in Å.
    well_tolerance : float
        Half-width of the well region around vdw_bond_length in Å.
    distance_minima_prominence : float
        Minimum prominence of a local minimum of R(t) in Å.
    vdw_well_occupancy : float or None, default=0.75
        Minimum fraction of the frames in the well region for the direct vdW assignment of rule 1. None
        disables the rule.
    n_jobs : int, default=1
        Number of parallel processes over the reference atoms. The value -1 uses all available CPUs.
    allowed_atom_indices : list of int, default=None
        If given, events in which none of these atoms is a partner are skipped, and only these atoms are
        considered as the main partner.

    Returns
    -------
    categorized_collisions : dict
        Dictionary with the keys 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering' and
        'vdw'. Each maps the atom index of a reference atom to the list of its events in that category, as
        DataFrames. Reference atoms without events in a category are omitted from it. Every processed event,
        including short events without a resolved minimum, is assigned to one category.
    """

    # Well region around the minimum of the pair potential
    vdw_bond_threshold = (vdw_bond_length - well_tolerance, vdw_bond_length + well_tolerance)

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
            num_minima_below_vdw = np.sum(main_atom_distances[minima_indices] < vdw_bond_threshold[1])

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
