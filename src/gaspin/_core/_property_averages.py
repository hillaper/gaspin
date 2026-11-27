"""
Time series and time-weighted category averages of properties along the classified collision events.
"""

# Imports
from collections.abc import Callable

import MDAnalysis as mda
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from gaspin._core._collisions import distance_series, main_partner_distance_series, main_partner_index
from gaspin._core._trajectory import center_and_wrap


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
