"""
Statistics of the classified collision events: lifetimes, numbers of atoms, cluster counts, equilibrium
constants, formation rates and atom-time fractions. Lifetimes are in ps and the time step in seconds.
"""

# Imports
import numpy as np
from tqdm import tqdm

from gaspin._core._collisions import main_partner_distance_series


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
                                         n_atoms: int,
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
    n_atoms : int
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
        avg_free_monomers = n_atoms - total_bound_monomers

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
        equilibrium_fractions[category] = avg_bound_monomers / n_atoms if n_atoms > 0 else 0

    # Fraction of monomers
    equilibrium_fractions['free_monomers'] = avg_free_monomers / n_atoms if n_atoms > 0 else 0

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
    frame, and therefore underestimates the rates of frequent events, increasingly so for longer intervals
    between the frames. The numbers are divided by the time spanned by the frames, (N_frames - 1) dt.

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

    For events from process_collisions(), each frame of a reference atom belongs to at most one of its events, so
    these fractions and the remaining fraction of free atoms sum to unity. Together with the time-weighted
    category averages of average_property_R_time_weighted(), they then decompose the trajectory average of a
    single-atom property exactly:

    .. math::

        \\langle A \\rangle = \\sum_c f_c \\langle A_c \\rangle.

    The partner-specific events of process_collisions_partner_specific() overlap in time, so that a frame can be
    counted several times. The fractions then do not sum to unity, the fraction of free atoms is not
    meaningful, and the decomposition does not hold.

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
