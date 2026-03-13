"""
MD trajectory analysis tool for gas phase MD simulations.
"""

###########################################################################################################
# Imports
###########################################################################################################
import numpy as np
import pandas as pd
import os

from scipy.signal import find_peaks

from tqdm import tqdm

from MDAnalysis.analysis import rdf
from MDAnalysis.analysis import distances
import MDAnalysis.transformations as mda_trans

from joblib import Parallel, delayed
from collections import Counter

###########################################################################################################
# Trajectory transformations
###########################################################################################################
def center_and_wrap(u, ts, group):
    """
    Wrap the coordinates back into the box and center the trajectory
    on the center of mass of the selected group.

    Parameters:
    - u: MDAnalysis Universe
        The MDAnalysis Universe object containing the trajectory and topology.
    - ts: MDAnalysis Timestep
        The current timestep of the trajectory to be transformed.
    - group: MDAnalysis AtomGroup
        The group of atoms to be centered.

    Returns:
    - ts: MDAnalysis Timestep
        The transformed timestep.
    """
    # NOTE: This is the correct order of transformations
    # Transform the trajectory by centering the selected group
    mda_trans.center_in_box(group)(ts)

    # Wrap the coordinates back into the box
    mda_trans.wrap(u.atoms)(ts)

###########################################################################################################
# General functions
###########################################################################################################
def rdf(group1,
        group2,
        nbins = 100,
        range=(0.0, 10.0)):
    """
    Calculate the radial distribution function (RDF) between two groups of atoms in a trajectory.

    Parameters:
    - group1: MDAnalysis AtomGroup
    The first group of atoms.
    - group2: MDAnalysis AtomGroup
    The second group of atoms.
    - nbins: int
    The number of bins to use for the RDF histogram (default is 100).
    - range: tuple
    The range of distances to consider for the RDF histogram (default is (0.0, 10.0)).

    Returns:
    - r: numpy array
    The array of distance values corresponding to the RDF bins.
    - g_r: numpy array
    The radial distribution function values for each bin.
    """
    # Create an RDF object and run the analysis
    if group1 == group2:
        rdf_analysis = rdf.InterRDF(
                        group1, 
                        group2, 
                        exclusion_block=(1, 1),  # Exclude self-pairs for the same group
                        range=range,
                        nbins=nbins)
    else:
        rdf_analysis = rdf.InterRDF(
                        group1, 
                        group2, 
                        range=range, 
                        nbins=nbins)
    rdf_analysis.run()

    return rdf_analysis.results.bins, rdf_analysis.results.rdf

def angular_momentum(group1, 
                     group2):
    """
    Calculate the angular momentum of two groups of atoms around their 
    combined center of mass for the current timestep.

    Parameters:
    - group1: MDAnalysis AtomGroup
        The first group of atoms.
    - group2: MDAnalysis AtomGroup
        The second group of atoms.

    Returns:
    - L: numpy array
        The angular momentum vector for the current timestep.
    """
    # Combine both groups
    combined_group = group1 + group2

    # Calculate the center of mass of the combined group
    com = combined_group.center_of_mass()
    
    # Initialize angular momentum for this timestep
    L = np.zeros(3)
    
    # Calculate angular momentum contribution from all atoms in the combined group
    r_rel = combined_group.positions - com  # Relative positions to COM
    v = combined_group.velocities  # Velocities of atoms
    masses = combined_group.masses[:, np.newaxis]  # Shape (N, 1) for broadcasting
    L = np.sum(np.cross(r_rel, masses * v), axis=0)

    return L

def connector(group1,
              group2):
    """
    Calculate the vector between the centers of mass of two groups of atoms for the current timestep.
    Note: The vector points from the COM of group1 to the COM of group2.

    Parameters:
    - group1: MDAnalysis AtomGroup
        The first group of atoms.
    - group2: MDAnalysis AtomGroup
        The second group of atoms.

    Returns:
    - r_vec: numpy array
        The vector from the center of mass of group1 to the center of mass of group2 for the current timestep.
    """
    # Calculate the center of mass for each group or use position if only one atom
    com1 = group1.center_of_mass() if len(group1) > 1 else group1.positions[0]
    com2 = group2.center_of_mass() if len(group2) > 1 else group2.positions[0]
    
    # Calculate the vector from com1 to com2
    r_vec = com2 - com1

    return r_vec

###########################################################################################################
# Atomic collision extraction and processing functions
###########################################################################################################
def find_collisions(u, 
                    group1, 
                    group2, 
                    cutoff):
    """
    Find collisions between atoms in two groups over the trajectory
    based on a cutoff distance. Returns a dictionary with atom indices
    from group1 as keys and a list of dictionaries for each frame as values.
    
    Inputs:
    - u: MDAnalysis Universe
        The MDAnalysis Universe object containing the trajectory and topology.
    - group1: MDAnalysis AtomGroup
        The first group of atoms.
    - group2: MDAnalysis AtomGroup
        The second group of atoms.
    - cutoff: float
        Distance cutoff for collision detection in Angstroms.
    
    Returns:
    - collisions: dict
        A dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of dictionaries, each representing a collision event for a specific frame.
          Each collision event dictionary contains:
          - 'frame': The frame number where the collision occurred.
          - 'atoms2': A list of atom indices from group2 that collided with the atom from group1.
          - 'distances': A list of distances (in Angstroms) between the group1 atom and the group2 atoms
                        that collided with it in the given frame.
    """    
    # Initialize dictionary to store collisions for each atom in group1
    collisions = {atom.index: [] for atom in group1}
    
    # Iterate over all frames in the trajectory
    for ts in tqdm(u.trajectory, desc='Finding collisions'):
        
        # Compute the distance matrix between atoms in group1 and group2
        dists = distances.distance_array(group1.positions,
                                         group2.positions,
                                         box=u.dimensions)
        
        # Find pairs of atoms where the distance is less than the cutoff
        g1_indices, g2_indices = np.where(dists < cutoff)

        # Loop over all pairs and avoid self-collisions
        for i, j in zip(g1_indices, g2_indices):
            if group1.indices[i] != group2.indices[j]:

                # Use a dictionary to track existing frames for each atom in group1
                if 'frame_dict' not in locals():
                    frame_dict = {atom.index: {} for atom in group1}

                # Check if the current frame already exists for the atom
                if ts.frame in frame_dict[group1.indices[i]]:
                    
                    # Retrieve the existing frame entry
                    frame_entry = frame_dict[group1.indices[i]][ts.frame]

                    # Append atom2 and distance to the existing frame entry
                    frame_entry['atoms2'].append(group2.indices[j])
                    frame_entry['distances'].append(dists[i, j])
                else:
                    # Create a new entry for the current frame with atom2 and distance
                    new_entry = {
                        'frame': ts.frame,
                        'atoms2': [group2.indices[j]],
                        'distances': [dists[i, j]]
                    }
                    collisions[group1.indices[i]].append(new_entry)
                    frame_dict[group1.indices[i]][ts.frame] = new_entry
    
    return collisions

# TODO: Later: Allowed atom indices can be generalized to MDAnalysis AtomGroups
def process_collisions(collisions, 
                       allowed_atom_indices=None):
    """
    Process the collisions dictionary to split distinct events based on frame gaps
    and the presence of the first atom in atoms2.
    Optionally, only record collisions where at least one atom index from allowed_atom_indices appears in atoms2.
    
    Inputs:
    - collisions: dict
        A dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of dictionaries, each representing a collision event for a specific frame.
    - allowed_atom_indices: list of int, optional
        If provided, only collisions where at least one allowed atom index appears in atoms2 are recorded.
    
    Returns:
    - processed_collisions: dict
        A dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames, each representing a distinct collision event.
    """
    processed_collisions = {}
    allowed_set = set(allowed_atom_indices) if allowed_atom_indices is not None else None
    
    for atom_index in tqdm(collisions, desc='Processing collisions'):
        # Initialize a list to store distinct events for the current atom
        events = []

        # Temporary list to store collisions for the current event
        current_event = []

        # Variable to track the frame of the previous collision
        previous_frame = None

        # Variable to track the first atom for the current event
        first_atom = None

        for collision in collisions[atom_index]:
            frame = collision['frame']
            atoms2 = collision['atoms2']

            # If allowed_atom_indices is given, skip collision if no allowed atom index in atoms2
            if allowed_set is not None and not any(atom in allowed_set for atom in atoms2):
                continue

            # Identify the first atom at the first frame of the event
            if first_atom is None and atoms2:
                first_atom = atoms2[0]

            # Check if the frame gap is greater than 1 or the first atom is no longer in atoms2
            if (previous_frame is not None and frame - previous_frame > 1) or (first_atom not in atoms2):
                
                # If so, finalize the current event and start a new one
                if current_event:  # Ensure the current event is not empty
                    events.append(current_event)
                current_event = []
                first_atom = None  # Reset the first atom for the new event

            # TODO: Does this need to be here? 
            # Ensure the first atom is still valid for the ongoing event
            if first_atom not in atoms2:
                first_atom = atoms2[0] if atoms2 else None

            # Add the current collision to the ongoing event
            current_event.append(collision)
            
            # Update the previous frame to the current frame
            previous_frame = frame

        # Add the last event if it exists
        if current_event:
            events.append(current_event)

        # Flatten events into separate entries as DataFrames
        processed_collisions[atom_index] = []
        for event in events:
            processed_collisions[atom_index].append(pd.DataFrame(event))
    
    return processed_collisions

# Helper function to identify the main colliding partner and extract its distance series
def get_main_partner_index(collision, 
                           allowed_atom_indices=None):
    """
    Identify the main partner atom index in a collision event based on the number of frames it is present in.

    The main partner is defined as the atom from group2 that is present in the most frames of the collision event.
    If allowed_atom_indices is provided, only those atoms are considered as potential main partners.

    Inputs:
    - collision: pandas DataFrame
        A DataFrame representing a collision, containing columns 'frame', 'atoms2', and 'distances'.
    - allowed_atom_indices: list of int, optional
        If provided, only these atom indices are considered as potential main partners.

    Returns:
    - main_partner_index: int
        The atom index of the main colliding partner.
    """
    # Build a counter for how many frames each partner atom is present in
    partner_counts = Counter()

    # Iterate over each row in the collision DataFrame
    for row in collision.itertuples():
        atoms2 = row.atoms2

        # Loop through each partner atom
        for atom in atoms2:
            # If allowed_atom_indices is provided, skip atoms not in the allowed list
            if allowed_atom_indices is not None and atom not in allowed_atom_indices:
                continue

            # Increment the count for this atom
            partner_counts[atom] += 1

    # If no partner atoms were found, return None
    if not partner_counts:
        return None

    # Select the atom with the highest count as the main partner
    main_partner_index = partner_counts.most_common(1)[0][0]

    return main_partner_index

# Helper function to extract the distance series for the main colliding partner in a collision event
def main_partner_distance_series(collision, 
                                 allowed_atom_indices=None):
    """
    Extract the distance series for the main colliding partner in a collision.

    The main partner is defined as the atom from group2 that has the smallest
    average distance to the atom from group1 over the entire collision event.
    If allowed_atom_indices is provided, only those atoms are considered as potential main partners.

    Inputs:
    - collision: pandas DataFrame
        A DataFrame representing a collision, containing columns 'frame', 'atoms2', and 'distances'.
    - allowed_atom_indices: list of int, optional
        If provided, only these atom indices are considered as potential main partners.

    Returns:
    - distances: numpy array
        An array of distances between the main colliding partner and the atom from group1 for each frame in the collision.
        If the main partner is not present in a frame, the distance for that frame will be NaN.
    """
    # Identify the main partner atom index using the helper function
    main_partner_index = get_main_partner_index(collision, 
                                                allowed_atom_indices=allowed_atom_indices)

    # Preallocate an array with NaN values to store distances for each frame
    distances = np.full(len(collision), np.nan)

    if main_partner_index is None:
        return distances

    # Iterate over each row in the collision DataFrame to extract the distance for the main partner
    for i, row in enumerate(collision.itertuples()):
        if main_partner_index in row.atoms2:
            partner_index = row.atoms2.index(main_partner_index)
            distances[i] = row.distances[partner_index]

    return distances

def categorize_collisions(processed_collisions, 
                          cutoff,
                          distance_minima_prominence=0.25, 
                          n_jobs=-1,
                          allowed_atom_indices=None,
                          vdw_bond_threshold=(3.9, 4.9)):
    """
    Categorize collisions into different categories based on collision properties.

    Inputs:
    - processed_collisions: dict
        A dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames, each representing a distinct collision.
    - cutoff: float
        Distance cutoff used for collision detection in Angstroms.
    - distance_minima_prominence: float
        The prominence for distance minima. Default is 0.25.
    - n_jobs: int
        Number of CPUs for parallel processing. Default is -1 (use all available).
    - allowed_atom_indices: list of int, optional
        If provided, only collisions where at least one allowed atom index appears in atoms2 are considered.
    - vdw_bond_threshold: tuple of float, optional
        The (min, max) threshold for average main partner distance to automatically classify as vdw.
        Default is (3.9, 4.9) Å for Xe-Xe interactions (+- 0.5 Å around Xe-Xe equilibrium distance).

    Returns:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    """
    # Helper function to process collisions for a single atom
    def process_atom(atom_index, collisions):

        # Initialize categories for the atom
        categorized = {'two_body_short': [], 
                       'multi_body_short': [], 
                       'two_body_vdw': [], 
                       'multi_body_vdw': []}
        
        # Create a set of allowed atom indices for quick lookup if provided
        allowed_set = set(allowed_atom_indices) if allowed_atom_indices is not None else None

        # Iterate over each collision for the atom
        for collision_df in collisions:

            # If allowed_atom_indices is given, skip collision if no allowed atom index in atoms2
            if allowed_set is not None:
                atoms2_all = set(atom for sublist in collision_df['atoms2'] for atom in sublist)
                if not atoms2_all.intersection(allowed_set):
                    continue

            # Check if the collision involves multiple colliding atoms
            is_multi_body = collision_df['atoms2'].str.len().max() > 1
            
            # Extract the distance series for the main colliding partner
            main_atom_distances = main_partner_distance_series(collision_df, 
                                                               allowed_atom_indices=allowed_atom_indices)
            
            # Check vdW bond threshold
            # Exclude the first and last 5% of the distance series to filter out extreme values
            n_points = len(main_atom_distances)
            start_idx = int(0.05 * n_points)
            end_idx = int(0.95 * n_points)
            valid_indices = np.arange(start_idx, end_idx)

            if valid_indices.size > 0:
                # Calculate the average distance for the valid indices
                avg_valid_dist = np.mean(main_atom_distances[valid_indices])

                # If the average valid distance falls within the vdw_bond_threshold range,
                # automatically classify the collision as a van der Waals (vdw) event
                if vdw_bond_threshold[0] <= avg_valid_dist <= vdw_bond_threshold[1]:
                    if is_multi_body:
                        categorized['multi_body_vdw'].append(collision_df)
                    else:
                        categorized['two_body_vdw'].append(collision_df)
                    continue  # Skip further categorization for this collision

            # Find the indices of local minima in the negated and shifted distance series
            shifted_neg_distances = -main_atom_distances + cutoff  # shift to start from zero
            minima_indices, _ = find_peaks(
                shifted_neg_distances, 
                prominence=distance_minima_prominence
            )
            
            # Count the number of local minima
            num_minima = len(minima_indices)

            # Categorize the collision based on its properties
            if is_multi_body:
                # Shallow minima might result in num_minima being 0, so we include that case in short collisions.
                # Also, we require at least 3 minima for a collision to be categorized as vdw.
                if num_minima <= 2:
                    categorized['multi_body_short'].append(collision_df)
                elif num_minima >= 3:
                    categorized['multi_body_vdw'].append(collision_df)
            else:
                if num_minima <= 2:
                    categorized['two_body_short'].append(collision_df)
                elif num_minima >= 3:
                    categorized['two_body_vdw'].append(collision_df)

        return atom_index, categorized

    # Parallel processing of collisions for all atoms
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_atom)(atom_index, collisions)
        for atom_index, collisions in tqdm(processed_collisions.items(), desc='Categorizing collisions')
    )

    # Combine results from all atoms into the final categorized collisions dictionary
    categorized_collisions = {'two_body_short': {}, 
                              'multi_body_short': {}, 
                              'two_body_vdw': {}, 
                              'multi_body_vdw': {}}

    # Iterate over the results and populate the categorized_collisions dictionary
    for atom_index, categorized in results:
        for category in categorized_collisions:
            if categorized[category]:
                categorized_collisions[category][atom_index] = categorized[category]

    return categorized_collisions

###########################################################################################################
# TODO: Everything above is more or less checked. Continue with the functions below.
# Atomic collision analysis functions
###########################################################################################################
def collision_counts(categorized_collisions):
    """
    Calculate the total number of collisions of each category over the entire trajectory.

    Inputs:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.

    Returns:
    - collision_counts: dict
        A dictionary with the same keys as categorized_collisions, where each value is the total number 
        of collisions for the corresponding category.
    """
    collision_counts = {category: 0 for category in categorized_collisions}

    for category in categorized_collisions:
        for atom_index in categorized_collisions[category]:
            collision_counts[category] += len(categorized_collisions[category][atom_index])

    return collision_counts

def collision_counts_by_lifetime(collision_stats, lifetime_ranges):
    """
    Calculate the number of collisions per category for given lifetime ranges.

    Inputs:
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each value is a list of dictionaries
        containing statistics for each event including 'lifetime_ps'.
    - lifetime_ranges: list of tuples
        A list of tuples defining the lifetime ranges in picoseconds. 
        Each tuple should be in the form (min_lifetime, max_lifetime),
        both in picoseconds. 

    Returns:
    - counts_by_lifetime: dict
        A dictionary with the same keys as collision_stats, where each value 
        is a list of counts corresponding to the defined lifetime ranges.
    """
    counts_by_lifetime = {category: [0] * len(lifetime_ranges) for category in collision_stats}

    for category, stats in collision_stats.items():
        for event in stats:
            lifetime = event['lifetime_ps']
            for i, (min_lifetime, max_lifetime) in enumerate(lifetime_ranges):
                if min_lifetime <= lifetime < max_lifetime:
                    counts_by_lifetime[category][i] += 1
                    break  # Stop checking after the first matching range

    return counts_by_lifetime

def collision_statistics(categorized_events, 
                         dt, 
                         total_frames):
    """
    Extract statistics from categorized events: Lifetime, number of atoms involved, and cluster counts over time.

    Inputs:
    - categorized_events: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized events.
    - dt: float
        The time step between frames in seconds.
    - total_frames: int
        The total number of frames in the trajectory.

    Returns:
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each value is a list of dictionaries
        containing statistics for each event:
        - 'atom_index': The index of the atom from group1.
        - 'event_index': The index of the event for the atom.
        - 'lifetime_ps': The lifetime of the event in picoseconds.
        - 'num_atoms_involved': The number of unique atoms involved in the event (including the atom itself).
    - cluster_counts: dict
        A dictionary with the same keys as categorized_events, where each value is an array of length total_frames.
        Each element in the array represents the number of clusters at that frame for the corresponding category.
    """
    # Initialize a dictionary to store statistics for each category
    collision_stats = {category: [] for category in categorized_events}
    
    # Initialize a dictionary to store cluster counts for each category
    cluster_counts = {category: np.zeros(total_frames, dtype=int) for category in categorized_events}
    
    # Create a dictionary to track unique clusters per frame for each category
    cluster_set = {category: {} for category in categorized_events}

    # Iterate over each category in the categorized events
    for category in categorized_events:

        # Process each atom in the current category
        for atom_index in tqdm(categorized_events[category], desc=f'Extracting statistics for {category}'):

            # Process each event for the current atom
            for event_index, event_df in enumerate(categorized_events[category][atom_index]):

                # Calculate the lifetime of the event in picoseconds
                lifetime = (event_df['frame'].max() - event_df['frame'].min() + 1) * dt * 1e12  # in ps
                
                # Identify the unique atoms involved in the event
                unique_atoms = set(atom for sublist in event_df['atoms2'] for atom in sublist)
                
                # Calculate the total number of atoms involved (including the main atom)
                num_atoms_involved = len(unique_atoms) + 1  # Add 1 for the atom itself
                
                # Append the statistics for the current event to the category's list
                collision_stats[category].append({
                    'atom_index': atom_index,
                    'event_index': event_index,
                    'lifetime_ps': lifetime,
                    'num_atoms_involved': num_atoms_involved
                })
                
                # Extract cluster counts over time
                main_partner_index = event_df.iloc[0]['atoms2'][0]

                # Iterate over each row in the event DataFrame to track clusters
                for row in event_df.itertuples():
                    frame = row.frame

                    # Create a unique identifier for the cluster (main atom index, partner index)
                    cluster_id = tuple(sorted((atom_index, main_partner_index)))  # NOTE: (A, B) is currently considered same as (B, A)
                    
                    # Use a set to avoid double counting the same cluster in the same frame
                    if frame not in cluster_set[category]:
                        cluster_set[category][frame] = set()
                    cluster_set[category][frame].add(cluster_id)

    # Count unique clusters per frame for each category
    for category, clusters in cluster_set.items():
        for frame, clusters_in_frame in clusters.items():
            cluster_counts[category][frame] = len(clusters_in_frame)

    # Return the dictionary containing collision statistics and cluster counts
    return collision_stats, cluster_counts

def sort_collision_statistics(collision_stats, 
                              property_name):
    """
    Sort collision statistics by a given property.

    Inputs:
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each value is a list of dictionaries
        containing statistics for each event.
    - property_name: str
        The name of the property to sort by (e.g., 'lifetime_ps').

    Returns:
    - sorted_stats: dict
        A dictionary with the same keys as collision_stats, where each value is a list of dictionaries
        sorted by the specified property in descending order.
    """
    sorted_stats = {}
    
    for category, stats in collision_stats.items():
        sorted_stats[category] = sorted(stats, key=lambda x: x[property_name], reverse=True)

    return sorted_stats

def cluster_formation_equilibrium_constants(cluster_counts, 
                                            collision_stats, 
                                            total_monomers, 
                                            volume=None):
    """
    Calculates the equilibrium constants for cluster formation.

    Inputs:
    - cluster_counts: dict
        A dictionary with the same keys as categorized_events, where each value is an array of length total_frames.
        Each element in the array represents the number of clusters at that frame for the corresponding category.
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each value is a list of dictionaries
        containing statistics for each event including 'num_atoms_involved'.
    - total_monomers: int
        The total number of monomers in the system.
    - volume: float, optional
        The system volume. If provided, the equilibrium constant is multiplied by the volume. Default is None.

    Returns:
    - equilibrium_constants: dict
        A dictionary with the same keys as cluster_counts, where each value is the calculated equilibrium 
        constant for cluster formation in that category.
    """
    equilibrium_constants = {}
    
    for category, counts in cluster_counts.items():
        # Calculate the average number of clusters over time
        avg_num_clusters = np.mean(counts)
        
        # Calculate the average number of monomers involved in clusters
        if collision_stats[category]:
            avg_monomers_per_cluster = np.mean([stat['num_atoms_involved'] for stat in collision_stats[category]])
            avg_bound_monomers = avg_num_clusters * avg_monomers_per_cluster
        else:
            avg_bound_monomers = 0
        
        # # Calculate the average number of free monomers and monomer pairs
        # avg_free_monomers = total_monomers - avg_bound_monomers
        # avg_free_monomer_pairs = avg_free_monomers * (avg_free_monomers - 1) / 2

        # NOTE: According to the formula K_eq = [Dimer] / [Monomer]^2
        # Calculate the average number of free monomer pairs
        avg_free_monomer_pairs = (total_monomers - avg_bound_monomers)**2
            
        # Avoid division by zero
        if avg_free_monomer_pairs > 0:
            equilibrium_constant = avg_num_clusters / avg_free_monomer_pairs
        else:
            equilibrium_constant = np.inf  # Infinite equilibrium constant if no monomer pairs are present
        
        # Multiply by volume if provided
        if volume is not None:
            equilibrium_constant *= volume
        
        equilibrium_constants[category] = equilibrium_constant

    return equilibrium_constants

def cluster_lifetime_rates(cluster_counts, 
                           dt):
    """
    Compute cluster formation and break-up rates from cluster counts over time.

    Inputs:
    - cluster_counts: dict
        A dictionary with the same keys as categorized_events, where each value is an array of length total_frames.
        Each element in the array represents the number of clusters at that frame for the corresponding category.
    - dt: float
        The time step between frames in seconds.

    Returns:
    - rates: dict
        A dictionary with the same keys as cluster_counts, where each value is a dictionary containing:
        - 'formation_rate': The average rate of cluster formation (clusters per second).
        - 'breakup_rate': The average rate of cluster break-up (clusters per second).
    """
    rates = {}
    
    for category, counts in cluster_counts.items():
        # Calculate the difference in cluster counts between consecutive frames
        count_diff = np.diff(counts)
        
        # Calculate formation and break-up events
        formations = np.sum(count_diff > 0)  # Count positive changes as formations
        breakups = np.sum(count_diff < 0)    # Count negative changes as break-ups
        
        # Calculate total time in seconds
        total_time = len(counts) * dt
        
        # Calculate rates
        formation_rate = formations / total_time if total_time > 0 else 0
        breakup_rate = breakups / total_time if total_time > 0 else 0
        
        rates[category] = {
            'formation_rate': formation_rate,
            'breakup_rate': breakup_rate
        }
    
    return rates

##############################################################################################################
# Trajectory analysis functions using categorized events
##############################################################################################################
def collision_property_time_series(u, 
                                   categorized_collisions, 
                                   category, 
                                   atom_index, 
                                   event_index, 
                                   property_function, 
                                   *args):
    """
    Calculate a time series of a given property for a specific collision event.

    Inputs:
    - u: MDAnalysis Universe
        The MDAnalysis Universe object containing the trajectory and topology.
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - category: str
        The category of the collision event (e.g., 'two_body_short').
    - atom_index: int
        The index of the atom from group1 involved in the collision event.
    - event_index: int
        The index of the specific event for the given atom in the specified category.
    - property_function: function
        A function that takes two MDAnalysis AtomGroups and additional arguments to calculate the desired property.
    - *args: additional arguments to be passed to the property_function.

    Returns:
    - time_series: numpy array
        An array of property values for each frame in the specified collision event.
    """
    # Retrieve the event DataFrame for the specified category, atom index, and event index
    event_df = categorized_collisions[category][atom_index][event_index]

    # Identify the main partner atom index from the first frame of the collision
    main_partner_index = event_df.iloc[0]['atoms2'][0]

    # Initialize a list to store the property values for each frame in the event
    time_series = []

    # Iterate over each row in the event DataFrame to calculate the property for each frame
    for row in event_df.itertuples():
        frame = row.frame
        
        # Set the trajectory to the current frame
        u.trajectory[frame]
                
        # Create AtomGroups for the current frame
        atom_group1 = u.atoms[[atom_index]]
        atom_group2 = u.atoms[[main_partner_index]]

        # Apply center and wrap transformation
        center_and_wrap(u, u.trajectory.ts, atom_group1)
        
        # Calculate the property using the provided function and additional arguments
        property_value = property_function(atom_group1, atom_group2, *args)
        
        # Append the calculated property value to the time series list
        time_series.append(property_value)
    
    return np.array(time_series)

###########################################################################################################
# Collision event data extraction and saving functions
###########################################################################################################
def get_collision_event_trajectories(u, 
                                     categorized_collisions, 
                                     category, 
                                     atom_index, 
                                     event_index):
    """
    Retrieve the positions and velocities of each atom involved in a specific collision event.

    Inputs:
    - u: MDAnalysis Universe
        The MDAnalysis Universe object containing the trajectory and topology.
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - category: str
        The category of the collision event (e.g., 'two_body_short').
    - atom_index: int
        The index of the atom from group1 involved in the collision event.
    - event_index: int
        The index of the specific event for the given atom in the specified category.

    Returns:
    - positions: numpy array of shape (N_frames, N_atoms, 3)
        Array of positions for each atom in each frame.
        Index 0: main atom (atom_index)
        Index 1: main partner atom
        Index 2+: other partner atoms
    - velocities: numpy array of shape (N_frames, N_atoms, 3)
        Array of velocities for each atom in each frame.
        Index 0: main atom (atom_index)
        Index 1: main partner atom
        Index 2+: other partner atoms
    """
    # Retrieve the event DataFrame for the specified category, atom index, and event index
    event_df = categorized_collisions[category][atom_index][event_index]

    # Get the main partner index from the first frame
    main_partner_index = event_df.iloc[0]['atoms2'][0]

    # Collect all unique partner indices throughout the event (excluding main partner)
    all_partner_indices = set(event_df['atoms2'].explode().unique())
    all_partner_indices.discard(main_partner_index)
    other_partner_indices = sorted(list(all_partner_indices))

    # Total number of atoms: main atom + main partner + other partners
    n_atoms = 2 + len(other_partner_indices)
    n_frames = len(event_df)

    # Initialize arrays to store positions and velocities
    positions = np.zeros((n_frames, n_atoms, 3))
    velocities = np.zeros((n_frames, n_atoms, 3))

    # Pre-fetch atom groups
    atom_groups = {idx: u.atoms[[idx]] for idx in [atom_index, main_partner_index] + other_partner_indices}

    # Iterate over each row in the event DataFrame
    for frame_idx, row in enumerate(event_df.itertuples()):
        frame = row.frame
        
        # Set the trajectory to the current frame
        u.trajectory[frame]
        
        # Index 0: main atom
        atom_group = atom_groups[atom_index]
        center_and_wrap(u, u.trajectory.ts, atom_group)
        positions[frame_idx, 0] = atom_group.positions[0]
        velocities[frame_idx, 0] = atom_group.velocities[0]
        
        # Index 1: main partner atom
        partner_group = atom_groups[main_partner_index]
        positions[frame_idx, 1] = partner_group.positions[0]
        velocities[frame_idx, 1] = partner_group.velocities[0]
        
        # Index 2+: other partner atoms
        for idx, partner_index in enumerate(other_partner_indices):
            partner_group = atom_groups[partner_index]
            positions[frame_idx, 2 + idx] = partner_group.positions[0]
            velocities[frame_idx, 2 + idx] = partner_group.velocities[0]
    
    return positions, velocities

###########################################################################################################
# Pipeline functions
###########################################################################################################
def get_categorized_collisions(u,
                               group1,
                               group2,
                               cutoff,
                               distance_minima_prominence=0.25,
                               n_jobs=-1,
                               return_all=False):
    """
    Perform the full analysis pipeline for collision extraction and processing.

    Inputs:
    - u: MDAnalysis Universe
        The MDAnalysis Universe object containing the trajectory and topology.
    - group1: MDAnalysis AtomGroup
        The first group of atoms.
    - group2: MDAnalysis AtomGroup
        The second group of atoms.
    - cutoff: float
        Distance cutoff for collision detection in Angstroms.
    - distance_minima_prominence: float
        The prominence for distance minima. Default is 0.25.
    - n_jobs: int
        Number of CPUs for parallel processing. Default is -1 (use all available).
    - return_all: bool
        If True, return collisions and processed_collisions in addition to categorized_collisions. Default is False.

    Returns:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - collisions: dict (only if return_all=True)
        The raw collisions dictionary from find_collisions.
    - processed_collisions: dict (only if return_all=True)
        The processed collisions dictionary from process_collisions.
    """
    # Step 1: Find collisions in the trajectory
    collisions = find_collisions(u, group1, group2, cutoff)

    # Step 2: Process collisions to split distinct collisions
    processed_collisions = process_collisions(collisions)

    # Step 3: Categorize collisions based on collision properties
    categorized_collisions = categorize_collisions(processed_collisions,
                                                cutoff=cutoff,
                                                distance_minima_prominence=distance_minima_prominence,
                                                n_jobs=n_jobs)
    
    if return_all:
        return collisions, processed_collisions, categorized_collisions
    else:
        return categorized_collisions
    
def save_collision_event_trajectories(u,
                                      categorized_collisions, 
                                      directory='collision_trajectories'):
    """
    Retrieve and save the positions and velocities of all collision events in the categorized collisions
    to .npy files with a structured naming convention.

    Inputs:
    - u: MDAnalysis Universe
        The MDAnalysis Universe object containing the trajectory and topology.
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - directory: str
        The directory where the files will be saved. Default is 'collision_trajectories'.

    Returns:
    - None
    """
    total_events = sum(len(categorized_collisions[category][atom_index]) for category in categorized_collisions for atom_index in categorized_collisions[category])
    
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    with tqdm(total=total_events, desc='Saving collision event trajectories') as pbar:
        for category in categorized_collisions:
            for atom_index in categorized_collisions[category]:
                for event_index in range(len(categorized_collisions[category][atom_index])):
                    
                    # Get trajectories for the specified collision event
                    positions, velocities = get_collision_event_trajectories(u, categorized_collisions, category, atom_index, event_index)

                    # Construct file names based on the category, atom index, and event index
                    positions_filename = f'{directory}/positions_{category}_atom{atom_index}_event{event_index}.npy'
                    velocities_filename = f'{directory}/velocities_{category}_atom{atom_index}_event{event_index}.npy'

                    # Save positions and velocities to .npy files
                    np.save(positions_filename, positions)
                    np.save(velocities_filename, velocities)

                    pbar.update(1)

###############################################################################################################
# Miscellaneous helper functions
###############################################################################################################
# def filter_collisions(collisions, 
#                       cutoff):
#     """
#     Filter out collision frames where the distances between the colliding
#     atoms are greater than the specified cutoff distance.

#     Inputs:
#     - collisions: dict
#         A dictionary where:
#         - Keys are atom indices from group1.
#         - Values are lists of dictionaries, each representing a collision event for a specific frame.
#     - cutoff: float
#         The distance cutoff in Angstroms.

#     Returns:
#     - filtered_collisions: dict
#         A dictionary with the same structure as the input collisions, 
#         but only including collision frames where at least one 
#         distance is less than the specified cutoff.
#     """
#     filtered_collisions = {}

#     # Iterate over each atom in the collisions dictionary
#     for atom_index, collision_events in collisions.items():
#         filtered_events = []

#         # Iterate over each collision event for the atom
#         for event in collision_events:

#             # Filter atoms2 and distances for those below cutoff
#             filtered_atoms2 = [atom for atom, dist in zip(event['atoms2'], event['distances']) if dist < cutoff]
#             filtered_distances = [dist for dist in event['distances'] if dist < cutoff]

#             # Only keep the event if at least one partner is within the cutoff
#             if filtered_atoms2:
#                 filtered_event = {
#                     'frame': event['frame'],
#                     'atoms2': filtered_atoms2,
#                     'distances': filtered_distances
#                 }
#                 filtered_events.append(filtered_event)

#         # Only add atom entry if it has filtered events
#         if filtered_events:
#             filtered_collisions[atom_index] = filtered_events

#     return filtered_collisions
