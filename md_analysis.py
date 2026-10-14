"""
MD trajectory analysis tool for gas phase MD simulations.
"""

# TODO: OMP parallelization, thread number control (to 1) 

###########################################################################################################
# Imports
###########################################################################################################
import numpy as np
import pandas as pd
import os

from scipy.signal import find_peaks

from tqdm import tqdm

from MDAnalysis.analysis.rdf import InterRDF
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
        rdf_analysis = InterRDF(group1,
                                group2,
                                exclusion_block=(1, 1),  # Exclude self-pairs for the same group
                                range=range,
                                nbins=nbins)
    else:
        rdf_analysis = InterRDF(group1,
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
                    new_entry = {'frame': ts.frame,
                                 'atoms2': [group2.indices[j]],
                                 'distances': [dists[i, j]]}
                    
                    collisions[group1.indices[i]].append(new_entry)
                    frame_dict[group1.indices[i]][ts.frame] = new_entry
    
    return collisions

# TODO: Later: Allowed atom indices can be generalized to MDAnalysis AtomGroups
def process_collisions(collisions, 
                       allowed_atom_indices=None):
    """
    Process frame-wise collision data into event blocks using the legacy event definition.

    In this version, each atom from group1 gets a sequence of collision events built
    from consecutive collision frames. An event is continued only while:
    1) frames remain contiguous, and
    2) the first partner atom seen for that event remains present in `atoms2`.

    This means the event identity is anchored to the first observed partner for that
    event. Unlike `process_collisions_by_partner`, overlapping partner-specific events
    are not created from the same frame.

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
        - Values are lists of pandas DataFrames, each representing a distinct collision event
          under the legacy first-partner continuity rule.
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

# TODO: Later: Allowed atom indices can be generalized to MDAnalysis AtomGroups
# TODO: Human check this function
def process_collisions_partner_specific(collisions, 
                                        allowed_atom_indices=None):
    """
    Process frame-wise collision data into partner-specific collision events.

    Unlike `process_collisions`, this version does not anchor an event to only the
    first observed partner. Instead, for each atom in group1, collision frames are
    split into separate events for each partner atom in `atoms2`. A partner-specific
    event starts when that partner appears and ends when:
    1) the partner is absent in the next processed frame, or
    2) a frame gap larger than 1 occurs.

    If multiple partners are present in the same frame, that frame can appear in more
    than one processed event. Within each event row, the selected partner is reordered
    to appear first in `atoms2`, and its distance is reordered to appear first in
    `distances`.

    Parameters:
    - collisions: dict
        Dictionary keyed by atom indices from group1. Each value is a list of
        dictionaries with keys 'frame', 'atoms2', and 'distances'.
    - allowed_atom_indices: list of int, optional
        If provided, only partner atoms with indices in this list are used to
        define processed events.

    Returns:
    - processed_collisions: dict
        Dictionary keyed by atom indices from group1. Each value is a list of
        pandas DataFrames, where each DataFrame represents one partner-specific
        collision event.
    """
    # Store the final processed events for each atom in group1.
    processed_collisions = {}

    # Convert the optional partner filter to a set for faster membership checks.
    allowed_set = set(allowed_atom_indices) if allowed_atom_indices is not None else None

    # Build a partner-specific version of a collision frame.
    # The selected partner is moved to the first position so downstream code
    # can treat it as the primary partner for that event.
    def collision_for_partner(collision, partner_atom):
        atoms2 = list(collision['atoms2'])
        distances2 = list(collision['distances'])
        partner_position = atoms2.index(partner_atom)

        # Start with the selected partner.
        reordered_atoms2 = [atoms2[partner_position]]
        reordered_distances = [distances2[partner_position]]

        # Preserve any additional simultaneous partners after the main one.
        for i, atom2 in enumerate(atoms2):
            if i == partner_position:
                continue

            reordered_atoms2.append(atom2)
            reordered_distances.append(distances2[i])

        return {'frame': collision['frame'],
                'atoms2': reordered_atoms2,
                'distances': reordered_distances}
    
    # Process collisions independently for each atom from group1.
    for atom_index in tqdm(collisions, desc='Processing collisions'):
        
        # Collect finalized events before converting them to DataFrames.
        # Each record stores:
        # (event start frame, partner atom index, list of frame dictionaries)
        event_records = []

        # Keep track of currently active events for each partner atom.
        # This allows multiple overlapping partner-specific events.
        active_events = {}

        # Iterate through the frame-wise collision records for this atom.
        for collision in collisions[atom_index]:
            frame = collision['frame']
            atoms2 = list(collision['atoms2'])

            # Apply optional filtering so only selected partners define events.
            if allowed_set is None:
                current_partners = atoms2
            else:
                current_partners = [atom for atom in atoms2 if atom in allowed_set]

            # Ignore frames with no relevant partners.
            if not current_partners:
                continue

            # Close any active partner event if:
            # 1) there is a frame gap larger than 1, or
            # 2) the partner is no longer present in the current frame.
            for partner_atom in list(active_events):
                previous_frame = active_events[partner_atom]['previous_frame']
                if frame - previous_frame > 1 or partner_atom not in current_partners:
                    event_records.append((active_events[partner_atom]['start_frame'],
                                          partner_atom,
                                          active_events[partner_atom]['rows']))
                    del active_events[partner_atom]

            # Update or start an event for every partner present in this frame.
            for partner_atom in current_partners:
                if partner_atom not in active_events:
                    active_events[partner_atom] = {'rows': [],
                                                   'previous_frame': frame,
                                                   'start_frame': frame}

                # Append a reordered frame entry where this partner is first.
                active_events[partner_atom]['rows'].append(collision_for_partner(collision, partner_atom))
                active_events[partner_atom]['previous_frame'] = frame

        # Finalize any events still active after the last frame.
        for partner_atom, event_data in active_events.items():
            event_records.append((event_data['start_frame'],
                                  partner_atom,
                                  event_data['rows']))

        # Sort events deterministically by start frame and partner index,
        # then convert each event to a DataFrame.
        processed_collisions[atom_index] = []
        for _, _, event_rows in sorted(event_records, key=lambda item: (item[0], item[1])):
            processed_collisions[atom_index].append(pd.DataFrame(event_rows))
    
    return processed_collisions

def categorize_collisions(processed_collisions, 
                          cutoff,
                          dt,
                          vdw_bond_length,
                          distance_minima_prominence=0.25,
                          n_jobs=-1,
                          allowed_atom_indices=None,
                          vdw_well_occupancy=0.75):
    """
    Categorize collisions into different categories based on collision properties.

    The classification uses the local minima of the distance series of the main colliding pair, the
    fraction of the event that the pair spends in the potential well, and the number of atoms involved.
    It involves no lifetime criterion.

    Inputs:
    - processed_collisions: dict
        A dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames, each representing a distinct collision.
        Note: This can be the output of either process_collisions() or process_collisions_partner_specific().
              Both versions produce valid input, but will yield different categorization results.
              process_collisions() uses legacy first-partner continuity, while 
              process_collisions_partner_specific() creates partner-specific events.
    - cutoff: float
        Distance cutoff used for collision detection in Angstroms.
    - dt: float
        Time step between frames in seconds.
    - vdw_bond_length: float
        The van der Waals bond length in Angstroms. This corresponds to the energy minimum
        distance of the two-atom pair. The classification threshold is taken as ±0.75 Å
        around this value.
    - distance_minima_prominence: float
        The prominence for distance minima. Default is 0.25.
    - n_jobs: int
        Number of CPUs for parallel processing. Default is -1 (use all available).
    - allowed_atom_indices: list of int, optional
        If provided, only collisions where at least one allowed atom index appears in atoms2 are considered.
    - vdw_well_occupancy: float or None
        An event is classified directly as vdW when at least this fraction of its main-pair distances lies
        within ±0.75 Å of vdw_bond_length. This captures bound pairs on nearly circular orbits, whose radial
        oscillations are too shallow to be resolved as distance minima. Default is 0.75. None disables the
        criterion. For gas-phase Xe, all events that meet the criterion last longer than 50 ps, so it
        reproduces the earlier rule that additionally required a lifetime above 10 ps.

    Returns:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
        Every processed collision event is assigned to one of these
        categories, including short partner-specific events with no
        detected local minimum.
    """
    # Define the vdW classification threshold as ±0.75 Å around the bond length
    vdw_bond_threshold = (vdw_bond_length - 0.75, vdw_bond_length + 0.75)

    # Helper function to process collisions for a single atom
    def process_atom(atom_index, collisions):

        # Initialize categories for the atom
        categorized = {'two_body_scattering': [], 
                    'multi_body_scattering': [], 
                    'perturbed_scattering': [],
                    'vdw': []}
        
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
            
            # Classify events in which the pair stays in the potential well for most of the time directly as vdW
            if vdw_well_occupancy is not None:

                # Calculate the fraction of distances within the vdW threshold range
                within_threshold = np.sum((main_atom_distances >= vdw_bond_threshold[0]) &
                                          (main_atom_distances <= vdw_bond_threshold[1]))
                fraction_within_threshold = within_threshold / len(main_atom_distances)

                # Classify as vdW if the fraction reaches the well-occupancy threshold
                if fraction_within_threshold >= vdw_well_occupancy:
                    categorized['vdw'].append(collision_df)
                    continue

            # Find the indices of local minima in the negated and shifted distance series
            shifted_neg_distances = -main_atom_distances + cutoff  # shift to start from zero
            minima_indices, _ = find_peaks(shifted_neg_distances,
                                           prominence=distance_minima_prominence)
            
            # Count the number of local minima
            num_minima = len(minima_indices)

            # Count the number of local minima that are below the vdW bond length + 0.75 Å
            num_minima_below_vdw = np.sum(main_atom_distances[minima_indices] < (vdw_bond_length + 0.75))
            # num_minima_below_vdw = np.sum(main_atom_distances[minima_indices] < (vdw_bond_length + 1.0)) # NOTE

            # Categorize the collision based on its properties.
            # Partner-specific processing can yield short events with no
            # resolved local minimum, but these must still be retained.
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

    # Parallel processing of collisions for all atoms
    results = Parallel(n_jobs=n_jobs)(delayed(process_atom)(atom_index, collisions)
                                      for atom_index, collisions in tqdm(processed_collisions.items(), 
                                                                         desc='Categorizing collisions'))

    # Combine results from all atoms into the final categorized collisions dictionary
    categorized_collisions = {'two_body_scattering': {}, 
                              'multi_body_scattering': {}, 
                              'perturbed_scattering': {},
                              'vdw': {}}

    # Iterate over the results and populate the categorized_collisions dictionary
    for atom_index, categorized in results:
        for category in categorized_collisions:
            if categorized[category]:
                categorized_collisions[category][atom_index] = categorized[category]

    return categorized_collisions

###########################################################################################################
# Helper functions
###########################################################################################################
def main_partner_index(collision, 
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

def main_partner_distance_series(collision, 
                                 allowed_atom_indices=None):
    """
    Extract the distance series for the main colliding partner in a collision.

    The main partner is defined as the atom from group2 that is present in the most frames
    of the collision event (see main_partner_index()).
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
    main_partner_idx = main_partner_index(collision, 
                                          allowed_atom_indices=allowed_atom_indices)

    # Preallocate an array with NaN values to store distances for each frame
    distances = np.full(len(collision), np.nan)

    if main_partner_idx is None:
        return distances

    # Iterate over each row in the collision DataFrame to extract the distance for the main partner
    for i, row in enumerate(collision.itertuples()):
        if main_partner_idx in row.atoms2:
            partner_index = row.atoms2.index(main_partner_idx)
            distances[i] = row.distances[partner_index]

    return distances

def distance_series(collision, 
                    allowed_atom_indices=None):
    """
    Extract the distance series for all colliding partners in a collision.
    If allowed_atom_indices is provided, only those atoms are considered as potential colliding partners.

    Inputs:
    - collision: pandas DataFrame
        A DataFrame representing a collision, containing columns 'frame', 'atoms2', and 'distances'.
    - allowed_atom_indices: list of int, optional
        If provided, only these atom indices are considered as potential colliding partners.

    Returns:
    - distance_dict: dict
        A dictionary where:
        - Keys are atom indices of colliding partners from group2.
        - Values are numpy arrays of distances between the corresponding partner and the atom from group1 for
            each frame in the collision. If a partner is not present in a frame, the distance for that frame will be NaN.
    """
    distance_dict = {}

    # Iterate over each frame (row) in the collision DataFrame
    for i, row in enumerate(collision.itertuples()):

        # Loop through all partner atoms in this frame
        for j, atom_index in enumerate(row.atoms2):

            # Skip atoms not in allowed_atom_indices if provided
            if allowed_atom_indices is not None and atom_index not in allowed_atom_indices:
                continue

            # Initialize the distance array for this partner if not already present
            if atom_index not in distance_dict:
                distance_dict[atom_index] = np.full(len(collision), np.nan)

            # Set the distance for this frame
            distance_dict[atom_index][i] = row.distances[j]

    return distance_dict

###########################################################################################################
# Atomic collision analysis functions
###########################################################################################################
# TODO: Check everything below
def collision_statistics(categorized_events, 
                         dt, 
                         total_frames):
    """
    Extract statistics from categorized events: Lifetime, number of atoms involved, and cluster counts over time.

    Inputs:
    - categorized_events: dict
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized events.
    - dt: float
        The time step between frames in seconds.
    - total_frames: int
        The total number of frames in the trajectory.

    Returns:
    - collision_stats: dict
        A dictionary with the same structure as categorized_events, where each category maps to a dictionary
        with atom indices as keys and lists of dictionaries as values. Each dictionary in the list contains
        statistics for each event:
        - 'event_index': The index of the event within the list of events
            for that atom in the category.
        - 'lifetime_ps': The lifetime of the event in picoseconds.
        - 'num_atoms_involved': The number of unique atoms involved in the event (including the atom itself).
        - 'avg_num_atoms': The average number of atoms present per frame during the event.
        - 'avg_main_partner_distance': The average distance in Angstroms
          between the reference atom and the main colliding partner during
          the event.
    - cluster_counts: dict
        A dictionary with the same keys as categorized_events, where each value is an array of length total_frames.
        Each element in the array represents the number of connected atom clusters at that frame for the corresponding category.
    """
    # Merge one set of cluster atoms into the list of frame-local connected
    # clusters, combining any overlapping atom sets into a single component.
    def merge_frame_cluster(frame_clusters, cluster_atoms):
        merged_cluster = set(cluster_atoms)
        remaining_clusters = []

        for existing_cluster in frame_clusters:
            if merged_cluster.intersection(existing_cluster):
                merged_cluster.update(existing_cluster)
            else:
                remaining_clusters.append(existing_cluster)

        remaining_clusters.append(merged_cluster)

        return remaining_clusters

    # Initialize a dictionary to store statistics for each category and atom index
    collision_stats = {category: {} for category in categorized_events}
    
    # Initialize a dictionary to store cluster counts for each category
    cluster_counts = {category: np.zeros(total_frames, dtype=int) for category in categorized_events}
    
    # Create a dictionary to track connected clusters per frame for each
    # category.
    cluster_set = {category: {} for category in categorized_events}

    # Iterate over each category in the categorized events
    for category in categorized_events:

        # Process each atom in the current category
        for atom_index in tqdm(categorized_events[category], desc=f'Extracting statistics for {category}'):

            # Initialize list to store statistics for this atom
            collision_stats[category][atom_index] = []

            # Process each event for the current atom.
            for event_index, event_df in enumerate(categorized_events[category][atom_index]):

                # Calculate the lifetime of the event in picoseconds
                event_duration = event_df['frame'].max() - event_df['frame'].min() + 1
                lifetime = event_duration * dt * 1e12  # in ps
                
                # Identify the unique atoms involved in the event
                unique_atoms = set(atom for sublist in event_df['atoms2'] for atom in sublist)
                
                # Calculate the total number of atoms involved (including the main atom)
                num_atoms_involved = len(unique_atoms) + 1  # Add 1 for the atom itself
                
                # Calculate the average number of atoms present per frame during the event
                num_frames = len(event_df)
                total_atoms_per_frame = sum(len(atoms2_list) + 1 for atoms2_list in event_df['atoms2'])
                avg_num_atoms = total_atoms_per_frame / num_frames if num_frames > 0 else 0

                # Calculate the average distance to the main colliding partner.
                main_partner_distances = main_partner_distance_series(event_df)
                if np.all(np.isnan(main_partner_distances)):
                    avg_main_partner_distance = np.nan
                else:
                    avg_main_partner_distance = np.nanmean(main_partner_distances)
                
                # Append the statistics for the current event to the atom's list
                collision_stats[category][atom_index].append({'event_index': event_index,
                                                              'lifetime_ps': lifetime,
                                                              'num_atoms_involved': num_atoms_involved,
                                                              'avg_num_atoms': avg_num_atoms,
                                                              'avg_main_partner_distance': avg_main_partner_distance})
                
                # Iterate over each row in the event DataFrame to track
                # connected clusters over time.
                for row in event_df.itertuples():
                    frame = row.frame

                    # Build the atom set present in this event row.
                    cluster_atoms = set([atom_index] + list(row.atoms2))

                    # Merge this atom set into the connected clusters already
                    # seen for the same frame.
                    if frame not in cluster_set[category]:
                        cluster_set[category][frame] = []
                    cluster_set[category][frame] = merge_frame_cluster(
                        cluster_set[category][frame],
                        cluster_atoms
                    )

    # Count unique clusters per frame for each category
    for category, clusters in cluster_set.items():
        for frame, clusters_in_frame in clusters.items():
            cluster_counts[category][frame] = len(clusters_in_frame)

    # Return the dictionary containing collision statistics and cluster counts
    return collision_stats, cluster_counts

def combine_collision_statistics(collision_statistics_results):
    """
    Combine multiple outputs of collision_statistics() in chronological order.

    Inputs:
    - collision_statistics_results: tuple-like
        Iterable of tuples returned by collision_statistics(). Each tuple must
        contain:
        - collision_stats: dict
            Nested dictionary of per-event statistics.
        - cluster_counts: dict
            Dictionary of per-frame cluster-count arrays.
        The first tuple is used as the structural reference. All following
        tuples are assumed to continue the trajectory in chronological order.

    Returns:
    - combined_collision_stats: dict
        Combined collision statistics with the same nested structure as one
        collision_statistics() output. Event indices are reassigned so that
        they remain consecutive within each atom and category after merging.
    - combined_cluster_counts: dict
        Dictionary of concatenated cluster-count arrays for each category.
    """
    # Require at least one result tuple to define the reference structure.
    if not collision_statistics_results:
        raise ValueError('collision_statistics_results must contain at least one result.')

    # Use the first result as the structural reference for categories and atoms.
    reference_collision_stats, reference_cluster_counts = collision_statistics_results[0]

    # Initialize the combined statistics by copying the reference structure.
    combined_collision_stats = {}
    for category, atoms_dict in reference_collision_stats.items():
        combined_collision_stats[category] = {}
        for atom_index, stats_list in atoms_dict.items():
            combined_collision_stats[category][atom_index] = [dict(event_stats)
                                                              for event_stats in stats_list]

    # Initialize concatenated cluster counts from the reference arrays.
    combined_cluster_counts = {category: np.array(counts, copy=True)
                               for category, counts in reference_cluster_counts.items()}

    # Append all following result tuples in chronological order.
    for collision_stats, cluster_counts in collision_statistics_results[1:]:

        # Extend the event statistics for each category and include any new
        # atom keys that appear only in later trajectory pieces.
        for category, next_atoms_dict in collision_stats.items():
            if category not in combined_collision_stats:
                raise ValueError(f"Missing collision_stats category '{category}' in reference result.")

            for atom_index, next_stats_list in next_atoms_dict.items():
                if atom_index not in combined_collision_stats[category]:
                    combined_collision_stats[category][atom_index] = []

                combined_stats_list = combined_collision_stats[category][atom_index]
                next_event_index = len(combined_stats_list)

                for event_stats in next_stats_list:
                    merged_event_stats = dict(event_stats)
                    merged_event_stats['event_index'] = next_event_index
                    combined_stats_list.append(merged_event_stats)
                    next_event_index += 1

        # Concatenate the per-category cluster-count time series.
        for category, combined_counts in combined_cluster_counts.items():
            next_counts = cluster_counts.get(category)
            if next_counts is None:
                raise ValueError(f"Missing cluster_counts category '{category}' in merged result.")
            combined_cluster_counts[category] = np.concatenate((combined_counts,
                                                                next_counts))

    return combined_collision_stats, combined_cluster_counts

def collision_counts(collision_stats):
    """
    Calculate the total number of collisions of each category over the entire trajectory.

    Inputs:
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each value is a dictionary
        where keys are atom indices from group1 and values are lists of dictionaries
        containing statistics for each event.

    Returns:
    - collision_counts: dict
        A dictionary with the same keys as collision_stats, where each value is the total number 
        of collisions for the corresponding category.
    """
    return {category: sum(len(stats) for stats in atoms_dict.values()) 
            for category, atoms_dict in collision_stats.items()}

def collision_counts_by_lifetime(collision_stats, lifetime_ranges):
    """
    Calculate the number of collisions per category for given lifetime ranges.

    Inputs:
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each value is a dictionary
        where keys are atom indices from group1 and values are lists of dictionaries
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
    # Initialize one counter per lifetime range for each collision category
    counts_by_lifetime = {category: [0] * len(lifetime_ranges) for category in collision_stats}

    # Loop over all categories and their recorded collision events
    for category, atoms_dict in collision_stats.items():
        for atom_index, stats_list in atoms_dict.items():
            for event in stats_list:
                lifetime = event['lifetime_ps']

                # Assign the event to the first matching lifetime interval
                for i, (min_lifetime, max_lifetime) in enumerate(lifetime_ranges):
                    if min_lifetime <= lifetime < max_lifetime:
                        counts_by_lifetime[category][i] += 1
                        break  # Stop after the first matching range

    return counts_by_lifetime

def cluster_formation_equilibrium_values(cluster_counts,
                                        collision_stats,
                                        total_monomers,
                                        volume=None):
    """
    Calculates the equilibrium constants and fractions for cluster formation.

    Inputs:
    - cluster_counts: dict
        A dictionary with the same keys as categorized_events, where each value is an array of length total_frames.
        Each element in the array represents the number of clusters at that frame for the corresponding category.
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each
        value is a dictionary keyed by atom index and each entry is a list of
        event statistics. The field 'avg_num_atoms' is used to estimate the
        average number of monomers in a cluster for each category.
    - total_monomers: int
        The total number of monomers in the system.
    - volume: float, optional
        The system volume. If provided, the equilibrium constant is multiplied by the volume. Default is None.

    Returns:
    - equilibrium_constants: dict
        A dictionary with the same keys as cluster_counts, where each value is the calculated equilibrium 
        constant for cluster formation in that category.
    - equilibrium_fractions: dict
        A dictionary with the same keys as cluster_counts, where each value is the fraction of monomers
        involved in clusters of that category relative to the total number of monomers.
    """
    equilibrium_constants = {}
    equilibrium_fractions = {}

    # Calculate the average number of atoms per cluster for each category
    # from the collision statistics.
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
    
    # Calculate total monomers bound across all categories
    total_bound_monomers = 0
    for category, counts in cluster_counts.items():
        avg_num_clusters = np.mean(counts)
        avg_bound_monomers = avg_num_clusters * avg_cluster_sizes[category]
        total_bound_monomers += avg_bound_monomers
    
    # Calculate free monomers once for all categories
    avg_free_monomers = total_monomers - total_bound_monomers
    
    # Now calculate equilibrium constants for each category
    for category, counts in cluster_counts.items():
        avg_num_clusters = np.mean(counts)
        
        # Calculate number of monomer pairs using K_eq = [Dimer] / [Monomer]^2
        avg_free_monomer_pairs = avg_free_monomers**2

        # Avoid division by zero
        if avg_free_monomer_pairs > 0:
            equilibrium_constant = avg_num_clusters / avg_free_monomer_pairs
        else:
            equilibrium_constant = np.inf  # Infinite equilibrium constant if no monomer pairs are present
        
        # Multiply by volume if provided
        if volume is not None:
            equilibrium_constant *= volume
        
        equilibrium_constants[category] = equilibrium_constant
        
        # Calculate bound monomers for this category
        avg_bound_monomers = avg_num_clusters * avg_cluster_sizes[category]
        equilibrium_fractions[category] = avg_bound_monomers / total_monomers if total_monomers > 0 else 0

    # Calculate the fraction of free monomers
    equilibrium_fractions['free_monomers'] = avg_free_monomers / total_monomers if total_monomers > 0 else 0

    return equilibrium_constants, equilibrium_fractions

def cluster_formation_rates(cluster_counts, 
                            dt,
                            volume=None):
    """
    Compute cluster formation and break-up rates from cluster counts over time.

    Inputs:
    - cluster_counts: dict
        A dictionary with the same keys as categorized_events, where each value is an array of length total_frames.
        Each element in the array represents the number of clusters at that frame for the corresponding category.
    - dt: float
        The time step between frames in seconds.
    - volume: float, optional
        The system volume in cubic Ångströms. If provided, rates are normalized by volume
        and returned in units of 1/(s*cm^3). Default is None.

    Returns:
    - rates: dict
        A dictionary with the same keys as cluster_counts, where each value is a dictionary containing:
        - 'formation_rate': The average rate of cluster formation (clusters per second, or 1/(s*cm^3) if volume is provided).
        - 'breakup_rate': The average rate of cluster break-up (clusters per second, or 1/(s*cm^3) if volume is provided).
    """
    rates = {}
    
    # Conversion factor from Ångströms^3 to cm^3
    angstrom_to_cm = 1e-8
    volume_cm3 = volume * (angstrom_to_cm ** 3) if volume is not None else None
    
    for category, counts in cluster_counts.items():
        # Calculate the difference in cluster counts between consecutive frames
        count_diff = np.diff(counts)
        
        # Calculate formation and break-up events by summing the positive and negative changes
        formations = np.sum(count_diff[count_diff > 0])  # Sum of positive changes as formations
        breakups = -np.sum(count_diff[count_diff < 0])   # Sum of negative changes as break-ups (negate to get positive count)
        
        # Calculate total time in seconds
        total_time = len(counts) * dt
        
        # Calculate rates
        formation_rate = formations / total_time if total_time > 0 else 0
        breakup_rate = breakups / total_time if total_time > 0 else 0
        
        # Normalize by volume if provided
        if volume_cm3 is not None:
            formation_rate /= volume_cm3
            breakup_rate /= volume_cm3
        
        rates[category] = {'formation_rate': formation_rate,
                           'breakup_rate': breakup_rate}

    return rates

def atom_time_fractions(collision_stats,
                        n_atoms,
                        total_frames,
                        dt):
    """
    Calculate the fraction of the total atom-time spent as a reference atom in events of each category.

    Each frame of a reference atom belongs to at most one of its events, so these fractions and the
    remaining fraction of free atoms sum to unity. Together with time-weighted category averages
    (see average_property_R_time_weighted()), they decompose the trajectory average of a single-atom
    property exactly: <A> = sum_c f_c <A_c>.

    Inputs:
    - collision_stats: dict
        Output of collision_statistics() or combine_collision_statistics(). The field 'lifetime_ps'
        of each event is used.
    - n_atoms: int
        Number of reference atoms.
    - total_frames: int
        Total number of frames in the trajectory.
    - dt: float
        Time step between frames in seconds.

    Returns:
    - time_fractions: dict
        A dictionary with the same category keys as collision_stats, plus the key 'free_monomers',
        where each value is the fraction of the total atom-time.
    """
    # Total atom-time of the trajectory in picoseconds
    total_atom_time_ps = n_atoms * total_frames * dt * 1e12

    # Sum the event lifetimes of each category and normalise by the total atom-time
    time_fractions = {}
    for category, atoms_dict in collision_stats.items():
        category_time_ps = sum(event['lifetime_ps'] for stats_list in atoms_dict.values() for event in stats_list)
        time_fractions[category] = category_time_ps / total_atom_time_ps

    # The remaining atom-time is spent outside any event
    time_fractions['free_monomers'] = 1.0 - sum(time_fractions.values())

    return time_fractions

##############################################################################################################
# Trajectory analysis functions using categorized events
##############################################################################################################
def property_time_series(u, 
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
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - category: str
        The category of the collision event (e.g., 'two_body_scattering').
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

    # Identify the main partner atom index 
    main_partner_idx = main_partner_index(event_df)

    # Initialize a list to store the property values for each frame in the event
    time_series = []

    # Iterate over each row in the event DataFrame to calculate the property for each frame
    for row in event_df.itertuples():
        frame = row.frame
        
        # Set the trajectory to the current frame
        u.trajectory[frame]
                
        # Create AtomGroups for the current frame
        atom_group1 = u.atoms[[atom_index]]
        atom_group2 = u.atoms[[main_partner_idx]]

        # Apply center and wrap transformation
        center_and_wrap(u, u.trajectory.ts, atom_group1)
        
        # Calculate the property using the provided function and additional arguments
        property_value = property_function(atom_group1, atom_group2, *args)
        
        # Append the calculated property value to the time series list
        time_series.append(property_value)
    
    return np.array(time_series)

def property_R_main_partner_time_series(categorized_collisions,
                                       category,
                                       atom_index,
                                       event_index,
                                       property_function,
                                       *args):
    """
    Calculate a time series of a given property that depends only on the distance between the main
    colliding partners for a specific collision event.

    Inputs:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - category: str
        The category of the collision event (e.g., 'two_body_scattering').
    - atom_index: int
        The index of the atom from group1 involved in the collision event.
    - event_index: int
        The index of the specific event for the given atom in the specified category.
    - property_function: function
        A function that takes a distance array and additional arguments to calculate the desired property.
    - *args: additional arguments to be passed to the property_function.

    Returns:
    - time_series: numpy array
        An array of property values for each frame in the specified collision event (main partner).
    """
    # Retrieve the event DataFrame for the specified category, atom index, and event index
    event_df = categorized_collisions[category][atom_index][event_index]

    # Extract the distance series for the main colliding partner using the helper function
    main_partner_distances = main_partner_distance_series(event_df)

    # Calculate the property for each frame in the event using the provided function and additional arguments
    time_series = property_function(main_partner_distances, *args)

    return np.array(time_series)

def property_R_all_partners_time_series(categorized_collisions,
                                        category,
                                        atom_index,
                                        event_index,
                                        property_function,
                                        *args,
                                        use_pairwise_sum=False):
    """
    Calculate a time series of a given property for all partners in a specific collision event,
    where the property depends only on the distance between the atoms.

    Inputs:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - category: str
        The category of the collision event (e.g., 'two_body_scattering').
    - atom_index: int
        The index of the atom from group1 involved in the collision event.
    - event_index: int
        The index of the specific event for the given atom in the specified category.
    - property_function: function
        A function that takes a distance array and additional arguments to calculate the desired property.
    - *args: additional arguments to be passed to the property_function.
    - use_pairwise_sum: bool
        If True, also return the pairwise additive contribution (sum of all partners). Default is False.

    Returns:
    - time_series: dict
        A dict of arrays for each partner, with property values for each frame in the specified collision event.
    - pairwise_sum: numpy array (only if use_pairwise_sum=True)
        Sum of property values from all partners for each frame.
    """
    # Retrieve the event DataFrame for the specified category, atom index, and event index
    event_df = categorized_collisions[category][atom_index][event_index]

    # Extract the distance series for all partners
    partner_distances = distance_series(event_df)
    
    # Compute the property for each partner's distance series
    time_series = {partner: property_function(distances, *args)
                   for partner, distances in partner_distances.items()}
    
    # If requested, compute the pairwise sum of contributions from all partners
    if use_pairwise_sum:
        # Sum contributions from all partners, handling NaN values
        pairwise_sum = np.nansum(list(time_series.values()), axis=0)
        return time_series, pairwise_sum
    
    return time_series

def average_property_R_per_event(categorized_collisions,
                                 property_function,
                                 *args,
                                 use_pairwise_sum=False):
    """
    Calculate the average value of a distance-dependent property for each
    individual collision event.

    Inputs:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_scattering',
        'multi_body_scattering', 'perturbed_scattering', and 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the
          categorized collisions.
    - property_function: function
        A function that takes a distance array R and additional arguments to
        calculate the desired property. The property must depend only on the
        distance R between atoms.
    - *args: additional arguments to be passed to the property_function.
    - use_pairwise_sum: bool
        If True, uses pairwise sum of all partners. If False (default), uses
        only the main partner distances.

    Returns:
    - event_average_properties: dict
        A dictionary with the same category and atom-index structure as
        categorized_collisions. Each event is represented by a dictionary
        containing:
        - 'average_property': The average value of the property over all
          distances R in that collision event.
    """
    # Initialise a nested dictionary to match the categorized collision
    # structure.
    event_average_properties = {category: {atom_index: [] for atom_index in categorized_collisions[category]}
                                for category in categorized_collisions}

    # Process one category at a time using the same tqdm pattern as the other
    # analysis functions.
    for category in categorized_collisions:

        # Process each atom in the current category with a category-specific
        # progress bar.
        for atom_index in tqdm(categorized_collisions[category],
                               desc=f'Calculating average collision properties for {category}'):

            # Iterate through each collision event for the current atom.
            for event_index, _ in enumerate(categorized_collisions[category][atom_index]):

                # Select the distance series definition requested by the user.
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

                # Compute the event-averaged property value.
                average_value = np.nanmean(property_values)

                # Store the per-event result in the nested output structure.
                event_average_properties[category][atom_index].append({'average_property': average_value})

    return event_average_properties

def average_property_R_over_events(event_average_properties):
    """
    Average per-event property values over events for each atom.

    Inputs:
    - event_average_properties: dict
        Output of average_property_R_per_event(). The dictionary is keyed by
        category and atom index, and each value is a list of per-event result
        dictionaries.

    Returns:
        - average_properties: dict
                Each category value is a dictionary where each atom index maps to a dictionary containing:
        - 'average_property': The average property over all events of that
          atom in the category.
        - 'total_num_events': The total number of events for that atom in the
          category.
    """
    # Build one event-level summary per atom within each category.
    average_properties = {}
    for category, atoms_dict in event_average_properties.items():
        average_properties[category] = {}

        # Summarise the event averages for each atom separately.
        for atom_index, atom_events in atoms_dict.items():
            if atom_events:
                average_property_values = [event['average_property'] for event in atom_events]
                average_properties[category][atom_index] = {'average_property': np.mean(average_property_values),
                                                            'total_num_events': len(atom_events)}
            else:
                average_properties[category][atom_index] = {'average_property': np.nan,
                                                            'total_num_events': 0}

    return average_properties

def average_property_R_over_atoms(average_properties_over_events):
    """
    Average per-atom property summaries over atoms within each category.

    Inputs:
    - average_properties_over_events: dict
        Output of average_property_R_over_events(). Each category value is a
        dictionary keyed by atom index and each atom value contains the
        summary fields 'average_property' and 'total_num_events'.

    Returns:
    - average_properties: dict
        Dictionary keyed by category, where each value is a category-level
        summary averaged over all atoms with recorded events in that
        category.
    """
    average_properties = {}

    for category, atoms_dict in average_properties_over_events.items():
        atom_summaries = [summary for summary in atoms_dict.values()
                          if summary['total_num_events'] > 0]

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

def average_property_R(categorized_collisions,
                       property_function,
                       *args,
                       average_over_events=True,
                       average_over_atoms=False,
                       use_pairwise_sum=False):
    """
    Calculate the average value of a given property that depends only on the
    distance R for collision events across all categories.

    This is a convenience wrapper that first computes per-event property
    averages and then processes them further over events and, optionally,
    over atoms.

    Inputs:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_scattering',
        'multi_body_scattering', 'perturbed_scattering', and 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized
          collisions.
    - property_function: function
        A function that takes a distance array R and additional arguments to
        calculate the desired property. The property must depend only on the
        distance R between atoms.
    - *args: additional arguments to be passed to the property_function.
    - average_over_events: bool
        If True (default), returns the average property value over all events
        in each category. If False, returns the average property value for
        each individual collision event instead.
    - average_over_atoms: bool
        If True, also average over atoms within each category. This option
        requires average_over_events=True, so averaging only over atoms is
        not allowed.
    - use_pairwise_sum: bool
        If True, uses pairwise sum of all partners. If False (default), uses
        only the main partner distances.

    Returns:
    - average_properties: dict
        The output of average_property_R_over_events() or, if
        average_over_atoms=True, the output of average_property_R_over_atoms().
    """
    # Calculate the average property for each individual event first.
    event_average_properties = average_property_R_per_event(categorized_collisions,
                                                            property_function,
                                                            *args,
                                                            use_pairwise_sum=use_pairwise_sum)

    # Return the per-event values unchanged when no further averaging is requested.
    if not average_over_events:
        return event_average_properties

    # Average the event-level values for each atom.
    average_properties = average_property_R_over_events(event_average_properties)

    # Optionally average the atom-level summaries over atoms within a category.
    if average_over_atoms:
        return average_property_R_over_atoms(average_properties)

    return average_properties

def average_property_R_time_weighted(categorized_collisions,
                                     property_function,
                                     *args,
                                     use_pairwise_sum=True,
                                     n_jobs=1):
    """
    Calculate the time-weighted average of a distance-dependent property for each event category.

    The property values of all frames of all events in a category are pooled, so that each event
    contributes in proportion to its lifetime. This is the category average that is consistent with
    the atom-time fractions of atom_time_fractions(), and the two together reproduce the trajectory
    average of the property exactly.

    Inputs:
    - categorized_collisions: dict
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering',
        'perturbed_scattering', and 'vdw'. Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - property_function: function
        A function that takes a distance array R and additional arguments and returns the property.
        The property must depend only on R.
    - *args: additional arguments to be passed to the property_function.
    - use_pairwise_sum: bool
        If True (default), the property of a frame is the sum over all partners within the cutoff
        (pairwise additivity). If False, only the main partner is used.
    - n_jobs: int
        Number of CPUs for parallel processing over atoms. Default is 1.

    Returns:
    - average_properties: dict
        Dictionary keyed by category, where each value contains:
        - 'average_property': The time-weighted average property of the category.
        - 'total_num_events': The total number of events in the category.
        - 'total_frames': The total number of frames of all events in the category.
        - 'property_sum': The sum of the property over all frames, for pooling several trajectories.
    """
    # Helper function returning the frame sum of the property and the number of frames for the events of one atom.
    # Only the event list of the atom is passed, so that parallel workers do not receive the whole dictionary.
    def process_atom(events):
        property_sum = 0.0
        num_frames = 0
        for event_df in events:
            if use_pairwise_sum:
                partner_distances = distance_series(event_df)
                property_values = [property_function(distances, *args) for distances in partner_distances.values()]
            else:
                property_values = property_function(main_partner_distance_series(event_df), *args)
            property_sum += np.nansum(property_values)
            num_frames += len(event_df)
        return property_sum, num_frames, len(events)

    average_properties = {}
    for category in categorized_collisions:

        # Pool the frame sums of all atoms in the category
        results = Parallel(n_jobs=n_jobs)(delayed(process_atom)(events)
                                          for events in tqdm(categorized_collisions[category].values(),
                                                             desc=f'Time-weighted averages for {category}'))
        property_sum = sum(result[0] for result in results)
        total_frames = sum(result[1] for result in results)
        total_num_events = sum(result[2] for result in results)

        average_properties[category] = {'average_property': property_sum / total_frames if total_frames > 0
                                                            else np.nan,
                                         'total_num_events': total_num_events,
                                         'total_frames': total_frames,
                                         'property_sum': property_sum}

    return average_properties

def average_property_R_statistics(average_properties,
                                  equilibrium_fractions,
                                  collision_stats):
    """
    Calculate category-level lifetime statistics and equilibrium-weighted
    average contributions for distance-dependent properties.

    This function assumes that average_property_R() was called with
    average_over_events=True and average_over_atoms=True, so that each
    category maps directly to a category-level summary.

    Inputs:
    - average_properties: dict
        Output of average_property_R() when both average_over_events and
        average_over_atoms are True. Each category value must contain the
        keys 'average_property' and 'total_num_events'.
    - equilibrium_fractions: dict
        Output of cluster_formation_equilibrium_values(). The category
        fractions are used to weight the average property contributions.
    - collision_stats: dict
        Output of collision_statistics(). Used only to obtain event
        lifetimes for the average lifetime statistics.

    Returns:
    - property_statistics_df: pandas DataFrame
        A DataFrame containing one row for each event category and one final
        all row. The columns are:
        - 'category': The event category label.
        - 'average_property': The average property value for the category.
        - 'total_num_events': The total number of events in the category.
        - 'average_lifetime_ps': The average event lifetime in picoseconds.
        - 'equilibrium_fraction': The equilibrium fraction assigned to the
          category.
        - 'weighted_average': The contribution of the category to the
          ensemble average.
    """
    # Initialise a list of rows for the output table.
    property_statistics_rows = []

    # Calculate one set of summary statistics for each category.
    for category, category_summary in average_properties.items():
        category_collision_stats = collision_stats.get(category, {})
        event_lifetimes = [event['lifetime_ps']
                           for stats_list in category_collision_stats.values()
                           for event in stats_list]
        equilibrium_fraction = equilibrium_fractions.get(category, np.nan)

        average_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan
        weighted_average = category_summary['average_property'] * equilibrium_fraction

        # Store the category contribution as one row of the output table.
        property_statistics_rows.append({'category': category,
                                        'average_property': category_summary['average_property'],
                                        'total_num_events': category_summary['total_num_events'],
                                        'average_lifetime_ps': average_lifetime_ps,
                                        'equilibrium_fraction': equilibrium_fraction,
                                        'weighted_average': weighted_average})

    # Sum the category contributions to obtain the ensemble-average
    # contribution from all event categories.
    equilibrium_fraction_values = [row['equilibrium_fraction'] for row in property_statistics_rows]
    all_equilibrium_fraction = np.sum(equilibrium_fraction_values)
    all_weighted_average = np.sum([row['average_property'] * row['equilibrium_fraction']
                                   for row in property_statistics_rows])

    event_lifetimes = [event['lifetime_ps']
                       for category_stats in collision_stats.values()
                       for stats_list in category_stats.values()
                       for event in stats_list]
    all_average_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan

    # Calculate the all-category average property represented by the included
    # event categories.
    all_average_property = all_weighted_average / all_equilibrium_fraction \
        if all_equilibrium_fraction > 0 else np.nan

    # Append a final row containing the all-category weighted average.
    property_statistics_rows.append({'category': 'all',
                                     'average_property': all_average_property,
                                     'total_num_events': np.sum([row['total_num_events'] for row in property_statistics_rows]),
                                     'average_lifetime_ps': all_average_lifetime_ps,
                                     'equilibrium_fraction': all_equilibrium_fraction,
                                     'weighted_average': all_weighted_average})

    # Convert the table to a DataFrame and format the numeric columns for
    # display in scientific notation with two decimals.
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

##################################################################################################
# Trajectory extraction functions using categorized events
##################################################################################################
# TODO: Human check this function
def collision_event_trajectories(u, 
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
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - category: str
        The category of the collision event (e.g., 'two_body_scattering').
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

    # Get the main partner index from the event
    main_partner_idx = main_partner_index(event_df)

    # Collect all unique partner indices throughout the event (excluding main partner)
    all_partner_indices = set(event_df['atoms2'].explode().unique())
    all_partner_indices.discard(main_partner_idx)
    other_partner_indices = sorted(list(all_partner_indices))

    # Total number of atoms: main atom + main partner + other partners
    n_atoms = 2 + len(other_partner_indices)
    n_frames = len(event_df)

    # Initialize arrays to store positions and velocities
    positions = np.zeros((n_frames, n_atoms, 3))
    velocities = np.zeros((n_frames, n_atoms, 3))

    # Pre-fetch atom groups
    atom_groups = {idx: u.atoms[[idx]] for idx in [atom_index, main_partner_idx] + other_partner_indices}

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
        partner_group = atom_groups[main_partner_idx]
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
# Placeholder

#############################################################################################################
# Data saving functions
#############################################################################################################  
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
        A dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps to a dictionary where:
        - Keys are atom indices from group1.
        - Values are lists of pandas DataFrames representing the categorized collisions.
    - directory: str
        The directory where the files will be saved. Default is 'collision_trajectories'.

    Returns:
    - None
    """
    total_events = sum(len(categorized_collisions[category][atom_index]) 
                       for category in categorized_collisions for atom_index in categorized_collisions[category])
    
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Iterate over all categorized collisions and save trajectories for each event
    with tqdm(total=total_events, desc='Saving collision event trajectories') as pbar:
        for category in categorized_collisions:
            for atom_index in categorized_collisions[category]:
                for event_index in range(len(categorized_collisions[category][atom_index])):
                    
                    # Get trajectories for the specified collision event
                    positions, velocities = collision_event_trajectories(u, categorized_collisions, 
                                                                         category, atom_index, event_index)

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
def sort_collision_statistics(collision_stats, 
                              property_name,
                              atomwise=True,
                              reverse=True):
    """
    Sort collision statistics by a given property.

    Inputs:
    - collision_stats: dict
        A dictionary with the same keys as categorized_events, where each
        category value is a dictionary with atom indices as keys and lists
        of dictionaries containing statistics for each event as values.
    - property_name: str
        The name of the property to sort by (e.g., 'lifetime_ps').
    - atomwise: bool
        If True (default), sort the event statistics separately for each
        atom within each category and preserve the nested dictionary
        structure. If False, combine the events of all atoms within each
        category into one list, sort that list, and store the source atom
        index for each event under the key 'atom_index'.
    - reverse: bool
        If True (default), sort in descending order. If False, sort in
        ascending order.

    Returns:
    - sorted_stats: dict
        If atomwise is True, a dictionary with the same nested structure as
        collision_stats, where the list of event statistics for each atom is
        sorted by the specified property in descending order.
        If atomwise is False, a dictionary keyed by category, where each
        value is a sorted list of event-statistics dictionaries collected
        from all atoms in that category. Each event dictionary then also
        contains the corresponding atom index under the key 'atom_index'.
    """
    sorted_stats = {}
    
    # Sort each category either atomwise or across all atoms.
    for category, atoms_dict in collision_stats.items():
        if atomwise:
            sorted_stats[category] = {}

            # Sort the event list for each atom by the requested property.
            for atom_index, stats_list in atoms_dict.items():
                sorted_stats[category][atom_index] = sorted(stats_list,
                                                            key=lambda x: x[property_name],
                                                            reverse=reverse)
        else:
            combined_stats = []

            # Collect all event statistics for the category into one list.
            for atom_index, stats_list in atoms_dict.items():
                for event_stats in stats_list:
                    event_stats_with_atom = dict(event_stats)
                    event_stats_with_atom['atom_index'] = atom_index
                    combined_stats.append(event_stats_with_atom)

            # Sort the combined event list across all atoms.
            sorted_stats[category] = sorted(combined_stats,
                                            key=lambda x: x[property_name],
                                            reverse=reverse)

    return sorted_stats
