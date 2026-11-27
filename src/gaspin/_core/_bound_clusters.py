"""
Energetically bound clusters of atoms (vdW dimers, trimers and larger complexes), found from the positions
and velocities independently of the event classification, and linked into cluster events.
"""

# Imports
from collections.abc import Callable

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import distances as mda_distances
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


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

    Usage: ``frame_clusters = find_bound_clusters(u, u.atoms, 9.0, V, 1.2027e6, min_size=3)``

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
        - 'first_atoms' and 'last_atoms': tuples of the atom indices of the cluster in the first and last frame,
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
                               'first_atoms': set(cluster_atoms), 'last_atoms': set(cluster_atoms)})
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
                if other['frames'][0] == event['frames'][0]:
                    event['first_atoms'] |= other['first_atoms']
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
                       'first_atoms': tuple(sorted(event['first_atoms'])),
                       'last_atoms': tuple(sorted(event['last_atoms'])),
                       'max_size': int(max(event['sizes'])),
                       'frames': frames,
                       'sizes': np.array(event['sizes'])})

    return events


def bridge_bound_cluster_events(u: mda.Universe,
                                events: list[dict],
                                cutoff: float,
                                dt: float,
                                min_shared: int=3) -> list[dict]:
    """
    Merge consecutive cluster events of the same atoms whose atoms stay together during the gap between them.

    The energy criterion of hill_bound_clusters() can split a cluster that is bound as a whole, when the energy of
    one atom relative to the others is positive for a while without the atom leaving. Two events are therefore
    merged if at least min_shared atoms of the last cluster of the earlier event are also in the first cluster of
    the later one, and if these shared atoms stay connected within the cutoff in every frame of the gap, i.e. the
    graph of their pairs closer than the cutoff is connected. Only an actual separation then ends an event. The
    merged event spans the gap, and its 'frames' and 'sizes' remain those of the frames with a bound cluster.
    Distances follow the minimum-image convention.

    Usage: ``events = bridge_bound_cluster_events(u, events, 9.0, 25e-15)``

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing the trajectory from which the events were found.
    events : list of dict
        Output of bound_cluster_events().
    cutoff : float
        Largest distance of a connected pair during a gap, in Å.
    dt : float
        Time step between frames in seconds.
    min_shared : int, default=3
        Smallest number of atoms shared by the clusters on both sides of a gap.

    Returns
    -------
    bridged_events : list of dict
        Cluster events sorted by their first frame, with the keys of bound_cluster_events() and 'n_bridges', the
        number of gaps bridged within the event.
    """

    # Events sorted by their first frame, copied so that the input is not changed
    pending = [dict(event, n_bridges=0) for event in sorted(events, key=lambda item: item['start_frame'])]
    bridged_events = []
    while pending:
        event = pending.pop(0)

        # Repeatedly merge the earliest later event that shares enough atoms and is reached without separation
        merged = True
        while merged:
            merged = False
            for later in pending:
                shared = sorted(set(event['last_atoms']) & set(later['first_atoms']))
                if later['start_frame'] <= event['end_frame'] or len(shared) < min_shared:
                    continue

                # Check that the shared atoms stay connected within the cutoff in every frame of the gap
                atoms = u.atoms[shared]
                connected = True
                for ts in u.trajectory[event['end_frame'] + 1:later['start_frame']]:
                    pair_distances = mda_distances.self_distance_array(atoms.positions, box=ts.dimensions)
                    i, j = np.triu_indices(len(shared), k=1)
                    close = pair_distances < cutoff
                    graph = coo_matrix((np.ones(np.count_nonzero(close)), (i[close], j[close])),
                                       shape=(len(shared), len(shared)))
                    if connected_components(graph, directed=False)[0] > 1:
                        connected = False
                        break
                if not connected:
                    continue

                # Merge the later event into this one
                event = dict(event,
                             end_frame=later['end_frame'],
                             lifetime_ps=(later['end_frame'] - event['start_frame'] + 1) * dt * 1e12,
                             atoms=tuple(sorted(set(event['atoms']) | set(later['atoms']))),
                             last_atoms=later['last_atoms'],
                             max_size=max(event['max_size'], later['max_size']),
                             frames=np.concatenate([event['frames'], later['frames']]),
                             sizes=np.concatenate([event['sizes'], later['sizes']]),
                             n_bridges=event['n_bridges'] + later['n_bridges'] + 1)
                pending.remove(later)
                merged = True
                break
        bridged_events.append(event)

    return bridged_events


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
