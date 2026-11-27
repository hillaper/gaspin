"""
Tests for the energetically bound clusters and the cluster events.
"""

import unittest

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

import gaspin as gs

# Cubic box in the MDAnalysis format and the cutoff in Å
BOX = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
CUTOFF = 9.0


def pair_potential(r: np.ndarray) -> np.ndarray:
    """
    Model pair potential with a well of unit depth at 4 Å, in reduced energy units.

    Parameters
    ----------
    r : ndarray
        Distances in Å.

    Returns
    -------
    V : ndarray
        Pair potential at the distances.
    """

    # Gaussian well of unit depth
    return -np.exp(-(np.asarray(r) - 4.0)**2)


def universe_from_positions(positions: np.ndarray,
                            velocities: np.ndarray | None=None) -> mda.Universe:
    """
    Create a Universe of unit-mass atoms from a trajectory in memory.

    Parameters
    ----------
    positions : ndarray
        Positions with the shape (N_frames, N_atoms, 3) in Å.
    velocities : ndarray, default=None
        Velocities with the same shape. If None, all velocities are zero.

    Returns
    -------
    u : Universe
        Universe with the trajectory in memory.
    """

    # Universe with unit masses and the trajectory in memory
    n_atoms = positions.shape[1]
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr('masses', np.ones(n_atoms))
    if velocities is None:
        velocities = np.zeros_like(positions)
    u.load_new(positions, format=MemoryReader, velocities=velocities, dimensions=BOX)

    return u


class TestHillBoundClusters(unittest.TestCase):
    """
    Test the pair and cluster energy criteria in single frames.
    """

    def test_pair(self):
        """
        Test that a pair at rest in the well is bound and a fast pair is not.
        """

        # Two atoms 4 Å apart with unit masses and a unit kinetic-energy factor
        positions = np.array([[48.0, 50.0, 50.0], [52.0, 50.0, 50.0]])
        masses = np.ones(2)

        # At rest the pair energy is the well depth, -1
        at_rest = np.zeros((2, 3))
        clusters = gs.hill_bound_clusters(positions, at_rest, masses, BOX, CUTOFF, pair_potential, 1.0, min_size=2)
        self.assertEqual(clusters, [(0, 1)])

        # A relative speed of 3 gives a kinetic energy of 2.25, which exceeds the well depth
        moving = np.array([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
        clusters = gs.hill_bound_clusters(positions, moving, masses, BOX, CUTOFF, pair_potential, 1.0, min_size=2)
        self.assertEqual(clusters, [])

    def test_cluster_criterion(self):
        """
        Test that an atom bound to a dimer as a whole, but to neither of its atoms, counts only with the
        cluster criterion.
        """

        # Dimer 4 Å apart vibrating with speeds ±0.8, and a third atom at rest 5.5 Å from both of its atoms
        height = np.sqrt(5.5**2 - 2.0**2)
        positions = np.array([[48.0, 50.0, 50.0], [52.0, 50.0, 50.0], [50.0, 50.0 + height, 50.0]])
        velocities = np.array([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])
        masses = np.ones(3)

        # The third atom has a positive energy with each atom of the dimer, so the pair criterion finds no trimer
        clusters = gs.hill_bound_clusters(positions, velocities, masses, BOX, CUTOFF, pair_potential, 1.0,
                                          min_size=3, criterion='pair')
        self.assertEqual(clusters, [])

        # Relative to the dimer at rest, it is bound by the sum of the two pair potentials
        clusters = gs.hill_bound_clusters(positions, velocities, masses, BOX, CUTOFF, pair_potential, 1.0,
                                          min_size=3, criterion='cluster')
        self.assertEqual(clusters, [(0, 1, 2)])

    def test_find_bound_clusters(self):
        """
        Test the frame-wise clusters of a trajectory of a trimer at rest.
        """

        # Equilateral trimer of side 4 Å in a single frame
        height = np.sqrt(3.0)*2.0
        positions = np.array([[[48.0, 50.0, 50.0], [52.0, 50.0, 50.0], [50.0, 50.0 + height, 50.0]]])
        u = universe_from_positions(positions)
        frame_clusters = gs.find_bound_clusters(u, u.atoms, CUTOFF, pair_potential, 1.0, min_size=3)
        self.assertEqual(frame_clusters, {0: [(0, 1, 2)]})


class TestClusterEvents(unittest.TestCase):
    """
    Test the linking of the frame-wise clusters into events and the bridging of gaps.
    """

    def test_gap_tolerance(self):
        """
        Test that a gap within the tolerance continues an event and a longer gap ends it.
        """

        # Cluster of atoms 1-3 in frames 0, 1 and 3, and a separate cluster in frame 10
        frame_clusters = {0: [(1, 2, 3)], 1: [(1, 2, 3)], 3: [(1, 2, 3)], 10: [(4, 5, 6)]}

        # Without a gap tolerance, frame 3 starts a new event
        events = gs.bound_cluster_events(frame_clusters, 1e-15)
        self.assertEqual([(event['start_frame'], event['end_frame']) for event in events], [(0, 1), (3, 3), (10, 10)])

        # A one-frame gap is bridged
        events = gs.bound_cluster_events(frame_clusters, 1e-15, max_gap_frames=1)
        self.assertEqual([(event['start_frame'], event['end_frame']) for event in events], [(0, 3), (10, 10)])
        self.assertAlmostEqual(events[0]['lifetime_ps'], 4e-3)
        np.testing.assert_array_equal(events[0]['frames'], [0, 1, 3])

    def test_merging_of_events(self):
        """
        Test that a cluster overlapping two ongoing events merges them.
        """

        # Two trimers that join into a hexamer
        frame_clusters = {5: [(1, 2, 3), (4, 5, 6)], 6: [(1, 2, 3, 4, 5, 6)]}
        events = gs.bound_cluster_events(frame_clusters, 1e-15)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['atoms'], (1, 2, 3, 4, 5, 6))
        self.assertEqual(events[0]['max_size'], 6)

    def test_bridging(self):
        """
        Test that two events are bridged only if their atoms stay together during the gap.
        """

        # Trimer of side 4 Å held in place over ten frames
        height = np.sqrt(3.0)*2.0
        trimer = np.array([[48.0, 50.0, 50.0], [52.0, 50.0, 50.0], [50.0, 50.0 + height, 50.0]])
        positions = np.repeat(trimer[np.newaxis], 10, axis=0)
        events = [{'start_frame': 0, 'end_frame': 2, 'lifetime_ps': 3e-3, 'atoms': (0, 1, 2),
                   'first_atoms': (0, 1, 2), 'last_atoms': (0, 1, 2), 'max_size': 3,
                   'frames': np.arange(0, 3), 'sizes': np.full(3, 3)},
                  {'start_frame': 6, 'end_frame': 9, 'lifetime_ps': 4e-3, 'atoms': (0, 1, 2),
                   'first_atoms': (0, 1, 2), 'last_atoms': (0, 1, 2), 'max_size': 3,
                   'frames': np.arange(6, 10), 'sizes': np.full(4, 3)}]

        # The atoms stay within the cutoff during the gap, so the events merge
        bridged = gs.bridge_bound_cluster_events(universe_from_positions(positions), events, CUTOFF, 1e-15)
        self.assertEqual(len(bridged), 1)
        self.assertEqual((bridged[0]['end_frame'], bridged[0]['n_bridges']), (9, 1))
        self.assertAlmostEqual(bridged[0]['lifetime_ps'], 10e-3)

        # An atom that leaves during the gap separates the events
        positions[4, 2] = [50.0, 90.0, 50.0]
        bridged = gs.bridge_bound_cluster_events(universe_from_positions(positions), events, CUTOFF, 1e-15)
        self.assertEqual(len(bridged), 2)

    def test_time_series_energies(self):
        """
        Test that the pair, atom and internal energies of a vibrating pair coincide.
        """

        # Pair 4 Å apart with speeds ±0.5, so that each energy is 0.25 - 1
        positions = np.array([[[48.0, 50.0, 50.0], [52.0, 50.0, 50.0]]])
        velocities = np.array([[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]])
        u = universe_from_positions(positions, velocities)
        series = gs.bound_cluster_time_series(u, (0, 1), 0, 0, pair_potential, 1.0)
        self.assertAlmostEqual(series['pair_energies'][0, 0], -0.75, places=6)
        np.testing.assert_allclose(series['atom_energies'][0], [-0.75, -0.75], atol=1e-6)
        self.assertAlmostEqual(series['internal_energy'][0], -0.75, places=6)


if __name__ == '__main__':
    unittest.main()
