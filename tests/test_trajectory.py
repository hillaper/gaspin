"""
Tests for the trajectory helpers and the extraction of event trajectories.
"""

import os
import tempfile
import unittest

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

import gaspin as gs
from tests.synthetic_systems import BOX_LENGTH, CUTOFF, DISTANCES_ATOM_1, DISTANCES_ATOM_2, three_atom_universe


class TestTrajectoryHelpers(unittest.TestCase):
    """
    Test the RDF, the angular momentum and the connecting vector.
    """

    def test_rdf_of_ideal_gas(self):
        """
        Test that uniformly random positions give an RDF close to unity.
        """

        # 400 random atoms in 50 frames of a 60 Å box
        rng = np.random.default_rng(1)
        positions = rng.uniform(0.0, 60.0, size=(50, 400, 3))
        u = mda.Universe.empty(400, trajectory=True)
        u.load_new(positions, format=MemoryReader, dimensions=[60.0, 60.0, 60.0, 90.0, 90.0, 90.0])

        # Mean RDF between 5 and 15 Å, where the statistics are good
        r, g_r = gs.rdf(u.atoms, u.atoms, nbins=60, range=(0.0, 15.0))
        self.assertAlmostEqual(np.mean(g_r[r > 5.0]), 1.0, delta=0.02)

    def test_angular_momentum_and_connector(self):
        """
        Test the angular momentum and the connecting vector of two counter-rotating atoms.
        """

        # Two unit masses at ±1 Å along x moving at ±1 Å per time unit along y
        positions = np.array([[[51.0, 50.0, 50.0], [49.0, 50.0, 50.0]]])
        velocities = np.array([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
        u = mda.Universe.empty(2, trajectory=True)
        u.add_TopologyAttr('masses', [1.0, 1.0])
        u.load_new(positions, format=MemoryReader, velocities=velocities,
                   dimensions=[100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

        # Angular momentum about the centre of mass along z and the vector from atom 0 to atom 1
        np.testing.assert_allclose(gs.angular_momentum(u.atoms[[0]], u.atoms[[1]]), [0.0, 0.0, 2.0])
        np.testing.assert_allclose(gs.connector(u.atoms[[0]], u.atoms[[1]]), [-2.0, 0.0, 0.0])


class TestEventTrajectories(unittest.TestCase):
    """
    Test the positions and velocities extracted for a collision event.
    """

    def setUp(self):
        """
        Find the first event of atom 0 in the three-atom trajectory.
        """

        # Event of atom 0 anchored to atom 1 in frames 2-5, during which atom 2 enters
        self.u = three_atom_universe()
        events = gs.process_collisions(gs.find_collisions(self.u, self.u.atoms, self.u.atoms, CUTOFF))
        self.categorized = {'two_body_scattering': {0: [events[0][0]]}}

    def test_atom_order_and_positions(self):
        """
        Test that the reference atom, the main partner and the other partners come in this order.
        """

        # Four frames and three atoms: atom 0, its main partner 1 and the third body 2
        positions, velocities = gs.collision_event_trajectories(self.u, self.categorized, 'two_body_scattering', 0, 0)
        self.assertEqual(positions.shape, (4, 3, 3))
        self.assertEqual(velocities.shape, (4, 3, 3))

        # The reference atom is at the centre of the box and the partners at their distances
        np.testing.assert_allclose(positions[:, 0], np.full((4, 3), BOX_LENGTH / 2), atol=1e-4)
        np.testing.assert_allclose(positions[:, 1, 0] - positions[:, 0, 0], DISTANCES_ATOM_1[2:6], atol=1e-4)
        np.testing.assert_allclose(positions[:, 2, 1] - positions[:, 0, 1], DISTANCES_ATOM_2[2:6], atol=1e-4)

    def test_save(self):
        """
        Test that the positions and velocities of every event are saved.
        """

        # One pair of files for the single event
        with tempfile.TemporaryDirectory() as directory:
            gs.save_collision_event_trajectories(self.u, self.categorized, directory=directory)
            saved = np.load(os.path.join(directory, 'positions_two_body_scattering_atom0_event0.npy'))
            self.assertEqual(saved.shape, (4, 3, 3))
            self.assertTrue(os.path.exists(os.path.join(directory,
                                                        'velocities_two_body_scattering_atom0_event0.npy')))


if __name__ == '__main__':
    unittest.main()
