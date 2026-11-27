"""
Tests for the detection, grouping and classification of collision events.
"""

import unittest

import numpy as np

import gaspin as gs
from tests.synthetic_systems import CUTOFF, event_dataframe, three_atom_universe

# Parameters of the classification: minimum of the pair potential, well tolerance and prominence in Å
R_E = 4.4
WELL_TOLERANCE = 0.75
PROMINENCE = 0.25


def dip_series(n_dips: int,
               r_far: float=8.9,
               r_min: float=3.8,
               r_between: float=6.5,
               n_points: int=30) -> np.ndarray:
    """
    Create a distance series that enters from r_far, dips n_dips times to r_min with maxima at r_between in
    between, and leaves to r_far.

    Parameters
    ----------
    n_dips : int
        Number of distance minima.
    r_far : float, default=8.9
        Distance at the start and the end of the series in Å.
    r_min : float, default=3.8
        Distance at each minimum in Å.
    r_between : float, default=6.5
        Distance at the maxima between the minima in Å.
    n_points : int, default=30
        Number of points on each falling or rising branch.

    Returns
    -------
    series : ndarray
        Distance series in Å.
    """

    # Turning points of the series: far, minimum, (between, minimum)..., far
    turning_points = [r_far] + [r_min, r_between] * (n_dips - 1) + [r_min, r_far]

    # Linear branches between consecutive turning points, without repeating the shared points
    branches = [np.linspace(turning_points[0], turning_points[1], n_points)]
    for start, stop in zip(turning_points[1:-1], turning_points[2:]):
        branches.append(np.linspace(start, stop, n_points)[1:])

    return np.concatenate(branches)


class TestFindAndProcessCollisions(unittest.TestCase):
    """
    Test the frame-wise collision records and their grouping into events on a three-atom trajectory.
    """

    def setUp(self):
        """
        Create the three-atom trajectory and find its collisions.
        """

        # Trajectory in which atoms 1 and 2 visit atom 0 one after the other
        self.u = three_atom_universe()
        self.collisions = gs.find_collisions(self.u, self.u.atoms, self.u.atoms, CUTOFF)

    def test_find_collisions(self):
        """
        Test the frames and partners recorded for each reference atom.
        """

        # Atom 0 sees atom 1 in frames 2-5 and atom 2 in frames 4-8
        frames_atom_0 = [record['frame'] for record in self.collisions[0]]
        self.assertEqual(frames_atom_0, [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(self.collisions[0][2]['atoms2'], [1, 2])
        np.testing.assert_allclose(self.collisions[0][2]['distances'], [5.0, 8.0], rtol=1e-6)

        # Atoms 1 and 2 see only atom 0
        self.assertEqual([record['frame'] for record in self.collisions[1]], [2, 3, 4, 5])
        self.assertEqual([record['frame'] for record in self.collisions[2]], [4, 5, 6, 7, 8])
        self.assertTrue(all(record['atoms2'] == [0] for record in self.collisions[1]))

    def test_process_collisions_first_partner_rule(self):
        """
        Test that the event of atom 0 ends when its anchor leaves and a new event starts with the next partner.
        """

        # Events of atom 0: anchored to atom 1 in frames 2-5, then to atom 2 in frames 6-8
        events = gs.process_collisions(self.collisions)
        self.assertEqual(len(events[0]), 2)
        self.assertEqual(list(events[0][0]['frame']), [2, 3, 4, 5])
        self.assertEqual(list(events[0][1]['frame']), [6, 7, 8])

        # Each frame of a reference atom belongs to exactly one of its events
        frames = np.concatenate([event['frame'].to_numpy() for event in events[0]])
        self.assertEqual(len(frames), len(set(frames)))

    def test_process_collisions_partner_specific(self):
        """
        Test that overlapping partner-specific events are created with the partner moved first.
        """

        # Events of atom 0: with atom 1 in frames 2-5 and with atom 2 in frames 4-8
        events = gs.process_collisions_partner_specific(self.collisions)
        self.assertEqual(len(events[0]), 2)
        self.assertEqual(list(events[0][0]['frame']), [2, 3, 4, 5])
        self.assertEqual(list(events[0][1]['frame']), [4, 5, 6, 7, 8])

        # The partner of the event comes first in every row, with its own distance first
        self.assertTrue(all(atoms2[0] == 2 for atoms2 in events[0][1]['atoms2']))
        np.testing.assert_allclose(events[0][1]['distances'].iloc[0], [8.0, 5.0], rtol=1e-6)

    def test_allowed_atom_indices(self):
        """
        Test that the partner filter keeps only the frames with an allowed partner.
        """

        # With only atom 2 allowed, atom 0 has one event in frames 4-8
        events = gs.process_collisions(self.collisions, allowed_atom_indices=[2])
        self.assertEqual(len(events[0]), 1)
        self.assertEqual(list(events[0][0]['frame']), [4, 5, 6, 7, 8])


class TestMainPartner(unittest.TestCase):
    """
    Test the identification of the main partner and the distance series.
    """

    def test_main_partner_index(self):
        """
        Test that the partner present in most frames is the main partner, with ties resolved by appearance.
        """

        # Partner 5 is present in three frames and partner 3 in two
        event_df = event_dataframe({3: [np.nan, 6.0, 7.0], 5: [8.0, 7.0, 6.0]})
        self.assertEqual(gs.main_partner_index(event_df), 5)

        # In a tie, the partner that appears first wins
        event_df = event_dataframe({3: [np.nan, 6.0], 5: [8.0, np.nan]})
        self.assertEqual(gs.main_partner_index(event_df), 5)

    def test_distance_series(self):
        """
        Test the distance series of the main partner and of all partners, with NaN where a partner is absent.
        """

        # Event with a main partner and a partner present only in the middle frame
        event_df = event_dataframe({1: [8.0, 6.0, 7.0], 2: [np.nan, 8.5, np.nan]})
        np.testing.assert_allclose(gs.main_partner_distance_series(event_df), [8.0, 6.0, 7.0])
        series = gs.distance_series(event_df)
        np.testing.assert_allclose(series[2], [np.nan, 8.5, np.nan])


class TestCategorizeCollisions(unittest.TestCase):
    """
    Test the classification rules on events with designed distance series.
    """

    def setUp(self):
        """
        Create one event of each kind for reference atom 0.
        """

        # Single minimum with one partner (TBS) and with a brief third body (MBS)
        tbs = dip_series(1)
        third_body = np.full(len(tbs), np.nan)
        third_body[25:35] = 8.5

        # Two minima in the well (PS), four minima in the well (vdW) and a shallow orbit in the well (vdW)
        ps = dip_series(2)
        vdw_minima = dip_series(4, r_min=5.0)
        vdw_orbit = R_E + 0.05*np.sin(np.linspace(0.0, 6*np.pi, 100))

        # Events of reference atom 0 in a known order
        self.events = {0: [event_dataframe({1: tbs}),
                           event_dataframe({1: tbs, 2: third_body}),
                           event_dataframe({1: ps}),
                           event_dataframe({1: vdw_minima}),
                           event_dataframe({1: vdw_orbit})]}

    def categorize(self,
                   **kwargs) -> dict:
        """
        Classify the designed events with the test parameters.

        Parameters
        ----------
        **kwargs
            Optional arguments passed to categorize_collisions().

        Returns
        -------
        counts : dict
            Category mapped to the number of events of atom 0 in it.
        """

        # Classify and count the events of each category
        categorized = gs.categorize_collisions(self.events, CUTOFF, R_E, WELL_TOLERANCE, PROMINENCE, **kwargs)
        return {category: len(categorized[category].get(0, [])) for category in categorized}

    def test_categories(self):
        """
        Test that each designed event falls into its category.
        """

        # One TBS, one MBS, one PS and two vdW events
        counts = self.categorize()
        self.assertEqual(counts, {'two_body_scattering': 1,
                                  'multi_body_scattering': 1,
                                  'perturbed_scattering': 1,
                                  'vdw': 2})

    def test_well_occupancy_rule(self):
        """
        Test that the shallow orbit is vdW only through the well-occupancy rule.
        """

        # Without the rule, the shallow orbit has no resolved minimum and becomes TBS
        counts = self.categorize(vdw_well_occupancy=None)
        self.assertEqual(counts['vdw'], 1)
        self.assertEqual(counts['two_body_scattering'], 2)

    def test_well_tolerance(self):
        """
        Test that the minima count as well minima only within the well tolerance.
        """

        # With a narrow well, the minima at 5.0 Å lie above it, so that the four-minimum event becomes PS
        categorized = gs.categorize_collisions(self.events, CUTOFF, R_E, 0.5, PROMINENCE, vdw_well_occupancy=None)
        self.assertEqual(len(categorized['perturbed_scattering'][0]), 2)

    def test_parallel_matches_serial(self):
        """
        Test that the parallel classification gives the same result as the serial one.
        """

        # Same counts with one and two processes
        self.assertEqual(self.categorize(n_jobs=1), self.categorize(n_jobs=2))


if __name__ == '__main__':
    unittest.main()
