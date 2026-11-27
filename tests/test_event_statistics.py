"""
Tests for the event statistics, the atom-time fractions and the time-weighted property averages.
"""

import unittest

import numpy as np

import gaspin as gs
from tests.synthetic_systems import event_dataframe

# Three atoms over 20 frames of 1 fs, with atom 2 free throughout
N_ATOMS = 3
TOTAL_FRAMES = 20
DT = 1e-15


def mirrored_event(partner: int,
                   distances: list,
                   start_frame: int):
    """
    Create an event of one atom with a single partner.

    Parameters
    ----------
    partner : int
        Atom index of the partner.
    distances : list
        Distance to the partner in each frame in Å.
    start_frame : int
        First frame of the event.

    Returns
    -------
    event_df : DataFrame
        Event in the format of process_collisions().
    """

    # Event with one partner present in every frame
    return event_dataframe({partner: np.array(distances, dtype=float)}, start_frame=start_frame)


class TestEventStatistics(unittest.TestCase):
    """
    Test the statistics of a hand-built set of classified events.
    """

    def setUp(self):
        """
        Create a TBS event in frames 2-5 and a vdW event in frames 10-14 between atoms 0 and 1.
        """

        # Distances of the two events, seen by both atoms of the pair
        tbs_distances = [8.0, 5.0, 4.0, 7.0]
        vdw_distances = [6.0, 4.5, 4.4, 4.6, 6.0]
        self.categorized = {'two_body_scattering': {0: [mirrored_event(1, tbs_distances, 2)],
                                                    1: [mirrored_event(0, tbs_distances, 2)]},
                            'multi_body_scattering': {},
                            'perturbed_scattering': {},
                            'vdw': {0: [mirrored_event(1, vdw_distances, 10)],
                                    1: [mirrored_event(0, vdw_distances, 10)]}}
        self.all_distances = 2*(tbs_distances + vdw_distances)

        # Statistics, cluster counts and cluster births and deaths
        self.stats, self.counts, self.changes = gs.collision_statistics(self.categorized,
                                                                        DT,
                                                                        TOTAL_FRAMES,
                                                                        return_cluster_changes=True)

    def test_lifetimes_and_counts(self):
        """
        Test the event lifetimes, the numbers of events and the cluster counts.
        """

        # Lifetimes in ps from the number of frames
        self.assertAlmostEqual(self.stats['two_body_scattering'][0][0]['lifetime_ps'], 4e-3)
        self.assertAlmostEqual(self.stats['vdw'][1][0]['lifetime_ps'], 5e-3)
        self.assertEqual(self.stats['vdw'][0][0]['num_atoms_involved'], 2)

        # The two views of the same pair form one cluster per frame
        self.assertEqual(gs.collision_counts(self.stats)['two_body_scattering'], 2)
        np.testing.assert_array_equal(np.flatnonzero(self.counts['two_body_scattering']), [2, 3, 4, 5])
        self.assertTrue(np.all(self.counts['two_body_scattering'][2:6] == 1))

        # One birth and one death of each cluster
        self.assertEqual(self.changes['vdw']['births'][10], 1)
        self.assertEqual(self.changes['vdw']['deaths'][15], 1)
        self.assertEqual(np.sum(self.changes['two_body_scattering']['births']), 1)

    def test_atom_time_fractions(self):
        """
        Test the atom-time fractions and their sum.
        """

        # Two atoms in each event, over the atom-time of three atoms and 20 frames
        fractions = gs.atom_time_fractions(self.stats, N_ATOMS, TOTAL_FRAMES, DT)
        self.assertAlmostEqual(fractions['two_body_scattering'], 8/60)
        self.assertAlmostEqual(fractions['vdw'], 10/60)
        self.assertAlmostEqual(sum(fractions.values()), 1.0)

    def test_exact_decomposition_of_average(self):
        """
        Test that the fractions and the time-weighted averages reproduce the trajectory average exactly.
        """

        # Time-weighted category averages of a property equal to the distance
        averages = gs.average_property_R_time_weighted(self.categorized, lambda r: r)
        fractions = gs.atom_time_fractions(self.stats, N_ATOMS, TOTAL_FRAMES, DT)

        # The property is zero for free atoms, so the trajectory average is the sum over the event frames
        decomposed = sum(fractions[category]*averages[category]['average_property']
                         for category in averages if averages[category]['total_frames'] > 0)
        direct = np.sum(self.all_distances) / (N_ATOMS*TOTAL_FRAMES)
        self.assertAlmostEqual(decomposed, direct)

        # The parallel evaluation gives the same averages
        parallel = gs.average_property_R_time_weighted(self.categorized, lambda r: r, n_jobs=2)
        self.assertAlmostEqual(parallel['vdw']['average_property'], averages['vdw']['average_property'])

    def test_equilibrium_constants_and_rates(self):
        """
        Test the equilibrium constant with the exact monomer number and the formation rate.
        """

        # Mean number of vdW clusters 5/20 and monomers f_0 N
        fractions = gs.atom_time_fractions(self.stats, N_ATOMS, TOTAL_FRAMES, DT)
        free_monomers = fractions['free_monomers'] * N_ATOMS
        constants, _ = gs.cluster_formation_equilibrium_values(self.counts,
                                                               self.stats,
                                                               N_ATOMS,
                                                               volume=1000.0,
                                                               free_monomers=free_monomers)
        self.assertAlmostEqual(constants['vdw'], (5/20) / free_monomers**2 * 1000.0)

        # One formation over the time spanned by the frames
        rates = gs.cluster_formation_rates(self.counts, DT, cluster_changes=self.changes)
        self.assertAlmostEqual(rates['vdw']['formation_rate'], 1 / ((TOTAL_FRAMES - 1)*DT))
        self.assertAlmostEqual(rates['vdw']['breakup_rate'], 1 / ((TOTAL_FRAMES - 1)*DT))

    def test_combine_and_sort(self):
        """
        Test the combination of two segments, the lifetime ranges and the sorting.
        """

        # Two copies of the segment, with renumbered events and concatenated counts
        combined_stats, combined_counts = gs.combine_collision_statistics([(self.stats, self.counts),
                                                                           (self.stats, self.counts)])
        self.assertEqual([event['event_index'] for event in combined_stats['vdw'][0]], [0, 1])
        self.assertEqual(len(combined_counts['vdw']), 2*TOTAL_FRAMES)

        # Lifetime ranges in ps
        by_lifetime = gs.collision_counts_by_lifetime(self.stats, [(0.0, 4.5e-3), (4.5e-3, np.inf)])
        self.assertEqual(by_lifetime['two_body_scattering'], [2, 0])
        self.assertEqual(by_lifetime['vdw'], [0, 2])

        # Pooled events sorted by lifetime, longest first, with their reference atoms
        sorted_stats = gs.sort_collision_statistics(combined_stats, 'lifetime_ps', atomwise=False)
        self.assertEqual(len(sorted_stats['vdw']), 4)
        self.assertIn(sorted_stats['vdw'][0]['atom_index'], [0, 1])


class TestPropertySeries(unittest.TestCase):
    """
    Test the property series of the main partner and of all partners.
    """

    def test_pairwise_sum(self):
        """
        Test that the pairwise sum adds the partners present in each frame.
        """

        # Event with a main partner and a partner present only in the middle frame
        categorized = {'multi_body_scattering': {0: [event_dataframe({1: np.array([8.0, 6.0, 7.0]),
                                                                       2: np.array([np.nan, 8.5, np.nan])})]}}
        series, pairwise_sum = gs.property_R_all_partners_time_series(categorized,
                                                                      'multi_body_scattering',
                                                                      0,
                                                                      0,
                                                                      lambda r: 2*r,
                                                                      use_pairwise_sum=True)
        np.testing.assert_allclose(pairwise_sum, [16.0, 29.0, 14.0])
        main_series = gs.property_R_main_partner_time_series(categorized, 'multi_body_scattering', 0, 0, lambda r: 2*r)
        np.testing.assert_allclose(main_series, [16.0, 12.0, 14.0])


if __name__ == '__main__':
    unittest.main()
