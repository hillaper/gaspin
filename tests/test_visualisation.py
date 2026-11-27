"""
Smoke tests for the plotting functions: each function runs on small data and returns a figure.
"""

import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import gaspin as gs
from tests.synthetic_systems import CUTOFF, event_dataframe


class TestPlots(unittest.TestCase):
    """
    Test that the plotting functions return figures.
    """

    def setUp(self):
        """
        Create one event in each category and their statistics.
        """

        # One event of atom 0 in each category, with a third body in the MBS event
        distances = np.array([8.0, 5.0, 4.0, 5.0, 8.0])
        third_body = np.array([np.nan, 8.5, 8.5, np.nan, np.nan])
        self.categorized = {category: {0: [event_dataframe({1: distances})]}
                            for category in gs.visualisation.CATEGORY_ORDER}
        self.categorized['multi_body_scattering'] = {0: [event_dataframe({1: distances, 2: third_body})]}
        self.stats, _ = gs.collision_statistics(self.categorized, 1e-15, 10)
        self.lifetime_ranges = [(0.0, 1.0), (1.0, np.inf)]

    def tearDown(self):
        """
        Close all figures.
        """

        # Free the memory of the figures
        plt.close('all')

    def test_event_plots(self):
        """
        Test the plots of single events and of bound-cluster time series.
        """

        # Distance and property series of one event
        gs.visualisation.set_plot_style()
        fig = gs.visualisation.plot_distance_series_for_event(self.categorized, 'multi_body_scattering', 0, 0, CUTOFF,
                                                              x_axis='time', dt=1e-15)
        self.assertIsInstance(fig, Figure)
        series = gs.property_R_all_partners_time_series(self.categorized, 'multi_body_scattering', 0, 0, lambda r: r)
        fig = gs.visualisation.plot_property_series_for_event(self.categorized, series, 'R', 'multi_body_scattering',
                                                              0, 0)
        self.assertIsInstance(fig, Figure)

        # A missing atom returns no figure
        self.assertIsNone(gs.visualisation.plot_distance_series_for_event(self.categorized, 'vdw', 7, 0, CUTOFF))

        # Time series of a bound pair
        series = {'frames': np.arange(3), 'pairs': [(0, 1)], 'distances': np.full((3, 1), 4.0),
                  'pair_energies': np.full((3, 1), -1.0), 'atoms': [0, 1], 'atom_energies': np.full((3, 2), -1.0),
                  'internal_energy': np.full(3, -1.0)}
        event = {'start_frame': 0, 'end_frame': 2}
        fig = gs.visualisation.plot_bound_cluster_event(series, 1e-15, CUTOFF, event=event, r_min=4.0)
        self.assertIsInstance(fig, Figure)

    def test_statistics_plots(self):
        """
        Test the plots of the event statistics.
        """

        # Counts, lifetimes and numbers of atoms
        figures = [gs.visualisation.plot_collision_counts(gs.collision_counts(self.stats)),
                   gs.visualisation.plot_event_lifetimes(self.stats, 'vdw'),
                   gs.visualisation.plot_all_event_lifetimes(self.stats),
                   gs.visualisation.plot_event_num_atoms(self.stats, 'multi_body_scattering'),
                   gs.visualisation.plot_all_event_num_atoms(self.stats)]

        # Counts in lifetime ranges
        counts = gs.collision_counts_by_lifetime(self.stats, self.lifetime_ranges)
        figures.append(gs.visualisation.plot_event_counts_by_lifetime(counts['vdw'], self.lifetime_ranges, 'vdw'))
        figures.append(gs.visualisation.plot_all_event_counts_by_lifetime(counts, self.lifetime_ranges,
                                                                          event_stats=self.stats))
        self.assertTrue(all(isinstance(fig, Figure) for fig in figures))


if __name__ == '__main__':
    unittest.main()
