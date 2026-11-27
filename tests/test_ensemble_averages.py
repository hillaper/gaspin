"""
Tests for the ensemble averages from the radial distribution function and the pair potential.
"""

import unittest

import numpy as np
import scipy.constants as const

import gaspin as gs


class TestEnsembleAverages(unittest.TestCase):
    """
    Test the radial integrals against analytic results.
    """

    def setUp(self):
        """
        Create a fine radial grid up to 10 Å.
        """

        # Radial grid in Å
        self.r_max = 10.0
        self.r = np.linspace(0.0, self.r_max, 20001)

    def test_coordination_number_of_ideal_gas(self):
        """
        Test that a uniform RDF gives the number of partners in a sphere.
        """

        # 99 partners in a volume of 10^6 Å^3 and a uniform RDF
        volume = 1e6
        number = gs.coordination_number_from_rdf(self.r, np.ones_like(self.r), volume, N_other_atoms=99)
        expected = 99 / volume * 4/3*np.pi*self.r_max**3
        self.assertAlmostEqual(number, expected, delta=1e-6*expected)

    def test_running_average(self):
        """
        Test that the running average ends at the full-range average.
        """

        # Property r^-6 on a grid from 3 to 10 Å with a uniform RDF
        r = np.linspace(3.0, self.r_max, 20001)
        average, running = gs.ensemble_average_from_rdf(r, r**-6, np.ones_like(r), 1e3, cumulative=True)
        self.assertAlmostEqual(running[-1], average)

        # Analytic integral of 4 pi r^-4 from 3 to 10 Å
        expected = 4*np.pi*(3.0**-3 - 10.0**-3)/3 / 1e3
        self.assertAlmostEqual(average, expected, delta=1e-5*expected)

    def test_second_virial_coefficient(self):
        """
        Test the virial coefficient without a potential and with a constant potential.
        """

        # Without a potential, the coefficient of a unit property is the volume of the sphere
        unit_property = np.ones_like(self.r)
        zero_potential = np.zeros_like(self.r)
        B_T = gs.second_virial_coefficient(self.r, unit_property, zero_potential, 300.0)
        sphere_volume = 4/3*np.pi*self.r_max**3
        self.assertAlmostEqual(B_T, sphere_volume, delta=1e-6*sphere_volume)

        # A constant potential of -k_B T scales it by e
        constant_potential = np.full_like(self.r, -const.k*300.0)
        B_T = gs.second_virial_coefficient(self.r, unit_property, constant_potential, 300.0)
        self.assertAlmostEqual(B_T, np.e*sphere_volume, delta=1e-6*sphere_volume)


if __name__ == '__main__':
    unittest.main()
