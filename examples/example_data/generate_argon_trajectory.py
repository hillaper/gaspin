"""
Generate the small argon trajectory of the gaspin example notebooks with OpenMM.

The system is a Lennard-Jones model of argon (sigma = 3.4 Å, epsilon / k_B = 120 K) with 343 atoms at 120 K and the
number density of an ideal gas at 5 atm. The atoms start on a cubic lattice and are equilibrated with a Langevin
thermostat for 200 ps. The production run is 50 ps of constant-energy (NVE) dynamics with a 5 fs time step, saved
every 25 fs. OpenMM is needed only here, not by gaspin itself.

Usage: ``python generate_argon_trajectory.py``
"""

# Imports
import os

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

# Output directory of the trajectory files
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lennard-Jones parameters and mass of argon (rough values, not from any reference)
SIGMA = 3.4 * unit.angstrom
EPSILON = 120.0 * unit.kelvin * unit.MOLAR_GAS_CONSTANT_R
MASS = 39.948 * unit.amu

# State point: temperature, pressure of the ideal gas and number of atoms on a 7 x 7 x 7 lattice
TEMPERATURE = 120.0 * unit.kelvin
PRESSURE = 5.0 * unit.atmosphere
N_SIDE = 7

# Time step, equilibration length, production length and the number of steps between saved frames
TIME_STEP = 5.0 * unit.femtosecond
EQUILIBRATION_STEPS = 40000
PRODUCTION_STEPS = 10000
REPORT_INTERVAL = 5

# Cutoff of the Lennard-Jones interaction and the random seed
LJ_CUTOFF = 12.0 * unit.angstrom
SEED = 2026


def build_system(box_length: unit.Quantity) -> tuple[openmm.System, app.Topology]:
    """
    Build the OpenMM system and topology of the Lennard-Jones argon gas.

    Parameters
    ----------
    box_length : Quantity
        Edge of the cubic periodic box.

    Returns
    -------
    system : System
        OpenMM system with the Lennard-Jones interactions.
    topology : Topology
        Topology with one argon atom per residue.
    """

    # Cubic periodic box
    n_atoms = N_SIDE**3
    box_vectors = [openmm.Vec3(box_length.value_in_unit(unit.nanometer), 0.0, 0.0),
                   openmm.Vec3(0.0, box_length.value_in_unit(unit.nanometer), 0.0),
                   openmm.Vec3(0.0, 0.0, box_length.value_in_unit(unit.nanometer))] * unit.nanometer
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    # Lennard-Jones interactions with a periodic cutoff and no charges
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(LJ_CUTOFF)
    for _ in range(n_atoms):
        system.addParticle(MASS)
        nonbonded.addParticle(0.0, SIGMA, EPSILON)
    system.addForce(nonbonded)

    # Topology with one argon atom per residue
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(n_atoms):
        residue = topology.addResidue('AR', chain)
        topology.addAtom('AR', app.Element.getBySymbol('Ar'), residue)
    topology.setPeriodicBoxVectors(box_vectors)

    return system, topology


def main() -> None:
    """
    Equilibrate the gas and write the production trajectory and the topology.
    """

    # Box length from the ideal-gas number density n = P / (k_B T)
    n_atoms = N_SIDE**3
    number_density = PRESSURE / (unit.BOLTZMANN_CONSTANT_kB * TEMPERATURE)
    box_length = ((n_atoms / number_density)**(1/3)).in_units_of(unit.angstrom)
    print(f'Box length {box_length}')

    # System and the cubic lattice of starting positions
    system, topology = build_system(box_length)
    spacing = box_length.value_in_unit(unit.angstrom) / N_SIDE
    grid = (np.arange(N_SIDE) + 0.5) * spacing
    positions = np.array(np.meshgrid(grid, grid, grid, indexing='ij')).reshape(3, -1).T * unit.angstrom

    # Equilibration with a Langevin thermostat
    integrator = openmm.LangevinMiddleIntegrator(TEMPERATURE, 1.0 / unit.picosecond, TIME_STEP)
    integrator.setRandomNumberSeed(SEED)
    platform = openmm.Platform.getPlatformByName('CPU')
    simulation = app.Simulation(topology, system, integrator, platform, {'Threads': '12'})
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE, SEED)
    simulation.step(EQUILIBRATION_STEPS)

    # Production with constant energy, continuing from the equilibrated positions and velocities
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    production = app.Simulation(topology, system, openmm.VerletIntegrator(TIME_STEP), platform, {'Threads': '12'})
    production.context.setPositions(state.getPositions())
    production.context.setVelocities(state.getVelocities())

    # Topology file with the first production frame, and the trajectory and energies during production
    app.PDBFile.writeFile(topology, state.getPositions(), open(os.path.join(OUTPUT_DIR, 'argon.pdb'), 'w'))
    production.reporters.append(app.XTCReporter(os.path.join(OUTPUT_DIR, 'argon.xtc'), REPORT_INTERVAL))
    production.reporters.append(app.StateDataReporter(os.path.join(OUTPUT_DIR, 'argon_energies.csv'),
                                                      100 * REPORT_INTERVAL,
                                                      time=True,
                                                      kineticEnergy=True,
                                                      totalEnergy=True,
                                                      temperature=True))
    production.step(PRODUCTION_STEPS)


if __name__ == '__main__':
    main()
