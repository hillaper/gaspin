# gaspin

gaspin is a Python package for the analysis of molecular dynamics simulations of atomic gases. It finds and
classifies the collisions between the atoms, from fleeting two-body encounters to bound van der Waals complexes,
and turns them into collision statistics and ensemble averages of distance-dependent properties, such as NMR
parameters.

## Description
gaspin analyses molecular dynamics (MD) trajectories of atomic gases. It finds the collision events between the
atoms and classifies them into two-body scattering, multi-body scattering, perturbed scattering and van der Waals
events, using only the interatomic distances and the pair potential. From the classified events it gives the event
lifetimes, the fractions of time spent in each kind of event, equilibrium constants and collision rates, and the
time-weighted averages of any property that depends on the interatomic distance, such as NMR shielding or spin-spin
coupling. The same ensemble averages are also available from the radial distribution function and from the second
virial coefficient of the property. Energetically bound clusters of three or more atoms are found independently
from the positions and velocities.

The trajectories are read with [MDAnalysis](https://www.mdanalysis.org), so any format it supports can be used.

## Installation
gaspin requires Python 3.11 or later. Install it from source:

```bash
git clone https://github.com/hillaper/gaspin.git
cd gaspin
pip install .
```

For development, install it in editable mode with the test dependencies:

```bash
pip install -e ".[test]"
```

## Usage
```python
import MDAnalysis as mda
import gaspin as gs

u = mda.Universe("gas.pdb", "gas.xtc")
collisions = gs.find_collisions(u, u.atoms, u.atoms, cutoff=8.0)
events = gs.process_collisions(collisions)
categorized = gs.categorize_collisions(events, 8.0, 3.816, 0.65, 0.2)
```

The example notebook in [`examples`](examples) goes through the whole workflow on a short argon trajectory.

## Tests
```bash
python -m pytest
```

## License
This project is licensed under the [MIT License](LICENSE).

## Citation
If you use gaspin, please cite the accompanying publication:

> [Citation of the publication, to be added once it is published.]

The software itself can be cited as described in [`CITATION.cff`](CITATION.cff).

## Use of AI
gaspin was developed by Perttu Hilla together with Claude (Anthropic), an AI assistant, which contributed to the
code, the tests and the documentation. All of it was reviewed by the author, who is responsible for its content.
