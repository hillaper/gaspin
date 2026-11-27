# Changelog

## 0.1.0 (unreleased)

First version as gaspin, previously the unpackaged module collection gpsp_tools.

- Package structure of `src/gaspin`, with the public functions under `import gaspin as gs` and the plotting
  functions under `gs.visualisation`.
- `categorize_collisions()` takes the well tolerance and the minimum prominence of the distance minima as
  arguments, so that the classification applies to any atomic gas. The unused `dt` argument is removed, and the
  default number of parallel processes is 1.
- `process_collisions()` anchors events only to the allowed atoms when `allowed_atom_indices` is given.
- `first_order_virial_coefficient()` is renamed `second_virial_coefficient()`.
- The `total_monomers` argument of `cluster_formation_equilibrium_values()` is renamed `n_atoms`.
- The event-weighted property averages (`average_property_R()` and its helpers) are removed. The time-weighted
  averages of `average_property_R_time_weighted()` are consistent with the atom-time fractions.
- `interpolate_to_match()` is removed, use `numpy.interp()`.
- The plotting style is applied with `gs.visualisation.set_plot_style()` instead of on import. The plotting
  functions return their figures instead of showing them, and the distance-axis limits and the energy unit are
  arguments.
- Tests and an example notebook on a short argon trajectory are added.
