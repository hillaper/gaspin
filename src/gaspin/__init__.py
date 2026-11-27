"""
Public package namespace for gaspin.

gaspin analyses molecular dynamics trajectories of atomic gases. It finds the collision events between the
atoms, classifies them, extracts their statistics, and gives the ensemble averages of distance-dependent pair
properties, such as NMR parameters. The package is intended to be imported as::

    import gaspin as gs

The typical workflow is:

1. find_collisions() records, for every frame, the atoms within a cutoff distance of each reference atom.
2. process_collisions() groups these records into events from the point of view of each reference atom.
3. categorize_collisions() classifies the events into two-body scattering (TBS), multi-body scattering (MBS),
   perturbed scattering (PS) and van der Waals (vdW) events.
4. collision_statistics() and the functions that follow it give the event lifetimes, the numbers of atoms
   involved, the atom-time fractions, the equilibrium constants and the collision rates.
5. average_property_R_time_weighted() gives the time-weighted category averages of a property that depends
   only on the interatomic distance R, and ensemble_average_from_rdf() and second_virial_coefficient() give its
   ensemble average from the radial distribution function and from the pair potential.
6. find_bound_clusters() and bound_cluster_events() find the energetically bound clusters of atoms
   independently of the classification, from the positions and velocities.

Distances are in Å and the time step between frames is in seconds. The plotting functions are in
``gs.visualisation``.
"""

# Re-export the public core functionality under the package namespace.
from gaspin._core import (
    # bound clusters
    bound_cluster_events,
    bound_cluster_time_series,
    bridge_bound_cluster_events,
    find_bound_clusters,
    hill_bound_clusters,

    # collisions
    categorize_collisions,
    distance_series,
    find_collisions,
    main_partner_distance_series,
    main_partner_index,
    process_collisions,
    process_collisions_partner_specific,

    # ensemble averages
    coordination_number_from_rdf,
    ensemble_average_from_rdf,
    second_virial_coefficient,

    # event statistics
    atom_time_fractions,
    cluster_formation_equilibrium_values,
    cluster_formation_rates,
    collision_counts,
    collision_counts_by_lifetime,
    collision_statistics,
    combine_collision_statistics,
    sort_collision_statistics,

    # event trajectories
    collision_event_trajectories,
    save_collision_event_trajectories,

    # property averages
    average_property_R_time_weighted,
    property_R_all_partners_time_series,
    property_R_main_partner_time_series,
    property_time_series,

    # trajectory
    angular_momentum,
    center_and_wrap,
    connector,
    rdf,
)

# Re-export the plotting namespace.
from gaspin import visualisation

__version__ = "0.1.0"

__all__ = [
    # bound clusters
    "bound_cluster_events",
    "bound_cluster_time_series",
    "bridge_bound_cluster_events",
    "find_bound_clusters",
    "hill_bound_clusters",

    # collisions
    "categorize_collisions",
    "distance_series",
    "find_collisions",
    "main_partner_distance_series",
    "main_partner_index",
    "process_collisions",
    "process_collisions_partner_specific",

    # ensemble averages
    "coordination_number_from_rdf",
    "ensemble_average_from_rdf",
    "second_virial_coefficient",

    # event statistics
    "atom_time_fractions",
    "cluster_formation_equilibrium_values",
    "cluster_formation_rates",
    "collision_counts",
    "collision_counts_by_lifetime",
    "collision_statistics",
    "combine_collision_statistics",
    "sort_collision_statistics",

    # event trajectories
    "collision_event_trajectories",
    "save_collision_event_trajectories",

    # property averages
    "average_property_R_time_weighted",
    "property_R_all_partners_time_series",
    "property_R_main_partner_time_series",
    "property_time_series",

    # trajectory
    "angular_momentum",
    "center_and_wrap",
    "connector",
    "rdf",

    # visualisation
    "visualisation",
]
