"""
Internal core namespace of gaspin.

This module centralises the re-export of the internal core functionality of gaspin. In normal use, the public
package namespace should be preferred::

    import gaspin as gs

Direct imports from :mod:`gaspin._core` remain possible, but they are intended primarily for internal use,
testing and development.
"""

from ._bound_clusters import (
    bound_cluster_events,
    bound_cluster_time_series,
    bridge_bound_cluster_events,
    find_bound_clusters,
    hill_bound_clusters,
)
from ._collisions import (
    categorize_collisions,
    distance_series,
    find_collisions,
    main_partner_distance_series,
    main_partner_index,
    process_collisions,
    process_collisions_partner_specific,
)
from ._ensemble_averages import (
    coordination_number_from_rdf,
    ensemble_average_from_rdf,
    second_virial_coefficient,
)
from ._event_statistics import (
    atom_time_fractions,
    cluster_formation_equilibrium_values,
    cluster_formation_rates,
    collision_counts,
    collision_counts_by_lifetime,
    collision_statistics,
    combine_collision_statistics,
    sort_collision_statistics,
)
from ._event_trajectories import (
    collision_event_trajectories,
    save_collision_event_trajectories,
)
from ._property_averages import (
    average_property_R_time_weighted,
    property_R_all_partners_time_series,
    property_R_main_partner_time_series,
    property_time_series,
)
from ._trajectory import (
    angular_momentum,
    center_and_wrap,
    connector,
    rdf,
)
