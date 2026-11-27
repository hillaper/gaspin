"""
Plotting functions of gaspin.

The functions are available under the package namespace as ``gs.visualisation``, for example::

    import gaspin as gs
    gs.visualisation.set_plot_style()
    fig = gs.visualisation.plot_collision_counts(gs.collision_counts(collision_stats))
"""

from ._plots import (
    CATEGORY_COLOURS,
    CATEGORY_ORDER,
    CATEGORY_SHORT_TITLES,
    CATEGORY_TITLES,
    plot_all_event_counts_by_lifetime,
    plot_all_event_lifetimes,
    plot_all_event_num_atoms,
    plot_bound_cluster_event,
    plot_collision_counts,
    plot_distance_series_for_event,
    plot_event_counts_by_lifetime,
    plot_event_lifetimes,
    plot_event_num_atoms,
    plot_property_series_for_event,
    set_plot_style,
)

__all__ = [
    "CATEGORY_COLOURS",
    "CATEGORY_ORDER",
    "CATEGORY_SHORT_TITLES",
    "CATEGORY_TITLES",
    "plot_all_event_counts_by_lifetime",
    "plot_all_event_lifetimes",
    "plot_all_event_num_atoms",
    "plot_bound_cluster_event",
    "plot_collision_counts",
    "plot_distance_series_for_event",
    "plot_event_counts_by_lifetime",
    "plot_event_lifetimes",
    "plot_event_num_atoms",
    "plot_property_series_for_event",
    "set_plot_style",
]
