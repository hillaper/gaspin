"""
Plotting functions for the collision events of gas-phase MD trajectories.

The functions plot the distance and property series of single events and the statistics of the classified
events (see gpsp_tools.md_analysis). The categories are drawn in a fixed order and colour: two-body scattering
(TBS) in blue, multi-body scattering (MBS) in orange, perturbed scattering (PS) in green and van der Waals (vdW)
events in red. A common plotting style is applied on import.
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np

import gpsp_tools.md_analysis as gpmda

# Order, colours and titles of the event categories
CATEGORY_ORDER = ['two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw']

CATEGORY_COLOURS = dict(zip(CATEGORY_ORDER,
                            ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']))

CATEGORY_TITLES = {
    'two_body_scattering': 'Two-body scattering',
    'multi_body_scattering': 'Multi-body scattering',
    'perturbed_scattering': 'Perturbed scattering',
    'vdw': 'vdW'
}

CATEGORY_SHORT_TITLES = {
    # 'two_body_scattering': 'TB-S',
    # 'multi_body_scattering': 'MB-S',
    # 'perturbed_scattering': 'P-S',
    'two_body_scattering': 'TBS',
    'multi_body_scattering': 'MBS',
    'perturbed_scattering': 'PS',
    'vdw': 'vdW'
}

# Smallest possible number of atoms in an event of each category, used as the first histogram bin
CATEGORY_MIN_NUM_ATOMS = {
    'two_body_scattering': 2,
    'multi_body_scattering': 3,
    'perturbed_scattering': 2,
    'vdw': 2,
}


def _category_event_stats(event_stats: dict,
                          category: str) -> list:
    """
    Return the event statistics of one category as a flat list.

    Both the nested structure (category, atom index, list of events) of
    gpsp_tools.md_analysis.collision_statistics() and the older flat structure (category, list of events) are
    supported. In the nested structure, the events are ordered by the atom index.

    Parameters
    ----------
    event_stats : dict
        Event statistics.
    category : str
        Category to extract.

    Returns
    -------
    flattened_stats : list
        Statistics dictionaries of the events of the category. Empty if the category is missing.
    """

    # Empty list for a missing category
    if category not in event_stats:
        return []

    category_stats = event_stats[category]

    # Nested structure: category, atom index, list of events
    if isinstance(category_stats, dict):
        flattened_stats = []
        for atom_index in sorted(category_stats):
            flattened_stats.extend(category_stats[atom_index])
        return flattened_stats

    # Older flat structure
    return list(category_stats)


###################################################################################################################
# Plotting style
###################################################################################################################
def set_gpsp_plot_style() -> None:
    """
    Set the common Matplotlib style of the gpsp_tools figures.

    The style is applied when the module is imported.
    """

    # Figure size, font sizes, line width, legend and grid, with a LaTeX-like serif font
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 100,

        'axes.titlesize': 20,
        'axes.labelsize': 20,

        'lines.linewidth': 2.5,

        'xtick.labelsize': 20,
        'ytick.labelsize': 20,

        'legend.fontsize': 14,
        'legend.loc': 'best',

        'grid.alpha': 0.7,

        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
    })


# Apply the style on import
set_gpsp_plot_style()


###################################################################################################################
# Plotting functions
###################################################################################################################
def plot_distance_series_for_event(categorized_events: dict,
                                   category: str,
                                   atom_index: int,
                                   event_index: int,
                                   collision_cutoff: float,
                                   x_axis: str='frames',
                                   dt: float | None=None,
                                   xlims: tuple[float, float] | None=None,
                                   savename: str | None=None) -> None:
    """
    Plot the distances between the reference atom and all partners of one event.

    The main partner is drawn in black and the third bodies in the default colours. The cutoff distance is
    shown as a red dashed line. A message is printed if the atom or the event is not found.

    Usage: ``plot_distance_series_for_event(categorized, 'vdw', atom_index, event_index, 9.0, x_axis='time',
    dt=50e-15)``

    Parameters
    ----------
    categorized_events : dict
        Output of gpsp_tools.md_analysis.categorize_collisions().
    category : str
        Category of the event: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering' or 'vdw'.
    atom_index : int
        Atom index of the reference atom.
    event_index : int
        Index of the event in the event list of the reference atom in the category.
    collision_cutoff : float
        Cutoff distance of the collision detection in Å.
    x_axis : str, default='frames'
        Quantity on the x-axis: 'frames' for the frame number or 'time' for the time in ps.
    dt : float, default=None
        Time step between frames in seconds. Required if x_axis is 'time'.
    xlims : tuple of float, default=None
        Limits of the x-axis in the units of the selected x-axis quantity.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.

    Raises
    ------
    ValueError
        Raised if x_axis is not 'frames' or 'time', or if dt is missing for x_axis='time'.
    """

    # Validate the x-axis option
    if x_axis not in ['frames', 'time']:
        raise ValueError("x_axis must be either 'frames' or 'time'.")

    # Require the time step for the time axis
    if x_axis == 'time' and dt is None:
        raise ValueError("dt must be provided when x_axis='time'.")

    # Check that the reference atom has events in the category
    if atom_index in categorized_events[category]:
        events = categorized_events[category][atom_index]

        # Check that the event index is within the range of events of the reference atom
        if event_index < len(events):
            event_df = events[event_index]

            # Values and label of the x-axis
            frames = event_df['frame']
            if x_axis == 'time':
                x_values = frames * dt * 1e12
                x_label = 'Time (ps)'
            else:
                x_values = frames
                x_label = 'Frame'

            # Main partner of the event
            main_partner_index = gpmda.main_partner_index(event_df)

            # Collect the x-values and distances of each partner
            distances_by_partner = {}
            for i, row in event_df.iterrows():
                for partner_atom, distance in zip(row['atoms2'], row['distances']):
                    if partner_atom not in distances_by_partner:
                        distances_by_partner[partner_atom] = {'frames': [], 'distances': []}
                    distances_by_partner[partner_atom]['frames'].append(x_values.iloc[i])
                    distances_by_partner[partner_atom]['distances'].append(distance)

            # Plot the main partner in black and the third bodies semi-transparent
            fig = plt.figure()
            for partner_atom, data in distances_by_partner.items():
                if partner_atom == main_partner_index:
                    # plt.plot(data['frames'], data['distances'], color='black', label=f'Main collider {partner_atom}')
                    plt.plot(data['frames'], data['distances'], color='black', label=f'Main collider')
                else:
                    # plt.plot(data['frames'], data['distances'], label=f'Third-body {partner_atom}', alpha=0.5)
                    plt.plot(data['frames'], data['distances'], label=f'Third-body', alpha=0.5)

            # Axis labels and limits, and the cutoff distance
            # plt.title(f'Atom {atom_index} - Event {event_index} - {CATEGORY_TITLES.get(category, category)}')
            # plt.title(f'{CATEGORY_TITLES.get(category, category)}')
            plt.xlabel(x_label)
            # plt.ylabel('Distance (Å)'
            plt.ylabel('$R$ (Å)')
            plt.ylim(2, collision_cutoff + 1)
            if xlims:
                plt.xlim(xlims)
            plt.axhline(y=collision_cutoff, color='r', linestyle='--', alpha=0.5)

            # Number of third bodies, used for the optional legend below
            num_other_atoms = len(distances_by_partner) - 1 if main_partner_index is not None \
                else len(distances_by_partner)
            # if num_other_atoms <= 8:
            #     plt.legend()

            # Finalise the layout, save the figure if requested and show it
            plt.tight_layout()
            if savename:
                fig.savefig(f"{savename}.png", dpi=200)
            plt.show()

        # Report an event index out of range
        else:
            print(f"Event index {event_index} is out of range for atom {atom_index} in category {category}.")

    # Report a reference atom without events in the category
    else:
        print(f"Atom {atom_index} not found in category {category}.")


def plot_property_series_for_event(categorized_events: dict,
                                   property_series: dict,
                                   property_name: str='Property',
                                   category: str | None=None,
                                   atom_index: int | None=None,
                                   event_index: int | None=None,
                                   xlims: tuple[float, float] | None=None,
                                   ylims: tuple[float, float] | None=None,
                                   savename: str | None=None) -> None:
    """
    Plot the property series of all partners of one event against the frame number.

    The main partner is drawn in black and the other partners in the default colours. The legend is shown for
    at most eight other partners. A message is printed if the atom or the event is not found, or if there are
    no data.

    Usage: ``plot_property_series_for_event(categorized, series, 'J (Hz)', 'vdw', atom_index, event_index)``

    Parameters
    ----------
    categorized_events : dict
        Output of gpsp_tools.md_analysis.categorize_collisions().
    property_series : dict
        First output of gpsp_tools.md_analysis.property_R_all_partners_time_series(): atom index of each partner
        mapped to the property in each frame of the event.
    property_name : str, default='Property'
        Label of the y-axis.
    category : str, default=None
        Category of the event.
    atom_index : int, default=None
        Atom index of the reference atom.
    event_index : int, default=None
        Index of the event in the event list of the reference atom in the category.
    xlims : tuple of float, default=None
        Limits of the x-axis.
    ylims : tuple of float, default=None
        Limits of the y-axis.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    # Check that the reference atom has events in the category
    if atom_index in categorized_events[category]:
        events = categorized_events[category][atom_index]

        # Check that the event index is within the range of events of the reference atom
        if event_index < len(events):
            event_df = events[event_index]

            # Frames of the event and its main partner
            frames = event_df['frame']
            main_partner_index = gpmda.main_partner_index(event_df)

            # Stop if there are no property data
            if not property_series:
                print('No property series data to plot.')
                return

            # Plot the main partner in black and the other partners semi-transparent
            fig = plt.figure()
            for partner_atom, values in property_series.items():
                if partner_atom == main_partner_index:
                    plt.plot(frames, values, color='black', label=f'Main Atom {partner_atom}')
                else:
                    plt.plot(frames, values, label=f'Atom {partner_atom}', alpha=0.5)

            # Title, axis labels and optional axis limits
            plt.title(f'Atom {atom_index} - Event {event_index} - {CATEGORY_TITLES.get(category, category)}')
            plt.xlabel('Frame')
            plt.ylabel(property_name)
            if xlims is not None:
                plt.xlim(xlims)
            if ylims is not None:
                plt.ylim(ylims)

            # Show the legend only for at most eight other partners
            num_other_atoms = len(property_series) - 1 if main_partner_index is not None else len(property_series)
            if num_other_atoms <= 8:
                plt.legend()

            # Finalise the layout, save the figure if requested and show it
            plt.tight_layout()
            if savename:
                fig.savefig(f"{savename}.png", dpi=200)
            plt.show()

        # Report an event index out of range
        else:
            print(f"Event index {event_index} is out of range for atom {atom_index} in category {category}.")

    # Report a reference atom without events in the category
    else:
        print(f"Atom {atom_index} not found in category {category}.")


def plot_collision_counts(collision_counts: dict,
                          savename: str | None=None) -> None:
    """
    Plot the number of events in each category as a bar chart on a logarithmic scale.

    Each bar is annotated with the count and its percentage of all events.

    Usage: ``plot_collision_counts(gpsp_tools.md_analysis.collision_counts(collision_stats))``

    Parameters
    ----------
    collision_counts : dict
        Category mapped to the number of events, as given by gpsp_tools.md_analysis.collision_counts().
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    # Labels and counts in the fixed category order
    titles = [CATEGORY_SHORT_TITLES.get(cat, cat) for cat in CATEGORY_ORDER]
    counts = [collision_counts.get(cat, 0) for cat in CATEGORY_ORDER]
    total = sum(counts)

    # Bar chart in the category colours
    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(titles,
                   counts,
                   color=[CATEGORY_COLOURS[category] for category in CATEGORY_ORDER],
                   edgecolor='black')

    # Annotate each bar with its count and percentage
    for bar, count in zip(bars, counts):
        percent = (count / total * 100) if total > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count} ({percent:.1f}%)",
            ha='center', va='bottom', fontsize=15
        )

    # Logarithmic y-axis for the wide range of counts
    plt.ylabel('Count')
    plt.yscale('log')

    # Headroom above the tallest bar for the annotations
    ymax = max(counts) if counts else 1
    # plt.ylim(0.8, ymax * 2.2)
    plt.ylim(0.8, ymax * 3)

    # plt.title('Total collision counts by category')

    # Finalise the layout, save the figure if requested and show it
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_event_lifetimes(event_stats: dict,
                         category: str,
                         bins: int=20,
                         savename: str | None=None) -> None:
    """
    Plot the histogram of the event lifetimes of one category.

    The y-axis is logarithmic for all categories except vdW. A message is printed if the category is missing or
    has no events.

    Usage: ``plot_event_lifetimes(collision_stats, 'vdw')``

    Parameters
    ----------
    event_stats : dict
        Event statistics, as given by gpsp_tools.md_analysis.collision_statistics(), or the older flat
        structure (category mapped to a list of event statistics).
    category : str
        Category to plot: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering' or 'vdw'.
    bins : int, default=20
        Number of histogram bins.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    # Stop for a missing category
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return

    title = CATEGORY_TITLES.get(category, category)

    # Event lifetimes of the category in ps
    category_events = _category_event_stats(event_stats, category)
    lifetimes = [event['lifetime_ps'] for event in category_events]

    # Stop if the category has no events
    if not lifetimes:
        print(f"No data for {category}.")
        return

    # Histogram in the category colour
    fig = plt.figure(figsize=(8, 5))
    plt.hist(lifetimes,
             bins=bins,
             color=CATEGORY_COLOURS.get(category, 'tab:blue'),
             edgecolor='black')
    plt.title(title)
    plt.xlabel('Lifetime (ps)')
    plt.ylabel('Count')

    # Logarithmic y-axis for all categories except vdW
    if category != 'vdw':
        plt.yscale('log')

    # Finalise the layout, save the figure if requested and show it
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_all_event_lifetimes(event_stats: dict,
                             bins: int=20,
                             savename: str | None=None) -> None:
    """
    Plot the histograms of the event lifetimes of all four categories in a 2x2 grid.

    Each histogram is drawn on a logarithmic y-axis with the bars annotated by their counts. A straight line is
    fitted to the logarithm of the non-zero counts against the bin centres, ln[N(tau)] = A - tau / <tau>, and the
    resulting exponential curve and fit equation are shown in each panel.

    Usage: ``plot_all_event_lifetimes(collision_stats, savename='event_lifetimes')``

    Parameters
    ----------
    event_stats : dict
        Event statistics, as given by gpsp_tools.md_analysis.collision_statistics(), or the older flat
        structure (category mapped to a list of event statistics).
    bins : int, default=20
        Number of histogram bins in each panel.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    def log_lifetime_fit(hist_counts: np.ndarray,
                         bin_edges: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        """
        Fit a straight line to the logarithm of the non-zero histogram counts against the bin centres.

        Parameters
        ----------
        hist_counts : ndarray
            Histogram counts.
        bin_edges : ndarray
            Edges of the histogram bins.

        Returns
        -------
        fit_results : tuple or None
            Bin centres of the non-zero counts, slope and intercept of the fit, or None if fewer than two
            counts are non-zero.
        """

        # Require at least two non-zero counts
        positive_mask = hist_counts > 0
        if np.count_nonzero(positive_mask) < 2:
            return None

        # Linear fit of the logarithmic counts against the bin centres
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        fit_x = bin_centers[positive_mask]
        fit_y = np.log(hist_counts[positive_mask])
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        return fit_x, slope, intercept

    # 2x2 grid of panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot the categories in the fixed order
    for idx, category in enumerate(CATEGORY_ORDER):
        title = CATEGORY_TITLES.get(category, category)
        if category in event_stats:
            category_events = _category_event_stats(event_stats, category)
            lifetimes = [event['lifetime_ps'] for event in category_events]
            if lifetimes:

                # Histogram in the category colour
                hist_counts, bin_edges, bars = axes[idx].hist(
                    lifetimes,
                    bins=bins,
                    color=CATEGORY_COLOURS.get(category, 'tab:blue'),
                    edgecolor='black'
                )

                # Annotate each populated bar with its count
                for count, bar in zip(hist_counts, bars):
                    if count > 0:
                        axes[idx].text(bar.get_x() + bar.get_width()/2,
                                       bar.get_height(),
                                       f'{int(count)}',
                                       ha='center',
                                       va='bottom',
                                       fontsize=10)

                # Fit the logarithmic lifetime distribution and overlay the exponential curve
                fit_results = log_lifetime_fit(hist_counts, bin_edges)
                if fit_results is not None:
                    fit_x, slope, intercept = fit_results
                    fit_x_plot = np.linspace(fit_x.min(), fit_x.max(), 200)
                    fit_counts = np.exp(intercept + slope*fit_x_plot)

                    # Lifetime parameter <tau> = -1/slope
                    lifetime_parameter = None
                    if not np.isclose(slope, 0.0):
                        lifetime_parameter = -1/slope

                    axes[idx].plot(
                        fit_x_plot,
                        fit_counts,
                        color='black',
                        linestyle='--',
                        linewidth=2,
                        alpha=0.4,
                    )

                    # Show the fitted equation in the panel
                    if lifetime_parameter is None:
                        fit_label = rf'$\ln(N) = A$, $A = {intercept:+.2f}$'
                    else:
                        fit_label = (
                            # rf'$\ln(N) = -\frac{{\tau}}{{{lifetime_parameter:.2f}}}'
                            # rf' {intercept:+.2f}$'
                            # rf'$\ln(N) = {intercept:.2f} - \frac{{\tau}}{{{lifetime_parameter:.2f}}}$'
                            rf'$\ln[N(\tau)] = {intercept:.2f} - \frac{{\tau}}{{{lifetime_parameter:.2f}}}$'
                        )

                    axes[idx].text(
                        0.95,
                        0.95,
                        fit_label,
                        transform=axes[idx].transAxes,
                        ha='right',
                        va='top',
                        # fontsize=20,
                        fontsize=24,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                    )

                # Label only the outer axes
                if idx >= 2:
                    axes[idx].set_xlabel('Lifetime (ps)')
                if idx % 2 == 0:
                    axes[idx].set_ylabel('Count')

                # Logarithmic y-axis for all categories
                # if category != 'vdw':
                axes[idx].set_yscale('log')

                # Headroom above the tallest bar for the annotations
                ymax = max(hist_counts) if len(hist_counts) > 0 else 1
                # if not category == 'vdw':
                axes[idx].set_ylim(0.8, ymax**1.1)
                # else:
                    # axes[idx].set_ylim(None, ymax * 1.1)
                axes[idx].grid(axis='y', alpha=0.7)
                axes[idx].set_title(title)

            # Placeholder panel for a category without events
            else:
                axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
                axes[idx].set_title(title)

        # Placeholder panel for a missing category
        else:
            axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            axes[idx].set_title(title)

    # Finalise the layout, save the figure if requested and show it
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_event_num_atoms(event_stats: dict,
                         category: str,
                         savename: str | None=None) -> None:
    """
    Plot the histogram of the number of atoms involved in the events of one category.

    The bins are integers starting from the smallest possible number of atoms of the category, and the y-axis is
    logarithmic. A message is printed if the category is missing or has no events.

    Usage: ``plot_event_num_atoms(collision_stats, 'vdw')``

    Parameters
    ----------
    event_stats : dict
        Event statistics, as given by gpsp_tools.md_analysis.collision_statistics(), or the older flat
        structure (category mapped to a list of event statistics).
    category : str
        Category to plot, typically 'multi_body_scattering', 'perturbed_scattering' or 'vdw'.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    # Stop for a missing category
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return

    title = CATEGORY_TITLES.get(category, category)

    # Number of atoms involved in each event, and stop if there are no events
    category_events = _category_event_stats(event_stats, category)
    num_atoms = [event['num_atoms_involved'] for event in category_events]
    if not num_atoms:
        print(f"No data for {category}.")
        return

    # Smallest possible number of atoms of the category
    min_num_atoms = CATEGORY_MIN_NUM_ATOMS.get(category, 2)

    # Histogram with integer bins on a logarithmic y-axis
    fig = plt.figure(figsize=(8, 5))
    plt.hist(num_atoms,
             bins=range(min_num_atoms, max(num_atoms)+2),
             align='left',
             color=CATEGORY_COLOURS.get(category, 'tab:blue'),
             edgecolor='black',
             log=True)
    plt.title(title)
    plt.xlabel('N(Atoms)')
    plt.ylabel('Count')
    plt.xticks(range(min_num_atoms, max(num_atoms)+1))

    # Finalise the layout, save the figure if requested and show it
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_all_event_num_atoms(event_stats: dict,
                             savename: str | None=None) -> None:
    """
    Plot the histograms of the number of atoms involved in the MBS, PS and vdW events in a 1x3 grid.

    The bars are annotated with their percentages (vdW) or with their counts and percentages (MBS, PS).

    Usage: ``plot_all_event_num_atoms(collision_stats, savename='event_num_atoms')``

    Parameters
    ----------
    event_stats : dict
        Event statistics, as given by gpsp_tools.md_analysis.collision_statistics(), or the older flat
        structure (category mapped to a list of event statistics).
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    # Categories in which three or more atoms can be involved, in a 1x3 grid
    categories = ['multi_body_scattering', 'perturbed_scattering', 'vdw']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot each category
    for idx, category in enumerate(categories):
        title = CATEGORY_TITLES.get(category, category)
        if category in event_stats:
            category_events = _category_event_stats(event_stats, category)
            num_atoms = [event['num_atoms_involved'] for event in category_events]
            if num_atoms:

                # Histogram with integer bins on a logarithmic y-axis
                min_num_atoms = CATEGORY_MIN_NUM_ATOMS.get(category, 2)
                hist_counts, _, bars = axes[idx].hist(num_atoms,
                                                      bins=range(min_num_atoms, max(num_atoms)+2),
                                                      align='left',
                                                      color=CATEGORY_COLOURS[category],
                                                      edgecolor='black',
                                                      log=True)
                axes[idx].set_title(title)
                axes[idx].set_xlabel('N(Atoms)')
                if idx == 0:
                    axes[idx].set_ylabel('Count')

                # Annotate each populated bar with its percentage, and with its count for MBS and PS
                total = len(num_atoms)
                for count, bar in zip(hist_counts, bars):
                    if count > 0:
                        percent = count / total * 100
                        if category == 'vdw':
                            # label = f'{int(count)}'
                            label = f'{percent:.0f}%'
                            font_size = 10
                        else:
                            label = f'{int(count)} ({percent:.1f}%)'
                            font_size = 12
                        axes[idx].text(bar.get_x() + bar.get_width()/2,
                                       bar.get_height(),
                                       label,
                                       ha='center',
                                       va='bottom',
                                       fontsize=font_size)

                # Headroom above the tallest bar for the annotations
                ymax = max(hist_counts) if len(hist_counts) > 0 else 1
                # axes[idx].set_ylim(0.8, ymax * 2.2)
                axes[idx].set_ylim(0.8, ymax**1.1)

                # Integer ticks, every second one if there are more than ten
                x_ticks = list(range(min_num_atoms, max(num_atoms)+1))
                if len(x_ticks) > 10:
                    x_ticks = x_ticks[::2]
                axes[idx].set_xticks(x_ticks)
                axes[idx].grid(axis='y', alpha=0.7)

            # Placeholder panel for a category without events
            else:
                axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
                axes[idx].set_title(title)

        # Placeholder panel for a missing category
        else:
            axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            axes[idx].set_title(title)

    # Finalise the layout, save the figure if requested and show it
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_event_counts_by_lifetime(counts: list,
                                  lifetime_ranges: list[tuple[float, float]],
                                  category: str,
                                  savename: str | None=None) -> None:
    """
    Plot the number of events of one category in given lifetime ranges as a bar chart.

    Empty ranges are omitted, the y-axis is logarithmic, and each bar is annotated with its count and
    percentage. A message is printed if all ranges are empty.

    Usage: ``plot_event_counts_by_lifetime(counts_by_lifetime['vdw'], lifetime_ranges, 'vdw')``

    Parameters
    ----------
    counts : list
        Event counts of the category, one per lifetime range, as given by
        gpsp_tools.md_analysis.collision_counts_by_lifetime().
    lifetime_ranges : list of tuple
        Lifetime ranges (min_lifetime, max_lifetime) in ps.
    category : str
        Category of the counts.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    title = CATEGORY_TITLES.get(category, category)

    # Remove the empty ranges
    filtered = [(c, r) for c, r in zip(counts, lifetime_ranges) if c > 0]
    if not filtered:
        print("No events to plot for the given lifetime ranges.")
        return

    # Counts, labels and positions of the remaining ranges
    filtered_counts, filtered_ranges = zip(*filtered)
    lifetime_labels = [f"{min_lifetime}-{max_lifetime}" for min_lifetime, max_lifetime in filtered_ranges]
    x = np.arange(len(lifetime_labels))
    total = sum(filtered_counts)

    # Bar chart in the category colour on a logarithmic y-axis
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x,
                  filtered_counts,
                  width=0.6,
                  color=CATEGORY_COLOURS.get(category, 'tab:blue'),
                  edgecolor='black')
    ax.set_xlabel('Lifetime Ranges (ps)')
    ax.set_ylabel('Number of Events')
    ax.set_title(f'{title}')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(lifetime_labels)

    # Headroom above the tallest bar for the annotations
    ymax = max(filtered_counts) if filtered_counts else 1
    # ax.set_ylim(0.8, ymax * 2.2)
    ax.set_ylim(0.8, ymax**1.1)

    # Annotate each bar with its count and percentage
    for bar, count in zip(bars, filtered_counts):
        percent = (count / total * 100) if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count} ({percent:.1f}%)",
            # ha='center', va='bottom', fontsize=14
            # ha='center', va='bottom', fontsize=12
            ha='center', va='bottom', fontsize=11
        )

    # Finalise the layout, save the figure if requested and show it
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_all_event_counts_by_lifetime(counts_by_lifetime: dict,
                                      lifetime_ranges: list[tuple[float, float]],
                                      event_stats: dict | None=None,
                                      savename: str | None=None) -> None:
    """
    Plot the number of events of each category in given lifetime ranges as a grouped bar chart.

    The legend gives the mean lifetime <tau> of each category, computed exactly from the event statistics if
    they are given, and otherwise estimated from the range midpoints (the lower limit for an open range).

    Usage: ``plot_all_event_counts_by_lifetime(counts_by_lifetime, lifetime_ranges, event_stats=collision_stats)``

    Parameters
    ----------
    counts_by_lifetime : dict
        Category mapped to the event counts in the lifetime ranges, as given by
        gpsp_tools.md_analysis.collision_counts_by_lifetime().
    lifetime_ranges : list of tuple
        Lifetime ranges (min_lifetime, max_lifetime) in ps.
    event_stats : dict, default=None
        Event statistics, as given by gpsp_tools.md_analysis.collision_statistics(), for the exact mean
        lifetimes.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    def histogram_mean_lifetime(counts: np.ndarray,
                                lifetime_ranges: list[tuple[float, float]]) -> float:
        """
        Estimate the mean lifetime from the counts in the lifetime ranges.

        Parameters
        ----------
        counts : ndarray
            Event counts, one per lifetime range.
        lifetime_ranges : list of tuple
            Lifetime ranges (min_lifetime, max_lifetime) in ps.

        Returns
        -------
        mean_lifetime : float
            Count-weighted mean of the range midpoints in ps, with the lower limit used for an open range. NaN
            if all counts are zero.
        """

        # No events
        total_count = np.sum(counts)
        if total_count == 0:
            return np.nan

        # Midpoint of each range, or the lower limit of an open range
        lifetime_midpoints = []
        for min_lifetime, max_lifetime in lifetime_ranges:
            if np.isinf(max_lifetime):
                lifetime_midpoints.append(float(min_lifetime))
            else:
                lifetime_midpoints.append(0.5*(min_lifetime + max_lifetime))

        return np.average(lifetime_midpoints, weights=counts)

    # Categories, counts, range labels and bar positions
    categories = list(counts_by_lifetime.keys())
    counts = np.array(list(counts_by_lifetime.values()))
    lifetime_labels = [f">{min_lifetime}" if np.isinf(max_lifetime) else f"{min_lifetime}-{max_lifetime}"
                       for min_lifetime, max_lifetime in lifetime_ranges]
    x = np.arange(len(lifetime_labels))
    width = 0.2

    # Grouped bars, offset for each category
    # fig, ax = plt.subplots(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(12, 8.5))
    for i, category in enumerate(categories):
        title = CATEGORY_TITLES.get(category, category)

        # Exact mean lifetime from the event statistics, or its estimate from the ranges
        if event_stats is not None:
            category_events = _category_event_stats(event_stats, category)
            event_lifetimes = [event['lifetime_ps'] for event in category_events]
            mean_lifetime_ps = np.mean(event_lifetimes) if event_lifetimes else np.nan
        else:
            mean_lifetime_ps = histogram_mean_lifetime(counts[i], lifetime_ranges)

        # Legend label with the mean lifetime
        if np.isnan(mean_lifetime_ps):
            legend_label = f"{title}, mean=n/a"
        else:
            # legend_label = f"{title}, mean={mean_lifetime_ps:.1f} ps"
            # legend_label = f"{title}, $\\tau = {mean_lifetime_ps:.1f}$ ps"
            legend_label = f"{title}, $\\langle \\tau \\rangle$ = {mean_lifetime_ps:.1f} ps"

        bars = ax.bar(x + i*width,
                      counts[i],
                      width,
                      label=legend_label,
                      color=CATEGORY_COLOURS.get(category, 'tab:blue'),
                      edgecolor='black')

        # Annotate each non-zero bar with its count
        for bar, count in zip(bars, counts[i]):
            if count > 0:
                # fontsize=10
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{int(count)}",
                        ha='center',
                        va='bottom',
                        fontsize=12)

    # Axis labels and the logarithmic y-axis
    ax.set_xlabel('Lifetime ranges (ps)')
    # ax.set_ylabel('Number of events')
    ax.set_ylabel('Count')
    # ax.set_title('Event Counts by Lifetime Ranges')
    # ax.set_title('Event lifetimes')
    ax.set_yscale('log')
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(lifetime_labels)

    # Horizontal grid lines
    ax.grid(axis='y', alpha=0.7)

    # Legend, layout, saving if requested and showing the figure
    # ax.legend()
    # ax.legend(fontsize=16)
    ax.legend(fontsize=18)
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()


def plot_bound_cluster_event(series: dict,
                             dt: float,
                             collision_cutoff: float,
                             event: dict | None=None,
                             r_min: float | None=None,
                             show_internal_energy: bool=True,
                             title: str | None=None,
                             savename: str | None=None) -> None:
    """
    Plot the pair distances and binding energies of the atoms of a bound-cluster event.

    The upper panel shows the distance R(t) of every pair of atoms, with the collision cutoff as a red dashed
    line and, optionally, the minimum of the pair potential as a grey dotted line. The lower panel shows the
    energy of each atom relative to the rest of the atoms and, optionally, the internal energy of the whole set,
    with zero as a dotted line, so that an atom is bound to the others where its curve is negative. The internal
    energy is meaningful only if the same atoms form the cluster throughout. If the event is given, the time is
    measured from its first frame, and the frames between its first and last frame are shaded.

    Usage: ``plot_bound_cluster_event(series, 25e-15, 9.0, event=event, r_min=4.378, savename='trimer_1')``

    Parameters
    ----------
    series : dict
        Output of gpsp_tools.md_analysis.bound_cluster_time_series().
    dt : float
        Time step between frames in seconds.
    collision_cutoff : float
        Cutoff distance of the collision detection in Å.
    event : dict, default=None
        Cluster event of gpsp_tools.md_analysis.bound_cluster_events(), whose frames are shaded.
    r_min : float, default=None
        Distance of the minimum of the pair potential in Å.
    show_internal_energy : bool, default=True
        If True, plot the internal energy of the whole set of atoms.
    title : str, default=None
        Title of the figure.
    savename : str, default=None
        Path of the saved figure without the extension. If None, the figure is not saved.
    """

    # Time axis in ps, measured from the first frame of the event if it is given
    start_frame = event['start_frame'] if event is not None else 0
    time_ps = (series['frames'] - start_frame) * dt * 1e12

    # Two panels sharing the time axis
    fig, (ax_distance, ax_energy) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Pair distances, with the cutoff and the minimum of the pair potential
    for k, (i, j) in enumerate(series['pairs']):
        ax_distance.plot(time_ps, series['distances'][:, k], linewidth=1.5, label=f'{i}–{j}')
    ax_distance.axhline(collision_cutoff, color='r', linestyle='--', alpha=0.5)
    if r_min is not None:
        ax_distance.axhline(r_min, color='grey', linestyle=':', alpha=0.8)
    ax_distance.set_ylabel('$R$ (Å)')
    ax_distance.set_ylim(2, collision_cutoff + 1)

    # Energy of each atom relative to the rest and the internal energy of the whole set of atoms
    for k, atom in enumerate(series['atoms']):
        ax_energy.plot(time_ps, series['atom_energies'][:, k], linewidth=1.2, alpha=0.8, label=f'Atom {atom}')
    if show_internal_energy:
        ax_energy.plot(time_ps, series['internal_energy'], color='black', linewidth=2, label='Internal energy')
    ax_energy.axhline(0.0, color='black', linestyle=':', alpha=0.8)
    ax_energy.set_xlabel('Time from the start of the event (ps)' if event is not None else 'Time (ps)')
    ax_energy.set_ylabel('$E$ (K)')

    # Shade the frames in which the bound cluster exists
    if event is not None:
        for ax in (ax_distance, ax_energy):
            ax.axvspan(0.0, (event['end_frame'] - start_frame) * dt * 1e12, color='tab:red', alpha=0.08)

    # Legends for at most ten pairs and atoms, title, layout, saving if requested and showing the figure
    if len(series['pairs']) <= 10:
        ax_distance.legend(fontsize=11, ncol=2, loc='upper right')
    if len(series['atoms']) <= 10:
        ax_energy.legend(fontsize=11, ncol=2, loc='lower right')
    if title:
        ax_distance.set_title(title, fontsize=14)
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()
