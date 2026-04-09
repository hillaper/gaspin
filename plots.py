"""
Plotting and visualization utilities for gpsp_tools.
"""

##########################################################################################################
# Imports
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import gpsp_tools.md_analysis as gpmda

CATEGORY_ORDER = ['two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw']

CATEGORY_TITLES = {
    'two_body_scattering': 'Two-body scattering',
    'multi_body_scattering': 'Multi-body scattering',
    'perturbed_scattering': 'Perturbed scattering',
    'vdw': 'vdW'
}

CATEGORY_SHORT_TITLES = {
    'two_body_scattering': 'TB-S',
    'multi_body_scattering': 'MB-S',
    'perturbed_scattering': 'P-S',
    'vdw': 'vdW'
}

def _category_event_stats(event_stats,
                          category):
    """
    Return event statistics for one category as a flat list.

    Parameters:
    - event_stats:
        Collision statistics dictionary.
    - category:
        Collision category to extract.

    Returns:
    - list
        Flat list of per-event statistics dictionaries.
    """
    # Return an empty list if the category is missing.
    if category not in event_stats:
        return []

    category_stats = event_stats[category]

    # Support the new nested structure: category -> atom index -> list of events.
    if isinstance(category_stats, dict):
        flattened_stats = []
        for atom_index in sorted(category_stats):
            flattened_stats.extend(category_stats[atom_index])
        return flattened_stats

    # Keep compatibility with the legacy flat list structure.
    return list(category_stats)

##########################################################################################################
# Set default plotting parameters
##########################################################################################################
def set_gpsp_plot_style():
    """
    Sets a consistent plotting style for gpsp_tools visualizations.
    """
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

        # Use a Latex-like font
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
    })
set_gpsp_plot_style()  # Apply the style settings on import

##########################################################################################################
# Plotting functions
##########################################################################################################
def plot_distance_series_for_event(categorized_events, 
                                   category, 
                                   atom_index, 
                                   event_index, 
                                   collision_cutoff,
                                   xlims=None,
                                   savename=None):
    """
    Plots distance series including all colliding partners for a specific event of a given atom in a category.

    Parameters:
    - categorized_events: 
        Dictionary containing categorized collision events (see md_analysis).
    - category: 
        Category of the event
        'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
    - atom_index: 
        Index of the atom for which to plot the event.
    - event_index: 
        Index of the specific event to plot for the atom.
    - collision_cutoff:
        Distance cutoff value for collision detection.
    - xlims: 
        Optional tuple specifying x-axis limits (min, max).
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (plots the distance series).
    """
    # Check if the atom index exists in the specified category
    if atom_index in categorized_events[category]:
        events = categorized_events[category][atom_index]
        
        # Check if the event index is within the range of events for the atom
        if event_index < len(events):
            event_df = events[event_index]
            
            # Extract frames and distances for all colliding partners in the event
            frames = event_df['frame']  # Use frame numbers directly for the x-axis
            distances_by_partner = {}

            # Identify the main partner atom index
            main_partner_index = gpmda.main_partner_index(event_df)
            
            # Organize distances by partner atom
            for i, row in event_df.iterrows():
                for partner_atom, distance in zip(row['atoms2'], row['distances']):
                    if partner_atom not in distances_by_partner:
                        distances_by_partner[partner_atom] = {'frames': [], 'distances': []}
                    distances_by_partner[partner_atom]['frames'].append(frames.iloc[i])
                    distances_by_partner[partner_atom]['distances'].append(distance)
            
            # Plot the distance series for all colliding partners
            fig = plt.figure()
            for partner_atom, data in distances_by_partner.items():
                if partner_atom == main_partner_index:
                    # Highlight the main partner atom with a distinct black color
                    plt.plot(data['frames'], data['distances'], color='black', label=f'Main Atom {partner_atom}')
                else:
                    # Plot other partner atoms with default colors and alpha=0.5
                    plt.plot(data['frames'], data['distances'], label=f'Atom {partner_atom}', alpha=0.5)
            
            # Add plot title, labels, and legend
            plt.title(f'Atom {atom_index} - Event {event_index} - {CATEGORY_TITLES.get(category, category)}')
            plt.xlabel('Frame')
            plt.ylabel('Distance (Å)')
            plt.ylim(2, collision_cutoff + 1)  
            if xlims:
                plt.xlim(xlims)
            plt.axhline(y=collision_cutoff, color='r', linestyle='--', alpha=0.5)

            # Only show legend if number of other atoms is 8 or fewer
            num_other_atoms = len(distances_by_partner) - 1 if main_partner_index is not None else len(distances_by_partner)
            if num_other_atoms <= 8:
                plt.legend()
            
            plt.tight_layout()
            if savename:
                fig.savefig(f"{savename}.png", dpi=200)
            plt.show()
        else:
            # Print a message if the event index is out of range
            print(f"Event index {event_index} is out of range for atom {atom_index} in category {category}.")
    else:
        # Print a message if the atom index is not found in the specified category
        print(f"Atom {atom_index} not found in category {category}.")

def plot_property_series_for_event(categorized_events,
                                   property_series,
                                   property_name='Property',
                                   category=None,
                                   atom_index=None,
                                   event_index=None,
                                   xlims=None,
                                   ylims=None,
                                   savename=None):
    """
    Plot a property series for all colliding partners in a single event.

    Parameters:
    - categorized_events:
        Dictionary containing categorized collision events (see md_analysis).
    - property_series:
        Dictionary returned by
        gpmda.collision_property_all_partners_R_time_series.
        Keys are partner atom indices and values are arrays of property
        values for each frame in the event.
    - property_name:
        Label for the plotted property.
    - category:
        Optional collision category used in the plot title.
    - atom_index:
        Optional index of the atom from group1 used in the plot title.
    - event_index:
        Optional index of the event used in the plot title.
    - xlims:
        Optional tuple specifying x-axis limits.
    - ylims:
        Optional tuple specifying y-axis limits.
    - savename:
        Optional string specifying the filename to save the figure.

    Returns:
    - None
    """
    # Check if the atom index exists in the specified category.
    if atom_index in categorized_events[category]:
        events = categorized_events[category][atom_index]

        # Check if the event index is within the range of events for the atom.
        if event_index < len(events):
            event_df = events[event_index]

            # Extract frames for the selected event to match the distance-series plot.
            frames = event_df['frame']

            # Identify the main partner atom index for consistent styling.
            main_partner_index = gpmda.main_partner_index(event_df)

            # Guard against missing or empty property data.
            if not property_series:
                print('No property series data to plot.')
                return

            # Create the figure before plotting the partner-resolved series.
            fig = plt.figure()

            # Plot the property series for each partner atom.
            for partner_atom, values in property_series.items():
                if partner_atom == main_partner_index:
                    # Highlight the main partner atom with a distinct black colour.
                    plt.plot(frames, values, color='black', label=f'Main Atom {partner_atom}')
                else:
                    # Plot other partner atoms with default colours and alpha=0.5.
                    plt.plot(frames, values, label=f'Atom {partner_atom}', alpha=0.5)

            # Add plot title, labels, and optional axis limits.
            plt.title(f'Atom {atom_index} - Event {event_index} - {CATEGORY_TITLES.get(category, category)}')
            plt.xlabel('Frame')
            plt.ylabel(property_name)
            if xlims is not None:
                plt.xlim(xlims)
            if ylims is not None:
                plt.ylim(ylims)

            # Only show legend if number of other atoms is 8 or fewer.
            num_other_atoms = len(property_series) - 1 if main_partner_index is not None else len(property_series)
            if num_other_atoms <= 8:
                plt.legend()

            # Finalise the layout and optionally save the figure to disk.
            plt.tight_layout()
            if savename:
                fig.savefig(f"{savename}.png", dpi=200)

            # Display the figure after all plot settings are applied.
            plt.show()
        else:
            # Print a message if the event index is out of range.
            print(f"Event index {event_index} is out of range for atom {atom_index} in category {category}.")
    else:
        # Print a message if the atom index is not found in the specified category.
        print(f"Atom {atom_index} not found in category {category}.")

def plot_collision_counts(collision_counts, savename=None):
    """
    Plot the total number of collisions for each category as a bar chart.

    Parameters:
    - collision_counts: 
        Dictionary with keys corresponding to collision categories and values 
        representing the total number of collisions for each category.
        Expected keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
    - savename: 
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays bar chart).
    """
    # Build x-axis labels and counts in a fixed category order.
    titles = [CATEGORY_SHORT_TITLES.get(cat, cat) for cat in CATEGORY_ORDER]
    counts = [collision_counts.get(cat, 0) for cat in CATEGORY_ORDER]
    total = sum(counts)

    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(titles, counts, color='tab:green', edgecolor='black')

    # Annotate each bar with its absolute count and percentage.
    for bar, count in zip(bars, counts):
        percent = (count / total * 100) if total > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            # f"{count}\n({percent:.1f}%)",
            f"{count} ({percent:.1f}%)",
            ha='center', va='bottom', fontsize=15
        )

    plt.ylabel('Count')
    # Use log scale to handle wide count ranges.
    plt.yscale('log')

    # Keep some headroom above the tallest bar for text labels.
    ymax = max(counts) if counts else 1
    plt.ylim(0.8, ymax * 2.2)

    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()

def plot_event_lifetimes(event_stats, 
                         category,
                         bins=20,
                         savename=None):
    """
    Plot histogram of event lifetimes for a specific category.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps either to a list of dictionaries containing statistics for each
        event, or to a dictionary with atom indices as keys and per-atom lists of
        event statistics as values.
    - category:
        String specifying the category to plot.
        Valid values: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
    - bins:
        Optional integer specifying the number of bins for the histogram (default: 20).
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays histogram plot).
    """
    # Guard against missing categories.
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return
    
    title = CATEGORY_TITLES.get(category, category)

    # Extract lifetime values (ps) for the selected category.
    category_events = _category_event_stats(event_stats, category)
    lifetimes = [event['lifetime_ps'] for event in category_events]

    # Stop early if there are no events in the selected category.
    if not lifetimes:
        print(f"No data for {category}.")
        return
    
    fig = plt.figure(figsize=(8, 5))
    plt.hist(lifetimes, bins=bins, color='tab:blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Lifetime (ps)')
    plt.ylabel('Count')

    # Keep vdW on linear scale; use log for other categories.
    if category != 'vdw':
        plt.yscale('log')

    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()

def plot_all_event_lifetimes(event_stats,
                             bins=20,
                             savename=None):
    """
    Plot histograms of event lifetimes for all four categories in a 2x2 subplot grid.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps either to a list of dictionaries containing statistics for each
        event, or to a dictionary with atom indices as keys and per-atom lists of
        event statistics as values.
    - bins:
        Optional integer specifying the number of bins for the histograms (default: 20).
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays subplot grid).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot each category in a fixed order to keep layout consistent.
    for idx, category in enumerate(CATEGORY_ORDER):
        title = CATEGORY_TITLES.get(category, category)
        if category in event_stats:
            category_events = _category_event_stats(event_stats, category)
            lifetimes = [event['lifetime_ps'] for event in category_events]
            if lifetimes:
                axes[idx].hist(lifetimes, bins=bins, color='tab:blue', edgecolor='black')

                # Label only the outer axes for a cleaner grid.
                if idx >= 2:
                    axes[idx].set_xlabel('Lifetime (ps)')
                if idx % 2 == 0:
                    axes[idx].set_ylabel('Count')

                # Use log scale for all except vdW.
                if category != 'vdw':
                    axes[idx].set_yscale('log')
                axes[idx].grid(axis='y', alpha=0.7)
            else:
                # Show message if category exists but has no events.
                axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')

            axes[idx].set_title(title)
        else:
            # Fallback panel if a category is missing.
            axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            axes[idx].set_title(title)
    
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()

def plot_event_num_atoms(event_stats, 
                         category,
                         savename=None):
    """
    Plot histogram of number of atoms involved for a selected event category.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps either to a list of dictionaries containing statistics for each
        event, or to a dictionary with atom indices as keys and per-atom lists of
        event statistics as values.
    - category:
        String specifying the category to plot.
        Recommended values: 'multi_body_scattering', 'perturbed_scattering', or 'vdw'.
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays histogram plot).
    """
    # Guard against missing categories.
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return
    
    title = CATEGORY_TITLES.get(category, category)

    # Pull number of atoms involved per event.
    category_events = _category_event_stats(event_stats, category)
    num_atoms = [event['num_atoms_involved'] for event in category_events]
    if not num_atoms:
        print(f"No data for {category}.")
        return
    
    fig = plt.figure(figsize=(8, 5))
    # Integer bins starting at 3 atoms; log y-scale via `log=True`.
    plt.hist(num_atoms, bins=range(3, max(num_atoms)+2), align='left', color='tab:orange', edgecolor='black', log=True)
    plt.title(title)
    plt.xlabel('N(Atoms)')
    plt.ylabel('Count')
    plt.xticks(range(3, max(num_atoms)+1))
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()

def plot_all_event_num_atoms(event_stats,
                             savename=None):
    """
    Plot histograms of number of atoms involved for selected categories in a 1x3 subplot grid.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_scattering', 'multi_body_scattering', 'perturbed_scattering', 'vdw'.
        Each key maps either to a list of dictionaries containing statistics for each
        event, or to a dictionary with atom indices as keys and per-atom lists of
        event statistics as values.
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays subplot grid).
    """
    # Focus on categories where number-of-atoms is typically informative.
    categories = ['multi_body_scattering', 'perturbed_scattering', 'vdw']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, category in enumerate(categories):
        title = CATEGORY_TITLES.get(category, category)
        if category in event_stats:
            category_events = _category_event_stats(event_stats, category)
            num_atoms = [event['num_atoms_involved'] for event in category_events]
            if num_atoms:
                axes[idx].hist(num_atoms, bins=range(3, max(num_atoms)+2), align='left', color='tab:orange', edgecolor='black', log=True)
                axes[idx].set_title(title)
                axes[idx].set_xlabel('N(Atoms)')
                if idx == 0:
                    axes[idx].set_ylabel('Count')
                axes[idx].set_xticks(range(3, max(num_atoms)+1))
                axes[idx].grid(axis='y', alpha=0.7)
            else:
                # Show message if category exists but has no events.
                axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
                axes[idx].set_title(title)
        else:
            # Show message if category key is missing.
            axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            axes[idx].set_title(title)
    
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()

def plot_event_counts_by_lifetime(counts, 
                                  lifetime_ranges, 
                                  category,
                                  savename=None):
    """
    Plot the number of events for a single category across given lifetime ranges.

    Inputs:
    - counts: list
        List of counts corresponding to the defined lifetime ranges for the category.
    - lifetime_ranges: list of tuples
        A list of tuples defining the lifetime ranges in picoseconds.
        Each tuple should be in the form (min_lifetime, max_lifetime).
    - category: str
        The category to plot.
    - savename: str or None
        Optional filename to save the figure.

    Returns:
    - None
    """
    title = CATEGORY_TITLES.get(category, category)

    # Remove empty bins to reduce clutter on the x-axis.
    filtered = [(c, r) for c, r in zip(counts, lifetime_ranges) if c > 0]
    if not filtered:
        print("No events to plot for the given lifetime ranges.")
        return

    filtered_counts, filtered_ranges = zip(*filtered)
    lifetime_labels = [f"{min_lifetime}-{max_lifetime}" for min_lifetime, max_lifetime in filtered_ranges]
    x = np.arange(len(lifetime_labels))

    total = sum(filtered_counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, filtered_counts, width=0.6, color='tab:green', edgecolor='black')
    ax.set_xlabel('Lifetime Ranges (ps)')
    ax.set_ylabel('Number of Events')
    ax.set_title(f'{title}')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(lifetime_labels)

    # Keep some extra vertical space for count annotations.
    ymax = max(filtered_counts) if filtered_counts else 1
    ax.set_ylim(0.8, ymax * 2.2)

    # Write counts and percentages above each bar.
    for bar, count in zip(bars, filtered_counts):
        percent = (count / total * 100) if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count} ({percent:.1f}%)",
            # ha='center', va='bottom', fontsize=14
            ha='center', va='bottom', fontsize=12
        )
        
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()

def plot_all_event_counts_by_lifetime(counts_by_lifetime, 
                                      lifetime_ranges, 
                                      savename=None):
    """
    Plot the number of events per category for given lifetime ranges.

    Inputs:
    - counts_by_lifetime: dict
        A dictionary with the same keys as event_stats, where each value 
        is a list of counts corresponding to the defined lifetime ranges.
    - lifetime_ranges: list of tuples
        A list of tuples defining the lifetime ranges in picoseconds.
        Each tuple should be in the form (min_lifetime, max_lifetime).
    - savename: str or None
        Optional filename to save the figure.

    Returns:
    - None
    """
    categories = list(counts_by_lifetime.keys())
    counts = np.array(list(counts_by_lifetime.values()))
    
    lifetime_labels = [f"{min_lifetime}-{max_lifetime}" for min_lifetime, max_lifetime in lifetime_ranges]
    x = np.arange(len(lifetime_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    # Offset bars per category to create grouped bars at each lifetime bin.
    for i, category in enumerate(categories):
        title = CATEGORY_TITLES.get(category, category)
        ax.bar(x + i*width, counts[i], width, label=title, edgecolor='black')

    ax.set_xlabel('Lifetime Ranges (ps)')
    ax.set_ylabel('Number of Events')
    ax.set_title('Event Counts by Lifetime Ranges')
    ax.set_yscale('log')
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(lifetime_labels)

    ax.legend()
    plt.tight_layout()
    if savename:
        fig.savefig(f"{savename}.png", dpi=200)
    plt.show()
