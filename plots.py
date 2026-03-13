"""
Plotting and visualization utilities for gpsp_tools.
"""

##########################################################################################################
# Imports
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import gpsp_tools.md_analysis as gpmda

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
                                   xlims=None):
    """
    Plots distance series including all colliding partners for a specific event of a given atom in a category.

    Parameters:
    - categorized_events: 
        Dictionary containing categorized collision events (see md_analysis).
    - category: 
        Category of the event
        'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
    - atom_index: 
        Index of the atom for which to plot the event.
    - event_index: 
        Index of the specific event to plot for the atom.
    - collision_cutoff:
        Distance cutoff value for collision detection.
    - xlims: 
        Optional tuple specifying x-axis limits (min, max).

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
            main_partner_index = gpmda.get_main_partner_index(event_df)
            
            # Organize distances by partner atom
            for i, row in event_df.iterrows():
                for partner_atom, distance in zip(row['atoms2'], row['distances']):
                    if partner_atom not in distances_by_partner:
                        distances_by_partner[partner_atom] = {'frames': [], 'distances': []}
                    distances_by_partner[partner_atom]['frames'].append(frames.iloc[i])
                    distances_by_partner[partner_atom]['distances'].append(distance)
            
            # Plot the distance series for all colliding partners
            plt.figure()
            for partner_atom, data in distances_by_partner.items():
                if partner_atom == main_partner_index:
                    # Highlight the main partner atom with a distinct black color
                    plt.plot(data['frames'], data['distances'], color='black', label=f'Main Atom {partner_atom}')
                else:
                    # Plot other partner atoms with default colors and alpha=0.5
                    plt.plot(data['frames'], data['distances'], label=f'Atom {partner_atom}', alpha=0.5)
            
            # Add plot title, labels, and legend
            plt.title(f'Atom {atom_index} - Event {event_index} - {category}')
            plt.xlabel('Frame')
            plt.ylabel('Distance (Å)')
            plt.ylim(0, collision_cutoff + 1)
            if xlims:
                plt.xlim(xlims)
            plt.axhline(y=collision_cutoff, color='r', linestyle='--', alpha=0.5)

            # Only show legend if number of other atoms is 8 or fewer
            num_other_atoms = len(distances_by_partner) - 1 if main_partner_index is not None else len(distances_by_partner)
            if num_other_atoms <= 8:
                plt.legend()
        else:
            # Print a message if the event index is out of range
            print(f"Event index {event_index} is out of range for atom {atom_index} in category {category}.")
    else:
        # Print a message if the atom index is not found in the specified category
        print(f"Atom {atom_index} not found in category {category}.")

def plot_collision_counts(collision_counts, savename=None):
    """
    Plot the total number of collisions for each category as a bar chart.

    Parameters:
    - collision_counts: 
        Dictionary with keys corresponding to collision categories and values 
        representing the total number of collisions for each category.
        Expected keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
    - savename: 
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays bar chart).
    """
    category_titles = {
        'two_body_short': 'TB-S',
        'multi_body_short': 'MB-S',
        'two_body_vdw': 'TB-vdW',
        'multi_body_vdw': 'MB-vdW'
    }
    categories = ['two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw']
    titles = [category_titles.get(cat, cat) for cat in categories]
    counts = [collision_counts.get(cat, 0) for cat in categories]

    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(titles, counts, color='tab:green', edgecolor='black')

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=15)

    plt.ylabel('Count')
    plt.yscale('log')

    # Set y-axis limit slightly higher to give space for labels
    ymax = max(counts) if counts else 1
    plt.ylim(0.8, ymax * 2)

    plt.tight_layout()
    if savename:
        fig.savefig(savename, dpi=200)
    plt.show()

def plot_event_lifetimes(event_stats, 
                         category,
                         bins=20,
                         savename=None):
    """
    Plot histogram of event lifetimes for a specific category.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a list of dictionaries containing statistics for each event.
    - category:
        String specifying the category to plot.
        Valid values: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
    - bins:
        Optional integer specifying the number of bins for the histogram (default: 20).
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays histogram plot).
    """
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return
    
    category_titles = {
        'two_body_short': 'Two-body short',
        'multi_body_short': 'Multi-body short',
        'two_body_vdw': 'Two-body vdW',
        'multi_body_vdw': 'Multi-body vdW'
    }
    title = category_titles.get(category, category)
    
    lifetimes = [event['lifetime_ps'] for event in event_stats[category]]
    
    fig = plt.figure(figsize=(8, 5))
    plt.hist(lifetimes, bins=bins, color='tab:blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Lifetime (ps)')
    plt.ylabel('Count')
    if 'short' in category:
        plt.yscale('log')
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    if savename:
        fig.savefig(savename, dpi=200)
    plt.show()
        
def plot_all_event_lifetimes(event_stats,
                             bins=20,
                             savename=None):
    """
    Plot histograms of event lifetimes for all four categories in a 2x2 subplot grid.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a list of dictionaries containing statistics for each event.
    - bins:
        Optional integer specifying the number of bins for the histograms (default: 20).
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays subplot grid).
    """
    category_titles = {
        'two_body_short': 'Two-body short',
        'multi_body_short': 'Multi-body short',
        'two_body_vdw': 'Two-body vdW',
        'multi_body_vdw': 'Multi-body vdW'
    }
    
    categories = ['two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        title = category_titles.get(category, category)
        if category in event_stats:
            lifetimes = [event['lifetime_ps'] for event in event_stats[category]]
            axes[idx].hist(lifetimes, bins=bins, color='tab:blue', edgecolor='black')
            axes[idx].set_title(title)
            if 'short' not in category:
                axes[idx].set_xlabel('Lifetime (ps)')
            if 'two_body' in category:
                axes[idx].set_ylabel('Count')
            if 'short' in category:
                axes[idx].set_yscale('log')
            axes[idx].grid(axis='y', alpha=0.7)
        else:
            axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            axes[idx].set_title(title)
    
    plt.tight_layout()
    if savename:
        fig.savefig(savename, dpi=200)
    plt.show()

def plot_event_num_atoms(event_stats, 
                         category,
                         savename=None):
    """
    Plot histogram of number of atoms involved for a multi-body event category.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a list of dictionaries containing statistics for each event.
    - category:
        String specifying the category to plot.
        Must be a multi-body category: 'multi_body_short' or 'multi_body_vdw'.
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays histogram plot).
    """
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return
    
    if 'multi_body' not in category:
        print(f"Category '{category}' is not a multi-body category.")
        return
    
    category_titles = {
        'multi_body_short': 'Multi-body short',
        'multi_body_vdw': 'Multi-body vdW'
    }
    title = category_titles.get(category, category)
    
    num_atoms = [event['num_atoms_involved'] for event in event_stats[category]]
    
    fig = plt.figure(figsize=(8, 5))
    plt.hist(num_atoms, bins=range(3, max(num_atoms)+2), align='left', color='tab:orange', edgecolor='black', log=True)
    plt.title(title)
    plt.xlabel('N(Atoms)')
    plt.ylabel('Count')
    plt.xticks(range(3, max(num_atoms)+1))
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    if savename:
        fig.savefig(savename, dpi=200)
    plt.show()

def plot_all_event_num_atoms(event_stats,
                             savename=None):
    """
    Plot histograms of number of atoms involved for multi-body categories in a 1x2 subplot grid.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a list of dictionaries containing statistics for each event.
    - savename:
        Optional string specifying the filename to save the figure (default: None).

    Returns:
    - None (displays subplot grid).
    """
    category_titles = {
        'multi_body_short': 'Multi-body short',
        'multi_body_vdw': 'Multi-body vdW'
    }
    
    categories = ['multi_body_short', 'multi_body_vdw']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, category in enumerate(categories):
        title = category_titles.get(category, category)
        if category in event_stats:
            num_atoms = [event['num_atoms_involved'] for event in event_stats[category]]
            if num_atoms:
                axes[idx].hist(num_atoms, bins=range(3, max(num_atoms)+2), align='left', color='tab:orange', edgecolor='black', log=True)
                axes[idx].set_title(title)
                axes[idx].set_xlabel('N(Atoms)')
                if 'short' in category:
                    axes[idx].set_ylabel('Count')
                axes[idx].set_xticks(range(3, max(num_atoms)+1))
                axes[idx].grid(axis='y', alpha=0.7)
            else:
                axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
                axes[idx].set_title(title)
        else:
            axes[idx].text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            axes[idx].set_title(title)
    
    plt.tight_layout()
    if savename:
        fig.savefig(savename, dpi=200)
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
    category_titles = {
        'two_body_short': 'Two-body short',
        'multi_body_short': 'Multi-body short',
        'two_body_vdw': 'Two-body vdW',
        'multi_body_vdw': 'Multi-body vdW'
    }
    title = category_titles.get(category, category)
    # Filter out zero-count ranges
    filtered = [(c, r) for c, r in zip(counts, lifetime_ranges) if c > 0]
    if not filtered:
        print("No events to plot for the given lifetime ranges.")
        return
    filtered_counts, filtered_ranges = zip(*filtered)
    lifetime_labels = [f"{min_lifetime}-{max_lifetime}" for min_lifetime, max_lifetime in filtered_ranges]
    x = np.arange(len(lifetime_labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, filtered_counts, width=0.6, color='tab:green', edgecolor='black')
    ax.set_xlabel('Lifetime Ranges (ps)')
    ax.set_ylabel('Number of Events')
    ax.set_title(f'{title}')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(lifetime_labels)

    # Set y-axis limit slightly higher to give space for labels
    ymax = max(filtered_counts) if filtered_counts else 1
    ax.set_ylim(0.8, ymax * 2)

    # Add count labels on top of each bar
    for bar, count in zip(bars, filtered_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(count), ha='center', va='bottom', fontsize=14)
        
    plt.tight_layout()
    if savename:
        fig.savefig(savename, dpi=200)
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
    category_titles = {
        'two_body_short': 'Two-body short',
        'multi_body_short': 'Multi-body short',
        'two_body_vdw': 'Two-body vdW',
        'multi_body_vdw': 'Multi-body vdW'
    }
    categories = list(counts_by_lifetime.keys())
    counts = np.array(list(counts_by_lifetime.values()))
    
    lifetime_labels = [f"{min_lifetime}-{max_lifetime}" for min_lifetime, max_lifetime in lifetime_ranges]
    x = np.arange(len(lifetime_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, category in enumerate(categories):
        title = category_titles.get(category, category)
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
        fig.savefig(savename, dpi=200)
    plt.show()
