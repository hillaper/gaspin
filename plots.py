"""
Plotting and visualization utilities for gpsp_tools.
"""

##########################################################################################################
# Imports
##########################################################################################################
import matplotlib.pyplot as plt

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

        'axes.titlesize': 18,
        'axes.labelsize': 18,

        'lines.linewidth': 2.5,

        'xtick.labelsize': 16,
        'ytick.labelsize': 16,

        'legend.fontsize': 12,
        'legend.loc': 'best',

        'grid.alpha': 0.7,
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

            # Identify the main partner atom index from the first frame of the event
            main_partner_index = event_df.iloc[0]['atoms2'][0] if event_df.iloc[0]['atoms2'] else None
            
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
                    # Plot other partner atoms with default colors
                    plt.plot(data['frames'], data['distances'], label=f'Atom {partner_atom}')
            
            # Add plot title, labels, and legend
            plt.title(f'Atom {atom_index} - Event {event_index} - {category}')
            plt.xlabel('Frame')
            plt.ylabel('Distance (Å)')
            plt.ylim(0, collision_cutoff + 1)
            if xlims:
                plt.xlim(xlims)
            plt.axhline(y=collision_cutoff, color='r', linestyle='--', alpha=0.5)
            plt.legend()
            plt.show()
        else:
            # Print a message if the event index is out of range
            print(f"Event index {event_index} is out of range for atom {atom_index} in category {category}.")
    else:
        # Print a message if the atom index is not found in the specified category
        print(f"Atom {atom_index} not found in category {category}.")

def plot_event_lifetimes(event_stats, 
                         category):
    """
    Plot histograms of event lifetimes and number of atoms involved for a specific category.

    Parameters:
    - event_stats: 
        Dictionary with four keys: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.
        Each key maps to a list of dictionaries containing statistics for each event.
    - category:
        String specifying the category to plot.
        Valid values: 'two_body_short', 'multi_body_short', 'two_body_vdw', 'multi_body_vdw'.

    Returns:
    - None (displays histogram plots).
    """
    if category not in event_stats:
        print(f"Category '{category}' not found in event_stats.")
        return
        
    lifetimes = [event['lifetime_ps'] for event in event_stats[category]]
    
    # Plot lifetime histogram for the category
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(lifetimes, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Lifetimes - {category}')
    plt.xlabel('Lifetime (ps)')
    plt.ylabel('Count')
    
    # Plot number of atoms involved only for multi-body categories
    if 'multi_body' in category:
        num_atoms = [event['num_atoms_involved'] for event in event_stats[category]]
        plt.subplot(1, 2, 2)
        plt.hist(num_atoms, bins=range(2, max(num_atoms)+2), align='left', color='salmon', edgecolor='black', log=True)
        plt.title(f'N(Atoms) - {category}')
        plt.xlabel('N(Atoms)')
        plt.ylabel('Count (log scale)')
        plt.xticks(range(2, max(num_atoms)+1))
    
    plt.tight_layout()
    plt.show()
