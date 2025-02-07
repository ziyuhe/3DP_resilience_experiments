import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch




#################################################################################
# INDEPENDENT: VARYING YIELDLOSS
#################################################################################

# Load data from .mat file
def load_data1():
    mat_data = scipy.io.loadmat('varying_disruption_distr_ind_for_python_shortfalls1.mat')
    return mat_data

# Reproduce the plots
def plot_data1(data):
    # Unpack data
    p_subset1 = [2, 6, 11]
    Box_plot_data11 = data['Box_plot_data11'][0]  # Access nested structure
    Box_plot_data12 = data['Box_plot_data12'][0]  # Access nested structure
    mean_plot_data11 = data['mean_plot_data11']  # Directly as NumPy array
    mean_plot_data12 = data['mean_plot_data12']  # Directly as NumPy array
    box_plot_pos11 = data['box_plot_pos11'][0]  # Flatten structure
    box_plot_pos12 = data['box_plot_pos12'][0]  # Flatten structure
    x_ticks_labels1 = [label[0] for label in data['x_ticks_labels1'][0]]  # Convert to list of strings
    x_ticks_pos1 = data['x_ticks_pos1'][0]  # Flatten structure
    xlimit1 = data['xlimit1'][0]  # X-axis limits
    ylimit1 = data['ylimit1'][0]  # Y-axis limits
    vertline_pos1 = data['vertline_pos1'][0]  # Vertical line positions

    # Colors
    Matlab_blue_trans = (0, 0.4470, 0.7410, 0.0333)  # Transparent blue for boxes
    Matlab_orange_trans = (0.8500, 0.3250, 0.0980, 0.0333)  # Transparent orange for boxes
    Matlab_blue_solid = (0, 0.4470, 0.7410)  # Original solid blue
    Matlab_orange_solid = (0.8500, 0.3250, 0.0980)  # Original solid orange

    # Darker shades for mean curves
    Matlab_blue_darker = tuple(max(c - 0.1, 0) for c in Matlab_blue_solid)  # Slightly darker blue
    Matlab_orange_darker = tuple(max(c - 0.1, 0) for c in Matlab_orange_solid)  # Slightly darker orange
    Solid_grey = [0.5, 0.5, 0.5]

    # Create output directory if it doesn't exist
    output_dir = "Python_Plots/Shortfalls"
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop over p_subset1
    for ppp, pp in enumerate(p_subset1):
        plt.figure(figsize=(4, 4))
        
        # Legend handles
        legend_handles = []
        
        # Loop over yield_loss_rate_subset1
        for yyy in range(len(box_plot_pos11)):
            # Box plots
            box_data1 = np.squeeze(Box_plot_data11[ppp][yyy])  # Squeeze to 1D
            box_data2 = np.squeeze(Box_plot_data12[ppp][yyy])  # Squeeze to 1D
            
            # First set of box plots
            plt.boxplot(
                [box_data1], 
                positions=[box_plot_pos11[yyy]], 
                widths=0.5, 
                patch_artist=True,
                whiskerprops=dict(color=Matlab_blue_solid, linestyle='--', linewidth=0.8),
                medianprops=dict(color=Matlab_blue_solid, linestyle='-', linewidth=1),
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, linewidth=1),
                flierprops=dict(marker='', markersize=0)  # No outliers
            )
            
            # Second set of box plots
            plt.boxplot(
                [box_data2], 
                positions=[box_plot_pos12[yyy]], 
                widths=0.5, 
                patch_artist=True,
                whiskerprops=dict(color=Matlab_orange_solid, linestyle='--', linewidth=0.8),
                medianprops=dict(color=Matlab_orange_solid, linestyle='-', linewidth=1),
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, linewidth=1),
                flierprops=dict(marker='', markersize=0)  # No outliers
            )
            
            # Add legend handles for the first iteration
            if yyy == 0:
                legend_handles.extend([
                    Patch(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, label='Distr. (with 3DP)',linewidth=1),
                    Patch(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, label='Distr. (no 3DP)',linewidth=1)
                ])
        
        # Plot mean curves with darker colors
        line1, = plt.plot(box_plot_pos11, mean_plot_data11[ppp, :], '-o', label='Mean (with 3DP)', color=Matlab_blue_darker, linewidth=1)
        line2, = plt.plot(box_plot_pos12, mean_plot_data12[ppp, :], '-^', label='Mean (no 3DP)', color=Matlab_orange_darker, linewidth=1)
        
        # Add the mean lines to the legend
        legend_handles.extend([line1, line2])
        
        # Add vertical lines (exclude positions at x-ticks)
        for pos in vertline_pos1:
            if not any(np.isclose(pos, x_ticks_pos1)):  # Ensure no overlap with tick positions
                plt.axvline(x=pos, color=Solid_grey, linestyle='-', linewidth=0.5)  # Solid and thin
        
        # Add x ticks
        plt.xticks(x_ticks_pos1, x_ticks_labels1)
        
        # Set x and y limits
        plt.xlim(xlimit1)
        plt.ylim(ylimit1)
        
        # Adjust y-axis ticks to percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add labels
        plt.xlabel(r'$\alpha_j$', fontsize=15)
        plt.ylabel('Shortfall', fontsize=15)
        
        # Add legend in the top-left corner
        legend = plt.legend(
            handles=legend_handles,
            loc='upper left',
            ncol=2, columnspacing=0.5,
            fontsize=9, 
            frameon=True  # Turn on the legend frame
        )
        legend.get_frame().set_edgecolor('black')  # Set black border
        legend.get_frame().set_facecolor('white')  # Optional: Set background color to white

        plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
        plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)
        
        # Only show gridlines along x-axis
        # plt.grid(axis='x')
        
        # Save plot with specified name
        filename = f"{output_dir}/boxplots_fixed_C3DPcase(11)_fixed_p_varying_yieldloss_case{pp:03}.pdf"
        plt.savefig(filename, bbox_inches='tight')  # Tight layout to fit legend
        plt.close()


#################################################################################
# INDEPENDENT: VARYING P
#################################################################################
# Load data from .mat file
def load_data2():
    mat_data = scipy.io.loadmat('varying_disruption_distr_ind_for_python_shortfalls2.mat')
    return mat_data

# Reproduce the plots
def plot_data2(data):
    # Unpack data
    yield_loss_rate_subset2 = [2, 5, 8]  # Example yield loss subset
    Box_plot_data21 = data['Box_plot_data21'][0]  # Access nested structure
    Box_plot_data22 = data['Box_plot_data22'][0]  # Access nested structure
    mean_plot_data21 = data['mean_plot_data21']  # Directly as NumPy array
    mean_plot_data22 = data['mean_plot_data22']  # Directly as NumPy array
    box_plot_pos21 = data['box_plot_pos21'][0]  # Flatten structure
    box_plot_pos22 = data['box_plot_pos22'][0]  # Flatten structure
    x_ticks_labels2 = [label[0] for label in data['x_ticks_labels2'][0]]  # Convert to list of strings
    x_ticks_pos2 = data['x_ticks_pos2'][0]  # Flatten structure
    xlimit2 = data['xlimit2'][0]  # X-axis limits
    ylimit2 = data['ylimit2'][0]  # Y-axis limits
    vertline_pos2 = data['vertline_pos2'][0]  # Vertical line positions

    # Colors
    Matlab_blue_trans = (0, 0.4470, 0.7410, 0.0333)  # Transparent blue for boxes
    Matlab_orange_trans = (0.8500, 0.3250, 0.0980, 0.0333)  # Transparent orange for boxes
    Matlab_blue_solid = (0, 0.4470, 0.7410)  # Solid blue
    Matlab_orange_solid = (0.8500, 0.3250, 0.0980)  # Solid orange
    Matlab_blue_darker = tuple(max(c - 0.1, 0) for c in Matlab_blue_solid)  # Slightly darker blue
    Matlab_orange_darker = tuple(max(c - 0.1, 0) for c in Matlab_orange_solid)  # Slightly darker orange
    Solid_grey = [0.5, 0.5, 0.5]

    # Create output directory if it doesn't exist
    output_dir = "Python_Plots/Shortfalls"
    os.makedirs(output_dir, exist_ok=True)

    # Loop over yield_loss_rate_subset2
    for yyy, yy in enumerate(yield_loss_rate_subset2):
        plt.figure(figsize=(4, 4))

        # Legend handles
        legend_handles = []

        # Loop over p_subset2
        for ppp in range(len(box_plot_pos21)):
            # Box plots
            box_data21 = np.squeeze(Box_plot_data21[yyy][ppp])  # Squeeze to 1D
            box_data22 = np.squeeze(Box_plot_data22[yyy][ppp])  # Squeeze to 1D

            # First set of box plots
            plt.boxplot(
                [box_data21],
                positions=[box_plot_pos21[ppp]],
                widths=0.5,
                patch_artist=True,
                whiskerprops=dict(color=Solid_grey, linestyle=':', linewidth=0.8),
                medianprops=dict(color=Matlab_blue_solid, linestyle='-', linewidth=1),
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, linewidth=1),
                flierprops=dict(marker='', markersize=0)  # No outliers
            )

            # Second set of box plots
            plt.boxplot(
                [box_data22],
                positions=[box_plot_pos22[ppp]],
                widths=0.5,
                patch_artist=True,
                whiskerprops=dict(color=Solid_grey, linestyle=':', linewidth=0.8),
                medianprops=dict(color=Matlab_orange_solid, linestyle='-', linewidth=1),
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, linewidth=1),
                flierprops=dict(marker='', markersize=0)  # No outliers
            )

            # Add legend handles for the first iteration
            if ppp == 0:
                legend_handles.extend([
                    Patch(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, label='Distri. (with 3DP)'),
                    Patch(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, label='Distri. (no 3DP)')
                ])

        # Plot mean curves with darker colors
        line1, = plt.plot(box_plot_pos21, mean_plot_data21[yyy, :], '-o', label='Mean (with 3DP)', color=Matlab_blue_darker, linewidth=1)
        line2, = plt.plot(box_plot_pos22, mean_plot_data22[yyy, :], '-^', label='Mean (no 3DP)', color=Matlab_orange_darker, linewidth=1)

        # Add the mean lines to the legend
        legend_handles.extend([line1, line2])

        # Add vertical lines
        for pos in vertline_pos2:
            plt.axvline(x=pos, color=Solid_grey, linestyle='-', linewidth=0.5)  # Solid and thin

        # Add x ticks
        plt.xticks(x_ticks_pos2, x_ticks_labels2)

        # Set x and y limits
        plt.xlim(xlimit2)
        plt.ylim(ylimit2)

        # Adjust y-axis ticks to percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Add labels
        plt.xlabel(r'$p_j$', fontsize=15)
        plt.ylabel('Shortfall', fontsize=15)

        # Add legend in the top-left corner
        legend = plt.legend(
            handles=legend_handles,
            loc='upper left',
            ncol=2, columnspacing=0.5,
            fontsize=9, 
            frameon=True  # Turn on the legend frame
        )
        legend.get_frame().set_edgecolor('black')  # Set black border
        legend.get_frame().set_facecolor('white')  # Optional: Set background color to white

        plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
        plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)

        # Save plot with specified name
        filename = f"{output_dir}/boxplots_fixed_C3DPcase(11)_fixed_yieldloss_varying_p_case{yy:03}.pdf"
        plt.savefig(filename, bbox_inches='tight')  # Tight layout to fit legend
        plt.close()












################################################################################################################################
# IND 3DP VS. NO3DP COMBINED PLOTS (1 X 4)
################################################################################################################################
def combined_plot(data1, data2):
    # Unpack data for plot_data1
    p_subset1 = [2, 6, 11]
    Box_plot_data11 = data1['Box_plot_data11'][0]
    Box_plot_data12 = data1['Box_plot_data12'][0]
    mean_plot_data11 = data1['mean_plot_data11']
    mean_plot_data12 = data1['mean_plot_data12']
    box_plot_pos11 = data1['box_plot_pos11'][0]
    box_plot_pos12 = data1['box_plot_pos12'][0]
    x_ticks_labels1 = [label[0] for label in data1['x_ticks_labels1'][0]]
    x_ticks_pos1 = data1['x_ticks_pos1'][0]
    xlimit1 = data1['xlimit1'][0]
    ylimit1 = data1['ylimit1'][0]

    # Unpack data for plot_data2
    yield_loss_rate_subset2 = [2, 5, 8]
    Box_plot_data21 = data2['Box_plot_data21'][0]
    Box_plot_data22 = data2['Box_plot_data22'][0]
    mean_plot_data21 = data2['mean_plot_data21']
    mean_plot_data22 = data2['mean_plot_data22']
    box_plot_pos21 = data2['box_plot_pos21'][0]
    box_plot_pos22 = data2['box_plot_pos22'][0]
    x_ticks_labels2 = [label[0] for label in data2['x_ticks_labels2'][0]]
    x_ticks_pos2 = data2['x_ticks_pos2'][0]
    xlimit2 = data2['xlimit2'][0]
    ylimit2 = data2['ylimit2'][0]

    # Set up the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)  # Reduced height
    fig.subplots_adjust(top=0.8, bottom=0.2, wspace=0.05)  # Adjust spacing to avoid squeezing

    # Colors
    Matlab_blue_trans = (0, 0.4470, 0.7410, 0.0333)
    Matlab_orange_trans = (0.8500, 0.3250, 0.0980, 0.0333)
    Matlab_blue_solid = (0, 0.4470, 0.7410)
    Matlab_orange_solid = (0.8500, 0.3250, 0.0980)
    Matlab_blue_darker = tuple(max(c - 0.1, 0) for c in Matlab_blue_solid)
    Matlab_orange_darker = tuple(max(c - 0.1, 0) for c in Matlab_orange_solid)

    # Titles for the subplots
    titles = [
        r"Fixing $p_j$ = 1x baseline",
        r"Fixing $p_j$ = 10x baseline",
        r"Fixing $\alpha_j$ = 1x baseline",
        r"Fixing $\alpha_j$ = 12x baseline"
    ]

    # Shaded title box properties
    title_bbox = dict(facecolor='lightgrey', edgecolor='none', alpha=0.5, boxstyle="round,pad=0.2")

    # First and third plot from data1
    for idx, (ppp, ax) in enumerate(zip([0, 2], axes[:2])):
        for yyy in range(len(box_plot_pos11)):
            box_data1 = np.squeeze(Box_plot_data11[ppp][yyy])
            box_data2 = np.squeeze(Box_plot_data12[ppp][yyy])
            ax.boxplot(
                [box_data1],
                positions=[box_plot_pos11[yyy]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid),
                medianprops=dict(color=Matlab_blue_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_blue_solid, linestyle='--'),
                flierprops=dict(marker='', markersize=0)
            )
            ax.boxplot(
                [box_data2],
                positions=[box_plot_pos12[yyy]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid),
                medianprops=dict(color=Matlab_orange_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_orange_solid, linestyle='--'),
                flierprops=dict(marker='', markersize=0)
            )
        ax.plot(box_plot_pos11, mean_plot_data11[ppp, :], '-o', color=Matlab_blue_darker, markersize=7)
        ax.plot(box_plot_pos12, mean_plot_data12[ppp, :], '-^', color=Matlab_orange_darker, markersize=7)
        ax.set_xlim(xlimit1)
        ax.set_ylim(ylimit1)
        ax.set_xticks(x_ticks_pos1)
        ax.set_xticklabels(x_ticks_labels1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_xlabel(r'$\alpha_j$', fontsize=16)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(titles[idx], fontsize=14, bbox=title_bbox)

    # First and third plot from data2
    for idx, (yyy, ax) in enumerate(zip([0, 2], axes[2:])):
        for ppp in range(len(box_plot_pos21)):
            box_data21 = np.squeeze(Box_plot_data21[yyy][ppp])
            box_data22 = np.squeeze(Box_plot_data22[yyy][ppp])
            ax.boxplot(
                [box_data21],
                positions=[box_plot_pos21[ppp]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid),
                medianprops=dict(color=Matlab_blue_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_blue_solid, linestyle=':'),
                flierprops=dict(marker='', markersize=0)
            )
            ax.boxplot(
                [box_data22],
                positions=[box_plot_pos22[ppp]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid),
                medianprops=dict(color=Matlab_orange_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_orange_solid, linestyle=':'),
                flierprops=dict(marker='', markersize=0)
            )
        ax.plot(box_plot_pos21, mean_plot_data21[yyy, :], '-o', color=Matlab_blue_darker, markersize=7)
        ax.plot(box_plot_pos22, mean_plot_data22[yyy, :], '-^', color=Matlab_orange_darker, markersize=7)
        ax.set_xlim(xlimit2)
        ax.set_ylim(ylimit2)
        ax.set_xticks(x_ticks_pos2)
        ax.set_xticklabels(x_ticks_labels2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_xlabel(r'$p_j$', fontsize=16)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(titles[idx + 2], fontsize=14, bbox=title_bbox)

    # Add shared legend
    handles = [
        Patch(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, label='Distribution (with 3DP)'),
        Patch(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, label='Distribution (no 3DP)'),
        Line2D([0], [0], color=Matlab_blue_darker, label='Mean (with 3DP)', marker='o', markersize=7),
        Line2D([0], [0], color=Matlab_orange_darker, label='Mean (no 3DP)', marker='^', markersize=7)
    ]
    fig.legend(
        handles=handles, loc='upper center', ncol=4, fontsize=12, frameon=True
    )

    # Save and show the combined plot
    output_dir = "Python_Plots/Shortfalls"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/combined_boxplot_ind.pdf", bbox_inches='tight')
    plt.show()




################################################################################################################################
# IND 3DP VS. NO3DP COMBINED PLOTS (2 X 3)
################################################################################################################################
def combined_plot2(data1, data2):
    # Unpack data for plot_data1
    p_subset1 = [2, 6, 11]
    Box_plot_data11 = data1['Box_plot_data11'][0]
    Box_plot_data12 = data1['Box_plot_data12'][0]
    mean_plot_data11 = data1['mean_plot_data11']
    mean_plot_data12 = data1['mean_plot_data12']
    box_plot_pos11 = data1['box_plot_pos11'][0]
    box_plot_pos12 = data1['box_plot_pos12'][0]
    x_ticks_labels1 = [label[0] for label in data1['x_ticks_labels1'][0]]
    x_ticks_pos1 = data1['x_ticks_pos1'][0]
    xlimit1 = data1['xlimit1'][0]
    ylimit1 = data1['ylimit1'][0]

    # Unpack data for plot_data2
    yield_loss_rate_subset2 = [2, 5, 8]
    Box_plot_data21 = data2['Box_plot_data21'][0]
    Box_plot_data22 = data2['Box_plot_data22'][0]
    mean_plot_data21 = data2['mean_plot_data21']
    mean_plot_data22 = data2['mean_plot_data22']
    box_plot_pos21 = data2['box_plot_pos21'][0]
    box_plot_pos22 = data2['box_plot_pos22'][0]
    x_ticks_labels2 = [label[0] for label in data2['x_ticks_labels2'][0]]
    x_ticks_pos2 = data2['x_ticks_pos2'][0]
    xlimit2 = data2['xlimit2'][0]
    ylimit2 = data2['ylimit2'][0]

    # Set up the figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharey='row')
    fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.5, wspace=0.1)

    # Colors
    Matlab_blue_trans = (0, 0.4470, 0.7410, 0.0333)
    Matlab_orange_trans = (0.8500, 0.3250, 0.0980, 0.0333)
    Matlab_blue_solid = (0, 0.4470, 0.7410)
    Matlab_orange_solid = (0.8500, 0.3250, 0.0980)
    Matlab_blue_darker = tuple(max(c - 0.1, 0) for c in Matlab_blue_solid)
    Matlab_orange_darker = tuple(max(c - 0.1, 0) for c in Matlab_orange_solid)

    # Titles for the subplots
    first_row_titles = [
        r"Fixing $p_j$ = 1x baseline",
        r"Fixing $p_j$ = 5x baseline",
        r"Fixing $p_j$ = 10x baseline"
    ]
    second_row_titles = [
        r"Fixing $\alpha_j$ = 1x baseline",
        r"Fixing $\alpha_j$ = 6x baseline",
        r"Fixing $\alpha_j$ = 12x baseline"
    ]

    # Add shaded titles to plots using `bbox`
    title_bbox = dict(facecolor='lightgrey', edgecolor='none', alpha=0.5, boxstyle="round,pad=0.1")

    # Plot all three plots from data1 in the first row
    for idx, (ppp, ax) in enumerate(zip(range(3), axes[0])):
        for yyy in range(len(box_plot_pos11)):
            box_data1 = np.squeeze(Box_plot_data11[ppp][yyy])
            box_data2 = np.squeeze(Box_plot_data12[ppp][yyy])
            ax.boxplot(
                [box_data1],
                positions=[box_plot_pos11[yyy]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid),
                medianprops=dict(color=Matlab_blue_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_blue_solid, linestyle='--'),
                flierprops=dict(marker='', markersize=0)
            )
            ax.boxplot(
                [box_data2],
                positions=[box_plot_pos12[yyy]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid),
                medianprops=dict(color=Matlab_orange_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_orange_solid, linestyle='--'),
                flierprops=dict(marker='', markersize=0)
            )
        ax.plot(box_plot_pos11, mean_plot_data11[ppp, :], '-o', color=Matlab_blue_darker, markersize=7)
        ax.plot(box_plot_pos12, mean_plot_data12[ppp, :], '-^', color=Matlab_orange_darker, markersize=7)
        ax.set_xlim(xlimit1)
        ax.set_ylim(ylimit1)
        ax.set_xticks(x_ticks_pos1)
        ax.set_xticklabels(x_ticks_labels1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_xlabel(r'$\alpha_j$', fontsize=14)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(first_row_titles[idx], fontsize=14, bbox=title_bbox)

    # Plot all three plots from data2 in the second row
    for idx, (yyy, ax) in enumerate(zip(range(3), axes[1])):
        for ppp in range(len(box_plot_pos21)):
            box_data21 = np.squeeze(Box_plot_data21[yyy][ppp])
            box_data22 = np.squeeze(Box_plot_data22[yyy][ppp])
            ax.boxplot(
                [box_data21],
                positions=[box_plot_pos21[ppp]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid),
                medianprops=dict(color=Matlab_blue_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_blue_solid, linestyle='--'),
                flierprops=dict(marker='', markersize=0)
            )
            ax.boxplot(
                [box_data22],
                positions=[box_plot_pos22[ppp]],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid),
                medianprops=dict(color=Matlab_orange_solid, linewidth=1),
                whiskerprops=dict(color=Matlab_orange_solid, linestyle='--'),
                flierprops=dict(marker='', markersize=0)
            )
        ax.plot(box_plot_pos21, mean_plot_data21[yyy, :], '-o', color=Matlab_blue_darker, markersize=7)
        ax.plot(box_plot_pos22, mean_plot_data22[yyy, :], '-^', color=Matlab_orange_darker, markersize=7)
        ax.set_xlim(xlimit2)
        ax.set_ylim(ylimit2)
        ax.set_xticks(x_ticks_pos2)
        ax.set_xticklabels(x_ticks_labels2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_xlabel(r'$p_j$', fontsize=14)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(second_row_titles[idx], fontsize=14, bbox=title_bbox)

    # Add shared legend
    handles = [
        Patch(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, label='Distribution (with 3DP)'),
        Patch(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, label='Distribution (no 3DP)'),
        Line2D([0], [0], color=Matlab_blue_darker, label='Mean (with 3DP)', marker='o', markersize=7),
        Line2D([0], [0], color=Matlab_orange_darker, label='Mean (no 3DP)', marker='^', markersize=7)
    ]
    fig.legend(
        handles=handles, loc='upper center', ncol=4, fontsize=12, frameon=True
    )

    fig.text(0.07, 0.5, "Shortfall (% of Max. Demand)", ha='center', va='center', rotation='vertical', fontsize=18)
    # Save and show the combined plot
    output_dir = "Python_Plots/Shortfalls"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/combined_boxplot_ind2.pdf", bbox_inches='tight')
    plt.show()













# Main execution
if __name__ == '__main__':
    mat_data1 = load_data1()
    plot_data1(mat_data1)

    mat_data2 = load_data2()
    plot_data2(mat_data2)

    combined_plot(mat_data1, mat_data2)
    combined_plot2(mat_data1, mat_data2)
