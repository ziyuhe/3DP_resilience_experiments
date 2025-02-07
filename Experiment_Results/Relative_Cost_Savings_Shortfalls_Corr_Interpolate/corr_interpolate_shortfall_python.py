import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

def plot_boxplots_and_means(box_csv, meta_csv, save_dir, filename_prefix, ylimit=[-0.333, 6]):
    """
    Reads CSV files and plots the boxplot with mean curves, including softened outliers.
    
    Parameters:
        box_csv (str): Path to the CSV file containing the boxplot data.
        meta_csv (str): Path to the CSV file containing metadata (box positions, means, xticks).
        save_dir (str): Directory where the plots will be saved.
        filename_prefix (str): Prefix for the saved filenames.
        ylimit (list): Y-axis limits in the format [y_min, y_max]. Default is [-0.333, 5].
    """
    # Load boxplot data (boxdata1 and boxdata2 stacked horizontally)
    box_data = pd.read_csv(box_csv, header=None)
    num_rows, num_cols = box_data.shape
    num_cols_half = num_cols // 2  # Since it's concatenated (boxdata1 | boxdata2)

    boxdata1 = box_data.iloc[:, :num_cols_half].values  # First half is boxdata1
    boxdata2 = box_data.iloc[:, num_cols_half:].values  # Second half is boxdata2

    # Load metadata
    metadata = pd.read_csv(meta_csv, header=None)

    # Extract data from metadata
    boxposition1 = metadata.iloc[:, 0].values
    boxposition2 = metadata.iloc[:, 1].values
    meandata1 = metadata.iloc[:, 2].values
    meandata2 = metadata.iloc[:, 3].values
    x_ticks_positions = metadata.iloc[:, 4].values  # X-axis tick positions
    x_ticks_labels = metadata.iloc[:, 5].astype(str).values  # Convert to string labels

    # Set up colors (similar to MATLAB styling)
    Matlab_blue_trans = (0, 0.4470, 0.7410, 0.2)  # Semi-transparent blue
    Matlab_orange_trans = (0.8500, 0.3250, 0.0980, 0.2)  # Semi-transparent orange
    Matlab_blue_solid = (0, 0.4470, 0.7410)  # Solid blue
    Matlab_orange_solid = (0.8500, 0.3250, 0.0980)  # Solid orange
    Solid_grey = [0.5, 0.5, 0.5]  # Gray for reference lines
    Outlier_color = (0.3, 0.3, 0.3, 0.2)  # Light gray outliers with transparency

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Start plotting
    plt.figure(figsize=(6, 4))

    # Legend handles (for custom legend)
    legend_handles = []

    # Loop over the number of plotted elements (rows in boxdata)
    for i in range(num_rows):
        # Extract row-wise data
        box_data_1 = np.squeeze(boxdata1[i, :])  # Convert to 1D array
        box_data_2 = np.squeeze(boxdata2[i, :])  # Convert to 1D array

        # First set of box plots (with 3DP)
        plt.boxplot(
            [box_data_1], 
            positions=[boxposition1[i]], 
            widths=0.5, 
            patch_artist=True,
            whiskerprops=dict(color=Matlab_blue_solid, linestyle='--', linewidth=0.8),
            medianprops=dict(color=Matlab_blue_solid, linestyle='-', linewidth=1),
            boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, linewidth=1),
            flierprops=dict(marker='', color=Outlier_color, markersize=3, alpha=0.3)  # Softened outliers
        )

        # Second set of box plots (without 3DP)
        plt.boxplot(
            [box_data_2], 
            positions=[boxposition2[i]], 
            widths=0.5, 
            patch_artist=True,
            whiskerprops=dict(color=Matlab_orange_solid, linestyle='--', linewidth=0.8),
            medianprops=dict(color=Matlab_orange_solid, linestyle='-', linewidth=1),
            boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, linewidth=1),
            flierprops=dict(marker='', color=Outlier_color, markersize=3, alpha=0.3)  # Softened outliers
        )

        # Add legend handles for the first iteration
        if i == 0:
            legend_handles.extend([
                Patch(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, label='Distr. (with 3DP)', linewidth=1),
                Patch(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, label='Distr. (no 3DP)', linewidth=1)
            ])

    # Plot mean curves
    line1, = plt.plot(boxposition1, meandata1, '-o', label='Mean (with 3DP)', color=Matlab_blue_solid, linewidth=1)
    line2, = plt.plot(boxposition2, meandata2, '-^', label='Mean (no 3DP)', color=Matlab_orange_solid, linewidth=1)

    # Add to legend
    legend_handles.extend([line1, line2])

    # Add vertical reference lines (if needed)
    for pos in x_ticks_positions:
        plt.axvline(x=pos, color=Solid_grey, linestyle='-', linewidth=0.5)

    # Set x-ticks and labels
    plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=12)

    # Adjust y-axis to show percentage
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Set custom y-axis limits
    plt.ylim(ylimit)

    # Labels and formatting
    plt.xlabel(r'$p_0/p$', fontsize=15)
    plt.ylabel('Demand Shortfalls', fontsize=15)

    # Add legend
    legend = plt.legend(handles=legend_handles, loc='upper left', ncol=2, fontsize=9, frameon=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')

    # Save the plot
    save_path = os.path.join(save_dir, f"{filename_prefix}_boxplot.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {save_path}")





def plot_combined_boxplots(box_csv1, meta_csv1, box_csv2, meta_csv2, save_dir, filename, ylimit=[-0.333, 6]):
    """
    Plots two boxplots side by side as subfigures, sharing the y-axis and legend.

    Parameters:
        box_csv1 (str): Path to the first CSV file for "Small Disruption".
        meta_csv1 (str): Path to the first metadata CSV file.
        box_csv2 (str): Path to the second CSV file for "Large Disruption".
        meta_csv2 (str): Path to the second metadata CSV file.
        save_dir (str): Directory where the plot will be saved.
        filename (str): Filename for saving the final figure.
        ylimit (list): Y-axis limits [y_min, y_max]. Default is [-0.333, 6].
    """

    # Load first dataset
    box_data1 = pd.read_csv(box_csv1, header=None)
    meta_data1 = pd.read_csv(meta_csv1, header=None)

    # Load second dataset
    box_data2 = pd.read_csv(box_csv2, header=None)
    meta_data2 = pd.read_csv(meta_csv2, header=None)

    # Extract metadata for the first dataset
    boxposition1_small = meta_data1.iloc[:, 0].values
    boxposition2_small = meta_data1.iloc[:, 1].values
    meandata1_small = meta_data1.iloc[:, 2].values
    meandata2_small = meta_data1.iloc[:, 3].values
    x_ticks_positions = meta_data1.iloc[:, 4].values
    x_ticks_labels = meta_data1.iloc[:, 5].astype(str).values

    # Extract metadata for the second dataset
    boxposition1_large = meta_data2.iloc[:, 0].values
    boxposition2_large = meta_data2.iloc[:, 1].values
    meandata1_large = meta_data2.iloc[:, 2].values
    meandata2_large = meta_data2.iloc[:, 3].values

    # Define colors
    Matlab_blue_trans = (0, 0.4470, 0.7410, 0.1)
    Matlab_orange_trans = (0.8500, 0.3250, 0.0980, 0.1)
    Matlab_blue_solid = (0, 0.4470, 0.7410)
    Matlab_orange_solid = (0.8500, 0.3250, 0.0980)
    Solid_grey = [0.5, 0.5, 0.5]

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.1, right=0.95, wspace=0.1)

    # Legend handles
    legend_handles = [
        Patch(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, label='Distr. (with 3DP)', linewidth=1.5),
        Patch(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, label='Distr. (no 3DP)', linewidth=1.5),
        plt.Line2D([0], [0], color=Matlab_blue_solid, linestyle="-", marker="o", label="Mean (with 3DP)"),
        plt.Line2D([0], [0], color=Matlab_orange_solid, linestyle="--", marker="^", label="Mean (no 3DP)")
    ]

    # Function to plot each subplot
    def plot_subplot(ax, box_data, box_positions1, box_positions2, mean_data1, mean_data2, title):
        num_rows, num_cols = box_data.shape
        num_cols_half = num_cols // 2  

        boxdata1 = box_data.iloc[:, :num_cols_half].values
        boxdata2 = box_data.iloc[:, num_cols_half:].values

        for i in range(num_rows):
            ax.boxplot(
                [np.squeeze(boxdata1[i, :])],
                positions=[box_positions1[i]],
                widths=0.5,
                patch_artist=True,
                whiskerprops=dict(color=Matlab_blue_solid, linestyle='--', linewidth=0.8),
                medianprops=dict(color=Matlab_blue_solid, linewidth=3),
                boxprops=dict(facecolor=Matlab_blue_trans, edgecolor=Matlab_blue_solid, linewidth=1.5),
                flierprops=dict(marker='', color='gray', markersize=3, alpha=0.3)
            )

            ax.boxplot(
                [np.squeeze(boxdata2[i, :])],
                positions=[box_positions2[i]],
                widths=0.5,
                patch_artist=True,
                whiskerprops=dict(color=Matlab_orange_solid, linestyle='--', linewidth=0.8),
                medianprops=dict(color=Matlab_orange_solid, linewidth=2),
                boxprops=dict(facecolor=Matlab_orange_trans, edgecolor=Matlab_orange_solid, linewidth=1.5),
                flierprops=dict(marker='', color='gray', markersize=3, alpha=0.3)
            )

        # Mean curves
        ax.plot(box_positions1, mean_data1, '-o', color=Matlab_blue_solid, linewidth=1, label="Mean (with 3DP)")
        ax.plot(box_positions2, mean_data2, '-^', color=Matlab_orange_solid, linewidth=1, label="Mean (no 3DP)")

        # Vertical reference lines
        for pos in x_ticks_positions:
            ax.axvline(x=pos, color=Solid_grey, linestyle='-', linewidth=0.5)

        ax.set_xticks(x_ticks_positions)
        ax.set_xticklabels(x_ticks_labels, fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylim(ylimit)  # Apply y-axis limits
        ax.set_xlabel(r"$p_0 / p$", fontsize=16)

    # Plot first subplot (Small Disruption)
    plot_subplot(axes[0], box_data1, boxposition1_small, boxposition2_small, meandata1_small, meandata2_small, "Small Disruption")

    # Plot second subplot (Large Disruption)
    plot_subplot(axes[1], box_data2, boxposition1_large, boxposition2_large, meandata1_large, meandata2_large, "Large Disruption")

    # Adjust y-axis ticks (show only on left)
    axes[0].set_ylabel("Demand Shortfalls", fontsize=16)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Shared legend at the top
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=13, frameon=True)

    # Save the figure
    save_path = os.path.join(save_dir, f"{filename}_combined.pdf")
    plt.savefig(save_path)#, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {save_path}")









if __name__ == "__main__":

    plot_boxplots_and_means(
        "corr_inter_shortfall(small_p_small_yieldloss)_boxdata.csv", 
        "corr_inter_shortfall(small_p_small_yieldloss)_otherdata.csv",
        "Python_Plots",
        "shortfalls(small_p_small_yieldloss)",
        ylimit=[-0.333, 6]
    )

    plot_boxplots_and_means(
        "corr_inter_shortfall(small_p_large_yieldloss)_boxdata.csv", 
        "corr_inter_shortfall(small_p_large_yieldloss)_otherdata.csv",
        "Python_Plots",
        "shortfalls(small_p_large_yieldloss)",
        ylimit=[-0.333, 6]
    )

    plot_combined_boxplots(
        "corr_inter_shortfall(small_p_small_yieldloss)_boxdata.csv", 
        "corr_inter_shortfall(small_p_small_yieldloss)_otherdata.csv",
        "corr_inter_shortfall(small_p_large_yieldloss)_boxdata.csv", 
        "corr_inter_shortfall(small_p_large_yieldloss)_otherdata.csv",
        "Python_Plots",
        "shortfalls_comparison",
        ylimit=[-0.333, 6]
    )
