import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Define a formatter function for the y-axis with decimals
def percent_formatter_decimal(x, _):
    return f"{x:.1f}%"  # Decimal format (e.g., 80.0%)

# Define a formatter function for the y-axis with integers
def percent_formatter_integer(x, _):
    return f"{int(x)}%"  # Integer format (e.g., 80%)

# Define the shared legend labels
legend_labels = {
    "SINGLE": "No 3DP",
    "Naive": "Naive 3DP",
    "Full": "Full 3DP",
}

# Colors for boxplots and median-like bars
matlab_blue = [0, 0.4470, 0.7410]
matlab_orange = [0.8500, 0.3250, 0.0980]
matlab_yellow = [0.9290, 0.6940, 0.1250]
colors = {
    "SINGLE": matlab_blue,
    "Naive": matlab_orange,
    "Full": matlab_yellow,
}

# Load bar values from CSV
bar_values = np.loadtxt('n=55_data_for_plot_python.csv', delimiter=',')

# Data files for each subplot
num_suppliers_set = [8, 15, 30, 45]
data_files = [
    [f"Data_NumSuppliers_TMbackup_{num_suppliers}.csv" for num_suppliers in num_suppliers_set[1:]],
    [f"Data_NumSuppliers_costsavings_{num_suppliers}.csv" for num_suppliers in num_suppliers_set[1:]],
    [f"Data_NumSuppliers_demandshortfall_{num_suppliers}.csv" for num_suppliers in num_suppliers_set[1:]],
]

# Titles and labels
titles = ["Dedicated Backups", "Cost Savings", "Mean Demand Shortfall"]
y_labels = ["Dedicated Backups", "Cost Savings", "Mean Demand Shortfall"]

# Set up the figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
fig.subplots_adjust(top=0.8, bottom=0.2, left=0.05, right=0.98, wspace=0.2)

for idx, (ax, files, title, y_label) in enumerate(zip(axes, data_files, titles, y_labels)):
    for num_suppliers, file in zip(num_suppliers_set[1:], files):
        try:
            # Load data
            df = pd.read_csv(file)
            SINGLE = df.filter(like="SINGLE").values.flatten()
            Naive = df.filter(like="Naive").values.flatten()
            Full = df.filter(like="Full").values.flatten()

            # Plot SINGLE
            ax.boxplot(
                SINGLE,
                positions=[num_suppliers - 3],
                widths=2.5,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=(*colors["SINGLE"], 0.1), edgecolor=matlab_blue, linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                medianprops=dict(color=colors["SINGLE"], linewidth=2.5),
            )

            # Plot Naive
            ax.boxplot(
                Naive,
                positions=[num_suppliers],
                widths=2.5,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=(*colors["Naive"], 0.1), edgecolor=matlab_orange, linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                medianprops=dict(color=colors["Naive"], linewidth=2.5),
            )

            # Plot Full
            ax.boxplot(
                Full,
                positions=[num_suppliers + 3],
                widths=2.5,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=(*colors["Full"], 0.1), edgecolor=matlab_yellow, linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                medianprops=dict(color=colors["Full"], linewidth=2.5),
            )

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Add horizontal bars at positions 55-3, 55, and 55+3
    bar_positions = [55-2, 55, 55+2]
    bar_values_for_subplot = bar_values[idx]  # Get the row corresponding to this subplot

    for pos, val, color in zip(bar_positions, bar_values_for_subplot, colors.values()):
        ax.hlines(y=val, xmin=pos-1.25, xmax=pos+1.25, color=color, linewidth=3)  # Mimic median bars

    # Customize axes
    ax.set_title(title, fontsize=14)
    ax.set_xlim([min(num_suppliers_set[1:]) - 10, 60])
    ax.set_xticks(num_suppliers_set[1:] + [55])  # Add 55 to x-axis ticks for clarity
    ax.set_xticklabels(num_suppliers_set[1:] + [55], fontsize=12)
    ax.set_xlabel(r"$n$", fontsize=14)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add percentage formatter to the y-axis
    if idx == 0:
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter_integer))  # Integer format for subplot 1
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter_decimal))  # Decimal format for subplot 2 and 3

    ax.tick_params(axis="y", labelsize=12)

# Shared legend
handles = [
    plt.Line2D([0], [0], color=(*colors["SINGLE"], 1), linewidth=4, label=legend_labels["SINGLE"]),
    plt.Line2D([0], [0], color=(*colors["Naive"], 1), linewidth=4, label=legend_labels["Naive"]),
    plt.Line2D([0], [0], color=(*colors["Full"], 1), linewidth=4, label=legend_labels["Full"]),
]
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=3,
    fontsize=14,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05),
)

# Save and show plot
plt.savefig("Combined_Boxplots_With_Horizontal_Bars.pdf", bbox_inches="tight")
plt.show()
