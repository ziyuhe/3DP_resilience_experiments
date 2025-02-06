import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


#########################################################################################################
# COSTSAVINGS VARYING C3DP LARGE FONTS
#########################################################################################################

# Read data from the CSV file
data = pd.read_csv('benchmark_for_python_costsavings.csv')

# Extract data
capacity = data['Capacity_3DP_Percentage']
curves = data.iloc[:, 1:-1]  # Exclude the last curve

# Define markers, colors, and line styles
markers = ['o', 's', '^', 'd', 'v']  # Different markers
colors = ['#1f77b4', '#6baed6',  # Shades of blue (1st and 4th lines)
          '#ff7f0e', '#fdae61']  # Shades of orange (2nd and 5th lines)
black_line_color = 'black'  # Color for the third line

# Line and marker assignments
line_marker_map = {
    0: {'color': colors[0], 'marker': markers[0], 'linewidth': 2},  # First line: Dark Blue, thinner
    1: {'color': colors[2], 'marker': markers[1], 'linewidth': 2},  # Second line: Dark Orange, thinner
    2: {'color': black_line_color, 'marker': markers[2], 'linewidth': 3},  # Third line: Black, thickest
    3: {'color': colors[1], 'marker': markers[3], 'linewidth': 2},  # Fourth line: Light Blue, thinner
    4: {'color': colors[3], 'marker': markers[4], 'linewidth': 2}   # Fifth line: Light Orange, thinner
}

# Function to format ticks as percentages
def percentage_formatter(x, pos):
    return f"{int(x)}%"

# Create the plot
plt.figure(figsize=(6.5, 6.5))

# Plot curves with specified colors, markers, and line widths
for i, col in enumerate(curves.columns):
    style = line_marker_map.get(i, {'color': 'grey', 'marker': markers[i % len(markers)], 'linewidth': 2})
    plt.plot(
        capacity, curves[col], 
        marker=style['marker'], linestyle='-', linewidth=style['linewidth'], markersize=12, 
        color=style['color'], label=f'Curve {i + 1}'
    )

# Add horizontal grey line at 0
plt.axhline(0, linestyle='--', linewidth=1.5, color='grey')

# Fill the area below 0 with light grey
plt.fill_between(capacity, -1e6, 0, color='lightgrey', alpha=0.5)

# Set axis limits
plt.xlim(0, 15)
plt.ylim(-1.1 * curves.values.max(), 1.1 * curves.values.max())

# Apply percentage formatting to ticks
plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=15)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)

# Set custom ticks for density control
plt.xticks(np.arange(0, 21, 3))  # Major ticks every 5 units
# plt.yticks(np.arange(-4, 4, 2))  # Major ticks every 0.5 units

# Add grid, labels, and legend
plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
# plt.xlabel('3DP Capacity (% of Max Demand)', fontsize=20)
plt.xlabel(r'$K$', fontsize=25)
plt.ylabel('Cost Savings (%)', fontsize=20)


# Add legend
# legend_labels = ['0.2x Baseline', '0.4x Baseline', '1x Baseline', '3x Baseline', '5x Baseline', '7x Baseline']
legend_labels = ['0.2x', '0.4x', '1x', '3x', '5x', '7x']
plt.legend(
    legend_labels[:len(curves.columns)], 
    fontsize=19, 
    loc='lower left', 
    ncol = 3,
    title=r"$c^{\mathsf{cap}}$ (multiple of baseline)",  # Adjusted phrasing and LaTeX interpretation
    title_fontsize=22
)

# Save the figure
plt.savefig('Python_Plots/CostSavings/Varying_C3DP_Coeff_CostSavings_Large_Fonts.pdf')
plt.show()
















#########################################################################################################
# SHORTFALL VARYING C3DP LARGE FONTS
#########################################################################################################


import pandas as pd
import matplotlib.pyplot as plt

# Load data
mean_data = pd.read_csv('benchmark_for_python_shortfalls1.csv')
boxplot_data = pd.read_csv('benchmark_for_python_shortfalls2.csv')

# Define tick labels
tick_labels = ['0.2x', '0.4x', '1x', '3x', '5x', 'No3DP']

# Define box fill color
box_color = (0.9062, 0.9062, 0.9062, 0.333)
MatlabBlue = (0, 0.4470, 0.7410, 0.777)

# Create figure
plt.figure(figsize=(7,7))

plt.plot(mean_data['Position'], mean_data['MeanShortfall'], '-o', label='Mean Demand Shortfall', linewidth=3, color=MatlabBlue, zorder=10, markersize=12)

# Plot boxplots with thicker median bars and filled boxes
positions = boxplot_data['Position'].unique()
for pos in positions:
    data = boxplot_data[boxplot_data['Position'] == pos]['Shortfall']
    bp = plt.boxplot(
        data,
        positions=[pos],
        widths=0.5,
        showfliers=False,
        patch_artist=True, zorder=1  # Enable patch_artist for box fill
    )
    # Customize boxplot appearance
    for box in bp['boxes']:
        box.set(facecolor=box_color)  # Light grey fill
        box.set_edgecolor('black')  # Solid black edge
        box.set_linewidth(3)  # Thicker box line
    for whisker in bp['whiskers']:
        whisker.set_linewidth(3)  # Thicker whiskers
    for cap in bp['caps']:
        cap.set_linewidth(3)  # Thicker caps
    for median in bp['medians']:
        median.set_linewidth(5)  # Thicker median lines
        median.set_color('#ff7f0e')  # Orange median line

# Plot mean shortfall after boxplots to bring it to the front
# plt.plot(mean_data['Position'], mean_data['MeanShortfall'], '-o', label='Mean Demand Shortfall', linewidth=1, color=MatlabBlue, zorder=10, markersize=7)

# Customize plot
plt.xticks(ticks=range(1, len(tick_labels) + 1), labels=tick_labels)
plt.xlabel('$c^{\mathsf{cap}}$', fontsize=22)
# plt.ylabel('Shortfall (% of Max Demand)', fontsize=20)
plt.ylabel('Shortfall (%)', fontsize=22)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Show only horizontal grid lines
plt.legend(['Mean Demand Shortfall'], fontsize=20)

# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=18)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks (if applicable)


# Correct y-axis ticks (values are already percentages)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Save the plot
plt.savefig('Python_Plots/Shortfalls/boxplots_varying_C3DP_all_in_one_Large_Fonts.pdf')
plt.show()











