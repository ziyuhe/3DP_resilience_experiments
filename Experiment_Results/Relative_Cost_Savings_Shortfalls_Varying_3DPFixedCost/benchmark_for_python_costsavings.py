import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
plt.figure(figsize=(6, 4.2))

# Plot curves with specified colors, markers, and line widths
for i, col in enumerate(curves.columns):
    style = line_marker_map.get(i, {'color': 'grey', 'marker': markers[i % len(markers)], 'linewidth': 2})
    plt.plot(
        capacity, curves[col], 
        marker=style['marker'], linestyle='-', linewidth=style['linewidth'], markersize=8, 
        color=style['color'], label=f'Curve {i + 1}'
    )

# Add horizontal grey line at 0
plt.axhline(0, linestyle='--', linewidth=1.5, color='grey')

# Fill the area below 0 with light grey
plt.fill_between(capacity, -1e6, 0, color='lightgrey', alpha=0.5)

# Set axis limits
plt.xlim(0, 20)
plt.ylim(-1.1 * curves.values.max(), 1.1 * curves.values.max())

# Apply percentage formatting to ticks
plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

# Add grid, labels, and legend
plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
plt.xlabel('3DP Capacity (% of Max Demand)', fontsize=13)
plt.ylabel('Cost Savings (%)', fontsize=14)

# Add legend
# legend_labels = ['0.2x Baseline', '0.4x Baseline', '1x Baseline', '3x Baseline', '5x Baseline', '7x Baseline']
legend_labels = ['0.2x', '0.4x', '1x', '3x', '5x', '7x']
plt.legend(
    legend_labels[:len(curves.columns)], 
    fontsize=10, 
    loc='lower left', 
    ncol = 3,
    title=r"$c^{\mathsf{cap}}$ (multiple of baseline)",  # Adjusted phrasing and LaTeX interpretation
    title_fontsize=10
)

# Save the figure
plt.savefig('Python_Plots/CostSavings/Varying_C3DP_Coeff_CostSavings.pdf')
plt.show()
