import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load data from the CSV file
data = pd.read_csv('varying_c3DP_for_python_costsavings.csv')

# Extract data
capacity = data['Capacity_3DP_Percentage']  # X-axis data
baseline = data['Baseline']  # Baseline curve (first curve)
# curves = data.iloc[:, [1]].join(data.iloc[:, 3:]) # Remaining columns for varying cost curves
curves = data.iloc[:, 1:]

# Define colors, markers, and labels
colors = ['tab:blue', 'black', 'tab:orange', '#EDB120']  # Black for baseline, blue and orange for other curves
markers = ['^', 'o', 's','d']  # Unique markers
# labels = ['1x Baseline', '2x Baseline', '3x Baseline']  # Legend labels
labels = ['0.5x', '1x', '2x', '3x']  # Legend labels
linewidths = [2, 3, 2, 2]  # Thicker line for the first curve
marker_size = 8  # Set marker size here (increase for larger markers)

# Function to format ticks as percentages
def percentage_formatter(x, pos):
    return f"{int(x)}%"

# Create the plot
plt.figure(figsize=(6, 4.2))

# Plot the baseline curve (first curve) with thicker black line and larger markers
# plt.plot(capacity, baseline, marker=markers[0], markersize=marker_size, linestyle='-', linewidth=linewidths[0], 
#          color=colors[0], label=labels[0])

# Plot the varying cost curves (second and third curves) with larger markers
for i, col in enumerate(curves.columns):
    plt.plot(capacity, curves[col], marker=markers[i], markersize=marker_size, linestyle='-', linewidth=linewidths[i], 
             color=colors[i], label=labels[i])

# Add a horizontal grey line at 0
plt.axhline(0, linestyle='--', linewidth=1.5, color='grey')

# Fill the area below 0 with light grey
plt.fill_between(capacity, -1e6, 0, color='lightgrey', alpha=0.5)

# Set axis limits
ylimit = max(curves.max().max(), baseline.max())
plt.ylim([-1.1 * ylimit, 1.1 * ylimit])
plt.xlim([0, 20])

# Format axes
plt.xlabel('3DP Capacity (% of Max Demand)', fontsize=13)
plt.ylabel('Cost Savings (%)', fontsize=14)
plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

# Add grid and legend
plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
# plt.legend(fontsize=12, loc='lower left', title='3DP variable cost $c^{\\mathsf{3DP}}$', title_fontsize=12)
plt.legend(fontsize=12, loc='lower left', title='$c^{\\mathsf{3DP}}$ (multiple of baseline)', title_fontsize=12, ncol =4)

# Save and show the figure
plt.savefig('Python_Plots/CostSavings/Varying_c3DP_(Fixed_C3DP_Case11)_CostSavings.pdf')
plt.show()
