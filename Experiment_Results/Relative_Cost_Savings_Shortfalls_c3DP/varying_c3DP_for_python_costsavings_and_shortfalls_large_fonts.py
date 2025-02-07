import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np








#########################################################################################################
# COSTSAVINGS VARYING c3DP LARGE FONTS
#########################################################################################################

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
marker_size = 12  # Set marker size here (increase for larger markers)

# Function to format ticks as percentages
def percentage_formatter(x, pos):
    return f"{int(x)}%"

# Create the plot
plt.figure(figsize=(6.5, 6.5))

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
plt.xlim([0, 15])

# Format axes
# plt.xlabel('3DP Capacity (% of Max Demand)', fontsize=20)
plt.xlabel(r'$K$', fontsize=25)
plt.ylabel('Cost Savings (%)', fontsize=20)

plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

# Add grid and legend
plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.5)
# plt.legend(fontsize=12, loc='lower left', title='3DP variable cost $c^{\\mathsf{3DP}}$', title_fontsize=12)
plt.legend(fontsize=20, loc='lower left', title='$c^{\\mathsf{3DP}}$ (multiple of baseline)', title_fontsize=22, ncol =2)



# Set custom ticks for density control
plt.xticks(np.arange(0, 21, 3))  # Major ticks every 5 units
# plt.yticks(np.arange(-4, 4, 2))  # Major ticks every 0.5 units

# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=15)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)

# Save and show the figure
plt.savefig('Python_Plots/CostSavings/Varying_c3DP_(Fixed_C3DP_Case11)_CostSavings_Large_Fonts.pdf')
plt.show()




















#########################################################################################################
# SHORTFALL VARYING c3DP LARGE FONTS
#########################################################################################################

# Load data
mean_data = pd.read_csv('varying_c3DP_for_python_shortfalls1.csv')
boxplot_data = pd.read_csv('varying_c3DP_for_python_shortfalls2.csv')

# Define tick labels
tick_labels = ['0.5x', '1x', '2x', '3x', 'No 3DP']

# Define box fill color
box_color = (0.9062, 0.9062, 0.9062, 0.333)
MatlabBlue = (0, 0.4470, 0.7410, 0.777)

# Create figure
plt.figure(figsize=(7,7))

# Plot mean shortfall after boxplots to bring it to the front
plt.plot(mean_data['Position'], mean_data['MeanShortfall'], '-o', label='Mean Demand Shortfall', linewidth=3, color=MatlabBlue, zorder=10, markersize=12)

# Plot boxplots
positions = boxplot_data['Position'].unique()
for pos in positions:
    data = boxplot_data[boxplot_data['Position'] == pos]['Shortfall']
    bp = plt.boxplot(
        data,
        positions=[pos],
        widths=0.5,
        showfliers=False,
        patch_artist=True
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

# Customize plot
plt.xticks(ticks=range(1, len(tick_labels) + 1), labels=tick_labels)
plt.xlabel('$c_j^{\mathsf{3DP}}$', fontsize=22)
# plt.ylabel('Shortfall (% of Max Demand)', fontsize=20)
plt.ylabel('Shortfall (%)', fontsize=22)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Only horizontal lines
plt.legend(['Mean Demand Shortfall'], fontsize=20,loc='upper left')

# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=18)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks (if applicable)

# Correct y-axis ticks (values are already percentages)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Save the plot
plt.savefig('Python_Plots/Shortfalls/boxplots_varying_c3DP_all_in_one(Fixed_C3DP_Case11)_Large_Fonts.pdf')
plt.show()

