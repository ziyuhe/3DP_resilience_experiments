import pandas as pd
import matplotlib.pyplot as plt

# Load data
mean_data = pd.read_csv('varying_c3DP_for_python_shortfalls1.csv')
boxplot_data = pd.read_csv('varying_c3DP_for_python_shortfalls2.csv')

# Define tick labels
tick_labels = ['0.5x', '1x', '2x', '3x', 'No 3DP']

# Define box fill color
box_color = (0.9062, 0.9062, 0.9062, 0.333)
MatlabBlue = (0, 0.4470, 0.7410, 0.777)

# Create figure
plt.figure(figsize=(6, 4.2))

# Plot mean shortfall after boxplots to bring it to the front
plt.plot(mean_data['Position'], mean_data['MeanShortfall'], '-o', label='Mean Demand Shortfall', linewidth=1, color=MatlabBlue, zorder=10, markersize=8)

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
        box.set_linewidth(1)  # Thicker box line
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1)  # Thicker whiskers
    for cap in bp['caps']:
        cap.set_linewidth(1)  # Thicker caps
    for median in bp['medians']:
        median.set_linewidth(3)  # Thicker median lines
        median.set_color('#ff7f0e')  # Orange median line

# Customize plot
plt.xticks(ticks=range(1, len(tick_labels) + 1), labels=tick_labels)
plt.xlabel('$c^{\mathsf{3DP}}$ (Multiple of Baseline)', fontsize=12)
plt.ylabel('Shortfall (% of Max Demand)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Only horizontal lines
plt.legend(['Mean Demand Shortfall'], fontsize=12)

# Correct y-axis ticks (values are already percentages)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Save the plot
plt.savefig('Python_Plots/Shortfalls/boxplots_varying_c3DP_all_in_one(Fixed_C3DP_Case11).pdf')
plt.show()
