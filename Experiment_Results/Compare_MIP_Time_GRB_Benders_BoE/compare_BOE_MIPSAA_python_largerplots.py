import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV files
optimality_gap_data = pd.read_csv('optimality_gap_data.csv', header=None)
time_comparison_data = pd.read_csv('time_comparison_data.csv', header=None)

# Extract data for the first plot (Optimality Gap)
n_values_gap = optimality_gap_data[0]
boe_gap = optimality_gap_data[1]
benders_gap = optimality_gap_data[2]

# Plot Optimality Gap
plt.figure(figsize=(6, 4))
plt.axhspan(-100, 0, color='grey', alpha=0.2, zorder=0)  # Fill below y=0
plt.plot(n_values_gap, boe_gap, linewidth=5, label='SuperMod. Approx.')
plt.plot(n_values_gap, benders_gap, '-o', linewidth=1.5, markersize=10, markerfacecolor='none', label='MIO-SAA-Benders')
plt.xlim(3, 55)
plt.ylim(-0.333,9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Shallower grid
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
plt.xlabel('$n$', fontsize=18)
plt.ylabel('Approx. Optimality Gap', fontsize=18)
plt.legend(fontsize=18, loc='lower right')
plt.xticks(fontsize=15)  # Increase x-axis tick size
plt.yticks(fontsize=15)  # Increase y-axis tick size
plt.tight_layout()
plt.savefig('optimality_gap_comparison_python_largerplots.pdf')
plt.close()

# Extract data for the second plot (Time Comparison)
n_values_time = time_comparison_data[0]
boe_time = time_comparison_data[1]
benders_time = time_comparison_data[2]

# Plot Time Comparison
plt.figure(figsize=(6, 4))
plt.axhspan(-100, 0, color='grey', alpha=0.2, zorder=0)  # Fill below y=0
plt.plot(n_values_time, boe_time*598/100., '-o', linewidth=2, markersize=5, markerfacecolor='none', label='SuperMod. Approx.')
plt.plot(n_values_time, benders_time*598/100., linewidth=3, label='MIO-SAA-Benders')
plt.xlim(3, 55)
plt.ylim(-10, 605)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Shallower grid
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
plt.xlabel('$n$', fontsize=18)
plt.ylabel('Termination Time (sec)', fontsize=18)
plt.legend(fontsize=18, loc='center right')
plt.xticks(fontsize=15)  # Increase x-axis tick size
plt.yticks(fontsize=15)  # Increase y-axis tick size
plt.tight_layout()
plt.savefig('time_comparison_python_lagerplots.pdf')
plt.close()
