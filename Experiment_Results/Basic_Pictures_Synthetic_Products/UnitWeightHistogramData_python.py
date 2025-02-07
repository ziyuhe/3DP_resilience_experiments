import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("UnitWeightHistogramData.csv")

# Extract the weights
weights = data['UnitWeight_g']

MatlabBlue = (0, 0.4470, 0.7410, 0.5)  # RGB with alpha for transparency
charcoal = (0.21, 0.27, 0.31)

# Plot the histogram
plt.figure(figsize=(4, 4))
plt.hist(weights, bins=range(int(min(weights)), int(max(weights)) + 100, 100), color=MatlabBlue, edgecolor=charcoal)
plt.xlabel('Unit Weight(g)', fontsize=18)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))  # Show ticks as percentages
plt.grid(True, linestyle='--', alpha=0.7)

plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)

# Save the plot
plt.savefig("UnitWeightHistogram_python.pdf", bbox_inches='tight')
plt.close()
