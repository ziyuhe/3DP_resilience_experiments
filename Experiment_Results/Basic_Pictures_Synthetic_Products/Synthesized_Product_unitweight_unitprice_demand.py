import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv("Synthesized_Product_unitweight_unitprice_demand.csv")

# Extract columns
weights = data['AverageWeightPerMonth_g']
demands = data['AverageDemandPerMonth_unit']
prices = data['UnitPrice']
sizes = data['MarkerSize']

# Define marker properties
# marker_color = (0.5, 0, 1, 0.333)  # RGB with alpha for transparency
grid_color = (0.5, 0.5, 0.5)
marker_color = (0, 0.4470, 0.7410, 0.5)
# grid_color = (0.21, 0.27, 0.31)

# Plot 1: Weight vs Unit Price
plt.figure(figsize=(4, 4))
plt.scatter(weights, prices, s=sizes, color=marker_color, edgecolors='none')
plt.xlabel('Avg. Demand(g)/Month', fontsize=15)
plt.ylabel('Unit Price($)', fontsize=16)
plt.axhline(y=32, color=grid_color, linewidth=1)
plt.axvline(x=3e8, color=grid_color, linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)
plt.savefig("Synthesized_Product_unitweight_unitprice_demand(weight)_python.pdf", bbox_inches='tight')
plt.close()

# Plot 2: Demand vs Unit Price
plt.figure(figsize=(4, 4))
plt.scatter(demands, prices, s=sizes, color=marker_color, edgecolors='none')
plt.xlabel('Average Demand per Month(unit)', fontsize=15)
plt.ylabel('Unit Price($)', fontsize=16)
plt.axhline(y=32, color=grid_color, linewidth=1)
plt.axvline(x=4e5, color=grid_color, linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)
plt.savefig("Synthesized_Product_unitweight_unitprice_demand(quantity)_python.pdf", bbox_inches='tight')
plt.close()
