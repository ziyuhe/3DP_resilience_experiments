import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("Calc_month_total_demand.csv")

# Extract the relevant columns
weights = data["Total Weight (kg)"].values * 1000
quantities = data["Total Quantity"].values

MatlabBlue = (0, 0.4470, 0.7410, 0.5)  # RGB with alpha for transparency
charcoal = (0.21, 0.27, 0.31)

# Create the histogram for Total Weight (as percentage)
plt.figure(figsize=(4, 4))
counts, bins, patches = plt.hist(weights, bins=20, color=MatlabBlue, alpha=0.5, edgecolor=charcoal)
percentages = (counts / len(weights)) * 100  # Convert counts to percentages
for rect, percent in zip(patches, percentages):
    rect.set_height(percent)  # Update bar heights to percentages
plt.xlabel("Total Demand(g)/Month", fontsize=15)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))  # Show ticks as percentages
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)
plt.savefig("Total_demand_month_weight.pdf", bbox_inches="tight")
plt.close()

# Create the histogram for Total Quantity (as percentage)
plt.figure(figsize=(4, 4))
counts, bins, patches = plt.hist(quantities, bins=20, color=MatlabBlue, alpha=0.5, edgecolor=charcoal)
percentages = (counts / len(quantities)) * 100  # Convert counts to percentages
for rect, percent in zip(patches, percentages):
    rect.set_height(percent)  # Update bar heights to percentages
plt.xlabel("Total Demand(units)/Month", fontsize=15)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))  # Show ticks as percentages
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=13)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=13)  # Minor ticks (if applicable)
plt.savefig("Total_demand_month_quantity.pdf", bbox_inches="tight")
plt.close()
