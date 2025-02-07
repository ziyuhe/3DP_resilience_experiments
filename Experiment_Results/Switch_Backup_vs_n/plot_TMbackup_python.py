import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define a formatter function for the y-axis
def percent_formatter(x, _):
    return f"{x:.0f}%"  # Format as 'value%'

# Define num_suppliers_set
num_suppliers_set = [8, 15, 30, 45]
data_files = [f"Data_NumSuppliers_TMbackup_{num_suppliers}.csv" for num_suppliers in num_suppliers_set[1:]]

# Custom RGB colors
matlab_blue = [0, 0.4470, 0.7410]
matlab_orange = [0.8500, 0.3250, 0.0980]
matlab_yellow = [0.9290, 0.6940, 0.1250]
colors = {
    "SINGLE": matlab_blue,    # Blue
    "Naive": matlab_orange,   # Orange
    "Full":  matlab_yellow    # Yellow
}

# Plot function
plt.figure(figsize=(8,6))
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

for num_suppliers, file in zip(num_suppliers_set[1:], data_files):
    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Extract and reshape SINGLE, Naive, and Full columns
        SINGLE = df.filter(like="SINGLE").values.flatten()
        Naive = df.filter(like="Naive").values.flatten()
        Full = df.filter(like="Full").values.flatten()

        # SINGLE boxplot
        plt.boxplot(
            SINGLE, 
            positions=[num_suppliers - 3], 
            widths=2.5, 
            patch_artist=True, 
            showfliers=False,
            boxprops=dict(facecolor=(*colors["SINGLE"], 0.1), edgecolor=matlab_blue, linewidth=2),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1), 
            medianprops=dict(color=colors["SINGLE"], linewidth=3)
        )

        # Naive boxplot
        plt.boxplot(
            Naive, 
            positions=[num_suppliers], 
            widths=2.5, 
            patch_artist=True, 
            showfliers=False,
            boxprops=dict(facecolor=(*colors["Naive"], 0.1), edgecolor=matlab_orange, linewidth=2),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1), 
            medianprops=dict(color=colors["Naive"], linewidth=3)
        )

        # Full boxplot
        plt.boxplot(
            Full, 
            positions=[num_suppliers + 3], 
            widths=2.5, 
            patch_artist=True, 
            showfliers=False,
            boxprops=dict(facecolor=(*colors["Full"], 0.1), edgecolor=matlab_yellow, linewidth=2),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1), 
            medianprops=dict(color=colors["Full"], linewidth=3)
        )

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Adjust x-axis limits and labels
plt.xlim([min(num_suppliers_set[1:]) - 10, max(num_suppliers_set) + 10])
plt.xticks(num_suppliers_set[1:], labels=num_suppliers_set[1:], fontsize=15)
plt.xlabel(r"$n$", fontsize=19)

# Format y-axis as percentages
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
plt.yticks(fontsize=15)
plt.ylabel("Dedicated Backups (%)", fontsize=19)

plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.savefig("Ratio_TMbackup_boxplots.pdf")
plt.show()
