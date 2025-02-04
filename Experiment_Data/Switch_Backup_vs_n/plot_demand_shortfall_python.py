import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define a formatter function for the y-axis
def percent_formatter(x, _):
    return f"{x:.1f}%"  # Format as 'value%'

# Define num_suppliers_set
num_suppliers_set = [8, 15, 30, 45]
data_files = [f"Data_NumSuppliers_demandshortfall_{num_suppliers}.csv" for num_suppliers in num_suppliers_set[1:]]

# Custom RGB colors
matlab_blue = [0, 0.4470, 0.7410]
matlab_orange = [0.8500, 0.3250, 0.0980]
matlab_yellow = [0.9290, 0.6940, 0.1250]
colors = {
    "SINGLE": matlab_blue,    # Blue
    "Naive": matlab_orange,   # Orange
    "Full":  matlab_yellow    # Yellow
}

# Initialize a figure
plt.figure(figsize=(8,6))

# Set lighter grid properties
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Loop through files and process data
for num_suppliers, file in zip(num_suppliers_set[1:], data_files):
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Extract and reshape SINGLE, Naive, and Full columns
        SINGLE = df.filter(like="SINGLE").values.flatten()  # Collect all SINGLE_1, SINGLE_2, ...
        Naive = df.filter(like="Naive").values.flatten()    # Collect all Naive_1, Naive_2, ...
        Full = df.filter(like="Full").values.flatten()      # Collect all Full_1, Full_2, ...
        
        # Create a new DataFrame
        reshaped_df = pd.DataFrame({
            "SINGLE": SINGLE,
            "Naive": Naive,
            "Full": Full,
            "NumSuppliers": [num_suppliers] * len(SINGLE)
        })
        
        # SINGLE boxplot
        plt.boxplot(
            reshaped_df['SINGLE'], 
            positions=[num_suppliers - 3], 
            widths=2.5, 
            patch_artist=True, 
            showfliers=False,  # Suppress outliers
            boxprops=dict(facecolor=(*colors["SINGLE"], 0.1), edgecolor=matlab_blue, linewidth=2),  # Transparent fill
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1), 
            medianprops=dict(color=colors["SINGLE"], linewidth=3)  # Median bar matches box color
        )

        # Naive boxplot
        plt.boxplot(
            reshaped_df['Naive'], 
            positions=[num_suppliers], 
            widths=2.5, 
            patch_artist=True, 
            showfliers=False,  # Suppress outliers
            boxprops=dict(facecolor=(*colors["Naive"], 0.1), edgecolor=matlab_orange, linewidth=2),  # Transparent fill
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1), 
            medianprops=dict(color=colors["Naive"], linewidth=3)  # Median bar matches box color
        )

        # Full boxplot
        plt.boxplot(
            reshaped_df['Full'], 
            positions=[num_suppliers + 3], 
            widths=2.5, 
            patch_artist=True, 
            showfliers=False,  # Suppress outliers
            boxprops=dict(facecolor=(*colors["Full"], 0.1), edgecolor=matlab_yellow, linewidth=2),  # Transparent fill
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1), 
            medianprops=dict(color=colors["Full"], linewidth=3)  # Median bar matches box color
        )

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Adjust x-axis limits and labels
plt.xlim([min(num_suppliers_set[1:]) - 10, max(num_suppliers_set) + 10])
plt.xticks(num_suppliers_set[1:], labels=num_suppliers_set[1:], fontsize=15)  # Larger x-ticks
plt.xlabel(r"$n$", fontsize=19)  # Larger x-label

# Format y-axis as percentages
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))  # Append % to y-axis values
plt.yticks(fontsize=15)  # Larger y-ticks
plt.ylabel("Mean Demand Shortfall", fontsize=19)  # Larger y-label

plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot
plt.savefig("Demand_shortfall_boxplots.pdf")
plt.show()
