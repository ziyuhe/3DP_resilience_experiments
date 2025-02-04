import os
import pandas as pd
from collections import defaultdict

# Define the base folder path
base_folder = os.path.join("..", "Mattel_Suppliers_Data")

# Initialize a dictionary to store the aggregated results
all_months = defaultdict(lambda: {"total_weight": 0, "total_quantity": 0})

# Traverse through subfolders
for sub_folder_name in os.listdir(base_folder):
    sub_folder_path = os.path.join(base_folder, sub_folder_name)
    
    # Check if it's a directory
    if not os.path.isdir(sub_folder_path):
        continue
    
    # Find CSV files that end with "aggregated by month.csv"
    for file_name in os.listdir(sub_folder_path):
        if file_name.endswith("(aggregated by month).csv"):
            file_path = os.path.join(sub_folder_path, file_name)
            
            # Read the CSV file
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            
            # Ensure required columns exist
            required_columns = ["Arrival Date", "Weight (kg, 0.8 Markdown)", "Quantity of Synthetic Product"]
            if not all(col in data.columns for col in required_columns):
                print(f"Skipping file {file_path} due to missing columns.")
                continue
            
            # Process each row in the file
            for _, row in data.iterrows():
                month = row["Arrival Date"]
                weight = row["Weight (kg, 0.8 Markdown)"]
                quantity = row["Quantity of Synthetic Product"]
                
                # Update totals for the month
                all_months[month]["total_weight"] += weight
                all_months[month]["total_quantity"] += quantity

# Create a DataFrame from the aggregated results
sorted_months = sorted(all_months.items(), key=lambda x: pd.to_datetime(x[0], format="Month %m of %Y"), reverse=True)
df_results = pd.DataFrame(
    [(month, values["total_weight"], values["total_quantity"]) for month, values in sorted_months],
    columns=["Month", "Total Weight (kg)", "Total Quantity"]
)

# Save the results to a CSV file
output_file = "Calc_month_total_demand.csv"
df_results.to_csv(output_file, index=False)
print(f"Aggregated results saved to {output_file}.")
