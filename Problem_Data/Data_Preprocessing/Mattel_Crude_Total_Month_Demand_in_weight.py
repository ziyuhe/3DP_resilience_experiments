# NOTE: This script processes Mattel's Bill-of-Lading (BOL) data, 
# which was retrieved from a third-party service and is **not included** 
# in this repository due to licensing restrictions. 
# To use this script, you must provide your own dataset with a similar structure.

# NOTE: This script builds on the results from:
#    - "Mattel_Suppliers_Synthetic_Products_Info.py"
#    - "Mattel_Suppliers_Synthetic_Products_Info_Costs.py"
#    - "Mattel_Suppliers_Shipments_Processing.py"
#    - "Mattel_Suppliers_Shipments_Processing_Aggregate_by_Month_Week.py"
#
# This script calculates the total weight and quantity sourced per month among all suppliers


import os
import pandas as pd

def calculate_weight_and_quantity_sum(directory):
    total_weight_sum = 0.0
    total_quantity_sum = 0.0
    
    # Walk through each folder under the current directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            for file in os.listdir(folder_path):
                if file.endswith("(aggregated by month).csv"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    avg_weight = df['Weight (kg, 0.8 Markdown)'].mean() * 1000
                    avg_quantity = df['Quantity of Synthetic Product'].mean()
                    total_weight_sum += avg_weight
                    total_quantity_sum += avg_quantity
    
    final_weight_result = total_weight_sum * 0.0005
    final_quantity_result = total_quantity_sum * 0.0005
    print(f"Final Weight Result after multiplying by 0.0005 (in grams): {final_weight_result:.2f}")
    print(f"Final Quantity Result after multiplying by 0.0005: {final_quantity_result:.2f}")

# Specify the root directory containing the folders
directory = '.'

# Run the function
calculate_weight_and_quantity_sum(directory)
