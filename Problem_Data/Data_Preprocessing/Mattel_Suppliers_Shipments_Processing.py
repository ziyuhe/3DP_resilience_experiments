# NOTE: This script processes Mattel's Bill-of-Lading (BOL) data, 
# which was retrieved from a third-party service and is **not included** 
# in this repository due to licensing restrictions. 
# To use this script, you must provide your own dataset with a similar structure.

# NOTE: This script builds on the results from:
#    - "Mattel_Suppliers_Synthetic_Products_Info.py"
#    - "Mattel_Suppliers_Synthetic_Products_Info_Costs.py"
#
# For each supplier, it converts sourcing records from **weight-based measurements** 
# to the **number of units** of synthesized products.

import os
import pandas as pd

def process_files(directory):
    # Load the synthetic products info data
    synthetic_products_info_file = "Mattel_Suppliers_Synthetic_Products_Info.csv"
    synthetic_df = pd.read_csv(synthetic_products_info_file)
    
    # Walk through each folder under the current directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            # Set a flag to check if we found the file we need
            file_found = False
            original_file_path = None
            
            # Check if a file with "Shipments.csv" exists
            for file in os.listdir(os.path.join(root, dir_name)):
                if file.endswith("Shipments.csv"):
                    original_file_path = os.path.join(root, dir_name, file)
                    file_found = True
                    break
            
            # If not found, check for a file with "Mattel_Only.csv"
            if not file_found:
                for file in os.listdir(os.path.join(root, dir_name)):
                    if file.endswith("Mattel_Only.csv"):
                        original_file_path = os.path.join(root, dir_name, file)
                        file_found = True
                        break
            
            if not file_found:
                print(f"No valid file found in {dir_name}. Skipping.")
                continue
            
            # Process the original file
            original_df = pd.read_csv(original_file_path)
            output_df = pd.DataFrame()
            
            # Retrieve and process columns
            output_df['Arrival Date'] = original_df['Arrival Date']
            output_df['Weight (kg, 0.8 Markdown)'] = (original_df['Weight'] * 0.8).round(2)
            
            # Find the synthetic product weight for the current company
            synthetic_product_weight = synthetic_df.loc[synthetic_df['Company Name'] == dir_name, 'Synthetic Product Weight (gram)'].values[0]
            
            # Calculate the quantity of synthetic product
            output_df['Quantity of Synthetic Product'] = ((output_df['Weight (kg, 0.8 Markdown)'] * 1000 / synthetic_product_weight).apply(lambda x: int(x)))
            
            # Save the processed file
            output_file_path = os.path.join(root, dir_name, f"{dir_name}_shipment_processed.csv")
            output_df.to_csv(output_file_path, index=False)
            print(f"Processed file saved to {output_file_path}")

# Specify the root directory containing the folders
directory = '.'

# Run the function
process_files(directory)
