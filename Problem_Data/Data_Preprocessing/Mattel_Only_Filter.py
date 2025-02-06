# NOTE: This script processes Mattel's Bill-of-Lading (BOL) data, 
# which was retrieved from a third-party service and is **not included** 
# in this repository due to licensing restrictions. 
# To use this script, you must provide your own dataset with a similar structure.

# This script filters out the non-Mattel BOL records of a Mattle's supplier.

import os
import pandas as pd

def filter_mattel_rows(csv_path):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='warn')
        # Filter rows where "Company Name" column contains "Mattel"
        filtered_df = df[df['Company Name'].astype(str).str.contains('Mattel', case=False, na=False)]
        return filtered_df
    except pd.errors.ParserError as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def process_folders(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("bols.csv"):
                file_path = os.path.join(root, file)
                filtered_df = filter_mattel_rows(file_path)
                if filtered_df is not None:
                    new_file_name = file.replace(".csv", "_Mattel_Only.csv")
                    new_file_path = os.path.join(root, new_file_name)  # Ensures the new file is stored in the same folder
                    filtered_df.to_csv(new_file_path, index=False)
                    print(f"Saved filtered data to {new_file_path}")

# Specify the root directory containing the folders
directory = '.'

# Run the function
process_folders(directory)
