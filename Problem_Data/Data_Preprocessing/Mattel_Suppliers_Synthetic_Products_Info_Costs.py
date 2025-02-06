# NOTE: This script processes Mattel's Bill-of-Lading (BOL) data, 
# which was retrieved from a third-party service and is **not included** 
# in this repository due to licensing restrictions. 
# To use this script, you must provide your own dataset with a similar structure.

# This script provides some further post-processing on the results obtained from "Mattel_Suppliers_Synthetic_Products_Info.py"
# Based on proportion mix of product categories within a supplier's BOL data.
#   - Sales price and cost parameters for the synthesized product are determined using a weighted average based on this proportion mix.

import pandas as pd

# Load the synthetic products info data
synthetic_products_info_file = "Mattel_Suppliers_Synthetic_Products_Info.csv"
synthetic_df = pd.read_csv(synthetic_products_info_file)

# Define the categories
categories = ["Dolls", "Cars & Trucks", "Action Figure + Roleplay", "Dinosaur", "Trains", "Track + Play Sets", "Baby Floor Seats + Educational Toys"]

# Normalize the counts to weights and calculate weighted sums
synthetic_df['Synthetic Product Weight (gram)'] = 0.0
synthetic_df['Synthetic Sourcing Cost'] = 0.0
synthetic_df['Synthetic Price'] = 0.0

# Load the sales markup data
sales_markup_file = "Mattel_Products_Sales_Markup.csv"
sales_markup_df = pd.read_csv(sales_markup_file)
sales_markup_df.set_index('File', inplace=True)

# Define the mapping of category names to the file names in the sales markup file
category_to_file = {
    "Dolls": "Mattel Products - Dolls.csv",
    "Cars & Trucks": "Mattel Products - Cars & Trucks.csv",
    "Action Figure + Roleplay": "Mattel Products - Action Figure + Roleplay.csv",
    "Dinosaur": "Mattel Products - Dinosaur.csv",
    "Trains": "Mattel Products - Trains.csv",
    "Track + Play Sets": "Mattel Products - Track + Play Sets.csv",
    "Baby Floor Seats + Educational Toys": "Mattel Products - Baby Floor Seats + Educational Toys.csv"
}

# Ensure all category_to_file keys exist in sales_markup_df index
for category, file_name in category_to_file.items():
    if file_name not in sales_markup_df.index:
        raise ValueError(f"File '{file_name}' not found in sales markup data")

# Calculate weighted sums for each supplier
for i, row in synthetic_df.iterrows():
    counts = row[categories]
    total_count = counts.sum()
    
    if total_count > 0:
        normalized_counts = counts / total_count
    else:
        normalized_counts = counts  # If the total count is zero, keep the counts as they are (they should all be zero)
    
    synthetic_weight = 0.0
    synthetic_sourcing_cost = 0.0
    synthetic_price = 0.0
    
    for category in categories:
        file_name = category_to_file[category]
        synthetic_weight += normalized_counts[category] * sales_markup_df.at[file_name, 'Ave. Weight']
        synthetic_sourcing_cost += normalized_counts[category] * sales_markup_df.at[file_name, 'Ave. Sourcing Cost']
        synthetic_price += normalized_counts[category] * sales_markup_df.at[file_name, 'Ave. Price']
    
    synthetic_df.at[i, 'Synthetic Product Weight (gram)'] = round(synthetic_weight, 2)
    synthetic_df.at[i, 'Synthetic Sourcing Cost'] = round(synthetic_sourcing_cost, 2)
    synthetic_df.at[i, 'Synthetic Price'] = round(synthetic_price, 2)

# Add columns for Synthetic 3DP Cost and Synthetic Expedition Cost
synthetic_df['Synthetic 3DP Cost'] = (synthetic_df['Synthetic Sourcing Cost'] * 2).round(2)
synthetic_df['Synthetic Expedition Cost'] = (synthetic_df['Synthetic Sourcing Cost'] * 1.5).round(2)

# Add columns for cost ratios
synthetic_df['Price/Sourcing Cost'] = (synthetic_df['Synthetic Price'] / synthetic_df['Synthetic Sourcing Cost']).round(2)
synthetic_df['Price/3DP Cost'] = (synthetic_df['Synthetic Price'] / synthetic_df['Synthetic 3DP Cost']).round(2)
synthetic_df['Price/Expedition Cost'] = (synthetic_df['Synthetic Price'] / synthetic_df['Synthetic Expedition Cost']).round(2)

# Save the updated synthetic products info data
output_file = "Mattel_Suppliers_Synthetic_Products_Info.csv"
synthetic_df.to_csv(output_file, index=False)

print(f"Updated synthetic products info saved to {output_file}")
