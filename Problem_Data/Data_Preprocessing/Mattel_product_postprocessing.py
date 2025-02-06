# NOTE: This script processes product information that is **not included** in this repository.
# It requires CSV files containing product details, where:
#   - Each CSV file corresponds to a specific product category (see "csv_files" below).
#   - For each category, the script calculates the average weight per unit and various per-unit costs.


import pandas as pd
import glob
import os

# Function to convert weight to grams
def convert_to_grams(weight_str):
    if pd.isna(weight_str):
        return None  # Handle NaNs appropriately
    weight_str = weight_str.replace('\u200e', '')  # Remove left-to-right mark
    parts = weight_str.split()
    if len(parts) != 2:
        raise ValueError(f"Unexpected weight format: {weight_str}")
    value, unit = parts
    value = float(value)
    if 'ounces' in unit:
        return round(value * 28.3495, 2)
    elif 'pounds' in unit:
        return round(value * 453.59237, 2)
    else:
        raise ValueError(f"Unexpected weight unit: {unit}")

# List of CSV files to process
csv_files = [
    "Mattel Products - Action Figure + Roleplay.csv",
    "Mattel Products - Baby Floor Seats + Educational Toys.csv",
    "Mattel Products - Cars & Trucks.csv",
    "Mattel Products - Dinosaur.csv",
    "Mattel Products - Dolls.csv",
    "Mattel Products - Track + Play Sets.csv",
    "Mattel Products - Train.csv"
]

# Initialize an empty DataFrame for the combined data
combined_data = pd.DataFrame()

# Initialize a list to store summary information for each file
summary_data = []

# Initialize a list to store all filtered weight items
all_filtered_weights = []

for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Convert the "Weight" column to grams
    df['Weight'] = df['Weight'].apply(convert_to_grams)
    
    # Calculate "Sourcing cost" and "3DP cost"
    df['Sourcing cost'] = round(df['Weight'] * 0.006, 2)
    df['3DP cost'] = round(df['Weight'] * 0.012, 2)
    
    # Extract and process the "Price" column
    df['Price'] = df['Price'].str.replace('$', '').astype(float)

    # Calculate "Selling markup"
    df['Selling markup'] = round(df['Price'] / df['Sourcing cost'], 2)
    
    # Filter out extreme weight values
    mean_weight = df['Weight'].mean()
    std_weight = df['Weight'].std()
    lower_bound = mean_weight - 2 * std_weight
    upper_bound = mean_weight + 2 * std_weight
    filtered_df = df[(df['Weight'] >= lower_bound) & (df['Weight'] <= upper_bound)]
    
    # Remember the filtered weights for overall calculation
    all_filtered_weights.extend(filtered_df['Weight'])
    
    # Calculate averages
    avg_weight = filtered_df['Weight'].mean()
    avg_sourcing_cost = filtered_df['Sourcing cost'].mean()
    avg_3dp_cost = filtered_df['3DP cost'].mean()
    avg_price = filtered_df['Price'].mean()
    avg_expedition_cost = avg_sourcing_cost * 1.5
    
    # Calculate Selling/Sourcing, Selling/3DP, and Selling/Expediting
    total_price = filtered_df['Price'].sum()
    total_sourcing_cost = filtered_df['Sourcing cost'].sum()
    selling_sourcing = total_price / total_sourcing_cost if total_sourcing_cost != 0 else 0
    selling_3dp = selling_sourcing / 2
    selling_expediting = selling_sourcing / 1.5
    
    # Append the summary data for the group
    summary_data.append({
        'File': file,
        'Ave. Weight': round(avg_weight, 2),
        'Ave. Sourcing Cost': round(avg_sourcing_cost, 2),
        'Ave. 3DP Cost': round(avg_3dp_cost, 2),
        'Ave. Expedition Cost': round(avg_expedition_cost, 2),
        'Ave. Price': round(avg_price, 2),
        'Total Price': round(total_price, 2),
        'Total Sourcing Cost': round(total_sourcing_cost, 2),
        'Selling/Sourcing': round(selling_sourcing, 2),
        'Selling/3DP': round(selling_3dp, 2),
        'Selling/Expediting': round(selling_expediting, 2)
    })
    
    # Select the required columns and save to a new CSV file
    processed_df = filtered_df[['Product Name', 'Weight', 'Sourcing cost', '3DP cost', 'Price', 'Selling markup', 'Size', 'Origin']]
    processed_filename = file.replace('.csv', '_processed.csv')
    processed_df.to_csv(processed_filename, index=False)
    
    # Append to the combined DataFrame
    combined_data = pd.concat([combined_data, processed_df], ignore_index=True)
    
    # Print the number of rows filtered
    print(f"Processed {file}: Filtered {len(df) - len(filtered_df)} rows out of {len(df)}")

# Save the combined data to a CSV file
combined_data.to_csv('Combined_Mattel_Products.csv', index=False)

# Calculate overall summary information
total_price_overall = combined_data['Price'].sum()
total_sourcing_cost_overall = combined_data['Sourcing cost'].sum()
total_3dp_cost_overall = combined_data['3DP cost'].sum()

overall_avg_weight = pd.Series(all_filtered_weights).mean()
overall_avg_sourcing_cost = combined_data['Sourcing cost'].mean()
overall_avg_3dp_cost = combined_data['3DP cost'].mean()
overall_avg_price = combined_data['Price'].mean()
overall_avg_expedition_cost = overall_avg_sourcing_cost * 1.5

overall_sourcing_selling = total_price_overall / total_sourcing_cost_overall
overall_three_dp_selling = total_price_overall / total_3dp_cost_overall
overall_expediting_selling = overall_sourcing_selling / 1.5

# Add overall summary information as the first row
summary_data.insert(0, {
    'File': 'Overall',
    'Ave. Weight': round(overall_avg_weight, 2),
    'Ave. Sourcing Cost': round(overall_avg_sourcing_cost, 2),
    'Ave. 3DP Cost': round(overall_avg_3dp_cost, 2),
    'Ave. Expedition Cost': round(overall_avg_expedition_cost, 2),
    'Ave. Price': round(overall_avg_price, 2),
    'Selling/Sourcing': round(overall_sourcing_selling, 2),
    'Selling/3DP': round(overall_three_dp_selling, 2),
    'Selling/Expediting': round(overall_expediting_selling, 2)
})

# Create a DataFrame for the summary data
summary_df = pd.DataFrame(summary_data)

# Save the summary data to a CSV file
summary_df.to_csv('Mattel_Products_Sales_Markup.csv', index=False)
