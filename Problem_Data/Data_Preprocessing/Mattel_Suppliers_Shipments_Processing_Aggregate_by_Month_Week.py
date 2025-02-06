# NOTE: This script processes Mattel's Bill-of-Lading (BOL) data, 
# which was retrieved from a third-party service and is **not included** 
# in this repository due to licensing restrictions. 
# To use this script, you must provide your own dataset with a similar structure.

# NOTE: This script builds on the results from:
#    - "Mattel_Suppliers_Synthetic_Products_Info.py"
#    - "Mattel_Suppliers_Synthetic_Products_Info_Costs.py"
#    - "Mattel_Suppliers_Shipments_Processing.py"
#
# For each supplier, we aggreate the bills to monthly and weekly sourcing qunatities (of the syntehsized products)


import os
import pandas as pd

def aggregate_by_period(file_path, period, date_format, output_suffix):
    # Read the file
    df = pd.read_csv(file_path)
    
    # Convert "Arrival Date" to datetime and extract year and period (month/week)
    df['Arrival Date'] = pd.to_datetime(df['Arrival Date'])
    df['Year'] = df['Arrival Date'].dt.year
    
    if period == 'month':
        df['Period'] = df['Arrival Date'].dt.month
    elif period == 'week':
        df['Period'] = df['Arrival Date'].dt.isocalendar().week
    
    # Aggregate by Year and Period, summing up the other columns (excluding datetime columns)
    agg_df = df.groupby(['Year', 'Period']).agg({
        'Weight (kg, 0.8 Markdown)': 'sum',
        'Quantity of Synthetic Product': 'sum'
    }).reset_index()
    
    # Calculate the number of shipments in each group
    shipments_count = df.groupby(['Year', 'Period']).size().reset_index(name='Shipments')
    
    # Merge the aggregated data with the shipments count
    agg_df = pd.merge(agg_df, shipments_count, on=['Year', 'Period'])
    
    # Sort by year and period in descending order
    agg_df = agg_df.sort_values(by=['Year', 'Period'], ascending=[False, False]).reset_index(drop=True)
    
    # Create "Arrival Month" or "Arrival Week" column
    agg_df['Arrival Date'] = agg_df.apply(lambda row: date_format.format(int(row['Period']), int(row['Year'])), axis=1)
    
    # Select the relevant columns and rename "Arrival Date"
    columns = ['Arrival Date', 'Shipments'] + [col for col in agg_df.columns if col not in ['Year', 'Period', 'Arrival Date', 'Shipments']]
    agg_df = agg_df[columns]
    
    # Round values to two decimal points
    agg_df = agg_df.round(2)
    
    # Generate output file name
    output_file = file_path.replace("processed.csv", output_suffix)
    agg_df.to_csv(output_file, index=False)
    print(f"Saved aggregated data to {output_file}")

def process_folders_for_aggregation(directory):
    # Walk through each folder under the current directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            for file in os.listdir(os.path.join(root, dir_name)):
                if file.endswith("_shipment_processed.csv"):
                    file_path = os.path.join(root, dir_name, file)
                    # Aggregate by month
                    aggregate_by_period(file_path, 'month', "Month {:02d} of {}", "_shipment_processed(aggregated by month).csv")
                    # Aggregate by week
                    aggregate_by_period(file_path, 'week', "Week {:02d} of {}", "_shipment_processed(aggregated by week).csv")

# Specify the root directory containing the folders
directory = '.'

# Run the function
process_folders_for_aggregation(directory)
