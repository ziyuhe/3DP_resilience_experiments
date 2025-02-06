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
# For each supplier, we draw the histogram of its sourcing quantities.

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(directory):
    # Walk through each folder under the current directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            for file in os.listdir(folder_path):
                if file.endswith("(aggregated by month).csv"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    
                    # Plot histogram for "Weight (kg, 0.8 Markdown)"
                    plt.figure()
                    df['Weight (kg, 0.8 Markdown)'].hist(bins=20)
                    plt.title(f'{dir_name} Monthly Weight Histogram')
                    plt.xlabel('Weight (kg, 0.8 Markdown)')
                    plt.ylabel('Frequency')
                    weight_histogram_path = os.path.join(folder_path, f'{dir_name}_Monthly_Weight_Histogram.png')
                    plt.savefig(weight_histogram_path)
                    plt.close()

                    # Plot histogram for "Quantity of Synthetic Product"
                    plt.figure()
                    df['Quantity of Synthetic Product'].hist(bins=20)
                    plt.title(f'{dir_name} Monthly Quantity Histogram')
                    plt.xlabel('Quantity of Synthetic Product')
                    plt.ylabel('Frequency')
                    quantity_histogram_path = os.path.join(folder_path, f'{dir_name}_Monthly_Quantity_Histogram.png')
                    plt.savefig(quantity_histogram_path)
                    plt.close()

# Specify the root directory containing the folders
directory = '.'

# Run the function
plot_histograms(directory)

