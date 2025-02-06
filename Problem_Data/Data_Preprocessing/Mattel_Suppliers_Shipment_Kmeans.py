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
# For each supplier, we perform K-means analysis on the monthly sourcing quantities of the synthesized product



import os
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def kmeans_analysis(directory, k_values):
    # Read the per unit weight information
    synthetic_product_info_path = os.path.join(directory, 'Mattel_Suppliers_Synthetic_Products_Info.csv')
    synthetic_product_info = pd.read_csv(synthetic_product_info_path)

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            for file in os.listdir(folder_path):
                if file.endswith("(aggregated by month).csv"):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    
                    # Find the per unit weight for the current company
                    company_info = synthetic_product_info[synthetic_product_info['Company Name'] == dir_name]
                    if company_info.empty:
                        print(f"Company {dir_name} not found in synthetic product info. Skipping...")
                        continue
                    
                    per_unit_weight = company_info['Synthetic Product Weight (gram)'].values[0]

                    # Create the output directory
                    output_dir = os.path.join(folder_path, "K-Means Analysis on Shipments(Demand)")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    for k in k_values:
                        # K-means analysis on "Weight (kg, 0.8 Markdown)"
                        kmeans_weight = KMeans(n_clusters=k, random_state=0).fit(df[['Weight (kg, 0.8 Markdown)']])
                        weight_centers = np.round(kmeans_weight.cluster_centers_.flatten(), 3) * 1000
                        weight_labels = kmeans_weight.labels_
                        weight_probabilities = np.round(np.bincount(weight_labels) / len(weight_labels), 3)
                        
                        weight_results = pd.DataFrame({
                            'Cluster Center (in grams)': weight_centers,
                            'Probability': weight_probabilities
                        })
                        weight_results.to_csv(os.path.join(output_dir, f"{dir_name}_Weight_{k}_clusters.csv"), index=False)
                        
                        # Calculate quantity centers based on weight centers and per unit weight
                        quantity_centers = np.round(weight_centers / per_unit_weight, 3)
                        
                        quantity_results = pd.DataFrame({
                            'Cluster Center': quantity_centers,
                            'Probability': weight_probabilities
                        })
                        quantity_results.to_csv(os.path.join(output_dir, f"{dir_name}_Quantity_{k}_clusters.csv"), index=False)

# Specify the root directory containing the folders
directory = '.'

# Define the K values to try
k_values = [2, 3, 4, 5, 10]

# Run the function
kmeans_analysis(directory, k_values)
