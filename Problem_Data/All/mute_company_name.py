import os
import pandas as pd

def anonymize_company_names():
    # Get all CSV files in the current directory
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    # Dictionary to map company names to unique numbers
    company_mapping = {}
    company_counter = 1  # Start numbering from 1

    for file in csv_files:
        df = pd.read_csv(file)  # Read CSV
        
        # Check if "Company" column exists
        if "Company" in df.columns:
            df["Company"] = df["Company"].astype(str)  # Ensure it's a string
            
            unique_companies = df["Company"].unique()
            
            # Assign unique numbers to company names
            for company in unique_companies:
                if company not in company_mapping:
                    company_mapping[company] = company_counter
                    company_counter += 1
            
            # Replace company names with numbers
            df["Company"] = df["Company"].map(company_mapping)

            # Save the modified CSV
            df.to_csv(file, index=False)
            print(f"Processed: {file}")


if __name__ == "__main__":
    anonymize_company_names()
