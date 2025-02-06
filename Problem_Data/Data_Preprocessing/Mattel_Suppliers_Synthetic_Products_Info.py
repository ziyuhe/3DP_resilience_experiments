# NOTE: This script processes Mattel's Bill-of-Lading (BOL) data, 
# which was retrieved from a third-party service and is **not included** 
# in this repository due to licensing restrictions. 
# To use this script, you must provide your own dataset with a similar structure.

# This script is the first step of creating synthesized product using keyword matching. Specifically:
#   - Keywords are matched across different categories in the BOL raw descriptions (take "categories_keywords" below as an example keyword list).
#   - The matching records are used to estimate the proportion mix of product categories within a supplier's BOL data.
# The output results are further processed by script "Mattel_Suppliers_Synthetic_Products_Info_Costs.py"




import os
import pandas as pd

# Define the categories and keywords
categories_keywords = {
    "Dolls": ["Barbie", "Dolls", "Doll", "American Girl", "Americangirl", "Frozen", "Princesses", "Monster", "Monster High", "Little People", "People", "Descendent"],
    "Cars & Trucks": ["Hot Wheels", "Hot", "Wheels", "Die-Cast", "Diecast", "Cast", "Truck", "Trucks", "Car", "Cars", "Vehicle", "Matchbox"],
    "Action Figure + Roleplay": ["Action", "Figure", "Disney", "Pixar", "WWE", "Minecraft", "Master of the Universe", "Master", "Universe"],
    "Dinosaur": ["Jurassic", "Dinosaur"],
    "Trains": ["Train", "Trains", "Thomas", "Friends"],
    "Track + Play Sets": ["Track", "Tracks", "Trackset", "Playset"],
    "Baby Floor Seats + Educational Toys": ["Fisher Price", "Fisher", "Price", "Educational", "Baby", "Toddler", "Seat", "Seats", "Floor", "Floor Seat"]
}

def search_keywords_in_file(csv_path, categories_keywords):
    df = pd.read_csv(csv_path)
    found_keywords = set()
    found_categories = set()
    category_keyword_counts = {category: 0 for category in categories_keywords.keys()}

    for category, keywords in categories_keywords.items():
        print(f"Checking category: {category}")
        for keyword in keywords:
            print(f"Searching for keyword: {keyword}")
            keyword_lower = keyword.lower()
            count = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any() + row.astype(str).str.contains(keyword_lower, case=False).any(), axis=1).sum()
            if count > 0:
                found_keywords.add(keyword)
                found_categories.add(category)
                category_keyword_counts[category] += count

    return list(found_keywords), list(found_categories), category_keyword_counts

def process_folders_and_generate_summary(directory, categories_keywords):
    summary_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("Mattel_Only.csv"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                found_keywords, found_categories, category_keyword_counts = search_keywords_in_file(file_path, categories_keywords)

                if found_keywords:
                    company_name = os.path.basename(root)
                    summary_row = [company_name, ', '.join(found_keywords), ', '.join(found_categories)]
                    summary_row.extend(category_keyword_counts[category] for category in categories_keywords.keys())
                    summary_data.append(summary_row)

    # Create summary DataFrame
    columns = ["Company Name", "Keywords", "Categories"] + list(categories_keywords.keys())
    summary_df = pd.DataFrame(summary_data, columns=columns)
    
    # Save summary DataFrame to CSV
    summary_csv_path = os.path.join(directory, "Mattel_Suppliers_Synthetic_Products_Info.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

# Specify the root directory containing the folders
directory = '.'

# Run the function
process_folders_and_generate_summary(directory, categories_keywords)

