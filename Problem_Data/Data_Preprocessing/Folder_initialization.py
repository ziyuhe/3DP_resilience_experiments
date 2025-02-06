import os
import shutil

def capitalize_words(name):
    words = name.split('-')
    capitalized_words = [word.capitalize() for word in words]
    return ' '.join(capitalized_words)

def process_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith("-bols.csv"):
            base_name = filename.rsplit("-bols.csv", 1)[0]
            folder_name = capitalize_words(base_name)
            
            folder_path = os.path.join(directory, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            src_path = os.path.join(directory, filename)
            dst_path = os.path.join(folder_path, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {folder_path}")

# Specify the directory containing your CSV files
directory = '.'

# Run the function
process_files(directory)

