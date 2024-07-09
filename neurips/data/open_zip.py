import zipfile
import os

# Define the path to your zip file and the directory to extract to
zip_file_path = '/root/Kaggle_NeurIPS2024/data/raw/train.parquet.zip'
extract_to_path = '/root/Kaggle_NeurIPS2024/data/raw/'

# Create the extraction directory if it does not exist
os.makedirs(extract_to_path, exist_ok=True)

# Open the zip file and extract all contents
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f'Extracted all files to {extract_to_path}')
