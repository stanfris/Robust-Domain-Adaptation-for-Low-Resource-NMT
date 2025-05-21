import os
import gzip
from urllib.request import urlretrieve
from datasets import Dataset, DatasetDict

# URLs for OPUS Wikimedia monolingual files
en_ru_url = "https://object.pouta.csc.fi/OPUS-wikimedia/v20210402/moses/en-ru.txt.zip"

# Paths
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
en_ru_file = os.path.join(data_dir, "en-ru.txt.zip")

# # Download files if they don't exist
# if not os.path.exists(en_ru_file):
#     print(f"Downloading {en_ru_url}...")
#     urlretrieve(en_ru_url, en_ru_file)
#     print("Download complete.")

# # Unzip the downloaded file
# if en_ru_file.endswith(".zip"):
#     import zipfile
#     with zipfile.ZipFile(en_ru_file, 'r') as zip_ref:
#         zip_ref.extractall(data_dir)
#     print("Unzipping complete.")

# Read .gz files and return lines
en_path = os.path.join(data_dir, "wikimedia.en-ru.en")
ru_path = os.path.join(data_dir, "wikimedia.en-ru.ru")

# Read and align lines
print("Reading files...")
with open(en_path, 'r', encoding='utf-8') as f:
    en_lines = f.readlines()
with open(ru_path, 'r', encoding='utf-8') as f:
    ru_lines = f.readlines()
print("Files read complete.")
# Create a list of dicts with keys 'en' and 'ru'
data = [{"en": en, "ru": ru} for en, ru in zip(en_lines, ru_lines)]

# Create dataset and wrap in DatasetDict
dataset = Dataset.from_list(data)
dataset_dict = DatasetDict({"train": dataset})

# Save to disk
save_path = os.path.join(data_dir, "opus_wikimedia_en_ru")
dataset_dict.save_to_disk(save_path)

print(f"\n✅ DatasetDict saved to: {save_path}")
print("You can now load it with:\nfrom datasets import load_from_disk\nload_dataset = load_from_disk('./data/opus_wikimedia_en_ru')")
    