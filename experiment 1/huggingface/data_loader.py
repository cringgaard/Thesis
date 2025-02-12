import os
import pandas as pd

print("Removing images not in metadata...")

# get list of files in data
files = os.listdir('data')
# get all images
images = [f for f in files if f.endswith('.jpg')]

metadata = pd.read_csv('data/metadata.csv')

# remove images that are not in metadata
missing = []
for image in images:
    if image not in metadata['file_name'].values:
        missing.append(image)

for image in missing:
    os.remove('data/' + image)

print("Pushing to Hugging Face...")

from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="data", drop_labels=True)

dataset.push_to_hub("cringgaard/boatsV2")

print("Done!")