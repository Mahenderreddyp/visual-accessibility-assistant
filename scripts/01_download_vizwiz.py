import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Features, Value, Image, Sequence, Split

# Configuration
DATA_DIR = Path("data/raw")
URLS = {
    "train_images": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
    "val_images": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "test_images": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip",
    "annotations": "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
}

def download_file(url, dest_path):
    """Downloads a file with a progress bar."""
    if dest_path.exists():
        print(f"File {dest_path.name} already exists. Skipping.")
        return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extracts a zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_split(split_name, annotation_file, image_dir):
    """Parses the JSON annotations and matches them with images."""
    print(f"Parsing {split_name} annotations...")
    
    # --- FIX START: Added encoding='utf-8' ---
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # --- FIX END ---
    
    # VizWiz JSON structure is a list of objects
    examples = []
    for entry in data:
        img_filename = entry['image']
        img_path = str(image_dir / split_name / img_filename)
        
        # Test set usually doesn't have answers in public datasets
        if split_name == "test":
            answers = []
            answerable = -1
        else:
            answers = [a['answer'] for a in entry['answers']]
            answerable = entry['answerable']

        examples.append({
            "image": img_path, 
            "filename": img_filename,
            "question": entry['question'],
            "answers": answers,
            "answerable": answerable
        })
    return examples

def main():
    # 1. Setup Directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "images").mkdir(exist_ok=True)
    (DATA_DIR / "annotations").mkdir(exist_ok=True)

    # 2. Download Files
    for key, url in URLS.items():
        filename = url.split('/')[-1]
        dest = DATA_DIR / filename
        download_file(url, dest)
        
        # Extract
        if "images" in key:
            extract_target = DATA_DIR / "images"
        else:
            extract_target = DATA_DIR / "annotations"
            
        # Only extract if folder doesn't strictly exist (simple check)
        # For robustness, we just extract. ZipFile handles overwrites fine.
        extract_zip(dest, extract_target)

    # 3. Build Datasets from extracted JSONs
    # Note: The extracted annotation folder structure is usually 'train.json', 'val.json' etc.
    splits_to_process = [
        ("train", DATA_DIR / "annotations" / "train.json", DATA_DIR / "images"),
        ("val", DATA_DIR / "annotations" / "val.json", DATA_DIR / "images"),
        # ("test", DATA_DIR / "annotations" / "test.json", DATA_DIR / "images") # Skip test for now if JSON missing or different format
    ]

    for split_name, json_path, img_root in splits_to_process:
        if not json_path.exists():
            print(f"Warning: {json_path} not found. Skipping {split_name}.")
            continue

        examples = load_split(split_name, json_path, img_root)
        
        # Define Features explicitly
        features = Features({
            "image": Image(),
            "filename": Value("string"),
            "question": Value("string"),
            "answers": Sequence(Value("string")),
            "answerable": Value("int32") # 0 or 1
        })

        # Create HF Dataset
        print(f"Creating Hugging Face Dataset for {split_name}...")
        hf_dataset = Dataset.from_list(examples, features=features)
        
        # Save to disk
        save_path = f"data/raw/vizwiz_{split_name}"
        hf_dataset.save_to_disk(save_path)
        print(f"Saved {split_name} dataset to {save_path}")

if __name__ == "__main__":
    main()