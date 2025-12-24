from datasets import load_from_disk, DatasetDict
import sys
import os
from tqdm import tqdm

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import format_vizwiz_entry

def filter_missing_images(example):
    """Checks if the image path actually exists."""
    image_path = example['image']
    # If it's a string, check it. If it's already a PIL Image (unlikely here), it's loaded.
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            return False
    return True

def process_split(split_name, input_path):
    print(f"\n--- Processing {split_name.upper()} ---")
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} does not exist. Skipping.")
        return None
        
    dataset = load_from_disk(input_path)
    initial_count = len(dataset)
    
    # 1. Filter Missing Images
    print(f"Verifying image paths for {initial_count} entries...")
    dataset = dataset.filter(filter_missing_images, desc="Checking files")
    final_count = len(dataset)
    
    if final_count < initial_count:
        print(f"⚠️ Removed {initial_count - final_count} entries pointing to missing files.")
    else:
        print("✅ All image files found!")

    # 2. Format for Chat (Majority Vote & Formatting)
    # num_proc=1 is CRITICAL for 16GB RAM.
    print(f"Formatting {split_name} (Majority Voting)...")
    processed = dataset.map(
        format_vizwiz_entry, 
        remove_columns=dataset.column_names,
        num_proc=8,              # <--- PROTECTS YOUR RAM
        writer_batch_size=200,   # <--- FLUSHES TO DISK OFTEN
        desc=f"Structuring {split_name}"
    )
    
    return processed

def main():
    # Define paths
    raw_train_path = "data/raw/vizwiz_train"
    raw_val_path = "data/raw/vizwiz_val"
    output_path = "data/processed/vizwiz_ready"
    
    # Process
    train_ds = process_split("train", raw_train_path)
    val_ds = process_split("val", raw_val_path)
    
    # Combine
    full_dataset = DatasetDict()
    if train_ds:
        full_dataset['train'] = train_ds
    if val_ds:
        full_dataset['test'] = val_ds 
        
    # Save
    print(f"\nSaving processed dataset to {output_path}...")
    full_dataset.save_to_disk(output_path)
    print("Success! Dataset is safe and ready for training.")

if __name__ == "__main__":
    main()