import os
from pathlib import Path
from datasets import load_from_disk
import humanize  # You might need to pip install humanize, or we can use a simple function

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def format_bytes(size):
    # Simple helper to avoid external dependencies
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"

def analyze_split(split_name, raw_hf_path, images_path):
    print(f"\n--- Analyzing {split_name.upper()} ---")
    
    # 1. Check Hugging Face Dataset (Metadata)
    if os.path.exists(raw_hf_path):
        try:
            ds = load_from_disk(raw_hf_path)
            count = len(ds)
            print(f"✅ HF Dataset loaded: {count:,} entries")
            print(f"   Example Q: {ds[0]['question']}")
            print(f"   Example A: {ds[0]['answers']}")
        except Exception as e:
            print(f"❌ Error loading HF dataset: {e}")
            return
    else:
        print(f"❌ HF Dataset path not found: {raw_hf_path}")
        return

    # 2. Check Physical Images
    if os.path.exists(images_path):
        # Fast count of files
        img_count = len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))])
        folder_size = get_dir_size(images_path)
        print(f"✅ Image Folder: {img_count:,} files")
        print(f"   Total Size: {format_bytes(folder_size)}")
        
        # 3. Integrity Check
        if count == img_count:
            print("✨ PERFECT MATCH: Dataset entries == Image files")
        else:
            diff = abs(count - img_count)
            print(f"⚠️ MISMATCH: Difference of {diff:,} files.")
            print("   (This is common if some images failed to unzip or hidden files exist)")
    else:
        print(f"❌ Image path not found: {images_path}")

def main():
    base_dir = Path("data/raw")
    
    # Analyze Train
    analyze_split(
        "Train", 
        base_dir / "vizwiz_train", 
        base_dir / "images/train"
    )
    
    # Analyze Val
    analyze_split(
        "Validation", 
        base_dir / "vizwiz_val", 
        base_dir / "images/val"
    )

if __name__ == "__main__":
    main()