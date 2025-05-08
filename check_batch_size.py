import json
from pathlib import Path
import os

def check_dataset_size():
    # Load metadata
    metadata_path = Path('data/raw/SpaceNet.FLARE.imam_alam/metadata/train_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Count valid images
    valid_images = 0
    empty_files = 0
    base_dir = Path('data/raw/SpaceNet.FLARE.imam_alam')
    
    for sample in metadata:
        image_path = base_dir / sample['image_path']
        if image_path.exists():
            if image_path.stat().st_size > 0:
                valid_images += 1
            else:
                empty_files += 1
    
    # Calculate batch count
    batch_size = 6  # Actual batch size from config
    total_batches = valid_images // batch_size
    
    print(f"Dataset Statistics:")
    print(f"Total images in metadata: {len(metadata)}")
    print(f"Valid images (non-empty): {valid_images}")
    print(f"Empty files: {empty_files}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches (with drop_last=True): {total_batches}")
    print(f"Expected batches (1335): {1335}")
    print(f"Difference: {total_batches - 1335}")

if __name__ == "__main__":
    check_dataset_size() 