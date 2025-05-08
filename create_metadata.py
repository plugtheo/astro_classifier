import os
import json
import random
from pathlib import Path
from typing import Dict, List, Set
import shutil

def create_metadata_files(dataset_root: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Create metadata files for train/val/test splits based on class folders.
    
    Args:
        dataset_root: Path to the dataset root directory containing class folders
        output_dir: Directory to save metadata files
        train_ratio: Ratio of training data (default: 0.7)
        val_ratio: Ratio of validation data (default: 0.15)
        # test_ratio will be 1 - train_ratio - val_ratio
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all class folders
    class_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    
    # Initialize metadata dictionaries
    train_metadata = []
    val_metadata = []
    test_metadata = []
    
    # Track all files to ensure no overlap
    all_files: Set[str] = set()
    
    for class_name in class_folders:
        class_path = os.path.join(dataset_root, class_name)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle files for random split
        random.shuffle(files)
        
        # Calculate split indices
        n_files = len(files)
        train_idx = int(n_files * train_ratio)
        val_idx = int(n_files * (train_ratio + val_ratio))
        
        # Split files
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        # Add to metadata
        for file in train_files:
            train_metadata.append({
                'image_path': os.path.join(class_name, file),
                'class': class_name
            })
            all_files.add(os.path.join(class_name, file))
            
        for file in val_files:
            val_metadata.append({
                'image_path': os.path.join(class_name, file),
                'class': class_name
            })
            all_files.add(os.path.join(class_name, file))
            
        for file in test_files:
            test_metadata.append({
                'image_path': os.path.join(class_name, file),
                'class': class_name
            })
            all_files.add(os.path.join(class_name, file))
    
    # Save metadata files
    with open(os.path.join(output_dir, 'train_metadata.json'), 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(output_dir, 'val_metadata.json'), 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    with open(os.path.join(output_dir, 'test_metadata.json'), 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    return train_metadata, val_metadata, test_metadata

def verify_metadata(dataset_root: str, metadata_dir: str):
    """
    Verify that metadata files match the dataset and have no overlaps.
    
    Args:
        dataset_root: Path to the dataset root directory
        metadata_dir: Directory containing metadata files
    """
    # Load metadata files
    with open(os.path.join(metadata_dir, 'train_metadata.json'), 'r') as f:
        train_metadata = json.load(f)
    with open(os.path.join(metadata_dir, 'val_metadata.json'), 'r') as f:
        val_metadata = json.load(f)
    with open(os.path.join(metadata_dir, 'test_metadata.json'), 'r') as f:
        test_metadata = json.load(f)
    
    # Check for overlaps
    train_files = {item['image_path'] for item in train_metadata}
    val_files = {item['image_path'] for item in val_metadata}
    test_files = {item['image_path'] for item in test_metadata}
    
    # Check overlaps
    train_val_overlap = train_files.intersection(val_files)
    train_test_overlap = train_files.intersection(test_files)
    val_test_overlap = val_files.intersection(test_files)
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("WARNING: Found overlapping files between splits!")
        if train_val_overlap:
            print(f"Train-Val overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"Train-Test overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"Val-Test overlap: {val_test_overlap}")
    else:
        print("No overlaps found between splits.")
    
    # Verify all files exist
    all_files = train_files.union(val_files).union(test_files)
    missing_files = []
    
    for file_path in all_files:
        full_path = os.path.join(dataset_root, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("WARNING: Found missing files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("All files in metadata exist in the dataset.")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(all_files)}")
    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")
    
    # Print class distribution
    def get_class_distribution(metadata):
        class_dist = {}
        for item in metadata:
            class_name = item['class']
            class_dist[class_name] = class_dist.get(class_name, 0) + 1
        return class_dist
    
    print("\nClass Distribution:")
    print("Train set:")
    for class_name, count in get_class_distribution(train_metadata).items():
        print(f"  {class_name}: {count}")
    print("Validation set:")
    for class_name, count in get_class_distribution(val_metadata).items():
        print(f"  {class_name}: {count}")
    print("Test set:")
    for class_name, count in get_class_distribution(test_metadata).items():
        print(f"  {class_name}: {count}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    dataset_root = "data/raw/SpaceNet.FLARE.imam_alam"  # Updated dataset path
    metadata_dir = "data/raw/SpaceNet.FLARE.imam_alam/metadata"  # Updated metadata path
    
    # Create metadata files
    print("Creating metadata files...")
    train_meta, val_meta, test_meta = create_metadata_files(dataset_root, metadata_dir)
    
    # Verify metadata
    print("\nVerifying metadata...")
    verify_metadata(dataset_root, metadata_dir) 