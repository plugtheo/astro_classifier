import json
from pathlib import Path

def check_dataset_splits():
    base_dir = Path('data/raw/SpaceNet.FLARE.imam_alam')
    splits = ['train', 'val', 'test']
    
    print("Dataset Split Statistics:")
    print("-" * 50)
    
    for split in splits:
        # Load metadata
        metadata_path = base_dir / 'metadata' / f"{split}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Count valid images
        valid_images = 0
        empty_files = 0
        class_counts = {}
        
        # Debug counters for galaxy
        total_galaxy = 0
        valid_galaxy = 0
        empty_galaxy = 0
        
        for sample in metadata:
            image_path = base_dir / sample['image_path']
            is_galaxy = sample['class'] == 'galaxy'
            
            if is_galaxy:
                total_galaxy += 1
                
            if image_path.exists():
                if image_path.stat().st_size > 0:
                    valid_images += 1
                    class_name = sample['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    if is_galaxy:
                        valid_galaxy += 1
                else:
                    empty_files += 1
                    if is_galaxy:
                        empty_galaxy += 1
        
        print(f"\n{split.upper()} Split:")
        print(f"Total images in metadata: {len(metadata)}")
        print(f"Valid images (non-empty): {valid_images}")
        print(f"Empty files: {empty_files}")
        
        print(f"\nGalaxy Debug Info:")
        print(f"Total galaxy in metadata: {total_galaxy}")
        print(f"Valid galaxy images: {valid_galaxy}")
        print(f"Empty galaxy files: {empty_galaxy}")
        
        print("\nClass Distribution:")
        for class_name in sorted(class_counts.keys()):
            print(f"  {class_name}: {class_counts[class_name]} images")

if __name__ == "__main__":
    check_dataset_splits() 