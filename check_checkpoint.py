import torch
import sys

def inspect_checkpoint(checkpoint_path):
    print(f"\nInspecting checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\nCheckpoint keys:", checkpoint.keys())
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"\nSub-keys in {key}:", checkpoint[key].keys())
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/astro-classifier-epoch=09-val_loss=0.32.ckpt"
    inspect_checkpoint(checkpoint_path) 