"""
Example script for classifying astronomical images using the trained model.
"""

import argparse
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from pathlib import Path
from utils.classification import AstroClassifierPredictor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Classify astronomical images')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty estimation')
    parser.add_argument('--save_visualization', type=str, default=None, help='Path to save the visualization')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Confidence threshold for detections')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AstroClassifierPredictor(model_path=args.model_path)
    
    # Make prediction
    results = predictor.predict(args.image_path, return_uncertainty=not args.no_uncertainty)
    
    # Print results
    print("\nClassification Results:")
    print("=" * 50)
    
    # Filter and sort results by confidence
    results = [r for r in results if r['confidence'] > args.confidence_threshold]
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    if not results:
        print("No objects detected with confidence above threshold.")
        return
    
    # Print top detections
    for i, result in enumerate(results, 1):
        print(f"\nDetection {i}:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result['uncertainty'] is not None:
            print(f"Uncertainty: {result['uncertainty']:.4f}")
    
    # Generate and save visualization
    if args.save_visualization:
        visualization, cam_info = predictor.visualize_cam(args.image_path)
        
        # Print CAM information
        print("\nLocalization Information:")
        print("=" * 50)
        
        for i, info in enumerate(cam_info, 1):
            print(f"\nObject {i}:")
            print(f"Class: {info['class']}")
            print(f"Classification Confidence: {info['classification_confidence']:.2%}")
            print(f"Localization Confidence: {info['localization_confidence']:.2%}")
            print(f"Combined Confidence: {info['combined_confidence']:.2%}")
            
            if info['bbox'] is not None:
                print("Bounding Box:")
                print(f"Top-left: ({info['bbox']['x1']}, {info['bbox']['y1']})")
                print(f"Bottom-right: ({info['bbox']['x2']}, {info['bbox']['y2']})")
        
        # Convert BGR to RGB for matplotlib
        visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        
        # Create figure with proper size based on image dimensions
        height, width = visualization_rgb.shape[:2]
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi
        
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        plt.imshow(visualization_rgb)
        plt.axis('off')
        
        # Save with high quality settings
        plt.savefig(args.save_visualization, 
                   bbox_inches='tight', 
                   pad_inches=0, 
                   dpi=dpi,
                   format='png',
                   transparent=False)
        plt.close()
        print(f"\nVisualization saved to: {args.save_visualization}")

if __name__ == '__main__':
    main() 