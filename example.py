#!/usr/bin/env python3
"""
Simple example demonstrating the image stitching module.

This script stitches pairs of images from the 'image pairs' directory
and displays the results.
"""

import os
from image_stitching import ImageStitcher
import matplotlib.pyplot as plt


def main():
    """Main function to demonstrate image stitching."""
    # Initialize the stitcher
    stitcher = ImageStitcher()
    
    # Define image pairs
    image_dir = "image pairs"
    image_pairs = [
        ("image pairs_01_01.jpg", "image pairs_01_02.jpg"),
        ("image pairs_02_01.png", "image pairs_02_02.png"),
        ("image pairs_03_01.jpg", "image pairs_03_02.jpg"),
        ("image pairs_04_01.jpg", "image pairs_04_02.jpg"),
    ]
    
    # Process each pair
    for i, (img1_name, img2_name) in enumerate(image_pairs, 1):
        img1_path = os.path.join(image_dir, img1_name)
        img2_path = os.path.join(image_dir, img2_name)
        
        print(f"\nProcessing pair {i}: {img1_name} + {img2_name}")
        
        try:
            # Stitch images
            result = stitcher.stitch_images(img1_path, img2_path)
            print(f"  ✓ Success! Result shape: {result.shape}")
            
            # Display result
            plt.figure(figsize=(12, 6))
            plt.imshow(result)
            plt.title(f"Stitched Panorama - Pair {i}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    main()
