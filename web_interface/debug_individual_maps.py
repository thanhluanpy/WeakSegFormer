#!/usr/bin/env python3
"""
Debug script to check individual class map colors
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def debug_individual_maps():
    """Debug individual class map color assignment"""
    
    print("🔍 Debugging Individual Class Map Colors")
    print("=" * 50)
    
    # Define expected colors
    expected_colors = {
        'Background': '#000000',  # Black
        'Necrotic': '#8B0000',   # Dark Red
        'Edema': '#228B22',      # Forest Green
        'Tumor': '#4169E1'       # Royal Blue
    }
    
    print("Expected colors:")
    for class_name, color in expected_colors.items():
        print(f"  {class_name}: {color}")
    
    # Simulate the individual class map creation process
    print("\n🔍 Simulating Individual Class Map Creation:")
    
    # Create a dummy segmentation result
    pred_class = np.array([
        [0, 0, 0, 0, 0],  # Background
        [0, 1, 1, 1, 0],  # Necrotic
        [0, 2, 2, 2, 0],  # Edema  
        [0, 3, 3, 3, 0],  # Tumor
        [0, 0, 0, 0, 0]   # Background
    ])
    
    class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
    present_classes = [0, 1, 2, 3]  # All classes present
    actual_class_names = [class_names[i] for i in present_classes]
    actual_class_colors = [expected_colors[name] for name in actual_class_names]
    
    print(f"Present classes: {present_classes}")
    print(f"Actual class names: {actual_class_names}")
    print(f"Actual class colors: {actual_class_colors}")
    
    # Create individual class maps
    individual_class_maps = {}
    
    for i, class_name in enumerate(actual_class_names):
        original_class_idx = present_classes[i]
        print(f"\n--- Processing {class_name} (index {original_class_idx}) ---")
        
        # Create binary mask for this class
        class_mask = (pred_class == original_class_idx).astype(np.uint8)
        print(f"Class mask shape: {class_mask.shape}")
        print(f"Class mask values: {np.unique(class_mask)}")
        print(f"Class mask sum: {np.sum(class_mask)}")
        
        if np.any(class_mask):
            # Create enhanced visualization for this class
            enhanced_class_map = np.zeros((5, 5, 3), dtype=np.uint8)
            print(f"Initial map shape: {enhanced_class_map.shape}")
            print(f"Initial map (all zeros): {np.all(enhanced_class_map == 0)}")
            
            # Get class color
            class_color = actual_class_colors[i]
            print(f"Class color: {class_color}")
            
            if class_color.startswith('#'):
                r = int(class_color[1:3], 16)
                g = int(class_color[3:5], 16)
                b = int(class_color[5:7], 16)
            else:
                r, g, b = 128, 128, 128
            
            print(f"RGB values: ({r}, {g}, {b})")
            
            # Fill the mask area with class color
            enhanced_class_map[class_mask > 0] = [r, g, b]
            print(f"After filling mask:")
            print(f"  Map shape: {enhanced_class_map.shape}")
            print(f"  Unique colors: {np.unique(enhanced_class_map.reshape(-1, 3), axis=0)}")
            print(f"  Non-zero pixels: {np.sum(np.any(enhanced_class_map > 0, axis=2))}")
            
            # Show the actual map
            print(f"Map content:")
            for row in range(enhanced_class_map.shape[0]):
                row_colors = []
                for col in range(enhanced_class_map.shape[1]):
                    pixel = enhanced_class_map[row, col]
                    if np.all(pixel == 0):
                        row_colors.append("BLACK")
                    elif np.all(pixel == [r, g, b]):
                        row_colors.append(f"CLASS({r},{g},{b})")
                    else:
                        row_colors.append(f"OTHER{pixel}")
                print(f"  Row {row}: {row_colors}")
            
            individual_class_maps[class_name] = enhanced_class_map
        else:
            print(f"No pixels found for {class_name}")
            individual_class_maps[class_name] = np.zeros((5, 5, 3), dtype=np.uint8)
    
    print(f"\n🔍 Final Individual Class Maps:")
    for class_name, class_map in individual_class_maps.items():
        print(f"\n{class_name}:")
        print(f"  Shape: {class_map.shape}")
        print(f"  Unique colors: {np.unique(class_map.reshape(-1, 3), axis=0)}")
        print(f"  Non-zero pixels: {np.sum(np.any(class_map > 0, axis=2))}")
        
        # Check if colors match expected
        expected_color = expected_colors[class_name]
        if expected_color.startswith('#'):
            expected_r = int(expected_color[1:3], 16)
            expected_g = int(expected_color[3:5], 16)
            expected_b = int(expected_color[5:7], 16)
        else:
            expected_r, expected_g, expected_b = 128, 128, 128
        
        # Check if any pixel has the expected color
        has_expected_color = np.any(np.all(class_map == [expected_r, expected_g, expected_b], axis=2))
        print(f"  Has expected color ({expected_r},{expected_g},{expected_b}): {has_expected_color}")
        
        # Check what colors are actually present
        unique_colors = np.unique(class_map.reshape(-1, 3), axis=0)
        print(f"  Actual colors present: {unique_colors.tolist()}")

if __name__ == "__main__":
    debug_individual_maps()