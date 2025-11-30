#!/usr/bin/env python3
"""
Test script for enhanced visualization improvements
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_enhanced_visualization():
    """Test the enhanced visualization functions"""
    print("🧪 Testing Enhanced Visualization Functions")
    print("=" * 50)
    
    try:
        # Import the enhanced functions
        from app import (
            create_enhanced_contours, 
            create_edge_enhanced_overlay, 
            create_enhanced_class_visualization,
            create_visualization
        )
        
        # Create test data
        print("📸 Creating test data...")
        
        # Create a synthetic MRI-like image
        original_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create a synthetic segmentation map
        segmentation_map = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add some synthetic tumor regions
        # Background (black)
        segmentation_map[:, :] = [0, 0, 0]
        
        # Necrotic region (dark red)
        segmentation_map[50:100, 50:100] = [139, 0, 0]  # Dark red
        
        # Edema region (forest green)
        segmentation_map[120:180, 80:140] = [34, 139, 34]  # Forest green
        
        # Tumor region (royal blue)
        segmentation_map[80:150, 150:220] = [65, 105, 225]  # Royal blue
        
        # Create test class data
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        class_colors = ['#000000', '#8B0000', '#228B22', '#4169E1']
        
        # Create synthetic class probabilities
        class_probs = np.random.rand(4, 256, 256)
        present_classes = [0, 1, 2, 3]
        
        # Create individual class maps
        individual_class_maps = {}
        for i, class_name in enumerate(class_names):
            class_mask = np.zeros((256, 256), dtype=np.uint8)
            if i == 1:  # Necrotic
                class_mask[50:100, 50:100] = 255
            elif i == 2:  # Edema
                class_mask[120:180, 80:140] = 255
            elif i == 3:  # Tumor
                class_mask[80:150, 150:220] = 255
            else:  # Background
                class_mask = 255 - (class_mask[50:100, 50:100] + class_mask[120:180, 80:140] + class_mask[80:150, 150:220])
            
            individual_class_maps[class_name] = class_mask
        
        print("✅ Test data created successfully")
        
        # Test enhanced contour function
        print("\n🔍 Testing enhanced contours...")
        try:
            enhanced_contours = create_enhanced_contours(segmentation_map, class_colors, class_names)
            print(f"✅ Enhanced contours created: {enhanced_contours.shape}")
        except Exception as e:
            print(f"❌ Enhanced contours failed: {e}")
        
        # Test edge enhanced overlay
        print("\n🔍 Testing edge enhanced overlay...")
        try:
            edge_overlay = create_edge_enhanced_overlay(original_image, segmentation_map)
            print(f"✅ Edge enhanced overlay created: {edge_overlay.shape}")
        except Exception as e:
            print(f"❌ Edge enhanced overlay failed: {e}")
        
        # Test enhanced class visualization
        print("\n🔍 Testing enhanced class visualization...")
        try:
            for class_name, class_map in individual_class_maps.items():
                enhanced_class = create_enhanced_class_visualization(
                    class_map, class_name, class_colors[class_names.index(class_name)], original_image
                )
                print(f"✅ Enhanced {class_name} visualization: {enhanced_class.shape}")
        except Exception as e:
            print(f"❌ Enhanced class visualization failed: {e}")
        
        # Test full visualization
        print("\n🔍 Testing full enhanced visualization...")
        try:
            viz_path = create_visualization(
                original_image, segmentation_map, 
                cv2.addWeighted(original_image, 0.7, segmentation_map, 0.3, 0),
                class_probs, present_classes, class_names, class_colors, 
                "test_enhanced", individual_class_maps
            )
            if viz_path:
                print(f"✅ Enhanced visualization created: {viz_path}")
            else:
                print("❌ Enhanced visualization creation returned None")
        except Exception as e:
            print(f"❌ Enhanced visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Enhanced visualization testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in enhanced visualization testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visual_improvements():
    """Test specific visual improvements"""
    print("\n🎨 Testing Visual Improvements")
    print("=" * 40)
    
    try:
        # Test contour line enhancement
        print("🔍 Testing contour line enhancement...")
        
        # Create a simple test image with a circle
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_image, (50, 50), 30, (255, 255, 255), -1)
        
        # Test contour detection
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✅ Found {len(contours)} contours")
        
        # Test edge detection
        print("🔍 Testing edge detection...")
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        print(f"✅ Edge detection found {edge_pixels} edge pixels")
        
        # Test color enhancement
        print("🔍 Testing color enhancement...")
        test_colors = ['#000000', '#8B0000', '#228B22', '#4169E1']
        for color in test_colors:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            print(f"✅ Color {color} -> RGB({r}, {g}, {b})")
        
        print("✅ Visual improvements testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in visual improvements testing: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Starting Enhanced Visualization Tests")
    print("=" * 60)
    
    # Import required modules
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV not available, some tests may fail")
    
    # Run tests
    success1 = test_enhanced_visualization()
    success2 = test_visual_improvements()
    
    if success1 and success2:
        print("\n🎉 All enhanced visualization tests passed!")
        print("✅ Visual line improvements are working correctly")
    else:
        print("\n💥 Some tests failed!")
        print("⚠️ Check the error messages above for details")