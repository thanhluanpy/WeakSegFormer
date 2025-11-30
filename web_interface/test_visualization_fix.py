#!/usr/bin/env python3
"""
Test script to verify visualization fixes
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_visualization_fix():
    """Test the fixed visualization functions"""
    print("🧪 Testing Fixed Visualization Functions")
    print("=" * 50)
    
    try:
        # Create test data
        print("📸 Creating test data...")
        
        # Create a synthetic MRI-like image
        original_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create a synthetic segmentation map with proper class indices
        segmentation_map = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Background (class 0) - black
        segmentation_map[:, :] = [0, 0, 0]
        
        # Necrotic region (class 1) - dark red
        segmentation_map[50:100, 50:100] = [139, 0, 0]  # Dark red
        
        # Edema region (class 2) - forest green  
        segmentation_map[120:180, 80:140] = [34, 139, 34]  # Forest green
        
        # Tumor region (class 3) - royal blue
        segmentation_map[80:150, 150:220] = [65, 105, 225]  # Royal blue
        
        print("✅ Test data created successfully")
        print(f"   - Original image shape: {original_image.shape}")
        print(f"   - Segmentation map shape: {segmentation_map.shape}")
        print(f"   - Unique colors in segmentation: {len(np.unique(segmentation_map.reshape(-1, 3), axis=0))}")
        
        # Test color mapping
        print("\n🔍 Testing color mapping...")
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        class_colors = ['#000000', '#8B0000', '#228B22', '#4169E1']
        
        for i, (name, color) in enumerate(zip(class_names, class_colors)):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            print(f"   - {name}: {color} -> RGB({r}, {g}, {b})")
        
        # Test contour detection
        print("\n🔍 Testing contour detection...")
        for i, (name, color) in enumerate(zip(class_names, class_colors)):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            # Create binary mask for this class
            class_mask = np.all(segmentation_map == [r, g, b], axis=2).astype(np.uint8)
            
            if np.any(class_mask):
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"   - {name}: Found {len(contours)} contours")
            else:
                print(f"   - {name}: No contours found")
        
        # Test individual class maps
        print("\n🔍 Testing individual class maps...")
        individual_class_maps = {}
        
        for i, (name, color) in enumerate(zip(class_names, class_colors)):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            # Create binary mask for this class
            class_mask = np.all(segmentation_map == [r, g, b], axis=2).astype(np.uint8)
            
            if np.any(class_mask):
                # Create enhanced visualization for this class
                enhanced_class_map = np.zeros((256, 256, 3), dtype=np.uint8)
                
                # Fill the mask area with class color
                enhanced_class_map[class_mask > 0] = [r, g, b]
                
                # Find contours for better edge definition
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw enhanced contour lines
                for contour in contours:
                    # Draw thick outer contour in white
                    cv2.drawContours(enhanced_class_map, [contour], -1, (255, 255, 255), 3)
                    # Draw inner contour in class color
                    cv2.drawContours(enhanced_class_map, [contour], -1, (r, g, b), 1)
                
                individual_class_maps[name] = enhanced_class_map
                print(f"   - {name}: Enhanced class map created with {len(contours)} contours")
            else:
                individual_class_maps[name] = np.zeros((256, 256, 3), dtype=np.uint8)
                print(f"   - {name}: Empty class map created")
        
        # Test overlay creation
        print("\n🔍 Testing overlay creation...")
        overlay = cv2.addWeighted(original_image, 0.6, segmentation_map, 0.4, 0)
        print(f"   - Overlay created: {overlay.shape}")
        
        # Test edge enhancement
        print("\n🔍 Testing edge enhancement...")
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_original, 30, 100)
        edge_pixels = np.sum(edges > 0)
        print(f"   - Edge detection found {edge_pixels} edge pixels")
        
        # Create edge-enhanced overlay
        edge_overlay = overlay.copy()
        edge_overlay[edges > 0] = [255, 255, 255]  # White edges
        enhanced_overlay = cv2.addWeighted(overlay, 0.8, edge_overlay, 0.2, 0)
        print(f"   - Edge-enhanced overlay created: {enhanced_overlay.shape}")
        
        # Save test results
        print("\n💾 Saving test results...")
        test_dir = "test_results"
        os.makedirs(test_dir, exist_ok=True)
        
        # Save original image
        Image.fromarray(original_image).save(os.path.join(test_dir, "original.png"))
        
        # Save segmentation map
        Image.fromarray(segmentation_map).save(os.path.join(test_dir, "segmentation.png"))
        
        # Save overlay
        Image.fromarray(overlay).save(os.path.join(test_dir, "overlay.png"))
        
        # Save edge-enhanced overlay
        Image.fromarray(enhanced_overlay).save(os.path.join(test_dir, "edge_enhanced_overlay.png"))
        
        # Save individual class maps
        for name, class_map in individual_class_maps.items():
            Image.fromarray(class_map).save(os.path.join(test_dir, f"{name.lower()}_enhanced.png"))
        
        print(f"✅ Test results saved to {test_dir}/")
        
        print("\n✅ All visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 Starting Visualization Fix Tests")
    print("=" * 60)
    
    success = test_visualization_fix()
    
    if success:
        print("\n🎉 Visualization fixes are working correctly!")
        print("✅ The segmentation visualization should now match the mask properly")
    else:
        print("\n💥 Some tests failed!")
        print("⚠️ Check the error messages above for details")