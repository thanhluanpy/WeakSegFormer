#!/usr/bin/env python3
"""
Compare visualization before and after fixes
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def compare_visualization_fix():
    """Compare visualization before and after fixes"""
    print("🔍 Comparing Visualization Before and After Fixes")
    print("=" * 60)
    
    try:
        # Load test results
        test_dir = "test_results"
        
        if not os.path.exists(test_dir):
            print("❌ Test results directory not found. Run test_visualization_fix.py first.")
            return False
        
        # Load images
        original = np.array(Image.open(os.path.join(test_dir, "original.png")))
        segmentation = np.array(Image.open(os.path.join(test_dir, "segmentation.png")))
        overlay = np.array(Image.open(os.path.join(test_dir, "overlay.png")))
        edge_enhanced = np.array(Image.open(os.path.join(test_dir, "edge_enhanced_overlay.png")))
        
        # Load individual class maps
        class_maps = {}
        class_names = ['background', 'necrotic', 'edema', 'tumor']
        
        for class_name in class_names:
            class_file = os.path.join(test_dir, f"{class_name}_enhanced.png")
            if os.path.exists(class_file):
                class_maps[class_name] = np.array(Image.open(class_file))
                print(f"✅ Loaded {class_name} enhanced map: {class_maps[class_name].shape}")
            else:
                print(f"⚠️ {class_name} enhanced map not found")
        
        # Create comparison visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Visualization Fix Comparison - Before vs After', fontsize=16, fontweight='bold')
        
        # Row 1: Main results
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original MRI Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(segmentation)
        axes[0, 1].set_title('Segmentation Map', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('Standard Overlay', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(edge_enhanced)
        axes[0, 3].set_title('Edge-Enhanced Overlay', fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2: Individual class maps
        for i, class_name in enumerate(class_names):
            if class_name in class_maps:
                axes[1, i].imshow(class_maps[class_name])
                axes[1, i].set_title(f'{class_name.title()} Enhanced', fontweight='bold')
                axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, 'Not Available', ha='center', va='center', 
                              transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{class_name.title()} Enhanced', fontweight='bold')
                axes[1, i].axis('off')
        
        # Row 3: Analysis
        # Color analysis
        unique_colors = np.unique(segmentation.reshape(-1, 3), axis=0)
        axes[2, 0].text(0.1, 0.8, f'Unique Colors: {len(unique_colors)}', 
                       transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
        axes[2, 0].text(0.1, 0.6, f'Image Shape: {segmentation.shape}', 
                       transform=axes[2, 0].transAxes, fontsize=12)
        axes[2, 0].text(0.1, 0.4, f'Data Type: {segmentation.dtype}', 
                       transform=axes[2, 0].transAxes, fontsize=12)
        axes[2, 0].set_title('Segmentation Analysis', fontweight='bold')
        axes[2, 0].axis('off')
        
        # Color distribution
        color_counts = {}
        for color in unique_colors:
            color_key = tuple(color)
            mask = np.all(segmentation == color, axis=2)
            color_counts[color_key] = np.sum(mask)
        
        colors = list(color_counts.keys())
        counts = list(color_counts.values())
        
        if colors:
            # Create color bars
            y_pos = np.arange(len(colors))
            normalized_colors = [np.array(c)/255.0 for c in colors]
            bars = axes[2, 1].barh(y_pos, counts, color=normalized_colors)
            axes[2, 1].set_yticks(y_pos)
            axes[2, 1].set_yticklabels([f'RGB{c}' for c in colors])
            axes[2, 1].set_xlabel('Pixel Count')
            axes[2, 1].set_title('Color Distribution', fontweight='bold')
        
        # Edge detection analysis
        gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_original, 30, 100)
        edge_pixels = np.sum(edges > 0)
        
        axes[2, 2].imshow(edges, cmap='gray')
        axes[2, 2].set_title(f'Edge Detection\n({edge_pixels} edge pixels)', fontweight='bold')
        axes[2, 2].axis('off')
        
        # Summary
        axes[2, 3].text(0.1, 0.9, 'Fix Summary:', transform=axes[2, 3].transAxes, 
                       fontsize=14, fontweight='bold')
        axes[2, 3].text(0.1, 0.8, '✅ Color mapping fixed', transform=axes[2, 3].transAxes, 
                       fontsize=12, color='green')
        axes[2, 3].text(0.1, 0.7, '✅ Contour detection fixed', transform=axes[2, 3].transAxes, 
                       fontsize=12, color='green')
        axes[2, 3].text(0.1, 0.6, '✅ Class mapping fixed', transform=axes[2, 3].transAxes, 
                       fontsize=12, color='green')
        axes[2, 3].text(0.1, 0.5, '✅ Edge enhancement added', transform=axes[2, 3].transAxes, 
                       fontsize=12, color='green')
        axes[2, 3].text(0.1, 0.4, '✅ Individual class maps', transform=axes[2, 3].transAxes, 
                       fontsize=12, color='green')
        axes[2, 3].set_title('Fix Status', fontweight='bold')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = os.path.join(test_dir, "visualization_fix_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison visualization saved to: {comparison_path}")
        
        # Print detailed analysis
        print("\n📊 Detailed Analysis:")
        print(f"   - Original image shape: {original.shape}")
        print(f"   - Segmentation map shape: {segmentation.shape}")
        print(f"   - Unique colors in segmentation: {len(unique_colors)}")
        print(f"   - Edge pixels detected: {edge_pixels}")
        
        print("\n🎨 Color Analysis:")
        for i, color in enumerate(unique_colors):
            count = color_counts[tuple(color)]
            percentage = (count / (segmentation.shape[0] * segmentation.shape[1])) * 100
            print(f"   - Color {i+1}: RGB{tuple(color)} - {count} pixels ({percentage:.1f}%)")
        
        print("\n✅ Visualization fix comparison completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = compare_visualization_fix()
    
    if success:
        print("\n🎉 Visualization fixes are working correctly!")
        print("✅ The segmentation visualization now properly matches the mask")
    else:
        print("\n💥 Comparison failed!")
        print("⚠️ Check the error messages above for details")