#!/usr/bin/env python3
"""
Compare Results: Before vs After Enhancement
===========================================

This script compares the results before and after the enhancement to ensure
we're getting better results, not worse ones.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_image():
    """Create a realistic test MRI image"""
    # Create a more realistic MRI-like image
    image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    # Add brain-like structure
    # Background (skull)
    image[20:30, :] = [50, 50, 50]
    image[-30:-20, :] = [50, 50, 50]
    image[:, 20:30] = [50, 50, 50]
    image[:, -30:-20] = [50, 50, 50]
    
    # Brain tissue
    image[80:180, 80:180] = [120, 120, 120]
    
    # CSF (darker)
    image[100:140, 100:140] = [80, 80, 80]
    
    # Potential tumor (brighter)
    image[130:160, 130:160] = [180, 180, 180]
    
    return image

def test_original_approach():
    """Test the original approach (without enhancement)"""
    print("🔄 Testing ORIGINAL approach (no enhancement)...")
    
    # Load model
    from web_interface.app import load_model_global
    if not load_model_global():
        print("❌ Failed to load model")
        return None
    
    from web_interface.app import model, device
    
    # Create test image
    test_image = create_test_image()
    image_tensor = torch.from_numpy(test_image).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    final_output, deep_sup1, deep_sup2, original_output = output[:4]
    
    # Test with just the final_output (original approach)
    from web_interface.app import postprocess_output
    
    result = postprocess_output(
        final_output,  # Only use final_output
        test_image,
        local_class_names=['Background', 'Necrotic', 'Edema', 'Tumor'],
        local_class_colors=['#000000', '#8B0000', '#228B22', '#4169E1']
        # No adaptive selection - just standard approach
    )
    
    # Unpack results
    (segmentation_map, colored_map, overlay, class_probs, present_classes, 
     actual_class_names, actual_class_colors, individual_class_maps,
     best_output_per_class, quality_scores_per_class) = result
    
    print(f"✅ Original approach completed")
    print(f"  Present classes: {present_classes}")
    print(f"  Individual maps: {list(individual_class_maps.keys())}")
    
    return {
        'approach': 'original',
        'segmentation_map': segmentation_map,
        'colored_map': colored_map,
        'present_classes': present_classes,
        'individual_class_maps': individual_class_maps
    }

def test_adaptive_approach():
    """Test the adaptive approach (with enhancement)"""
    print("\n🔄 Testing ADAPTIVE approach (with enhancement)...")
    
    # Load model
    from web_interface.app import load_model_global
    if not load_model_global():
        print("❌ Failed to load model")
        return None
    
    from web_interface.app import model, device
    
    # Create test image
    test_image = create_test_image()
    image_tensor = torch.from_numpy(test_image).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    final_output, deep_sup1, deep_sup2, original_output = output[:4]
    
    # Test with adaptive selection
    from web_interface.app import postprocess_output
    
    result = postprocess_output(
        final_output,
        test_image,
        local_class_names=['Background', 'Necrotic', 'Edema', 'Tumor'],
        local_class_colors=['#000000', '#8B0000', '#228B22', '#4169E1'],
        final_output=final_output,
        deep_sup1=deep_sup1,
        deep_sup2=deep_sup2,
        original_output_raw=original_output
    )
    
    # Unpack results
    (segmentation_map, colored_map, overlay, class_probs, present_classes, 
     actual_class_names, actual_class_colors, individual_class_maps,
     best_output_per_class, quality_scores_per_class) = result
    
    print(f"✅ Adaptive approach completed")
    print(f"  Present classes: {present_classes}")
    print(f"  Individual maps: {list(individual_class_maps.keys())}")
    
    if best_output_per_class:
        print(f"  Best output selection:")
        for class_name, info in best_output_per_class.items():
            print(f"    {class_name}: {info['output_name']} (score: {info['score']:.4f})")
    
    return {
        'approach': 'adaptive',
        'segmentation_map': segmentation_map,
        'colored_map': colored_map,
        'present_classes': present_classes,
        'individual_class_maps': individual_class_maps,
        'best_output_per_class': best_output_per_class
    }

def compare_results(original_result, adaptive_result):
    """Compare the two results"""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\n📊 CLASS DETECTION:")
    print(f"  Original: {len(original_result['present_classes'])} classes - {original_result['present_classes']}")
    print(f"  Adaptive: {len(adaptive_result['present_classes'])} classes - {adaptive_result['present_classes']}")
    
    print(f"\n🖼️ INDIVIDUAL CLASS MAPS:")
    print(f"  Original: {len(original_result['individual_class_maps'])} maps")
    print(f"  Adaptive: {len(adaptive_result['individual_class_maps'])} maps")
    
    # Check if adaptive approach is actually better
    original_classes = set(original_result['present_classes'])
    adaptive_classes = set(adaptive_result['present_classes'])
    
    if adaptive_classes == original_classes:
        print(f"\n✅ CLASS DETECTION: Same classes detected")
    elif adaptive_classes.issuperset(original_classes):
        print(f"\n✅ CLASS DETECTION: Adaptive detected more classes")
    elif original_classes.issuperset(adaptive_classes):
        print(f"\n⚠️ CLASS DETECTION: Original detected more classes")
    else:
        print(f"\n❓ CLASS DETECTION: Different classes detected")
    
    # Check segmentation quality
    print(f"\n🔍 SEGMENTATION QUALITY:")
    
    # Count non-zero pixels in colored maps
    original_nonzero = np.sum(original_result['colored_map'] > 0)
    adaptive_nonzero = np.sum(adaptive_result['colored_map'] > 0)
    
    print(f"  Original non-zero pixels: {original_nonzero:,}")
    print(f"  Adaptive non-zero pixels: {adaptive_nonzero:,}")
    
    # Check if adaptive approach preserved structure
    if abs(original_nonzero - adaptive_nonzero) < original_nonzero * 0.1:  # Within 10%
        print(f"  ✅ Structure preservation: Good (within 10% difference)")
    else:
        print(f"  ⚠️ Structure preservation: Significant difference detected")
    
    # Final verdict
    print(f"\n🎯 VERDICT:")
    if adaptive_classes == original_classes and abs(original_nonzero - adaptive_nonzero) < original_nonzero * 0.1:
        print(f"  ✅ Adaptive approach maintains quality while adding intelligence")
        print(f"  ✅ Enhancement is working correctly")
    else:
        print(f"  ⚠️ Adaptive approach may be affecting quality")
        print(f"  ⚠️ Consider reviewing the enhancement logic")

def main():
    """Main comparison function"""
    print("🧪 COMPARING ORIGINAL vs ADAPTIVE APPROACHES")
    print("="*60)
    
    # Test original approach
    original_result = test_original_approach()
    if original_result is None:
        print("❌ Original test failed")
        return
    
    # Test adaptive approach
    adaptive_result = test_adaptive_approach()
    if adaptive_result is None:
        print("❌ Adaptive test failed")
        return
    
    # Compare results
    compare_results(original_result, adaptive_result)

if __name__ == '__main__':
    main()