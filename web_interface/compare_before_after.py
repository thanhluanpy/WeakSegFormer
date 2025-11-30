#!/usr/bin/env python3
"""
Compare segmentation results before and after fixes
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

def test_old_vs_new_approach():
    """Compare old approach (using only first output) vs new approach (using ensemble)"""
    print("🔍 Comparing Old vs New Approach")
    print("=" * 50)
    
    try:
        # Create test image
        print("📸 Creating test image...")
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image_path = "comparison_test.jpg"
        Image.fromarray(img).save(test_image_path)
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args_path = os.path.join(parent_dir, 'advanced_results', 'args.json')
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        import models_enhanced
        model = models_enhanced.deit_small_EnhancedWeakTr_patch16_224(
            pretrained=False,
            num_classes=4,
            drop_rate=args.get('drop', 0.4),
            drop_path_rate=args.get('drop_path', 0.3),
            reduction=args.get('reduction', 4),
            pool_type=args.get('pool_type', 'avg'),
            feat_reduction=args.get('feat_reduction', 4)
        )
        
        model_path = os.path.join(parent_dir, 'advanced_results', 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        # Preprocess image
        from app import preprocess_image
        image_tensor, original_image = preprocess_image(test_image_path)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            print(f"✅ Model output type: {type(output)}")
            print(f"✅ Model output length: {len(output)}")
            
            # OLD APPROACH: Use only first output
            print("\n🔴 OLD APPROACH (First Output Only):")
            old_output = output[0]
            print(f"✅ Old output shape: {old_output.shape}")
            print(f"✅ Old output range: {old_output.min().item():.4f} to {old_output.max().item():.4f}")
            
            # Apply softmax to old output
            old_probs = torch.softmax(old_output, dim=1)
            old_class_means = torch.mean(old_probs, dim=[0, 2, 3])
            print(f"✅ Old class probabilities: {old_class_means.cpu().numpy()}")
            
            # NEW APPROACH: Use ensemble final output
            print("\n🟢 NEW APPROACH (Ensemble Final Output):")
            final_output, deep_sup1, deep_sup2, original_output = output
            print(f"✅ Final output shape: {final_output.shape}")
            print(f"✅ Final output range: {final_output.min().item():.4f} to {final_output.max().item():.4f}")
            
            # Apply softmax to final output
            new_probs = torch.softmax(final_output, dim=1)
            new_class_means = torch.mean(new_probs, dim=[0, 2, 3])
            print(f"✅ New class probabilities: {new_class_means.cpu().numpy()}")
            
            # Compare outputs
            print("\n📊 COMPARISON:")
            print(f"✅ Difference between old and new: {torch.mean(torch.abs(old_output - final_output)).item():.6f}")
            
            # Check individual class differences
            class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
            for i, class_name in enumerate(class_names):
                old_prob = old_class_means[i].item()
                new_prob = new_class_means[i].item()
                diff = abs(old_prob - new_prob)
                print(f"✅ {class_name}: Old={old_prob:.4f}, New={new_prob:.4f}, Diff={diff:.4f}")
            
            # Test postprocessing with both approaches
            print("\n🔄 Testing Postprocessing:")
            
            # Old approach postprocessing
            from app import postprocess_output
            try:
                old_seg, old_colored, old_overlay, old_probs, old_present, old_names, old_colors, old_individual = postprocess_output(old_output, original_image)
                print(f"✅ Old postprocessing successful")
                print(f"✅ Old present classes: {old_present}")
                print(f"✅ Old class names: {old_names}")
            except Exception as e:
                print(f"❌ Old postprocessing failed: {e}")
            
            # New approach postprocessing
            try:
                new_seg, new_colored, new_overlay, new_probs, new_present, new_names, new_colors, new_individual = postprocess_output(final_output, original_image)
                print(f"✅ New postprocessing successful")
                print(f"✅ New present classes: {new_present}")
                print(f"✅ New class names: {new_names}")
            except Exception as e:
                print(f"❌ New postprocessing failed: {e}")
            
            # Compare segmentation results
            if 'old_seg' in locals() and 'new_seg' in locals():
                print(f"\n📈 Segmentation Comparison:")
                print(f"✅ Old segmentation unique classes: {np.unique(old_seg)}")
                print(f"✅ New segmentation unique classes: {np.unique(new_seg)}")
                
                # Check if segmentations are different
                if np.array_equal(old_seg, new_seg):
                    print("⚠️ Warning: Old and new segmentations are identical!")
                else:
                    print("✅ Old and new segmentations are different (as expected)")
                    diff_pixels = np.sum(old_seg != new_seg)
                    total_pixels = old_seg.size
                    diff_percentage = (diff_pixels / total_pixels) * 100
                    print(f"✅ Different pixels: {diff_pixels}/{total_pixels} ({diff_percentage:.2f}%)")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        print("\n✅ Comparison completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_old_vs_new_approach()
    if success:
        print("\n🎉 Comparison test completed successfully!")
        print("✅ The fixes should improve segmentation quality")
    else:
        print("\n💥 Comparison test failed!")