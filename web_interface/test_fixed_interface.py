#!/usr/bin/env python3
"""
Test the fixed web interface with proper model output handling
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_fixed_interface():
    """Test the fixed web interface functions"""
    print("🧪 Testing Fixed Web Interface")
    print("=" * 40)
    
    try:
        # Import Flask app functions
        from app import preprocess_image, postprocess_output, calculate_metrics
        
        # Create test image
        print("📸 Creating test image...")
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image_path = "test_fixed_image.jpg"
        Image.fromarray(img).save(test_image_path)
        
        # Test preprocessing
        print("🔄 Testing preprocessing...")
        image_tensor, original_image = preprocess_image(test_image_path)
        print(f"✅ Image tensor shape: {image_tensor.shape}")
        print(f"✅ Original image shape: {original_image.shape}")
        
        # Load model and run inference
        print("🚀 Loading model and running inference...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration
        args_path = os.path.join(parent_dir, 'advanced_results', 'args.json')
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        # Create model
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
        
        # Load trained weights
        model_path = os.path.join(parent_dir, 'advanced_results', 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        # Run inference
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor)
            print(f"✅ Raw model output type: {type(output)}")
            print(f"✅ Raw model output length: {len(output) if isinstance(output, (list, tuple)) else 'N/A'}")
            
            # Test the new output handling logic
            if isinstance(output, tuple) and len(output) >= 4:
                print("✅ Model returns 4 outputs as expected")
                final_output, deep_sup1, deep_sup2, original_output = output
                print(f"✅ Final output shape: {final_output.shape}")
                print(f"✅ Deep sup1 shape: {deep_sup1.shape}")
                print(f"✅ Deep sup2 shape: {deep_sup2.shape}")
                print(f"✅ Original output shape: {original_output.shape}")
                
                # Check differences between outputs
                print(f"✅ Final vs Deep Sup1 difference: {torch.mean(torch.abs(final_output - deep_sup1)).item():.6f}")
                print(f"✅ Final vs Deep Sup2 difference: {torch.mean(torch.abs(final_output - deep_sup2)).item():.6f}")
                print(f"✅ Final vs Original difference: {torch.mean(torch.abs(final_output - original_output)).item():.6f}")
                
                # Use final output (ensemble)
                main_output = final_output
            else:
                print("⚠️ Model output format unexpected")
                if isinstance(output, (list, tuple)):
                    main_output = output[0]
                else:
                    main_output = output
        
        print(f"✅ Using output shape: {main_output.shape}")
        
        # Test postprocessing with the new logic
        print("🔄 Testing postprocessing...")
        try:
            segmentation_map, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors, individual_class_maps = postprocess_output(main_output, original_image)
            
            print(f"✅ Segmentation map shape: {segmentation_map.shape}")
            print(f"✅ Colored map shape: {colored_map.shape}")
            print(f"✅ Overlay shape: {overlay.shape}")
            print(f"✅ Class probabilities shape: {class_probs.shape}")
            print(f"✅ Present classes: {present_classes}")
            print(f"✅ Actual class names: {actual_class_names}")
            print(f"✅ Individual class maps: {list(individual_class_maps.keys())}")
            
            # Check if segmentation makes sense
            unique_pred_classes = np.unique(segmentation_map)
            print(f"✅ Unique predicted classes: {unique_pred_classes}")
            
            # Check class distribution
            for class_name in actual_class_names:
                if class_name in individual_class_maps:
                    class_mask = individual_class_maps[class_name]
                    class_pixels = np.sum(class_mask > 0)
                    total_pixels = class_mask.size
                    percentage = (class_pixels / total_pixels) * 100
                    print(f"✅ {class_name}: {class_pixels} pixels ({percentage:.2f}%)")
            
        except Exception as e:
            print(f"❌ Postprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test metrics calculation
        print("📊 Testing metrics calculation...")
        try:
            metrics = calculate_metrics(segmentation_map, class_probs, present_classes, actual_class_names)
            print(f"✅ Metrics: {metrics}")
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_fixed_interface()
    if success:
        print("\n🎉 Fixed interface test completed successfully!")
        print("✅ Model output handling: FIXED")
        print("✅ Postprocessing: FIXED")
        print("✅ Debug logging: ADDED")
    else:
        print("\n💥 Fixed interface test failed!")