#!/usr/bin/env python3
"""
Simple inference test using Flask app functions
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_simple_inference():
    """Test simple inference using Flask app functions"""
    print("🧪 Testing Simple Inference")
    print("=" * 30)
    
    try:
        # Import Flask app functions
        from app import preprocess_image, postprocess_output, calculate_metrics
        
        # Create test image
        print("📸 Creating test image...")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :] = [100, 100, 100]  # Simple gray image
        test_image_path = "simple_test.jpg"
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
            if isinstance(output, tuple):
                main_output = output[0]
            else:
                main_output = output
        
        print(f"✅ Model output shape: {main_output.shape}")
        
        # Test postprocessing
        print("🔄 Testing postprocessing...")
        segmentation_map, colored_map, overlay, class_probs, present_classes, actual_class_names, actual_class_colors = postprocess_output(main_output, original_image)
        
        print(f"✅ Segmentation map shape: {segmentation_map.shape}")
        print(f"✅ Colored map shape: {colored_map.shape}")
        print(f"✅ Overlay shape: {overlay.shape}")
        print(f"✅ Class probabilities shape: {class_probs.shape}")
        print(f"✅ Present classes: {present_classes}")
        print(f"✅ Actual class names: {actual_class_names}")
        
        # Test metrics calculation
        print("📊 Testing metrics calculation...")
        metrics = calculate_metrics(segmentation_map, class_probs, present_classes, actual_class_names)
        print(f"✅ Metrics: {metrics}")
        
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
    import json
    success = test_simple_inference()
    if success:
        print("\n🎉 Simple inference test completed successfully!")
    else:
        print("\n💥 Simple inference test failed!") 