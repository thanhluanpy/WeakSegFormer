#!/usr/bin/env python3
"""
Debug inference to find exact location of index out of bounds error
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def debug_inference():
    """Debug inference step by step"""
    print("🔍 Debugging Inference Step by Step")
    print("=" * 40)
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
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
        
        print("✅ Model loaded successfully!")
        
        # Create test image
        print("📸 Creating test image...")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :] = [100, 100, 100]  # Simple gray image
        test_image_path = "debug_test.jpg"
        Image.fromarray(img).save(test_image_path)
        
        # Preprocess image
        print("🔄 Preprocessing image...")
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ToTensorV2(),
        ])
        
        transformed = transform(image=img)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        print(f"✅ Image tensor shape: {image_tensor.shape}")
        
        # Run inference
        print("🚀 Running inference...")
        with torch.no_grad():
            output = model(image_tensor)
            if isinstance(output, tuple):
                main_output = output[0]
            else:
                main_output = output
        
        print(f"✅ Model output shape: {main_output.shape}")
        
        # Step 1: Apply softmax
        print("📊 Step 1: Applying softmax...")
        probs = F.softmax(main_output, dim=1)
        print(f"✅ Probabilities shape: {probs.shape}")
        
        # Step 2: Get predicted classes
        print("🎯 Step 2: Getting predicted classes...")
        pred_class = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        print(f"✅ Predicted classes shape: {pred_class.shape}")
        
        # Step 3: Find unique classes
        print("🔍 Step 3: Finding unique classes...")
        unique_classes = np.unique(pred_class)
        present_classes = sorted(unique_classes.tolist())
        print(f"✅ Present classes: {present_classes}")
        
        # Step 4: Create class names and colors
        print("🎨 Step 4: Creating class names and colors...")
        class_names = ['Background', 'Necrotic', 'Edema', 'Tumor']
        class_colors = ['#000000', '#FF0000', '#00FF00', '#0000FF']
        
        actual_class_colors = [class_colors[i] for i in present_classes]
        actual_class_names = [class_names[i] for i in present_classes]
        print(f"✅ Actual class names: {actual_class_names}")
        print(f"✅ Actual class colors: {actual_class_colors}")
        
        # Step 5: Create colored map
        print("🖼️ Step 5: Creating colored map...")
        colored_map = np.zeros((256, 256, 3), dtype=np.uint8)
        
        for old_idx in present_classes:
            color = class_colors[old_idx]
            mask = (pred_class == old_idx)
            colored_map[mask] = [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
        
        print(f"✅ Colored map shape: {colored_map.shape}")
        
        # Step 6: Calculate class probabilities
        print("📈 Step 6: Calculating class probabilities...")
        class_probs = probs.squeeze().cpu().numpy()
        print(f"✅ Class probabilities shape: {class_probs.shape}")
        
        # Step 7: Extract present class probabilities
        print("📊 Step 7: Extracting present class probabilities...")
        present_class_probs = class_probs[present_classes]
        print(f"✅ Present class probabilities shape: {present_class_probs.shape}")
        
        # Step 8: Calculate metrics
        print("📊 Step 8: Calculating metrics...")
        metrics = {}
        total_pixels = pred_class.shape[0] * pred_class.shape[1]
        
        for i, class_name in enumerate(actual_class_names):
            original_class_idx = present_classes[i]
            class_pixels = np.sum(pred_class == original_class_idx)
            percentage = (class_pixels / total_pixels) * 100
            metrics[f'{class_name}_percentage'] = round(percentage, 2)
            print(f"   - {class_name}: {percentage:.2f}%")
        
        for i, class_name in enumerate(actual_class_names):
            avg_confidence = np.mean(present_class_probs[i])
            metrics[f'{class_name}_confidence'] = round(avg_confidence * 100, 2)
            print(f"   - {class_name} confidence: {avg_confidence*100:.2f}%")
        
        print("✅ All steps completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import json
    success = debug_inference()
    if success:
        print("\n🎉 Debug completed successfully!")
    else:
        print("\n💥 Debug failed!") 