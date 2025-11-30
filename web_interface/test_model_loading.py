#!/usr/bin/env python3
"""
Test model loading independently
"""

import os
import sys
import torch
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_model_loading():
    """Test model loading step by step"""
    print("🧪 Testing Model Loading")
    print("=" * 40)
    
    try:
        # 1. Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
        
        # 2. Check model files
        args_path = os.path.join(parent_dir, 'advanced_results', 'args.json')
        model_path = os.path.join(parent_dir, 'advanced_results', 'best_model.pth')
        
        print(f"Args file exists: {os.path.exists(args_path)}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(args_path):
            print("❌ Args file not found!")
            return False
            
        if not os.path.exists(model_path):
            print("❌ Model file not found!")
            return False
        
        # 3. Load config
        with open(args_path, 'r') as f:
            args = json.load(f)
        print("✅ Config loaded")
        
        # 4. Import model
        import models_enhanced
        print("✅ models_enhanced imported")
        
        # 5. Create model
        model = models_enhanced.deit_small_EnhancedWeakTr_patch16_224(
            pretrained=False,
            num_classes=4,
            drop_rate=args.get('drop', 0.4),
            drop_path_rate=args.get('drop_path', 0.3),
            reduction=args.get('reduction', 4),
            pool_type=args.get('pool_type', 'avg'),
            feat_reduction=args.get('feat_reduction', 4)
        )
        print("✅ Model created")
        
        # 6. Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        print("✅ Model loaded and ready")
        
        # 7. Test inference
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
            print(f"✅ Inference test passed. Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test passed!")
    else:
        print("\n💥 Model loading test failed!")
        sys.exit(1) 