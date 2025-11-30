#!/usr/bin/env python3
"""
Debug actual model output after loading weights
"""

import os
import sys
import torch
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def debug_actual_output():
    """Debug actual model output after loading weights"""
    print("🔍 Debugging Actual Model Output")
    print("=" * 40)
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model configuration
        args_path = os.path.join(parent_dir, 'advanced_results', 'args.json')
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        # Create model with 4 classes
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
        
        print("✅ Model created with 4 classes")
        
        # Check model structure before loading weights
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
            print(f"Output shape before loading weights: {output.shape}")
        
        # Load trained weights
        model_path = os.path.join(parent_dir, 'advanced_results', 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"Checkpoint keys: {checkpoint.keys()}")
        print(f"Model state dict keys: {list(checkpoint['model'].keys())}")
        
        # Check if there are any mismatches
        model_dict = model.state_dict()
        checkpoint_dict = checkpoint['model']
        
        print(f"Model state dict has {len(model_dict)} keys")
        print(f"Checkpoint has {len(checkpoint_dict)} keys")
        
        # Check for key mismatches
        model_keys = set(model_dict.keys())
        checkpoint_keys = set(checkpoint_dict.keys())
        
        missing_in_checkpoint = model_keys - checkpoint_keys
        missing_in_model = checkpoint_keys - model_keys
        
        if missing_in_checkpoint:
            print(f"Keys missing in checkpoint: {missing_in_checkpoint}")
        if missing_in_model:
            print(f"Keys missing in model: {missing_in_model}")
        
        # Load weights
        model.load_state_dict(checkpoint['model'])
        print("✅ Weights loaded successfully!")
        
        # Test output after loading weights
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]
            print(f"Output shape after loading weights: {output.shape}")
            print(f"Number of classes in output: {output.shape[1]}")
            
            # Check if output has 4 classes
            if output.shape[1] == 4:
                print("✅ Model outputs 4 classes as expected")
                return 4
            elif output.shape[1] == 3:
                print("⚠️ Model outputs 3 classes (not 4)")
                return 3
            else:
                print(f"❌ Unexpected number of classes: {output.shape[1]}")
                return output.shape[1]
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    num_classes = debug_actual_output()
    print(f"\n🎯 Model actually outputs {num_classes} classes") 