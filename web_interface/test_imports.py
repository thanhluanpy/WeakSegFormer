#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported successfully")
    except ImportError as e:
        print(f"❌ opencv-python import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print("✅ albumentations imported successfully")
    except ImportError as e:
        print(f"❌ albumentations import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    # Test local imports
    try:
        import models_enhanced
        print("✅ models_enhanced imported successfully")
    except ImportError as e:
        print(f"❌ models_enhanced import failed: {e}")
        return False
    
    try:
        from datasets import build_transform
        print("✅ build_transform imported successfully")
    except ImportError as e:
        print(f"❌ build_transform import failed: {e}")
        print("⚠️  Using fallback build_transform function")
    
    return True

def test_model_loading():
    """Test if model can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        import json
        import torch
        
        # Check if model files exist
        args_file = os.path.join(parent_dir, 'advanced_results', 'args.json')
        model_file = os.path.join(parent_dir, 'advanced_results', 'best_model.pth')
        
        if not os.path.exists(args_file):
            print(f"❌ Model config file not found: {args_file}")
            return False
        
        if not os.path.exists(model_file):
            print(f"❌ Model weights file not found: {model_file}")
            return False
        
        print("✅ Model files found")
        
        # Load config
        with open(args_file, 'r') as f:
            args = json.load(f)
        print("✅ Model config loaded")
        
        # Test model creation (without loading weights)
        import models_enhanced
        device = torch.device('cpu')  # Use CPU for testing
        
        model = models_enhanced.deit_small_EnhancedWeakTr_patch16_224(
            pretrained=False,
            num_classes=4,
            drop_rate=args.get('drop', 0.4),
            drop_path_rate=args.get('drop_path', 0.3),
            reduction=args.get('reduction', 4),
            pool_type=args.get('pool_type', 'avg'),
            feat_reduction=args.get('feat_reduction', 4)
        )
        print("✅ Model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Enhanced WeakTR Web Interface - Import Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed!")
        return 1
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed!")
        return 1
    
    print("\n✅ All tests passed! Web interface should work correctly.")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 