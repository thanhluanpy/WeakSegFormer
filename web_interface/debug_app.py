#!/usr/bin/env python3
"""
Debug script to identify import issues
"""

import os
import sys
import traceback

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def debug_imports():
    """Debug import issues step by step"""
    print("🔍 Debugging imports step by step...")
    
    try:
        print("1. Testing basic imports...")
        import torch
        import numpy as np
        import cv2
        from PIL import Image
        import matplotlib.pyplot as plt
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from flask import Flask
        print("✅ Basic imports successful")
        
        print("2. Testing local model imports...")
        import models_enhanced
        print("✅ models_enhanced imported")
        
        print("3. Testing datasets import...")
        from datasets import build_transform
        print("✅ build_transform imported")
        
        print("4. Testing app.py imports...")
        # Test each import in app.py
        import io
        import base64
        import json
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch
        import seaborn as sns
        from datetime import datetime
        import uuid
        from werkzeug.utils import secure_filename
        print("✅ All app.py imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def debug_model_loading():
    """Debug model loading issues"""
    print("\n🔍 Debugging model loading...")
    
    try:
        import json
        import torch
        
        # Check model files
        args_file = os.path.join(parent_dir, 'advanced_results', 'args.json')
        model_file = os.path.join(parent_dir, 'advanced_results', 'best_model.pth')
        
        print(f"Args file exists: {os.path.exists(args_file)}")
        print(f"Model file exists: {os.path.exists(model_file)}")
        
        if not os.path.exists(args_file):
            print(f"❌ Args file not found: {args_file}")
            return False
            
        if not os.path.exists(model_file):
            print(f"❌ Model file not found: {model_file}")
            return False
        
        # Load config
        with open(args_file, 'r') as f:
            args = json.load(f)
        print("✅ Config loaded")
        
        # Test model creation
        import models_enhanced
        device = torch.device('cpu')
        
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
        
        # Test model loading
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        print("✅ Model loaded and ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def debug_app_creation():
    """Debug Flask app creation"""
    print("\n🔍 Debugging Flask app creation...")
    
    try:
        # Test Flask app creation
        from flask import Flask
        app = Flask(__name__)
        print("✅ Flask app created")
        
        # Test app configuration
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
        app.config['UPLOAD_FOLDER'] = 'uploads'
        app.config['RESULTS_FOLDER'] = 'results'
        print("✅ App config set")
        
        # Test directory creation
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        print("✅ Directories created")
        
        return True
        
    except Exception as e:
        print(f"❌ App creation error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🐛 Enhanced WeakTR Web Interface - Debug Mode")
    print("=" * 50)
    
    # Test imports
    if not debug_imports():
        print("\n❌ Import debugging failed!")
        return 1
    
    # Test model loading
    if not debug_model_loading():
        print("\n❌ Model loading debugging failed!")
        return 1
    
    # Test app creation
    if not debug_app_creation():
        print("\n❌ App creation debugging failed!")
        return 1
    
    print("\n✅ All debug tests passed!")
    print("The issue might be in the app.py file itself.")
    print("Try running: python app.py directly")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 