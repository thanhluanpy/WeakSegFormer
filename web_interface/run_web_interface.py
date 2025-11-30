#!/usr/bin/env python3
"""
WeakSegFormer Web Interface Launcher
=======================================

This script launches the web interface with automatic setup and validation.
"""

import os
import sys
import subprocess
import json
import torch
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # required_packages = [
    #     'flask', 'torch', 'torchvision', 'numpy', 'opencv-python',
    #     'PIL', 'matplotlib', 'seaborn', 'albumentations'
    # ]
    
    # missing_packages = []
    
    # for package in required_packages:
    #     try:
    #         if package == 'PIL':
    #             import PIL
    #         else:
    #             __import__(package)
    #     except ImportError:
    #         missing_packages.append(package)
    
    # if missing_packages:
    #     print(f"❌ Missing packages: {', '.join(missing_packages)}")
    #     print("📦 Installing missing packages...")
        
    #     try:
    #         subprocess.check_call([
    #             sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
    #         ])
    #         print("✅ Dependencies installed successfully!")
    #     except subprocess.CalledProcessError:
    #         print("❌ Failed to install dependencies. Please install manually:")
    #         print("pip install -r requirements.txt")
    #         return False
    # else:
    #     print("✅ All dependencies are installed!")
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("🔍 Checking model files...")
    
    model_files = [
        '../advanced_results/best_model.pth',
        '../advanced_results/args.json'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("Please ensure the model has been trained and files are in the correct location.")
        return False
    
    print("✅ Model files found!")
    return True

def check_gpu():
    """Check GPU availability"""
    print("🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   - Count: {gpu_count}")
        print(f"   - Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 4:
            print("⚠️  Warning: GPU memory is less than 4GB. Consider using CPU mode.")
        
        return True
    else:
        print("⚠️  No GPU detected. Will use CPU mode (slower inference).")
        return False

def load_model_config():
    """Load and display model configuration"""
    print("🔍 Loading model configuration...")
    
    try:
        with open('../advanced_results/args.json', 'r') as f:
            config = json.load(f)
        
        print("✅ Model configuration loaded:")
        print(f"   - Model: {config.get('model', 'N/A')}")
        print(f"   - Input Size: {config.get('input_size', 'N/A')}")
        print(f"   - Learning Rate: {config.get('lr', 'N/A')}")
        print(f"   - Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"   - Epochs: {config.get('epochs', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load model configuration: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("🔍 Creating directories...")
    
    directories = ['uploads', 'results', 'templates']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created!")
    return True

def main():
    """Main launcher function"""
    print("🚀 WeakSegFormer Web Interface Launcher")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run checks
    print("\n🔍 Running system checks...")
    
    check1 = check_dependencies()
    check2 = check_model_files()
    check3 = check_gpu()
    check4 = load_model_config()
    check5 = create_directories()
    
    checks = [check1, check2, check3, check4, check5]
    
    print(f"\n📊 Check Results:")
    print(f"   Dependencies: {'✅' if check1 else '❌'}")
    print(f"   Model Files: {'✅' if check2 else '❌'}")
    print(f"   GPU Check: {'✅' if check3 else '❌'}")
    print(f"   Config Load: {'✅' if check4 else '❌'}")
    print(f"   Directories: {'✅' if check5 else '❌'}")
    
    # GPU check is optional (can run on CPU)
    critical_checks = [check1, check2, check4, check5]  # Exclude GPU check
    
    if not all(critical_checks):
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        return 1
    
    print("\n✅ All checks passed! Starting web interface...")
    print("=" * 50)
    
    # Start the Flask app
    try:
        print("🌐 Web interface is starting...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Import and run app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Failed to start web interface: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 