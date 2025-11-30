#!/usr/bin/env python3
"""
Simple test script to test the upload functionality
"""

import requests
import os
import sys

def test_upload():
    """Test the upload endpoint with a simple image"""
    
    # URL of the web interface
    base_url = "http://localhost:5000"
    
    print("🔄 Testing web interface...")
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Model loaded: {data['model']['loaded']}")
            print(f"   Overall healthy: {data['overall_healthy']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Debug model
    print("\n2️⃣ Testing debug model...")
    try:
        response = requests.get(f"{base_url}/debug/model")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Debug model passed")
            print(f"   Model status: {data['model_status']['loaded']}")
            print(f"   Test inference: {data['test_inference']['status']}")
        else:
            print(f"❌ Debug model failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Debug model error: {e}")
        return False
    
    # Test 3: Upload a test image
    print("\n3️⃣ Testing file upload...")
    
    # Check if we have a test image
    test_image_path = "simple_test_data/images/BraTS-GLI-00000-000_slice050.jpg"
    if not os.path.exists(test_image_path):
        print(f"⚠️ Test image not found: {test_image_path}")
        print("   Creating a dummy test image...")
        
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        # Create a 256x256 test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        test_image.save("test_dummy_image.jpg")
        test_image_path = "test_dummy_image.jpg"
        print(f"   Created dummy image: {test_image_path}")
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/upload", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload successful!")
            print(f"   Filename: {data['filename']}")
            print(f"   Class names: {data['class_names']}")
            print(f"   Metrics: {list(data['metrics'].keys())}")
            
            # Clean up dummy image if we created it
            if test_image_path == "test_dummy_image.jpg":
                os.remove(test_image_path)
                print("   Cleaned up dummy image")
                
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting web interface test...")
    
    success = test_upload()
    
    if success:
        print("\n🎉 All tests passed! Web interface is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the logs above for details.")
        sys.exit(1) 