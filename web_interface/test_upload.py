#!/usr/bin/env python3
"""
Test script to verify upload functionality
"""

import requests
import os
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a simple test MRI-like image"""
    # Create a 256x256 test image with some patterns
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Add some structure to make it look more like an MRI
    # Create a circular region in the center
    center = (128, 128)
    y, x = np.ogrid[:256, :256]
    mask = (x - center[0])**2 + (y - center[1])**2 <= 50**2
    img_array[mask] = [200, 200, 200]  # Light gray center
    
    # Add some noise
    noise = np.random.randint(-30, 30, (256, 256, 3))
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_upload():
    """Test the upload endpoint"""
    print("🧪 Testing upload functionality...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare files for upload
        files = {
            'file': ('test_mri.png', test_image, 'image/png')
        }
        
        print("🔄 Sending test image to server...")
        
        # Send request
        response = requests.post('http://localhost:5000/upload', files=files, timeout=60)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"📊 Response keys: {list(data.keys())}")
            
            if 'error' in data:
                print(f"❌ Server error in response: {data['error']}")
                return False
            else:
                print("✅ No errors in response")
                return True
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"❌ Error details: {error_data}")
            except:
                print(f"❌ Error text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the server running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout - server might be overloaded")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == '__main__':
    success = test_upload()
    if success:
        print("\n🎉 Upload test passed!")
    else:
        print("\n❌ Upload test failed!")