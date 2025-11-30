#!/usr/bin/env python3
"""
Test script to verify server is running with updated color code
"""

import requests
import json
import base64
import io
from PIL import Image
import numpy as np

def test_server_colors():
    """Test if server is running with updated color code"""
    
    print("🧪 Testing Server Color Implementation")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            health_data = response.json()
            print(f"   - Status: {health_data.get('status', 'unknown')}")
            print(f"   - Model loaded: {health_data.get('model', {}).get('loaded', False)}")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Please make sure server is running on http://localhost:5000")
        return False
    
    # Test 2: Check debug endpoint
    try:
        response = requests.get('http://localhost:5000/debug/model', timeout=10)
        if response.status_code == 200:
            print("✅ Debug endpoint accessible")
            debug_data = response.json()
            print(f"   - Model type: {debug_data.get('model_status', {}).get('type', 'unknown')}")
            print(f"   - Test inference: {debug_data.get('test_inference', {}).get('status', 'unknown')}")
        else:
            print(f"❌ Debug endpoint returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Debug endpoint failed: {e}")
    
    # Test 3: Create a test image and upload it
    print("\n🔍 Testing with a sample image...")
    
    # Create a simple test image
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    test_image[100:150, 100:150] = [255, 255, 255]  # White square
    
    # Convert to PIL Image
    pil_image = Image.fromarray(test_image)
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Test upload
    try:
        files = {'file': ('test_image.png', img_buffer, 'image/png')}
        response = requests.post('http://localhost:5000/upload', files=files, timeout=30)
        
        if response.status_code == 200:
            print("✅ Upload successful")
            result_data = response.json()
            
            # Check if we have the expected data structure
            print(f"   - Has class_names: {'class_names' in result_data}")
            print(f"   - Has class_colors: {'class_colors' in result_data}")
            print(f"   - Has individual_class_maps: {'individual_class_maps' in result_data}")
            
            if 'class_names' in result_data:
                print(f"   - Class names: {result_data['class_names']}")
            
            if 'class_colors' in result_data:
                print(f"   - Class colors: {result_data['class_colors']}")
                
                # Check if colors match expected mapping
                expected_colors = {
                    'Background': '#000000',
                    'Necrotic': '#8B0000', 
                    'Edema': '#228B22',
                    'Tumor': '#4169E1'
                }
                
                print("\n🔍 Color Consistency Check:")
                for i, class_name in enumerate(result_data.get('class_names', [])):
                    if i < len(result_data['class_colors']):
                        actual_color = result_data['class_colors'][i]
                        expected_color = expected_colors.get(class_name, 'unknown')
                        match = actual_color == expected_color
                        print(f"   - {class_name}: {actual_color} {'✅' if match else '❌'} (expected: {expected_color})")
                    else:
                        print(f"   - {class_name}: No color provided")
            
            # Check individual class maps
            if 'individual_class_maps' in result_data:
                print(f"\n🔍 Individual Class Maps:")
                for class_name, class_map_b64 in result_data['individual_class_maps'].items():
                    print(f"   - {class_name}: Available")
                    
                    # Decode and analyze the image
                    try:
                        img_data = base64.b64decode(class_map_b64)
                        img = Image.open(io.BytesIO(img_data))
                        img_array = np.array(img)
                        
                        # Get unique colors
                        unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)
                        print(f"     Unique colors: {unique_colors.tolist()}")
                        
                        # Check if it has the expected class color
                        expected_color = expected_colors.get(class_name, None)
                        if expected_color:
                            # Convert hex to RGB
                            if expected_color.startswith('#'):
                                r = int(expected_color[1:3], 16)
                                g = int(expected_color[3:5], 16)
                                b = int(expected_color[5:7], 16)
                                expected_rgb = [r, g, b]
                            else:
                                expected_rgb = [128, 128, 128]
                            
                            has_expected_color = np.any(np.all(img_array == expected_rgb, axis=2))
                            print(f"     Has expected color {expected_rgb}: {'✅' if has_expected_color else '❌'}")
                        
                    except Exception as e:
                        print(f"     Error analyzing image: {e}")
            
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    print("\n✅ Server test completed!")
    return True

if __name__ == "__main__":
    test_server_colors()