#!/usr/bin/env python3
"""
Test script to check prediction accuracy and debug hybrid approach
"""

import requests
import os
import numpy as np
from PIL import Image
import io
import json

def create_realistic_mri_image():
    """Create a more realistic MRI-like test image"""
    # Create a 256x256 test image
    img_array = np.random.randint(50, 100, (256, 256, 3), dtype=np.uint8)
    
    # Create brain-like structure
    center = (128, 128)
    y, x = np.ogrid[:256, :256]
    
    # Brain outline (larger circle)
    brain_mask = (x - center[0])**2 + (y - center[1])**2 <= 100**2
    img_array[brain_mask] = [120, 120, 120]  # Gray brain tissue
    
    # Background (outside brain)
    img_array[~brain_mask] = [20, 20, 20]  # Dark background
    
    # Add a small tumor-like region (should be small percentage)
    tumor_center = (140, 140)
    tumor_mask = (x - tumor_center[0])**2 + (y - tumor_center[1])**2 <= 15**2
    img_array[tumor_mask] = [200, 200, 200]  # Bright tumor
    
    # Add some edema around tumor (slightly larger)
    edema_mask = (x - tumor_center[0])**2 + (y - tumor_center[1])**2 <= 25**2
    edema_mask = edema_mask & ~tumor_mask  # Don't overlap with tumor
    img_array[edema_mask] = [150, 150, 150]  # Medium gray edema
    
    # Add some necrotic regions (small spots)
    necrotic_centers = [(100, 100), (160, 120), (120, 160)]
    for nec_center in necrotic_centers:
        nec_mask = (x - nec_center[0])**2 + (y - nec_center[1])**2 <= 8**2
        img_array[nec_mask] = [80, 80, 80]  # Dark necrotic
    
    # Add noise
    noise = np.random.randint(-20, 20, (256, 256, 3))
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_prediction_accuracy():
    """Test prediction accuracy with realistic MRI image"""
    print("🧪 Testing prediction accuracy with realistic MRI image...")
    
    try:
        # Create realistic test image
        test_image = create_realistic_mri_image()
        
        # Prepare files for upload
        files = {
            'file': ('realistic_mri.png', test_image, 'image/png')
        }
        
        print("🔄 Sending realistic MRI image to server...")
        
        # Send request
        response = requests.post('http://localhost:5000/upload', files=files, timeout=120)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            
            if 'error' in data:
                print(f"❌ Server error in response: {data['error']}")
                return False
            
            # Analyze results
            print("\n📊 Analysis Results:")
            print(f"Classes detected: {data.get('class_names', [])}")
            print(f"Present classes: {data.get('present_classes', [])}")
            
            # Check metrics
            metrics = data.get('metrics', {})
            print("\n📈 Metrics Analysis:")
            
            total_percentage = 0
            for class_name in data.get('class_names', []):
                percentage = metrics.get(f'{class_name}_percentage', 0)
                confidence = metrics.get(f'{class_name}_confidence', 0)
                total_percentage += percentage
                print(f"  {class_name}: {percentage}% (confidence: {confidence}%)")
            
            print(f"\nTotal percentage: {total_percentage}%")
            
            # Check for unrealistic results
            tumor_percentage = metrics.get('Tumor_percentage', 0)
            background_percentage = metrics.get('Background_percentage', 0)
            
            print(f"\n🔍 Reality Check:")
            print(f"  Tumor percentage: {tumor_percentage}%")
            print(f"  Background percentage: {background_percentage}%")
            
            # Flag unrealistic results
            if tumor_percentage > 50:
                print("⚠️  WARNING: Tumor percentage seems too high (>50%)")
            if background_percentage < 20:
                print("⚠️  WARNING: Background percentage seems too low (<20%)")
            if total_percentage < 95 or total_percentage > 105:
                print("⚠️  WARNING: Total percentage should be close to 100%")
            
            # Check if results make sense
            if tumor_percentage > 80:
                print("❌ CRITICAL: Tumor percentage is unrealistically high (>80%)")
                print("   This suggests a problem with the model or hybrid approach")
                return False
            elif tumor_percentage < 1:
                print("❌ CRITICAL: Tumor percentage is unrealistically low (<1%)")
                print("   This suggests the model is not detecting tumors properly")
                return False
            else:
                print("✅ Tumor percentage seems reasonable")
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
    success = test_prediction_accuracy()
    if success:
        print("\n🎉 Prediction accuracy test passed!")
    else:
        print("\n❌ Prediction accuracy test failed!")
        print("   The model may have issues with hybrid approach or class mapping")