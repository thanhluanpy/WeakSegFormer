#!/usr/bin/env python3
"""
Simple test for Flask app
"""

import requests
import time
import subprocess
import sys
import os

def test_flask():
    """Test Flask app directly"""
    print("🧪 Simple Flask Test")
    print("=" * 30)
    
    # Start Flask app
    print("🚀 Starting Flask app...")
    flask_process = subprocess.Popen([
        sys.executable, 'app.py'
    ], cwd=os.path.dirname(__file__))
    
    # Wait for Flask to start
    print("⏳ Waiting for Flask to start...")
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("🔍 Testing health endpoint...")
        response = requests.get('http://localhost:5000/health', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health response: {data}")
            
            if data.get('model_loaded'):
                print("✅ Model is loaded!")
            else:
                print("❌ Model is not loaded!")
                print(f"Model type: {data.get('model_type')}")
                print(f"Device: {data.get('device')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Test main page
        print("🔍 Testing main page...")
        response = requests.get('http://localhost:5000/', timeout=10)
        
        if response.status_code == 200:
            print("✅ Main page loaded!")
        else:
            print(f"❌ Main page failed: {response.status_code}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Stop Flask
        print("🛑 Stopping Flask...")
        flask_process.terminate()
        flask_process.wait()

if __name__ == '__main__':
    test_flask() 