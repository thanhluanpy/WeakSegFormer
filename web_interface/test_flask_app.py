#!/usr/bin/env python3
"""
Test Flask app and model loading
"""

import os
import sys
import requests
import time

def test_flask_app():
    """Test Flask app functionality"""
    print("🧪 Testing Flask App")
    print("=" * 40)
    
    try:
        # Start Flask app in background
        import subprocess
        import threading
        
        def run_flask():
            subprocess.run([sys.executable, 'app.py'], cwd=os.path.dirname(__file__))
        
        # Start Flask in background
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # Wait for Flask to start
        print("⏳ Waiting for Flask app to start...")
        time.sleep(5)
        
        # Test health endpoint
        print("🔍 Testing health endpoint...")
        response = requests.get('http://localhost:5000/health', timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            
            if health_data.get('model_loaded'):
                print("✅ Model is loaded!")
            else:
                print("❌ Model is not loaded!")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test main page
        print("🔍 Testing main page...")
        response = requests.get('http://localhost:5000/', timeout=10)
        
        if response.status_code == 200:
            print("✅ Main page loaded successfully!")
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    success = test_flask_app()
    if success:
        print("\n🎉 Flask app test passed!")
    else:
        print("\n💥 Flask app test failed!")
        sys.exit(1) 